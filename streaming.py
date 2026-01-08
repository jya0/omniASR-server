"""
StreamingTranscriber - Gold Standard Audio Pipeline.

Features:
- "Throttling" to prevent GPU overload (0.35s interval)
- "Hysteresis" to prevent cutting words on short silence (0.6s)
- "Pre-roll" to preserve word starts after silence (0.25s)
- "DeepFilterNet" & "Pedalboard" for audio cleaning
"""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable
import numpy as np
import time

from config import config
from audio_buffer import VAD
from audio_processor import AudioProcessor
from model import ASRModel
from local_agreement import LocalAgreement

import torch
# from df.enhance import init_df # Removed direct usage here to keep clean, handled inside class if needed
from df.enhance import init_df, df_features
from df.utils import as_complex
from df.model import ModelParams
import os
import io
import base64
import soundfile as sf

# Global DF model instance (singleton)
_DF_MODEL = None

class OutputFilter:
    """Filters hallucinations and nonsensical output."""
    BLOCKLIST = [] # (Kept simple for this iteration)
    
    @staticmethod
    def is_valid_text(text: str) -> bool:
        if not text: return True
        # Basic hallucination filter
        if len(text) > 0 and text == text[0] * len(text) and len(text) > 10:
             return False
        return True

def get_df_model():
    """Get or initialize the DeepFilterNet model globally."""
    global _DF_MODEL
    if _DF_MODEL is None:
        if "XDG_CACHE_HOME" not in os.environ:
             base_path = os.path.dirname(os.path.abspath(__file__))
             os.environ["XDG_CACHE_HOME"] = os.path.join(base_path, "models-cache")
        model, _, _ = init_df(model_base_dir="DeepFilterNet3", config_allow_defaults=True)
        device = config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
        _DF_MODEL = model.to(device)
        _DF_MODEL.eval()
    return _DF_MODEL


@dataclass
class StreamingResult:
    """Result from streaming transcription."""
    text: str                  # Current hypothesis (full buffer)
    confirmed_text: str        # Actually stable/committed text (from previous sentences)
    pending_text: str          # Current changing text
    is_final: bool             # Is this a committed final sentence?
    latency_ms: float = 0      
    audio_duration: float = 0  
    debug_audio_file: str = None 
    debug_audio_file_vad: str = None 


@dataclass
class StreamingTranscriber:
    """
    Gold Standard Streaming ASR.
    
    Logic:
    1. Accumulate audio in a linear buffer.
    2. Run ASR every `inference_interval` (0.35s) -> Partial Update.
    3. If VAD detects silence > `min_silence_duration` (0.6s) -> Final Commit & Reset.
    4. On Reset, keep `preroll_duration` (0.25s) to catch next word start.
    """

    model: ASRModel = None
    vad: VAD = None
    processor: AudioProcessor = None
    local_agreement: LocalAgreement = None
    
    # DeepFilterNet state
    df_state: object = None

    # Callbacks
    on_result: Callable[[StreamingResult], None] = None

    # State variables
    _started: bool = False
    
    # Audio Buffers
    _audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    
    # Throttling & Hysteresis State
    _silence_start_time: float = None
    _last_inference_time: float = 0.0
    _confirmed_text_history: str = "" # Accumulate confirmed sentences
    _has_speech_in_buffer: bool = False # Track if current buffer contains ANY speech
    
    # Debug
    _debug_audio: list = field(default_factory=list)

    def __post_init__(self):
        print("DEBUG: ProductionStreamer Active (Gold Standard)")
        if self.model is None:
            self.model = ASRModel.get_instance()
        if self.vad is None:
            self.vad = VAD()
        if self.processor is None:
            self.processor = AudioProcessor()
        if self.local_agreement is None:
            self.local_agreement = LocalAgreement()

    def start(self) -> None:
        """Start a new streaming session."""
        self.model.ensure_loaded()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._silence_start_time = None
        self._last_inference_time = 0.0
        self._confirmed_text_history = ""
        self._has_speech_in_buffer = False
        self._debug_audio = []
        
        # Initialize DF State
        if config.noise_removal.enabled:
            _, self.df_state, _ = init_df(model_base_dir="DeepFilterNet3", config_allow_defaults=True)
        else:
            self.df_state = None
            
        self.local_agreement.reset()
        
        self._started = True

    def process(self, audio: np.ndarray) -> list[StreamingResult]:
        """Process incoming audio chunk."""
        if not self._started:
            self.start()

        # 1. Normalize Audio
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.max() > 1.0:
            audio = audio / 32768.0 

        # 2. Audio Processing Pipeline (Compressor -> Noise Removal -> Gate)
        # Compressor
        audio = self.processor.process_compressor(audio)

        # DeepFilterNet Noise Removal
        if self.df_state is not None:
             audio = self._apply_deepfilternet(audio)

        # Noise Gate
        audio = self.processor.process_noise_gate(audio)
        
        self._debug_audio.append(audio)

        # 3. VAD Process (State Machine)
        # This returns chunks ONLY if speech is confirmed (or in hangover).
        # It handles buffering PRE_SPEECH internally.
        speech_chunks, speech_ended = self.vad.process(audio)
        
        if speech_chunks:
            for chunk in speech_chunks:
                self._audio_buffer = np.concatenate([self._audio_buffer, chunk])
        
        current_time = time.time()
        results = []

        # --- LOGIC SAFETY: MAX DURATION CAP (Prevent Quadratic Explosion) ---
        # If buffer exceeds max duration (e.g. 20s), FORCE COMMIT & RESET
        max_samples = config.streaming.max_buffer_duration * config.audio.sample_rate
        forced_reset = False
        
        if len(self._audio_buffer) > max_samples:
            print("DEBUG: Max buffer duration exceeded. Forcing commit.")
            forced_reset = True

        # --- LOGIC A: Handle Silence (Commit & Reset) ---
        # Trigger if VAD says speech ended (hangover finished) OR we forced a reset
        if speech_ended or forced_reset:
            
            # Only transcribe if we have audio in buffer
            # Note: With VAD.process, buffer only contains speech+hangover, so it SHOULD be valid.
            # But we keep size check just in case.
            if len(self._audio_buffer) > (0.1 * config.audio.sample_rate):
                    # Transcribe Final
                    transcribe_result = self.model.transcribe_audio(self._audio_buffer)
                    final_text_raw = transcribe_result.text.strip()
                    
                    if final_text_raw and OutputFilter.is_valid_text(final_text_raw):
                        # Use LocalAgreement to finalize
                        la_result = self.local_agreement.process(final_text_raw, is_final=True)
                        
                        # Accumulate confirmed text
                        self._confirmed_text_history += la_result.confirmed_text + " "
                        
                        # Emit Final Result
                        results.append(StreamingResult(
                            text=self._confirmed_text_history.strip(),
                            confirmed_text=self._confirmed_text_history.strip(),
                            pending_text="",
                            is_final=True,
                            latency_ms=transcribe_result.latency * 1000,
                            audio_duration=len(self._audio_buffer)/config.audio.sample_rate
                        ))

            # RESET BUFFER
            # If forced reset, keep explicit overlap.
            if forced_reset:
                keep_samples = int(config.streaming.forced_reset_overlap * config.audio.sample_rate)
                if len(self._audio_buffer) > keep_samples:
                    self._audio_buffer = self._audio_buffer[-keep_samples:]
                else:
                    pass
            else:
                # Normal VAD end -> Clear buffer completely
                # VAD State machine handles transition to IDLE
                self._audio_buffer = np.array([], dtype=np.float32)
            
            return results # Return final result and stop


        # --- LOGIC B: Throttling (Partial Updates) ---
        # Only run if VAD says we are speaking (or in hangover)
        if self.vad.is_speaking and (current_time - self._last_inference_time) > config.streaming.inference_interval:
            
            # Only run if buffer has decent size (e.g. > 0.5s to avoid hallucinating on tiny preroll)
            if len(self._audio_buffer) > (0.5 * config.audio.sample_rate):
                
                draft_result = self.model.transcribe_audio(self._audio_buffer)
                draft_text_raw = draft_result.text.strip()

                if OutputFilter.is_valid_text(draft_text_raw):
                    # Use LocalAgreement for Partial Results
                    la_result = self.local_agreement.process(draft_text_raw, is_final=False)
                    
                    # Full Text = History + Local Confirmed + Local Pending
                    # We do NOT update self._confirmed_text_history here, only at final commit.
                    full_text = f"{self._confirmed_text_history} {la_result.confirmed_text} {la_result.pending_text}".strip()
                    
                    results.append(StreamingResult(
                        text=full_text,
                        confirmed_text=f"{self._confirmed_text_history} {la_result.confirmed_text}".strip(),
                        pending_text=la_result.pending_text,
                        is_final=False,
                        latency_ms=draft_result.latency * 1000,
                        audio_duration=len(self._audio_buffer)/config.audio.sample_rate
                    ))
                
                self._last_inference_time = current_time

        if results:
            for r in results:
                if self.on_result: self.on_result(r)

        return results

    def _apply_deepfilternet(self, audio):
        """Apply DF noise removal."""
        model = get_df_model()
        device = config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Same interpolation logic as before
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        audio_tensor_48k = torch.nn.functional.interpolate(
             audio_tensor.unsqueeze(0), scale_factor=3, mode='linear', align_corners=False
        ).squeeze(0)
        
        # Stream Enhance
        enhanced_48k = self._stream_enhance(model, self.df_state, audio_tensor_48k)
        
        # Back to 16k
        enhanced_16k = torch.nn.functional.interpolate(
             enhanced_48k.unsqueeze(0), scale_factor=1/3, mode='linear', align_corners=False
        ).squeeze(0)
        
        audio_out = enhanced_16k.squeeze(0).cpu().detach().numpy()
        
        # Mix
        att = config.noise_removal.attenuation
        if att < 1.0:
            min_len = min(len(audio), len(audio_out))
            audio_out = audio_out[:min_len] * att + audio[:min_len] * (1 - att)
            
        return audio_out

    def _stream_enhance(self, model, df_state, audio):
        """DF Enhance without reset."""
        model.eval()
        nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
        device = config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
        spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=device)
        enhanced = model(spec.clone(), erb_feat, spec_feat)[0].cpu()
        enhanced = as_complex(enhanced.squeeze(1))
        audio_out = torch.as_tensor(df_state.synthesis(enhanced.detach().numpy()))
        return audio_out

    def end(self) -> StreamingResult | None:
        """End session, flush remaining buffer."""
        if not self._started: return None
        
        result = None
        # Transcribe final buffer content if significant
        if len(self._audio_buffer) > (0.1 * config.audio.sample_rate):
             transcribe_result = self.model.transcribe_audio(self._audio_buffer)
             text = transcribe_result.text.strip()
             
             if text:
                 # Flush LocalAgreement
                 la_result = self.local_agreement.process(text, is_final=True)
                 if la_result.full_text:
                     self._confirmed_text_history += la_result.full_text + " "
                     
                 result = StreamingResult(
                     text=self._confirmed_text_history.strip(),
                     confirmed_text=self._confirmed_text_history.strip(),
                     pending_text="",
                     is_final=True
                 )

        self._started = False
        self._save_debug_audio(result)
        return result

    def _save_debug_audio(self, result):
        if self._debug_audio:
            try:
                 full_audio = np.concatenate(self._debug_audio)
                 buf = io.BytesIO()
                 sf.write(buf, full_audio, 16000, format='WAV')
                 buf.seek(0)
                 b64_data = base64.b64encode(buf.read()).decode('utf-8')
                 data_uri = f"data:audio/wav;base64,{b64_data}"
                 
                 if result:
                     result.debug_audio_file = data_uri
                 elif result is None:
                     # If we ended without result, maybe return a dummy one just for audio?
                     # For now, just print logic
                     pass
            except Exception as e:
                print(f"Debug Audio Error: {e}")

    def reset(self) -> None:
        self.start()


class AsyncStreamingTranscriber:
    """Async wrapper."""
    def __init__(self):
        self.transcriber = StreamingTranscriber()

    async def start(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.transcriber.start)

    async def process(self, audio: np.ndarray) -> list[StreamingResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcriber.process, audio)

    async def end(self) -> StreamingResult | None:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcriber.end)

    def reset(self) -> None:
        """Reset transcriber state."""
        self.transcriber.reset()

    async def stream(self, audio_iterator: AsyncIterator[np.ndarray]) -> AsyncIterator[StreamingResult]:
        await self.start()
        async for audio_chunk in audio_iterator:
            results = await self.process(audio_chunk)
            for result in results:
                yield result
        final = await self.end()
        if final:
            yield final

