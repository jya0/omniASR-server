"""
StreamingTranscriber - orchestrates streaming ASR with LocalAgreement.

Combines:
- AudioBuffer for chunking with overlap
- VAD for speech detection
- ASRModel for transcription
- LocalAgreement for stable output
"""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable
import numpy as np

from config import config
from audio_buffer import AudioBuffer, VAD, AudioChunk
from local_agreement import LocalAgreement, TranscriptionResult
from model import ASRModel, TranscribeResult

import torch
import torchaudio.transforms as T
from df.enhance import enhance, init_df, load_audio, df_features
from df.utils import as_complex
from df.model import ModelParams
import os
import re
import soundfile as sf
import datetime
import io
import base64

# Global DF model instance (singleton)
_DF_MODEL = None

class OutputFilter:
    """Filters hallucinations and nonsensical output."""
    
    BLOCKLIST = [
        # "Amara.org",
        # "Subtitles by",
        # "Translated by",
        # "Copyright",
        # "All rights reserved",
        # "Paradox Interactive",
        # "Mojang AB",
        # "Step Id:",
        # "Master Scribes",
        # "Nomad Scribes",
        # "Miraculous Ladybug",
        # "The following content has been modified",
        # "monsieur lucky",
        # "marie mathew",
        # "thiệt hơn chị bản",
        # "dem crite",
        # "électrique",
    ]

    @staticmethod
    def is_valid_text(text: str) -> bool:
        """
        Check if text is valid English or Arabic.
        Discard if it contains too many characters from other scripts (e.g. Vietnamese, French specific).
        Also discard blocklisted phrases.
        """
        if not text:
            return True

        # 1. Blocklist check
        lower_text = text.lower()
        for phrase in OutputFilter.BLOCKLIST:
            if phrase.lower() in lower_text:
                return False

        # 2. Character Ratio Check
        # Allowed: Arabic, English (Latin), Numbers, Punctuation, Common Symbols
        # Regex for valid characters:
        # \u0600-\u06FF (Arabic)
        # a-zA-Z (English)
        # 0-9 (Numbers)
        # \s (Whitespace)
        # .,?!'":;-() (Punctuation)
        
        # We count valid chars and total chars
        # Note: "Monsieur" (French) is mostly Latin, so it passes this check. Blocklist handles it.
        # "thiet hon chi ban" (Vietnamese) involves latin chars with tone marks.
        # Simple latin range a-zA-Z won't catch accented chars like 'ệ', 'ị', 'ả'. 
        # So if we strictly allow only a-zA-Z, we filter out Vietnamese/French accents.
        
        valid_pattern = re.compile(r'[\u0600-\u06FFa-zA-Z0-9\s.,?!\'"\-:;()]')
        
        valid_chars = len(valid_pattern.findall(text))
        total_chars = len(text)
        
        if total_chars == 0:
            return True
            
        ratio = valid_chars / total_chars
        
        # If less than 80% of characters are valid (Arabic/English/Common), we discard.
        # This effectively filters out CJK, Cyrillic, and extended Latin (Vietnamese tones, French accents).
        if ratio < 0.8:
            return False
            
        # 3. Repetition Check
        # Catch "oooooooooo" type hallucinations (common with whisper on silence/music)
        # Any character repeated 10 times or more
        if re.search(r'(.)\1{9,}', text):
            return False
            
        return True

def get_df_model():
    """Get or initialize the DeepFilterNet model globally."""
    global _DF_MODEL
    if _DF_MODEL is None:
        # Use local models-cache if not set
        if "XDG_CACHE_HOME" not in os.environ:
             base_path = os.path.dirname(os.path.abspath(__file__))
             os.environ["XDG_CACHE_HOME"] = os.path.join(base_path, "models-cache")
             
        # Initialize model (load weights)
        # We use a dummy state just to get the model, individual states are per-stream
        model, _, _ = init_df(model_base_dir="DeepFilterNet3", config_allow_defaults=True)
        _DF_MODEL = model.to(config.model.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        _DF_MODEL.eval()
        
    return _DF_MODEL


@dataclass
class StreamingResult:
    """Result from streaming transcription."""
    text: str                  # Current hypothesis (full)
    confirmed_text: str        # Stable part
    pending_text: str          # Changing part
    is_final: bool             # Is this the final result?
    latency_ms: float = 0      # Last inference latency in ms
    audio_duration: float = 0  # Audio duration processed
    debug_audio_file: str = None # Path to debug recording (if any)
    debug_audio_file_vad: str = None # Path to post-VAD debug recording


@dataclass
class StreamingTranscriber:
    """
    Streaming ASR using chunked processing with LocalAgreement.

    Usage:
        transcriber = StreamingTranscriber()
        transcriber.start()

        # Feed audio chunks
        for chunk in audio_stream:
            results = transcriber.process(chunk)
            for result in results:
                print(result.text)

        # End stream
        final = transcriber.end()
    """

    model: ASRModel = None
    buffer: AudioBuffer = None
    vad: VAD = None
    agreement: LocalAgreement = None
    
    # DeepFilterNet state
    df_state: object = None

    # Callbacks
    on_result: Callable[[StreamingResult], None] = None

    # State
    _started: bool = False
    _total_audio_duration: float = 0
    _debug_audio: list = field(default_factory=list) # Accumulate audio for debug
    _debug_audio_vad: list = field(default_factory=list) # Accumulate VAD audio for debug

    def __post_init__(self):
        print("DEBUG: BASE64 VERSION ACTIVE (StreamingTranscriber initialized)")
        if self.model is None:
            self.model = ASRModel.get_instance()
        if self.buffer is None:
            self.buffer = AudioBuffer()
        if self.vad is None:
            self.vad = VAD()
        if self.agreement is None:
            self.agreement = LocalAgreement()
            
        # Initialize DeepFilterNet state for this session
        # We need a fresh state for each stream/transcriber to handle overlaps/buffers correctly
        # init_df returns (model, state, suffix), we just want the state class/object
        # But init_df actually returns an instantiated state. 
        # Ideally we should construct DF() directly if we could import it, but init_df configures it.
        # So we call init_df again? No, that loads model.
        # Let's inspect how to create state. 
        # Actually init_df documentation says it returns "df_state (DF): Deep filtering state".
        # We can probably clone it or create new.
        # For now, let's call init_df with a known config if possible, or just use the one from get_df_model?
        # WAIT: DF state maintains buffer for STFT. It IS stateful.
        # We need a new state per session.
        # Optimization: We can reconstruct DF state using params from loaded config.
        # For simplicity and correctness given we can't easily import DF class without libdf:
        # We'll just use init_df again but it might be slow if it checks model every time.
        # Alternative: We used `from df.enhance import init_df` which does model loading.
        # Let's try to grab the class from the global model init if possible? 
        # Actually, let's look at `df.enhance.init_df`. It creates `df_state = DF(...)`. 
        # We can just call init_df. To avoid reloading model, maybe we can just ignore the model return.
        # But `init_df` loads model from efficient checkpoint.
        # Let's simply call init_df. It's safe but maybe slightly overhead on init.
        
        # We will initialize state in start() to ensure reset
        pass

    def start(self) -> None:
        """Start a new streaming session."""
        self.model.ensure_loaded()
        self.buffer.clear()
        self.vad.reset()
        self.agreement.reset()
        
        # Initialize specific DF state for this stream
        # This ensures buffers are clear
        # Initialize DF State
        if config.noise_removal.enabled:
            _, self.df_state, _ = init_df(model_base_dir="DeepFilterNet3", config_allow_defaults=True)
            # Resamplers replaced by functional interpolate
        else:
            self.df_state = None
        
        self._started = True
        self._total_audio_duration = 0
        self._debug_audio = []

    def process(self, audio: np.ndarray) -> list[StreamingResult]:
        """
        Process incoming audio chunk.

        Args:
            audio: Audio samples (numpy array, float32 or int16)

        Returns:
            List of StreamingResult (may be empty if no output yet)
        """
        if not self._started:
            self.start()

        # Normalize audio
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.max() > 1.0:
            audio = audio / 32768.0  # Convert from int16
            
        # --- DeepFilterNet Noise Removal ---
        if self.df_state is not None:
             model = get_df_model()
             device = config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
             
             # Prepare tensor
             # audio is (N,) numpy, DF expects (C, T) tensor
             # enhance expects CPU tensor because it converts to numpy for feature extraction
             audio_tensor = torch.from_numpy(audio).unsqueeze(0) # [1, T]
             
             # Resample 16k -> 48k using Linear Interpolation to avoid boundary artifacts
             # T.Resample (Sinc) introduces zero-padding at edges which sounds like "cuts" in streaming.
             # Linear is safer for chunked streaming.
             # Input: [1, T] -> Unsqueeze to [1, 1, T] for interpolate
             audio_tensor_48k = torch.nn.functional.interpolate(
                 audio_tensor.unsqueeze(0), 
                 scale_factor=3, 
                 mode='linear', 
                 align_corners=False
             ).squeeze(0)
             
             # Enhance (Stream mode)
             # We manually call DF features and model to avoid calling 'reset_h0' which 'enhance()' does.
             # This preserves the RNN state for continuous streaming.
             enhanced_audio_tensor_48k = self._stream_enhance(model, self.df_state, audio_tensor_48k)
             
             # Resample 48k -> 16k
             # Use same linear interpolation strategy
             enhanced_audio_tensor = torch.nn.functional.interpolate(
                 enhanced_audio_tensor_48k.unsqueeze(0), 
                 scale_factor=1/3, 
                 mode='linear', 
                 align_corners=False
             ).squeeze(0)
             
             # Convert back to numpy
             # enhanced is [C, T], we want [T]
             audio_out_numpy = enhanced_audio_tensor.squeeze(0).cpu().detach().numpy()
             
             # Dry/Wet Mix (Tone down)
             # audio (original) vs audio_out_numpy (enhanced)
             # Use config attenuation
             att = config.noise_removal.attenuation
             if att < 1.0:
                 # Ensure lengths match for mixing (interpolation might vary by 1 sample?)
                 min_len = min(len(audio), len(audio_out_numpy))
                 audio_out_numpy = audio_out_numpy[:min_len] * att + audio[:min_len] * (1 - att)
             
             audio = audio_out_numpy
         # -----------------------------------
        # -----------------------------------
        
        # Collect for debug recording (Linear stream, post-enhancement)
        self._debug_audio.append(audio)

        # Check VAD (only if enabled)
        is_speech = True
        speech_ended = False
        
        if config.vad.enabled:
            is_speech, speech_ended = self.vad.process(audio)

        # Only add to buffer if speech is detected (or VAD disabled)
        # Note: VAD.is_speech remains True during the silence_duration (hangover),
        # so we correctly capture the tail of the speech.
        if is_speech:
            self.buffer.add(audio)
            self._debug_audio_vad.append(audio)

        results = []

        # Growing Window Strategy
        # Instead of chopping into fixed chunks, we accumulate and process the
        # entire window. This preserves context for LocalAgreement.
        
        current_duration = self.buffer.duration
        
        # Determine if we should run inference
        # 1. Enough audio accumulated since start or last inference
        # 2. But don't run too frequently to save compute (e.g. every 1.0s)
        
        MIN_INFERENCE_DURATION = 0.5  # seconds
        
        should_infer = False
        
        if self.buffer.duration >= self._total_audio_duration + MIN_INFERENCE_DURATION:
             should_infer = True
             
        # Force inference if buffer is getting too full (safety)
        if self.buffer.duration > config.streaming.max_buffer_duration:
             should_infer = True
             speech_ended = True # Force end of utterance
             
        if should_infer:
            chunk = self.buffer.get_current_window()
            if chunk:
                result = self._process_chunk(chunk)
                if result:
                    results.append(result)
                    if self.on_result:
                        self.on_result(result)

        # If speech ended (transition from speech to silence), finalize
        if speech_ended:
            # Get final snapshot (mark as final)
            chunk = self.buffer.get_current_window()
            if chunk:
                # Manually set is_final on the chunk
                chunk.is_final = True
                result = self._process_chunk(chunk)
                if result:
                    results.append(result)
                    if self.on_result:
                        self.on_result(result)

            # Reset for next utterance
            self.buffer.clear()
            self.agreement.reset()
            self._total_audio_duration = 0  # Reset tracker

        return results

    def _process_chunk(self, chunk: AudioChunk) -> StreamingResult | None:
        """Process a single audio chunk."""
        # Transcribe
        transcribe_result = self.model.transcribe_audio(chunk.audio)

        # --- Output Filtering ---
        if not OutputFilter.is_valid_text(transcribe_result.text):
            # Treat as silence / ignore
            transcribe_result.text = "" 
        # ------------------------

        self._total_audio_duration = chunk.end_time

        # Run through LocalAgreement
        agreement_result = self.agreement.process(
            transcribe_result.text,
            is_final=chunk.is_final,
        )

        return StreamingResult(
            text=agreement_result.full_text,
            confirmed_text=agreement_result.confirmed_text,
            pending_text=agreement_result.pending_text,
            is_final=agreement_result.is_final,
            latency_ms=transcribe_result.latency * 1000,
            audio_duration=chunk.end_time,
        )

    def _stream_enhance(self, model, df_state, audio):
        """
        Custom enhance function for streaming that DOES NOT reset model state.
        Replicates logic from df.enhance.enhance but without reset_h0.
        """
        model.eval()
        # No model.reset_h0() call here!
        
        # Determine nb_df (num deep filter bands)
        # Try to get from model or default
        nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
        
        device = config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate features (spec, erb, spec_feat)
        # df_features handles STFT
        spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=device)
        
        # Run model
        # model expects (spec, erb_feat, spec_feat)
        # Output is complex spec
        enhanced = model(spec.clone(), erb_feat, spec_feat)[0].cpu()
        
        # Convert to complex tensor
        enhanced = as_complex(enhanced.squeeze(1))
        
        # Synthesis (ISTFT)
        # df_state.synthesis handles overlap-add
        audio_out = torch.as_tensor(df_state.synthesis(enhanced.detach().numpy()))
        
        return audio_out

    def end(self) -> StreamingResult | None:
        """
        End the streaming session and get final result.

        Returns:
            Final StreamingResult or None if no audio
        """
        if not self._started:
            return None

        # Process any remaining audio
        remaining = self.buffer.get_remaining(mark_final=True)
        result = None

        if remaining:
            result = self._process_chunk(remaining)
            if result and self.on_result:
                self.on_result(result)

        self._started = False
        
        # Save debug recording (Base64 Data URI)
        if self._debug_audio:
            try:
                 # 1. Full Audio (Original/Enhanced)
                 full_audio = np.concatenate(self._debug_audio)
                 
                 # Write to memory buffer
                 buf = io.BytesIO()
                 sf.write(buf, full_audio, 16000, format='WAV')
                 buf.seek(0)
                 b64_data = base64.b64encode(buf.read()).decode('utf-8')
                 data_uri = f"data:audio/wav;base64,{b64_data}"
                 
                 if result:
                     result.debug_audio_file = data_uri # Storing URI in 'file' field for transport
                 else:
                     result = StreamingResult(
                         text="", confirmed_text="", pending_text="", is_final=True, 
                         debug_audio_file=data_uri
                     )
                 
                 # 2. VAD Audio
                 print(f"DEBUG: VAD Buffer Status - Has Data? {bool(self._debug_audio_vad)}, Chunk Count: {len(self._debug_audio_vad)}")
                 if self._debug_audio_vad:
                     full_audio_vad = np.concatenate(self._debug_audio_vad)
                     
                     buf_vad = io.BytesIO()
                     sf.write(buf_vad, full_audio_vad, 16000, format='WAV')
                     buf_vad.seek(0)
                     b64_vad = base64.b64encode(buf_vad.read()).decode('utf-8')
                     data_uri_vad = f"data:audio/wav;base64,{b64_vad}"
                     
                     result.debug_audio_file_vad = data_uri_vad
                 else:
                     print("DEBUG: _debug_audio_vad IS EMPTY. No VAD recording generation.")
                     
            except Exception as e:
                 print(f"DEBUG: Failed to generate debug audio: {e}")
                 
        return result

    def reset(self) -> None:
        """Reset transcriber state."""
        self.buffer.clear()
        self.vad.reset()
        self.agreement.reset()
        self._started = False
        self._total_audio_duration = 0


class AsyncStreamingTranscriber:
    """
    Async version of StreamingTranscriber for use with FastAPI/WebSocket.

    Usage:
        transcriber = AsyncStreamingTranscriber()
        await transcriber.start()

        async for result in transcriber.stream(audio_chunks):
            yield result.text
    """

    def __init__(self):
        self.transcriber = StreamingTranscriber()
        self._queue: asyncio.Queue[StreamingResult] = asyncio.Queue()

    async def start(self) -> None:
        """Start streaming session."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.transcriber.start)

    async def process(self, audio: np.ndarray) -> list[StreamingResult]:
        """Process audio chunk asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcriber.process, audio)

    async def end(self) -> StreamingResult | None:
        """End streaming session."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcriber.end)

    async def stream(
        self,
        audio_iterator: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[StreamingResult]:
        """
        Stream transcription results from an async audio iterator.

        Args:
            audio_iterator: Async iterator yielding audio chunks

        Yields:
            StreamingResult for each processed chunk
        """
        await self.start()

        async for audio_chunk in audio_iterator:
            results = await self.process(audio_chunk)
            for result in results:
                yield result

        final = await self.end()
        if final:
            yield final

    def reset(self) -> None:
        """Reset transcriber."""
        self.transcriber.reset()
