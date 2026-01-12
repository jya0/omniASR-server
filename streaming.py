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
from audio_processor import AudioProcessor
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

class HallucinationDetector:
    """
    Robust hallucination detection using multiple techniques:
    1. Energy Gating: Skip low-energy audio before transcription
    2. Consistency Check: Compare full transcription vs halves
    3. N-gram Detection: Detect abnormal word repetition patterns
    """
    
    BLOCKLIST = [
        "noise",
        "Amara.org",
        "Subtitles by",
        "Translated by",
        "Copyright",
    ]
    
    def __init__(self, model=None):
        """
        Args:
            model: ASRModel instance for consistency check (optional)
        """
        self._model = model
        
        # Initialize flags from global config
        self.energy_gating_enabled = config.hallucination.energy_gating_enabled
        self.min_rms_threshold = config.hallucination.min_rms_threshold
        
        self.consistency_check_enabled = config.hallucination.consistency_check_enabled
        self.consistency_threshold = config.hallucination.consistency_threshold
        
        self.ngram_detection_enabled = config.hallucination.ngram_detection_enabled
        self.max_words_per_second = config.hallucination.max_words_per_second
        self.max_word_repeat = config.hallucination.max_word_repeat
        self.max_same_word_ratio = config.hallucination.max_same_word_ratio
    
    @staticmethod
    def get_rms(audio: np.ndarray) -> float:
        """Calculate RMS energy of audio."""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    
    def energy_gate(self, audio: np.ndarray) -> bool:
        """
        Check if audio has enough energy to be speech.
        
        Returns:
            True if audio should be transcribed, False if too quiet
        """
        if not self.energy_gating_enabled:
            return True
            
        rms = self.get_rms(audio)
        return rms >= self.min_rms_threshold
    
    def consistency_check(self, audio: np.ndarray, full_text: str) -> bool:
        """
        Verify transcription by comparing full vs two halves.
        Real speech produces consistent results; hallucinations vary wildly.
        
        Returns:
            True if transcription is consistent (likely real), False if inconsistent
        """
        if not self.consistency_check_enabled:
            return True
            
        if self._model is None:
            return True  # Can't check without model
            
        if len(audio) < 3200:  # Too short to split (< 0.2s at 16kHz)
            return True
            
        # Split audio in half
        mid = len(audio) // 2
        first_half = audio[:mid]
        second_half = audio[mid:]
        
        # Transcribe halves
        try:
            result_first = self._model.transcribe_audio(first_half)
            result_second = self._model.transcribe_audio(second_half)
            
            combined_text = f"{result_first.text} {result_second.text}".strip()
            
            # Compare using fuzzy matching
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, full_text.lower(), combined_text.lower()).ratio()
            
            return ratio >= self.consistency_threshold
            
        except Exception:
            return True  # On error, assume valid
    
    def ngram_check(self, text: str, audio_duration: float = None) -> bool:
        """
        Detect abnormal word patterns:
        1. Repetition (same word 4+ times)
        2. Speaking rate (>9 words/second is impossible)
        
        Args:
            text: Transcription text
            audio_duration: Duration of audio in seconds (for speaking rate check)
        
        Returns:
            True if text looks normal, False if suspicious pattern detected
        """
        if not self.ngram_detection_enabled:
            return True
            
        if not text or len(text.strip()) < 3:
            return False  # Too short is suspicious
            
        words = text.lower().split()
        if len(words) < 2:
            return True  # Too few words to check
        
        # Speaking rate check: Max ~9 words/second (fastest human speech)
        if audio_duration and audio_duration > 0.1:
            words_per_second = len(words) / audio_duration
            if words_per_second > self.max_words_per_second:
                return False  # Impossible speaking rate
            
        # Check consecutive repetition
        repeat_count = 1
        max_repeat = self.max_word_repeat
        
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repeat_count += 1
                if repeat_count > max_repeat:
                    return False
            else:
                repeat_count = 1
        
        # Check overall same-word ratio
        if len(words) >= 4:
            from collections import Counter
            word_counts = Counter(words)
            _, top_count = word_counts.most_common(1)[0]
            ratio = top_count / len(words)
            
            if ratio > self.max_same_word_ratio and top_count >= 3:
                return False
        
        return True
    
    def is_valid_text(self, text: str, audio_duration: float = None) -> bool:
        """
        Combined validation: blocklist + n-gram check + speaking rate.
        (Energy gating and consistency check are done separately on audio)
        
        Args:
            text: Transcription text
            audio_duration: Audio duration in seconds (for speaking rate check)
        """
        if not text:
            return True
            
        text = text.strip()
        
        # Blocklist check
        lower_text = text.lower()
        for phrase in self.BLOCKLIST:
            if phrase.lower() in lower_text:
                return False
        
        # N-gram repetition + speaking rate check
        if not self.ngram_check(text, audio_duration):
            return False
        
        # Character ratio check (Arabic/English)
        valid_pattern = re.compile(r'[\u0600-\u06FFa-zA-Z0-9\s.,?!\'"\-:;()]')
        valid_chars = len(valid_pattern.findall(text))
        total_chars = len(text)
        
        if total_chars > 0 and (valid_chars / total_chars) < 0.8:
            return False
            
        return True


# Backward compatibility alias
OutputFilter = HallucinationDetector

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
    processor: AudioProcessor = None
    
    # DeepFilterNet state
    df_state: object = None
    
    # Hallucination detector
    hallucination_detector: HallucinationDetector = None

    # Callbacks
    on_result: Callable[[StreamingResult], None] = None

    # Instance Config (Defaults from global config)
    vad_enabled: bool = field(default_factory=lambda: config.vad.enabled)
    noise_removal_enabled: bool = field(default_factory=lambda: config.noise_removal.enabled)
    max_buffer_duration: float = field(default_factory=lambda: config.streaming.max_buffer_duration)
    
    _noise_removal_attenuation: float = field(default_factory=lambda: config.noise_removal.attenuation)
    
    # Debug Audio Config
    debug_audio_enabled: bool = True  # Default enabled for backward compat
    debug_audio_interval: float = 10.0  # Flush every 10 seconds

    # State
    _started: bool = False
    _total_audio_duration: float = 0
    _debug_audio: list = field(default_factory=list) # Accumulate audio for debug
    _debug_audio_vad: list = field(default_factory=list) # Accumulate VAD audio for debug
    _last_debug_flush_time: float = 0.0  # Track last debug audio flush time
    _raw_audio_duration: float = 0.0  # Track raw input audio duration for debug flush

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
        if self.processor is None:
            self.processor = AudioProcessor()
        
        # Initialize hallucination detector with model reference
        if self.hallucination_detector is None:
            self.hallucination_detector = HallucinationDetector(model=self.model)

    def start(self) -> None:
        """Start a new streaming session."""
        self.model.ensure_loaded()
        self.buffer.clear()
        self.vad.reset()
        self.agreement.reset()
        
        # Initialize specific DF state for this stream
        if config.noise_removal.enabled:
            _, self.df_state, _ = init_df(model_base_dir="DeepFilterNet3", config_allow_defaults=True)
        else:
            self.df_state = None
        
        self._started = True
        self._total_audio_duration = 0
        self._debug_audio = []
        self._debug_audio_vad = []
        self._last_debug_flush_time = 0.0
        self._raw_audio_duration = 0.0
    
    def get_debug_audio(self, clear: bool = True) -> tuple[str | None, str | None]:
        """
        Get accumulated debug audio as base64 data URIs and optionally clear buffers.
        
        Returns:
            Tuple of (full_audio_uri, vad_audio_uri) - either may be None
        """
        if not self.debug_audio_enabled:
            return None, None
            
        full_uri = None
        vad_uri = None
        
        try:
            if self._debug_audio:
                full_audio = np.concatenate(self._debug_audio)
                buf = io.BytesIO()
                sf.write(buf, full_audio, 16000, format='WAV')
                buf.seek(0)
                b64_data = base64.b64encode(buf.read()).decode('utf-8')
                full_uri = f"data:audio/wav;base64,{b64_data}"
                
            if self._debug_audio_vad:
                vad_audio = np.concatenate(self._debug_audio_vad)
                buf_vad = io.BytesIO()
                sf.write(buf_vad, vad_audio, 16000, format='WAV')
                buf_vad.seek(0)
                b64_vad = base64.b64encode(buf_vad.read()).decode('utf-8')
                vad_uri = f"data:audio/wav;base64,{b64_vad}"
                
        except Exception as e:
            print(f"DEBUG: Failed to generate debug audio: {e}")
            
        if clear:
            self._debug_audio = []
            self._debug_audio_vad = []
            
        return full_uri, vad_uri

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
        
        # Track raw audio duration for debug flush (independent of processing)
        audio_duration = len(audio) / 16000.0  # Assuming 16kHz sample rate
        self._raw_audio_duration += audio_duration
            
        # --- 1. Dynamic Range Compression (Pedalboard) ---
        audio = self.processor.process_compressor(audio)
            
        # --- 2. DeepFilterNet Noise Removal ---
        if self.df_state is not None and self.noise_removal_enabled:
             model = get_df_model()
             device = config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
             
             # Prepare tensor
             audio_tensor = torch.from_numpy(audio).unsqueeze(0) # [1, T]
             
             # Resample 16k -> 48k (DF requirement)
             audio_tensor_48k = torch.nn.functional.interpolate(
                 audio_tensor.unsqueeze(0), 
                 scale_factor=3, 
                 mode='linear', 
                 align_corners=False
             ).squeeze(0)
             
             # Enhance
             enhanced_audio_tensor_48k = self._stream_enhance(model, self.df_state, audio_tensor_48k)
             
             # Resample 48k -> 16k
             enhanced_audio_tensor = torch.nn.functional.interpolate(
                 enhanced_audio_tensor_48k.unsqueeze(0), 
                 scale_factor=1/3, 
                 mode='linear', 
                 align_corners=False
             ).squeeze(0)
             
             # Convert back to numpy
             audio_out_numpy = enhanced_audio_tensor.squeeze(0).cpu().detach().numpy()
             
             # Dry/Wet Mix (Tone down)
             att = self._noise_removal_attenuation
             if att < 1.0:
                 min_len = min(len(audio), len(audio_out_numpy))
                 audio_out_numpy = audio_out_numpy[:min_len] * att + audio[:min_len] * (1 - att)
             
             audio = audio_out_numpy

        # --- 3. Noise Gate (Pedalboard) ---
        # Post-noise removal to mute any remaining low-level noise
        audio = self.processor.process_noise_gate(audio)
        
        # Collect for debug recording
        self._debug_audio.append(audio)

        # --- 4. VAD Filter ---
        chunks_to_process = []
        speech_ended = False
        
        if self.vad_enabled:
            # New VAD logic returns chunks and end-flag
            chunks_to_process, speech_ended = self.vad.process(audio)
        else:
            chunks_to_process = [audio]

        # Add passed chunks to buffer
        for chunk in chunks_to_process:
            self.buffer.add(chunk)
            self._debug_audio_vad.append(chunk)

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
        # EARLY COMMITMENT: When buffer approaches capacity, commit current text
        # and reset to bound latency while preserving all transcribed text
        # Use instance config for max_buffer_duration
        soft_limit = self.max_buffer_duration * config.streaming.soft_reset_threshold
        
        if self.buffer.duration >= soft_limit:
            should_infer = True
            # Force early commitment - treat as end of turn
            speech_ended = True
             
        if should_infer:
            chunk = self.buffer.get_current_window()
            if chunk:
                result = self._process_chunk(chunk)
                if result:
                    results.append(result)
                    if self.on_result:
                        self.on_result(result)

        # --- Periodic Debug Audio Flush ---
        # Check if we should flush debug audio based on time elapsed
        # Must be done BEFORE speech_ended reset to avoid timing issues
        
        # Debug: Log audio duration tracking
        if self.debug_audio_enabled:
            elapsed_since_flush = self._raw_audio_duration - self._last_debug_flush_time
            print(f"DEBUG flush check: raw_duration={self._raw_audio_duration:.2f}s, last_flush={self._last_debug_flush_time:.2f}s, elapsed={elapsed_since_flush:.2f}s, interval={self.debug_audio_interval}s")
        
        if self.debug_audio_enabled and self._raw_audio_duration - self._last_debug_flush_time >= self.debug_audio_interval:
            print(f"DEBUG: Triggering debug audio flush at {self._raw_audio_duration:.2f}s")
            full_uri, vad_uri = self.get_debug_audio(clear=True)
            self._last_debug_flush_time = self._raw_audio_duration
            
            print(f"DEBUG: Got debug audio URIs - full: {bool(full_uri)}, vad: {bool(vad_uri)}, results count: {len(results)}")
            
            # If we have results, attach debug audio to the last one
            # Otherwise create a minimal result just for debug audio
            if results and (full_uri or vad_uri):
                results[-1].debug_audio_file = full_uri
                results[-1].debug_audio_file_vad = vad_uri
                print(f"DEBUG: Attached debug audio to existing result")
            elif full_uri or vad_uri:
                # Create a debug-only result
                print(f"DEBUG: Creating debug-only result")
                results.append(StreamingResult(
                    text="",
                    confirmed_text=self.agreement._confirmed if hasattr(self.agreement, '_confirmed') else "",
                    pending_text="",
                    is_final=False,
                    debug_audio_file=full_uri,
                    debug_audio_file_vad=vad_uri,
                ))

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
            
            # NOTE: Do NOT reset _raw_audio_duration and _last_debug_flush_time here!
            # Debug audio should accumulate across buffer resets (soft-limit triggers)
            # Only reset when connection closes or explicit stop

        return results

    def _process_chunk(self, chunk: AudioChunk) -> StreamingResult | None:
        """Process a single audio chunk with hallucination detection."""
        
        # 1. Energy Gating: Skip if audio is too quiet
        if not self.hallucination_detector.energy_gate(chunk.audio):
            # Return empty result (silence)
            return StreamingResult(
                text="",
                confirmed_text=self.agreement._confirmed,
                pending_text="",
                is_final=chunk.is_final,
                latency_ms=0,
                audio_duration=chunk.end_time,
            )
        
        # 2. Transcribe
        transcribe_result = self.model.transcribe_audio(chunk.audio)
        
        # 3. N-gram, blocklist, and speaking rate check
        audio_duration = chunk.end_time - chunk.start_time
        if not self.hallucination_detector.is_valid_text(transcribe_result.text, audio_duration):
            transcribe_result.text = ""
        
        # 4. Consistency Check (for non-empty, non-final results)
        if transcribe_result.text and not chunk.is_final:
            if not self.hallucination_detector.consistency_check(chunk.audio, transcribe_result.text):
                # Inconsistent = likely hallucination
                transcribe_result.text = ""

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

    def apply_config(self, session_config: dict) -> None:
        """
        Apply session-specific config overrides.
        Note: This modifies the instance state for this session only.
        """
        if not session_config:
            return
        
        print(f"DEBUG apply_config: Received config: {session_config}")
            
        # VAD Settings
        if "vad_enabled" in session_config:
            old_val = self.vad_enabled
            self.vad_enabled = session_config["vad_enabled"]
            print(f"DEBUG apply_config: vad_enabled changed {old_val} -> {self.vad_enabled}")
            
        if self.vad and "vad_silence_duration" in session_config:
            self.vad.silence_duration = session_config["vad_silence_duration"]
             
        # Noise Removal Settings
        if "noise_removal_enabled" in session_config:
            old_val = self.noise_removal_enabled
            self.noise_removal_enabled = session_config["noise_removal_enabled"]
            print(f"DEBUG apply_config: noise_removal_enabled changed {old_val} -> {self.noise_removal_enabled}")
            
        if "noise_removal_attenuation" in session_config:
            self._noise_removal_attenuation = session_config["noise_removal_attenuation"]
            
        # Audio Processor Settings (Compressor / Noise Gate)
        if self.processor:
            if "compressor_enabled" in session_config:
                self.processor.compressor_enabled = session_config["compressor_enabled"]
            if "compressor_threshold_db" in session_config:
                self.processor.compressor_threshold = session_config["compressor_threshold_db"]
            if "compressor_ratio" in session_config:
                self.processor.compressor_ratio = session_config["compressor_ratio"]
                 
            if "noise_gate_enabled" in session_config:
                self.processor.noise_gate_enabled = session_config["noise_gate_enabled"]
            if "noise_gate_threshold_db" in session_config:
                self.processor.noise_gate_threshold = session_config["noise_gate_threshold_db"]

        # Streaming Settings
        if "max_buffer_duration" in session_config:
            self.max_buffer_duration = session_config["max_buffer_duration"]

        # Hallucination Settings
        # Update flags in detector
        if self.hallucination_detector:
             if "hallucination_energy_gating" in session_config:
                 self.hallucination_detector.energy_gating_enabled = session_config["hallucination_energy_gating"]
                 
             if "hallucination_min_rms" in session_config:
                 self.hallucination_detector.min_rms_threshold = session_config["hallucination_min_rms"]
             
             if "hallucination_ngram_enabled" in session_config:
                 self.hallucination_detector.ngram_detection_enabled = session_config["hallucination_ngram_enabled"]
                 
             if "hallucination_max_word_repeat" in session_config:
                 self.hallucination_detector.max_word_repeat = session_config["hallucination_max_word_repeat"]

        # Debug Audio Settings
        if "debug_audio_enabled" in session_config:
            self.debug_audio_enabled = session_config["debug_audio_enabled"]
            
        if "debug_audio_interval" in session_config:
            self.debug_audio_interval = session_config["debug_audio_interval"]

        # Store for reference
        self._session_config = session_config

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

    def apply_config(self, session_config: dict) -> None:
        """Apply session-specific config overrides."""
        self.transcriber.apply_config(session_config)

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
