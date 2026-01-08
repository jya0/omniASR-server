"""Audio buffer with overlap chunking for streaming ASR."""

import numpy as np
from dataclasses import dataclass
from config import config


try:
    from ten_vad import TenVad
except ImportError:
    TenVad = None


class TenVADWrapper:
    """Wrapper for TenVAD handling buffering and conversion."""
    
    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        if TenVad is None:
            raise ImportError("TenVAD not installed. Use 'rms' VAD type or install ten-vad.")
            
        self.vad = TenVad(hop_size, threshold)
        self.hop_size = hop_size
        self._buffer = np.array([], dtype=np.int16)
        
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.
        
        Args:
            audio: Float32 audio chunk (-1.0 to 1.0)
            
        Returns:
            True if any frame in the chunk is detected as speech.
        """
        # Convert to int16
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # Add to buffer
        self._buffer = np.concatenate([self._buffer, audio_int16])
        
        # Process all complete frames
        has_speech = False
        
        while len(self._buffer) >= self.hop_size:
            frame = self._buffer[:self.hop_size]
            self._buffer = self._buffer[self.hop_size:]
            
            _, is_speech_frame = self.vad.process(frame)
            if is_speech_frame:
                has_speech = True
                
        return has_speech

    def reset(self):
        """Reset buffer."""
        self._buffer = np.array([], dtype=np.int16)

@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    audio: np.ndarray
    start_time: float  # start time in the stream
    end_time: float    # end time in the stream
    is_final: bool = False  # is this the final chunk (silence detected)?


class AudioBuffer:
    """
    Ring buffer for audio with overlap chunking.

    Handles:
    - Accumulating incoming audio
    - Producing overlapping chunks for processing
    - Tracking timestamps
    """

    def __init__(
        self,
        sample_rate: int = None,
        chunk_duration: float = None,
        overlap_ratio: float = None,
        max_duration: float = None,
    ):
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.chunk_duration = chunk_duration or config.streaming.chunk_duration
        self.overlap_ratio = overlap_ratio or config.streaming.overlap_ratio
        self.max_duration = max_duration or config.streaming.max_buffer_duration

        # Calculate sizes in samples
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.overlap_samples = int(self.chunk_samples * self.overlap_ratio)
        self.step_samples = self.chunk_samples - self.overlap_samples
        self.max_samples = int(self.max_duration * self.sample_rate)

        # Buffer state
        self._buffer: list[np.ndarray] = []
        self._total_samples = 0
        self._processed_until = 0  # samples already processed
        self._stream_time = 0.0    # total time in stream

    def add(self, audio: np.ndarray) -> None:
        """Add audio samples to the buffer."""
        audio = np.asarray(audio, dtype=np.float32).flatten()
        self._buffer.append(audio)
        self._total_samples += len(audio)
        self._stream_time = self._total_samples / self.sample_rate

        # Prevent unbounded growth
        if self._total_samples > self.max_samples:
            self._trim_buffer()

    def _trim_buffer(self) -> None:
        """Trim old audio from buffer."""
        # Keep only recent audio
        keep_samples = self.max_samples // 2

        full_buffer = self._get_full_buffer()
        if len(full_buffer) > keep_samples:
            trimmed = full_buffer[-keep_samples:]
            self._buffer = [trimmed]
            self._processed_until = max(0, self._processed_until - (len(full_buffer) - keep_samples))
            self._total_samples = len(trimmed)

    def _get_full_buffer(self) -> np.ndarray:
        """Concatenate all buffer segments."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._buffer)

    def get_unprocessed(self) -> np.ndarray:
        """Get all unprocessed audio."""
        full = self._get_full_buffer()
        return full[self._processed_until:]

    def has_chunk(self) -> bool:
        """Check if there's enough audio for a chunk."""
        unprocessed = self._total_samples - self._processed_until
        return unprocessed >= self.chunk_samples

    def get_next_chunk(self) -> AudioChunk | None:
        """
        Get the next chunk for processing (with overlap).
        Returns None if not enough audio.
        """
        if not self.has_chunk():
            return None

        full = self._get_full_buffer()

        # Get chunk with overlap from previous
        start = max(0, self._processed_until - self.overlap_samples)
        end = self._processed_until + self.step_samples + self.overlap_samples
        end = min(end, len(full))

        chunk_audio = full[start:end]

        # Calculate timestamps
        start_time = start / self.sample_rate
        end_time = end / self.sample_rate

        # Advance processed pointer
        self._processed_until += self.step_samples

        return AudioChunk(
            audio=chunk_audio,
            start_time=start_time,
            end_time=end_time,
            is_final=False,
        )

    def get_remaining(self, mark_final: bool = True) -> AudioChunk | None:
        """
        Get remaining unprocessed audio (for end of stream).
        Used when silence is detected or stream ends.
        """
        unprocessed = self._total_samples - self._processed_until
        if unprocessed < int(config.streaming.min_chunk_duration * self.sample_rate):
            return None

        full = self._get_full_buffer()

        # Include overlap from previous
        start = max(0, self._processed_until - self.overlap_samples)
        chunk_audio = full[start:]

        start_time = start / self.sample_rate
        end_time = len(full) / self.sample_rate

        # Mark all as processed
        self._processed_until = len(full)

        return AudioChunk(
            audio=chunk_audio,
            start_time=start_time,
            end_time=end_time,
            is_final=mark_final,
        )

    def get_current_window(self) -> AudioChunk | None:
        """
        Get all accumulated audio in the buffer (Growing Window).
        Does NOT advance the processed pointer.
        """
        full = self._get_full_buffer()
        if len(full) == 0:
            return None
            
        return AudioChunk(
            audio=full,
            start_time=0.0,  # Relative to start of buffer/utterance
            end_time=len(full) / self.sample_rate,
            is_final=False
        )

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer = []
        self._total_samples = 0
        self._processed_until = 0

    def clear_confirmed(self, until_time: float) -> None:
        """
        Clear buffer up to a confirmed timestamp.
        Called after LocalAgreement confirms a prefix.
        """
        until_samples = int(until_time * self.sample_rate)

        full = self._get_full_buffer()
        if until_samples >= len(full):
            self.clear()
            return

        # Keep audio from until_samples onwards
        remaining = full[until_samples:]
        self._buffer = [remaining] if len(remaining) > 0 else []
        self._total_samples = len(remaining)
        self._processed_until = max(0, self._processed_until - until_samples)

    @property
    def duration(self) -> float:
        """Total duration of audio in buffer."""
        return self._total_samples / self.sample_rate

    @property
    def unprocessed_duration(self) -> float:
        """Duration of unprocessed audio."""
        return (self._total_samples - self._processed_until) / self.sample_rate


class VAD:
    """Simple Voice Activity Detection based on RMS energy."""

    def __init__(
        self,
        threshold: float = None,
        silence_duration: float = None,
        sample_rate: int = None,
    ):
        self.threshold = threshold or config.vad.silence_threshold
        self.silence_duration = silence_duration or config.vad.silence_duration
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.vad_type = config.vad.type

        if self.vad_type == "ten_vad":
            self.ten_vad = TenVADWrapper(
                hop_size=config.vad.ten_vad.hop_size,
                threshold=config.vad.ten_vad.threshold
            )
        else:
            self.ten_vad = None

        self._silence_samples = 0
        self._is_speaking = False

    def get_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS energy of audio."""
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def process(self, audio: np.ndarray) -> tuple[bool, bool]:
        """
        Process audio chunk and detect voice activity.

        Returns:
            (is_speech, speech_ended):
            - is_speech: True if current chunk contains speech
            - speech_ended: True if speech just ended (silence detected after speech)
        """
        if self.vad_type == "ten_vad":
            is_speech = self.ten_vad.is_speech(audio)
        else:
            rms = self.get_rms(audio)
            is_speech = rms > self.threshold
        
        speech_ended = False

        if is_speech:
            self._silence_samples = 0
            self._is_speaking = True
        else:
            if self._is_speaking:
                self._silence_samples += len(audio)
                if self._silence_samples >= self.silence_duration * self.sample_rate:
                    speech_ended = True
                    self._is_speaking = False
                    self._silence_samples = 0

        return is_speech, speech_ended

    def reset(self) -> None:
        """Reset VAD state."""
        self._silence_samples = 0
        self._is_speaking = False
        if self.ten_vad:
            self.ten_vad.reset()

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speech."""
        return self._is_speaking
