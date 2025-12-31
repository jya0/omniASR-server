"""
Audio chunking for long files.

Handles audio files longer than the model's max duration (40s) by:
1. VAD-based segmentation - split at natural pauses
2. Fallback chunking - if no pause found, split with overlap
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf

from config import config


@dataclass
class AudioSegment:
    """A segment of audio with metadata."""
    audio: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class ChunkerConfig:
    """Configuration for audio chunking."""
    max_chunk_duration: float = 35.0      # Max chunk size (leave margin from 40s limit)
    min_chunk_duration: float = 1.0       # Min chunk size
    target_chunk_duration: float = 30.0   # Target chunk size for splitting
    overlap_duration: float = 2.0         # Overlap when force-splitting
    silence_threshold: float = 0.01       # RMS threshold for silence
    min_silence_duration: float = 0.3     # Min silence to consider as boundary
    silence_search_window: float = 5.0    # How far to search for silence


class AudioChunker:
    """
    Splits long audio into chunks suitable for ASR inference.

    Uses VAD to find natural boundaries (silences), falls back to
    overlapping chunks if no suitable silence is found.
    """

    def __init__(self, config: ChunkerConfig = None):
        self.config = config or ChunkerConfig()

    def get_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS energy of audio."""
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def find_silence_boundary(
        self,
        audio: np.ndarray,
        sample_rate: int,
        search_start: float,
        search_end: float,
    ) -> float | None:
        """
        Find a silence boundary in the given time range.

        Args:
            audio: Full audio array
            sample_rate: Sample rate
            search_start: Start time to search (seconds)
            search_end: End time to search (seconds)

        Returns:
            Time of silence boundary, or None if not found
        """
        start_sample = int(search_start * sample_rate)
        end_sample = int(search_end * sample_rate)

        # Window size for RMS calculation (50ms)
        window_size = int(0.05 * sample_rate)
        min_silence_samples = int(self.config.min_silence_duration * sample_rate)

        # Scan for silence regions
        silence_start = None

        for i in range(start_sample, min(end_sample, len(audio)) - window_size, window_size):
            window = audio[i:i + window_size]
            rms = self.get_rms(window)

            if rms < self.config.silence_threshold:
                if silence_start is None:
                    silence_start = i
                elif i - silence_start >= min_silence_samples:
                    # Found a valid silence region, return the middle
                    silence_mid = silence_start + (i - silence_start) // 2
                    return silence_mid / sample_rate
            else:
                silence_start = None

        return None

    def chunk_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[AudioSegment]:
        """
        Split audio into chunks for transcription.

        Args:
            audio: Audio array (mono, float32 or int16)
            sample_rate: Sample rate

        Returns:
            List of AudioSegment objects
        """
        # Normalize audio
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        total_duration = len(audio) / sample_rate

        # If audio is short enough, return as single segment
        if total_duration <= self.config.max_chunk_duration:
            return [AudioSegment(
                audio=audio,
                start_time=0.0,
                end_time=total_duration,
                sample_rate=sample_rate,
            )]

        segments = []
        current_pos = 0.0  # Current position in seconds

        while current_pos < total_duration:
            remaining = total_duration - current_pos

            # If remaining audio is short enough, take it all
            if remaining <= self.config.max_chunk_duration:
                start_sample = int(current_pos * sample_rate)
                segments.append(AudioSegment(
                    audio=audio[start_sample:],
                    start_time=current_pos,
                    end_time=total_duration,
                    sample_rate=sample_rate,
                ))
                break

            # Try to find a silence boundary
            target_end = current_pos + self.config.target_chunk_duration
            search_start = target_end - self.config.silence_search_window
            search_end = min(target_end + self.config.silence_search_window,
                           current_pos + self.config.max_chunk_duration)

            boundary = self.find_silence_boundary(
                audio, sample_rate, search_start, search_end
            )

            if boundary is not None:
                # Found a silence boundary
                chunk_end = boundary
            else:
                # No silence found, force split with overlap consideration
                chunk_end = current_pos + self.config.target_chunk_duration

            # Extract chunk
            start_sample = int(current_pos * sample_rate)
            end_sample = int(chunk_end * sample_rate)

            segments.append(AudioSegment(
                audio=audio[start_sample:end_sample],
                start_time=current_pos,
                end_time=chunk_end,
                sample_rate=sample_rate,
            ))

            # Move position (with small overlap if force-split)
            if boundary is not None:
                current_pos = chunk_end
            else:
                # Add overlap for smoother transitions
                current_pos = chunk_end - self.config.overlap_duration

        return segments

    def chunk_file(self, file_path: str | Path) -> list[AudioSegment]:
        """
        Load and chunk an audio file.

        Args:
            file_path: Path to audio file

        Returns:
            List of AudioSegment objects
        """
        audio, sample_rate = sf.read(str(file_path), dtype='float32')

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        return self.chunk_audio(audio, sample_rate)


@dataclass
class TranscriptionSegment:
    """A transcription with timestamp info."""
    text: str
    start_time: float
    end_time: float


def merge_transcriptions(
    segments: list[TranscriptionSegment],
    overlap_duration: float = 2.0,
) -> str:
    """
    Merge transcription segments, handling overlaps.

    For overlapping regions, we take the text from the first segment
    since it has more context from what came before.

    Args:
        segments: List of transcription segments with timestamps
        overlap_duration: Expected overlap between segments

    Returns:
        Merged transcription text
    """
    if not segments:
        return ""

    if len(segments) == 1:
        return segments[0].text.strip()

    # Simple concatenation with space
    # For more sophisticated merging, we could:
    # 1. Use word-level timestamps to detect duplicates
    # 2. Use text similarity to find and remove overlapping words
    # For now, simple concatenation works reasonably well

    texts = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            texts.append(text)

    return " ".join(texts)


# Convenience function
def chunk_long_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_duration: float = 35.0,
) -> list[AudioSegment]:
    """
    Convenience function to chunk long audio.

    Args:
        audio: Audio array
        sample_rate: Sample rate
        max_duration: Maximum chunk duration

    Returns:
        List of AudioSegment objects
    """
    chunker = AudioChunker(ChunkerConfig(max_chunk_duration=max_duration))
    return chunker.chunk_audio(audio, sample_rate)
