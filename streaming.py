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


@dataclass
class StreamingResult:
    """Result from streaming transcription."""
    text: str                  # Current text (confirmed + pending)
    confirmed_text: str        # Confirmed (stable) text
    pending_text: str          # Pending text
    is_final: bool             # Is this the final result?
    latency_ms: float = 0      # Last inference latency in ms
    audio_duration: float = 0  # Audio duration processed


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

    # Callbacks
    on_result: Callable[[StreamingResult], None] = None

    # State
    _started: bool = False
    _total_audio_duration: float = 0

    def __post_init__(self):
        if self.model is None:
            self.model = ASRModel.get_instance()
        if self.buffer is None:
            self.buffer = AudioBuffer()
        if self.vad is None:
            self.vad = VAD()
        if self.agreement is None:
            self.agreement = LocalAgreement()

    def start(self) -> None:
        """Start a new streaming session."""
        self.model.ensure_loaded()
        self.buffer.clear()
        self.vad.reset()
        self.agreement.reset()
        self._started = True
        self._total_audio_duration = 0

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

        # Add to buffer
        self.buffer.add(audio)

        # Check VAD (only if enabled)
        speech_ended = False
        if config.vad.enabled:
            is_speech, speech_ended = self.vad.process(audio)

        results = []

        # Process chunks if available
        while self.buffer.has_chunk():
            chunk = self.buffer.get_next_chunk()
            if chunk:
                result = self._process_chunk(chunk)
                if result:
                    results.append(result)
                    if self.on_result:
                        self.on_result(result)

        # If speech ended (VAD detected silence), process remaining and finalize
        if speech_ended:
            remaining = self.buffer.get_remaining(mark_final=True)
            if remaining:
                result = self._process_chunk(remaining)
                if result:
                    results.append(result)
                    if self.on_result:
                        self.on_result(result)

            # Reset for next utterance
            self.buffer.clear()
            self.agreement.reset()

        return results

    def _process_chunk(self, chunk: AudioChunk) -> StreamingResult | None:
        """Process a single audio chunk."""
        # Transcribe
        transcribe_result = self.model.transcribe_audio(chunk.audio)

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
