"""
LocalAgreement algorithm for streaming ASR.

Based on whisper_streaming's LocalAgreement-n policy:
- Confirm transcription when n consecutive updates agree on a prefix
- Prevents flickering/unstable output from batch model in streaming mode
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from config import config


@dataclass
class TranscriptionResult:
    """Result from LocalAgreement processing."""
    confirmed_text: str      # Text that is confirmed (stable)
    pending_text: str        # Text that is pending confirmation
    is_final: bool = False   # Is this the final result (end of utterance)?

    @property
    def full_text(self) -> str:
        """Get full text (confirmed + pending)."""
        if self.confirmed_text and self.pending_text:
            return f"{self.confirmed_text} {self.pending_text}".strip()
        return self.confirmed_text or self.pending_text


@dataclass
class LocalAgreement:
    """
    LocalAgreement algorithm implementation.

    Confirms text when n consecutive transcriptions agree on a prefix.
    This provides stable streaming output from a batch ASR model.
    """

    min_agreement: int = None
    prefix_match_ratio: float = None

    # Internal state
    _history: list[str] = field(default_factory=list)
    _confirmed: str = ""
    _last_pending: str = ""

    def __post_init__(self):
        if self.min_agreement is None:
            self.min_agreement = config.local_agreement.min_agreement
        if self.prefix_match_ratio is None:
            self.prefix_match_ratio = config.local_agreement.prefix_match_ratio
        self._history = []
        self._confirmed = ""
        self._last_pending = ""

    def _get_common_prefix(self, text1: str, text2: str) -> str:
        """Get the longest common prefix between two texts."""
        words1 = text1.split()
        words2 = text2.split()

        common = []
        for w1, w2 in zip(words1, words2):
            if w1.lower() == w2.lower():
                common.append(w1)
            else:
                break

        return " ".join(common)

    def _texts_agree(self, text1: str, text2: str) -> bool:
        """Check if two texts agree (similar enough)."""
        if not text1 or not text2:
            return False

        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        if t1 == t2:
            return True

        # Check if one is prefix of other
        if t1.startswith(t2) or t2.startswith(t1):
            return True

        # Check similarity ratio
        ratio = SequenceMatcher(None, t1, t2).ratio()
        return ratio >= self.prefix_match_ratio

    def _find_agreed_prefix(self) -> str | None:
        """Find prefix that all recent transcriptions agree on."""
        if len(self._history) < self.min_agreement:
            return None

        recent = self._history[-self.min_agreement:]

        # Start with first text as candidate
        candidate = recent[0]
        if not candidate:
            return None

        # Find common prefix across all
        for text in recent[1:]:
            candidate = self._get_common_prefix(candidate, text)
            if not candidate:
                return None

        # Verify all agree on this prefix
        for text in recent:
            if not text.lower().startswith(candidate.lower()):
                return None

        return candidate

    def process(self, transcription: str, is_final: bool = False) -> TranscriptionResult:
        """
        Process a new transcription and return confirmed/pending text.

        Args:
            transcription: New transcription from ASR model
            is_final: Is this the final transcription (end of utterance)?

        Returns:
            TranscriptionResult with confirmed and pending text
        """
        transcription = transcription.strip()

        if is_final:
            # Final transcription - confirm everything
            result = TranscriptionResult(
                confirmed_text=self._confirmed,
                pending_text=transcription[len(self._confirmed):].strip() if transcription.startswith(self._confirmed) else transcription,
                is_final=True,
            )
            # Include all as confirmed for final
            full = result.full_text
            self.reset()
            return TranscriptionResult(confirmed_text=full, pending_text="", is_final=True)

        # Add to history
        self._history.append(transcription)

        # Trim history to avoid unbounded growth
        if len(self._history) > self.min_agreement * 3:
            self._history = self._history[-self.min_agreement * 2:]

        # Find agreed prefix
        agreed_prefix = self._find_agreed_prefix()

        if agreed_prefix and len(agreed_prefix) > len(self._confirmed):
            # New confirmed text
            self._confirmed = agreed_prefix
            self._history = []  # Reset history after confirmation

        # Pending is what's beyond confirmed in latest transcription
        if transcription.lower().startswith(self._confirmed.lower()):
            pending = transcription[len(self._confirmed):].strip()
        else:
            pending = transcription

        self._last_pending = pending

        return TranscriptionResult(
            confirmed_text=self._confirmed,
            pending_text=pending,
            is_final=False,
        )

    def reset(self) -> None:
        """Reset state for new utterance."""
        self._history = []
        self._confirmed = ""
        self._last_pending = ""

    @property
    def confirmed(self) -> str:
        """Get currently confirmed text."""
        return self._confirmed

    @property
    def has_confirmed(self) -> bool:
        """Check if there's any confirmed text."""
        return bool(self._confirmed)
