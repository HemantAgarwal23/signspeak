"""M6 - the Letter Merger: noisy per-frame predictions -> stable committed text.

A classifier fires 30 opinions a second and disagrees with itself constantly.
Four stages turn that stream into letters a person would accept:

    Stage 1  confidence filtering   drop predictions below tau
    Stage 2  buffer accumulation    count consecutive frames agreeing on a label
                                    and track their running mean confidence
    Stage 3  hold-time validation   commit only when frames >= MIN_FRAMES AND
                                    elapsed >= MIN_HOLD_TIME AND mean confidence
                                    >= tau, all three at once
    Stage 4  duplicate suppression  never commit the same letter twice in a row

Each stage can be disabled independently (see `Stages`), which is what M11's
ablation measures - the point is to quantify what the merger buys rather than
assert it.

Time is injected rather than read from the clock inside, so the ablation can
replay recorded sequences at arbitrary speed and still get identical results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from .config import MIN_FRAMES, MIN_HOLD_TIME, TAU
except ImportError:  # running as a plain script
    from config import MIN_FRAMES, MIN_HOLD_TIME, TAU


@dataclass
class Stages:
    """Which stages are active. All on = the full four-stage merger."""
    confidence_filter: bool = True
    accumulation: bool = True
    hold_time: bool = True
    duplicate_suppression: bool = True

    @classmethod
    def none(cls) -> "Stages":
        """Raw per-frame output - every prediction commits immediately."""
        return cls(False, False, False, False)


@dataclass
class _Candidate:
    """The label currently being accumulated toward a commit."""
    label: str
    frames: int = 0
    confidence_sum: float = 0.0
    started_at: float = 0.0
    committed: bool = False

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.frames if self.frames else 0.0

    def elapsed(self, now: float) -> float:
        return now - self.started_at


class LetterMerger:
    """Stateful. One per session; call `reset()` between words or demos."""

    def __init__(self, tau: float = TAU, min_frames: int = MIN_FRAMES,
                 min_hold_time: float = MIN_HOLD_TIME,
                 stages: Optional[Stages] = None,
                 idle_frames_to_forget: int = 15) -> None:
        self.tau = tau
        self.min_frames = min_frames
        self.min_hold_time = min_hold_time
        self.stages = stages or Stages()
        # How many rejected/absent frames clear the duplicate guard. Without
        # this, stage 4 makes double letters unspellable - "HELLO" would come
        # out "HELO". Dropping your hand between the two Ls is the natural
        # gesture, and this is what makes it work. The build spec does not
        # mention it; it is a deviation, and M11 measures it.
        self.idle_frames_to_forget = idle_frames_to_forget

        self.committed: List[str] = []
        self._candidate: Optional[_Candidate] = None
        self._last_committed: Optional[str] = None
        self._idle_frames = 0

    # -- main entry point ---------------------------------------------------
    def update(self, label: Optional[str], confidence: float,
               now: Optional[float] = None) -> Optional[str]:
        """Feed one frame's prediction. Returns a letter if one committed.

        `label=None` means no hand was visible this frame.
        """
        now = time.perf_counter() if now is None else now

        # Stage 1 - confidence filtering.
        if label is None or (self.stages.confidence_filter
                             and confidence < self.tau):
            self._on_idle_frame()
            return None

        # Stage 2 disabled: no accumulation, every accepted frame commits.
        if not self.stages.accumulation:
            return self._commit(label)

        self._idle_frames = 0

        # Stage 2 - buffer accumulation.
        if self._candidate is None or self._candidate.label != label:
            self._candidate = _Candidate(label=label, started_at=now)
        candidate = self._candidate
        candidate.frames += 1
        candidate.confidence_sum += confidence

        # Stage 3 - hold-time validation.
        if candidate.committed:
            return None
        if self.stages.hold_time and not self._is_held(candidate, now):
            return None

        candidate.committed = True
        return self._commit(label)

    def _is_held(self, candidate: _Candidate, now: float) -> bool:
        """All three conditions must hold simultaneously."""
        return (candidate.frames >= self.min_frames
                and candidate.elapsed(now) >= self.min_hold_time
                and candidate.mean_confidence >= self.tau)

    def _commit(self, label: str) -> Optional[str]:
        # Stage 4 - duplicate suppression.
        if self.stages.duplicate_suppression and label == self._last_committed:
            return None
        self.committed.append(label)
        self._last_committed = label
        return label

    def _on_idle_frame(self) -> None:
        self._idle_frames += 1
        self._candidate = None
        if self._idle_frames >= self.idle_frames_to_forget:
            # Long enough gap: the next occurrence of the same letter is a
            # deliberate new letter, not jitter on the old one.
            self._last_committed = None

    # -- UI surface ---------------------------------------------------------
    @property
    def state(self) -> Dict:
        """Everything a UI needs to show progress toward the next commit."""
        candidate = self._candidate
        if candidate is None:
            return {"label": None, "frames": 0, "progress": 0.0,
                    "mean_confidence": 0.0, "committed": False}

        now = time.perf_counter()
        by_frames = candidate.frames / self.min_frames if self.min_frames else 1.0
        by_time = (candidate.elapsed(now) / self.min_hold_time
                   if self.min_hold_time else 1.0)
        # Progress is the *slowest* of the two conditions - showing the fastest
        # would let the bar sit full while nothing commits.
        progress = min(1.0, min(by_frames, by_time))
        return {
            "label": candidate.label,
            "frames": candidate.frames,
            "progress": progress,
            "mean_confidence": candidate.mean_confidence,
            "committed": candidate.committed,
        }

    @property
    def text(self) -> str:
        return "".join(self.committed)

    def undo(self) -> Optional[str]:
        """Remove the last committed letter."""
        if not self.committed:
            return None
        removed = self.committed.pop()
        self._last_committed = self.committed[-1] if self.committed else None
        return removed

    def reset(self) -> None:
        self.committed.clear()
        self._candidate = None
        self._last_committed = None
        self._idle_frames = 0
