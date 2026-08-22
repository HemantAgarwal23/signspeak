"""Collapse a sequence of landmark vectors into one vector.

Per-coordinate median over the time axis. Median rather than mean because
MediaPipe occasionally emits a badly wrong frame - a momentary misdetection or
a hand half out of shot - and a single outlier frame drags a mean but barely
moves a median.

This is the interface between "sequences on disk" and "one row per sample" that
every classifier consumes.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

try:
    from .config import FEATURE_DIM, SEQUENCE_LENGTH
except ImportError:  # running as a plain script
    from config import FEATURE_DIM, SEQUENCE_LENGTH


def aggregate(sequence: np.ndarray) -> np.ndarray:
    """(n_frames, 126) -> (126,). Also accepts (n_samples, n_frames, 126)."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected last axis {FEATURE_DIM}, got {array.shape}")
    if array.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got {array.shape}")
    return np.median(array, axis=-2).astype(np.float32)


class RollingBuffer:
    """The live counterpart to a recorded sequence.

    Holds the most recent `length` landmark vectors and collapses them the same
    way training samples were collapsed. Training and inference must aggregate
    identically or the model sees a different kind of input than it learned on.

    Frames where no hand was detected are simply not added, so the buffer holds
    the last `length` *good* frames rather than a window with holes in it -
    matching how data_collector records.
    """

    def __init__(self, length: int = SEQUENCE_LENGTH) -> None:
        self.length = length
        self._frames: deque = deque(maxlen=length)
        self._hand_mask: Optional[tuple] = None

    @staticmethod
    def _mask(vector: np.ndarray) -> tuple:
        """Which hand slots this frame filled, e.g. (False, True)."""
        half = len(vector) // 2
        return (bool(np.abs(vector[:half]).sum() > 0),
                bool(np.abs(vector[half:]).sum() > 0))

    def push(self, vector: Optional[np.ndarray]) -> None:
        if vector is None:
            return
        vector = np.asarray(vector, dtype=np.float32)

        # A window that mixes one-handed and two-handed frames medians into a
        # vector that never occurred: once over half the frames lack a hand,
        # that hand's 63 coordinates collapse to zero and a two-handed gesture
        # silently becomes a one-handed one. Changing hand count starts a fresh
        # window instead.
        mask = self._mask(vector)
        if mask != self._hand_mask:
            self._frames.clear()
            self._hand_mask = mask

        self._frames.append(vector)

    def value(self) -> Optional[np.ndarray]:
        """Median-aggregated 126-D vector, or None until the buffer fills."""
        if not self.ready:
            return None
        return aggregate(np.stack(self._frames))

    @property
    def ready(self) -> bool:
        return len(self._frames) == self.length

    @property
    def fill(self) -> float:
        return len(self._frames) / self.length

    @property
    def hands(self) -> int:
        """How many hands the current window holds. 0 when empty."""
        return sum(self._hand_mask) if self._hand_mask else 0

    def clear(self) -> None:
        self._frames.clear()
        self._hand_mask = None
