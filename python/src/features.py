"""Feature representation: how a raw 126-D landmark vector becomes model input.

Kept deliberately separate from recording. `data/raw/` and `data/packs/` always
hold **raw absolute** MediaPipe coordinates; any representation change happens
here, at training and inference time. That means:

- switching representation needs a retrain, never a re-record or re-import;
- every representation can be evaluated on the identical dataset, which is a
  free ablation row for the report.

Two transforms, applied in this order:

    wrist-relative   subtract each hand's wrist -> where the hand sits in the
                     frame stops mattering
    scale-normalise  divide each hand by its own size -> how big the hand looks
                     stops mattering

Both matter because training data comes from cropped 200x200 dataset images
while inference runs on a wide webcam frame. Without them, the two occupy
different coordinate ranges and nothing transfers.

Whatever is chosen here must be reimplemented identically in the browser
(`web/src/buffer.js`), applied at the same point in the pipeline.
"""
from __future__ import annotations

import numpy as np

try:
    from .config import (COORDS, FEATURE_DIM, FEATURE_SCALE_NORMALISE,
                         FEATURE_WRIST_RELATIVE, LANDMARKS_PER_HAND, MAX_HANDS)
except ImportError:  # running as a plain script
    from config import (COORDS, FEATURE_DIM, FEATURE_SCALE_NORMALISE,
                        FEATURE_WRIST_RELATIVE, LANDMARKS_PER_HAND, MAX_HANDS)

_EPS = 1e-8


def _as_hand_blocks(x: np.ndarray) -> np.ndarray:
    """(..., 126) -> (..., 2, 21, 3), as a copy."""
    array = np.asarray(x, dtype=np.float32)
    if array.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected last axis {FEATURE_DIM}, got {array.shape}")
    return array.reshape(*array.shape[:-1], MAX_HANDS, LANDMARKS_PER_HAND,
                         COORDS).copy()


def _flatten(blocks: np.ndarray, like: np.ndarray) -> np.ndarray:
    return blocks.reshape(*np.asarray(like).shape[:-1],
                          FEATURE_DIM).astype(np.float32)


def _present(blocks: np.ndarray) -> np.ndarray:
    """Mask of hands that were actually detected, shaped for broadcasting.

    An absent hand is all-zero and must stay all-zero. Without this mask it
    would pick up a negative wrist offset and start looking like real data.
    """
    return np.abs(blocks).sum(axis=(-1, -2), keepdims=True) > 0


def to_wrist_relative(x: np.ndarray) -> np.ndarray:
    """Subtract each hand's wrist (landmark 0) from that hand's 21 landmarks."""
    blocks = _as_hand_blocks(x)
    out = np.where(_present(blocks), blocks - blocks[..., :1, :], blocks)
    return _flatten(out, x)


def to_scale_normalised(x: np.ndarray) -> np.ndarray:
    """Divide each hand by its own size: the largest distance from its wrist.

    Assumes the input is already wrist-relative - scaling raw absolute
    coordinates would be meaningless. `transform()` enforces the ordering.
    """
    blocks = _as_hand_blocks(x)
    present = _present(blocks)
    size = np.linalg.norm(blocks, axis=-1).max(axis=-1)[..., None, None]
    out = np.where(present, blocks / np.maximum(size, _EPS), blocks)
    return _flatten(out, x)


def transform(x: np.ndarray,
              wrist_relative: bool = FEATURE_WRIST_RELATIVE,
              scale_normalise: bool = FEATURE_SCALE_NORMALISE) -> np.ndarray:
    """Raw landmark vector(s) -> model input. Shape is unchanged.

    Accepts a single (126,) vector, a (n, 126) batch, or a (n, 30, 126)
    sequence batch. Scale normalisation implies wrist-relative.
    """
    array = np.asarray(x, dtype=np.float32)
    if scale_normalise:
        wrist_relative = True
    if wrist_relative:
        array = to_wrist_relative(array)
    if scale_normalise:
        array = to_scale_normalised(array)
    return array


def describe() -> str:
    """One line naming the active representation. Log this at train time."""
    return (f"features: wrist_relative={FEATURE_WRIST_RELATIVE} "
            f"scale_normalise={FEATURE_SCALE_NORMALISE} (dim {FEATURE_DIM})")
