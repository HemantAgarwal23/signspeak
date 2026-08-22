"""Webcam helpers shared by every capture loop.

One place owns the mirroring decision. Everything downstream - extraction,
drawing, recording, prediction - sees the already-mirrored frame, so the
landmark vectors are consistent across the whole project.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2

try:
    from .config import MIRROR_INPUT
except ImportError:  # running as a plain script
    from config import MIRROR_INPUT


class Camera:
    """Context-managed webcam that yields mirrored BGR frames."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 mirror: bool = MIRROR_INPUT) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "Camera":
        # CAP_DSHOW avoids the multi-second MSMF startup stall on Windows.
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.index}. Close any other app using "
                f"the webcam, or pass a different --camera index."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self

    def read(self):
        """Return a mirrored BGR frame, or None if the grab failed."""
        if self.cap is None:
            raise RuntimeError("Camera.open() was never called")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


class FPSMeter:
    """Rolling frames-per-second estimate over the last `window` frames."""

    def __init__(self, window: int = 30) -> None:
        self.times: deque[float] = deque(maxlen=window)
        self.last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self.times.append(now - self.last)
        self.last = now
        mean = sum(self.times) / len(self.times)
        return 1.0 / mean if mean > 0 else 0.0
