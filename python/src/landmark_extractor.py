"""MediaPipe Hands wrapper: one BGR frame in, one 126-D float32 vector out.

Uses the MediaPipe **Tasks** API (`HandLandmarker`). This is deliberate: the
browser build (M15) uses `@mediapipe/tasks-vision` HandLandmarker with the same
`hand_landmarker.task` asset, so both sides run identical graphs and the M16
numeric parity check has a chance of passing. The legacy `mp.solutions.hands`
API does not exist in mediapipe >= 1.0.

VECTOR LAYOUT - this ordering is contractual. The browser port must reproduce it
exactly or the exported model returns nonsense.

    index   0 ..  62  -> hand slot 0 = user's real LEFT hand,  21 x (x, y, z)
    index  63 .. 125  -> hand slot 1 = user's real RIGHT hand, 21 x (x, y, z)

Within a hand block the landmarks are in MediaPipe's native order
(0 = wrist, 4 = thumb tip, 8 = index tip, ... 20 = pinky tip) and each
landmark contributes exactly three consecutive floats: x, then y, then z.

Coordinates are always RAW and ABSOLUTE here. Any representation change
(wrist-relative and so on) belongs in features.py, applied at train/inference
time, so the recorded dataset never has to be thrown away.

A missing hand leaves its 63-float block as zeros. Handedness - not detection
order - decides the slot, so the same physical hand always lands in the same
half of the vector.

Frames are mirrored at capture time (see camera.Camera) but MediaPipe reports
handedness for the UN-mirrored view, so its label is inverted relative to the
user's real hand. config.FLIP_HANDEDNESS corrects this. Slot assignment is
always expressed in terms of the user's REAL hands.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker,
                                           HandLandmarkerOptions,
                                           HandLandmarksConnections,
                                           RunningMode)

try:
    from .config import (COORDS, DETECTION_CONFIDENCE, FEATURE_DIM,
                         FLIP_HANDEDNESS, HAND_BLOCK, HAND_SLOTS,
                         HAND_TASK_PATH, LANDMARKS_PER_HAND, MAX_HANDS,
                         PRESENCE_CONFIDENCE, TRACKING_CONFIDENCE)
except ImportError:  # running as a plain script
    from config import (COORDS, DETECTION_CONFIDENCE, FEATURE_DIM,
                        FLIP_HANDEDNESS, HAND_BLOCK, HAND_SLOTS,
                        HAND_TASK_PATH, LANDMARKS_PER_HAND, MAX_HANDS,
                        PRESENCE_CONFIDENCE, TRACKING_CONFIDENCE)

# (start, end) landmark index pairs for the skeleton overlay.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = tuple(
    (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
)

_JOINT_COLOUR = (60, 190, 255)
_BONE_COLOUR = (80, 220, 100)


class LandmarkExtractor:
    """Stateful wrapper around a HandLandmarker running in VIDEO mode.

    Stateful on purpose: VIDEO mode tracks hands across frames, which is both
    faster and steadier than re-detecting every frame. Create one instance per
    capture loop and reuse it. Not thread-safe.
    """

    def __init__(self,
                 max_hands: int = MAX_HANDS,
                 detection_confidence: float = DETECTION_CONFIDENCE,
                 presence_confidence: float = PRESENCE_CONFIDENCE,
                 tracking_confidence: float = TRACKING_CONFIDENCE,
                 flip_handedness: bool = FLIP_HANDEDNESS,
                 static: bool = False,
                 model_path=HAND_TASK_PATH) -> None:
        model_path = str(model_path)
        if not HAND_TASK_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipe model asset missing: {model_path}\n"
                f"Run:  python -m src.download_models"
            )

        self.flip_handedness = flip_handedness
        self.static = static
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE if static else RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._result = None
        # VIDEO mode demands strictly increasing timestamps, so we keep our own
        # monotonic counter rather than trusting wall-clock resolution.
        self._epoch = time.perf_counter()
        self._last_timestamp_ms = -1

    # -- inference ---------------------------------------------------------
    def process(self, frame_bgr: np.ndarray,
                force_slot: Optional[int] = None) -> Optional[np.ndarray]:
        """Run detection on a mirrored BGR frame.

        Returns a (126,) float32 array, or None when no hand was found. The
        raw result is cached so draw_overlay() can reuse it without a second
        inference pass.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        if self.static:
            # No temporal tracking: every image is judged on its own.
            result = self._landmarker.detect(image)
        else:
            timestamp_ms = int((time.perf_counter() - self._epoch) * 1000)
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
            result = self._landmarker.detect_for_video(image, timestamp_ms)

        self._result = result
        if not result.hand_landmarks:
            return None
        return self._to_vector(result, force_slot)

    def _to_vector(self, result, force_slot: Optional[int] = None) -> np.ndarray:
        vector = np.zeros(FEATURE_DIM, dtype=np.float32)
        filled = [False] * MAX_HANDS
        handedness = result.handedness or []

        for i, landmarks in enumerate(result.hand_landmarks):
            raw = (handedness[i][0].category_name
                   if i < len(handedness) and handedness[i] else HAND_SLOTS[0])
            label = self._user_hand(raw)
            slot = HAND_SLOTS.index(label) if label in HAND_SLOTS else 0
            if force_slot is not None:
                # Third-party stills give no reliable clue whether the image was
                # mirrored, so handedness cannot be trusted. Pinning the slot
                # sidesteps the question entirely for single-hand data.
                slot = force_slot

            # Two hands classified the same way (happens when hands overlap):
            # the second one takes whichever slot is still free.
            if filled[slot]:
                alternative = 1 - slot
                if alternative >= MAX_HANDS or filled[alternative]:
                    continue
                slot = alternative

            block = np.array([[p.x, p.y, p.z] for p in landmarks],
                             dtype=np.float32)          # (21, 3)
            start = slot * HAND_BLOCK
            vector[start:start + HAND_BLOCK] = block.reshape(-1)
            filled[slot] = True

        return vector

    def _user_hand(self, raw_label: str) -> str:
        """MediaPipe's handedness label -> the user's real hand."""
        if not self.flip_handedness:
            return raw_label
        return "Right" if raw_label == "Left" else "Left"

    # -- preview -----------------------------------------------------------
    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw the cached skeleton onto the frame, in place.

        The Tasks API ships no drawing helper, so this is hand-rolled: scale the
        normalised coordinates back to pixels, stroke the bones, dot the joints.
        """
        result = self._result
        if not result or not result.hand_landmarks:
            return frame_bgr

        height, width = frame_bgr.shape[:2]
        for landmarks in result.hand_landmarks:
            points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
            for start, end in HAND_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(frame_bgr, points[start], points[end],
                             _BONE_COLOUR, 2)
            for point in points:
                cv2.circle(frame_bgr, point, 3, _JOINT_COLOUR, -1)
        return frame_bgr

    @property
    def hands_detected(self) -> int:
        result = self._result
        if not result or not result.hand_landmarks:
            return 0
        return len(result.hand_landmarks)

    @property
    def handedness_labels(self) -> List[str]:
        """The user's REAL hands, e.g. ["Left"] when they raise their left."""
        return [self._user_hand(raw) for raw in self.raw_handedness_labels]

    @property
    def raw_handedness_labels(self) -> List[str]:
        """Uncorrected MediaPipe labels. Debugging only."""
        result = self._result
        if not result or not result.handedness:
            return []
        return [h[0].category_name for h in result.handedness if h]

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def describe_layout() -> str:
    """Human-readable index map. Paste this into the browser-port prompt."""
    lines = [f"FEATURE_DIM = {FEATURE_DIM} "
             f"({MAX_HANDS} hands x {LANDMARKS_PER_HAND} landmarks x {COORDS} coords)"]
    for slot, name in enumerate(HAND_SLOTS):
        start = slot * HAND_BLOCK
        lines.append(
            f"  slot {slot} (user's real {name} hand): "
            f"idx {start}..{start + HAND_BLOCK - 1} -> "
            f"landmark k at idx {start}+3k (x), +3k+1 (y), +3k+2 (z)"
        )
    lines.append(
        f"  frame is mirrored before detection; MediaPipe handedness is flipped "
        f"to match the user's real hand (FLIP_HANDEDNESS={FLIP_HANDEDNESS})"
    )
    return "\n".join(lines)
