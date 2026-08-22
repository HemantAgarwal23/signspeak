"""M1 verification script: live webcam + skeleton overlay + FPS.

Run from the `python/` directory:

    python -m src.preview
    python -m src.preview --camera 1 --print-vector

Keys:  q or ESC = quit,  v = print the current 126-D vector once.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

try:
    from .camera import Camera, FPSMeter
    from .config import FEATURE_DIM
    from .landmark_extractor import LandmarkExtractor, describe_layout
except ImportError:  # running as a plain script
    from camera import Camera, FPSMeter
    from config import FEATURE_DIM
    from landmark_extractor import LandmarkExtractor, describe_layout

GREEN = (80, 220, 100)
GREY = (170, 170, 170)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live landmark preview")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--print-vector", action="store_true",
                        help="print every frame's vector stats to stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(describe_layout())
    print("Opening camera... (q or ESC to quit, v to dump one vector)")

    fps_meter = FPSMeter()
    printed_shape = False

    with Camera(args.camera, args.width, args.height) as cam, \
            LandmarkExtractor() as extractor:
        while True:
            frame = cam.read()
            if frame is None:
                print("Dropped frame from camera; retrying.")
                continue

            vector = extractor.process(frame)
            extractor.draw_overlay(frame)
            fps = fps_meter.tick()

            if vector is not None and not printed_shape:
                print(f"vector shape={vector.shape} dtype={vector.dtype} "
                      f"nonzero={int(np.count_nonzero(vector))}/{FEATURE_DIM}")
                printed_shape = True
            if vector is not None and args.print_vector:
                print(f"hands={extractor.hands_detected} "
                      f"min={vector.min():+.3f} max={vector.max():+.3f}")

            hands = extractor.hands_detected
            # Handedness is shown so you can confirm the mirroring is right:
            # raise your left hand, it must say "Left".
            labels = ", ".join(extractor.handedness_labels)
            status = f"hands: {hands} [{labels}]" if hands else "no hand detected"
            cv2.putText(frame, f"FPS {fps:5.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            cv2.putText(frame, status, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        GREEN if hands else GREY, 2)

            cv2.imshow("M1 - landmark preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v") and vector is not None:
                np.set_printoptions(precision=4, suppress=True, linewidth=120)
                print(vector)


if __name__ == "__main__":
    main()
