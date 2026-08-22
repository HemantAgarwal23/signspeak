"""M2 - interactive recording tool.

Per sample: live preview -> 3-second countdown -> capture exactly 30 frames in
which a hand was detected -> save (30, 126) to disk -> 1-second pause.

Frames with no hand are skipped and retried, never zero-filled, so every stored
sequence is complete. A sample that cannot gather 30 good frames within
--timeout seconds is abandoned and retried.

Run from the `python/` directory:

    python -m src.data_collector --label A --count 60 --subject s01
    python -m src.data_collector --label A --count 5 --subject s01 --dry-run

Keys during a run:
    q or ESC  quit (samples already saved are kept)
    r         abandon the sample being captured and retry it
    p         pause; any key resumes
"""
from __future__ import annotations

import argparse
import time
from typing import List, Optional

import cv2
import numpy as np

try:
    from .camera import Camera, FPSMeter
    from .config import SEQUENCE_LENGTH, ensure_dirs
    from .dataset import (add_sample, load_manifest, next_index, save_manifest,
                          summary)
    from .landmark_extractor import LandmarkExtractor
except ImportError:  # running as a plain script
    from camera import Camera, FPSMeter
    from config import SEQUENCE_LENGTH, ensure_dirs
    from dataset import (add_sample, load_manifest, next_index, save_manifest,
                         summary)
    from landmark_extractor import LandmarkExtractor

GREEN = (80, 220, 100)
AMBER = (60, 190, 255)
RED = (60, 60, 235)
WHITE = (245, 245, 245)
WINDOW = "M2 - data collector"

QUIT = "quit"
RETRY = "retry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record gesture samples")
    parser.add_argument("--label", required=True, help="class label, e.g. A")
    parser.add_argument("--count", type=int, default=60,
                        help="number of samples to record in this run")
    parser.add_argument("--subject", required=True,
                        help="subject id, e.g. s01 - needed for M9 cross-subject eval")
    parser.add_argument("--countdown", type=float, default=3.0,
                        help="seconds of countdown before each capture")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="seconds of rest between samples")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="abandon a sample if 30 good frames take longer than this")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true",
                        help="run the full flow but write nothing to disk")
    return parser.parse_args()


def make_header(label: str, subject: str, recorded: int, count: int):
    """Build a per-frame header renderer.

    Returns a function of the current FPS, so the reading is refreshed every
    frame rather than once per sample.
    """
    def header(fps: float) -> List[tuple]:
        return [
            (f"label {label}   subject {subject}", WHITE),
            (f"sample {recorded + 1}/{count}   fps {fps:4.1f}", WHITE),
        ]
    return header


def banner(frame: np.ndarray, lines: List[tuple], y0: int = 30) -> None:
    """Draw (text, colour) lines stacked from the top-left."""
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


def progress_bar(frame: np.ndarray, fraction: float, colour=GREEN) -> None:
    height, width = frame.shape[:2]
    x0, x1, y0, y1 = 10, width - 10, height - 34, height - 14
    cv2.rectangle(frame, (x0, y0), (x1, y1), WHITE, 1)
    filled = int((x1 - x0 - 2) * max(0.0, min(1.0, fraction)))
    if filled > 0:
        cv2.rectangle(frame, (x0 + 1, y0 + 1), (x0 + 1 + filled, y1 - 1), colour, -1)


def poll_key() -> Optional[str]:
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return QUIT
    if key == ord("r"):
        return RETRY
    if key == ord("p"):
        cv2.waitKey(0)                      # blocks until any key
    return None


def countdown(cam: Camera, extractor: LandmarkExtractor, seconds: float,
              header, fps_meter: FPSMeter) -> Optional[str]:
    """Show a live countdown. Returns QUIT or RETRY if the user pressed a key."""
    start = time.perf_counter()
    while True:
        remaining = seconds - (time.perf_counter() - start)
        if remaining <= 0:
            return None
        frame = cam.read()
        if frame is None:
            continue
        extractor.process(frame)
        extractor.draw_overlay(frame)

        hands = extractor.hands_detected
        banner(frame, header(fps_meter.tick()) + [
            (f"get ready: {remaining:0.1f}s", AMBER if hands else RED),
            ("hand detected" if hands else "SHOW YOUR HAND",
             GREEN if hands else RED),
        ])
        progress_bar(frame, 1.0 - remaining / seconds, AMBER)
        cv2.imshow(WINDOW, frame)

        signal = poll_key()
        if signal:
            return signal


def capture_sequence(cam: Camera, extractor: LandmarkExtractor, timeout: float,
                     header, fps_meter: FPSMeter) -> tuple:
    """Collect SEQUENCE_LENGTH vectors from frames where a hand was found.

    Returns (sequence, signal): a (30, 126) array on success, or a
    QUIT / RETRY / None signal (None meaning it timed out).
    """
    frames: List[np.ndarray] = []
    dropped = 0
    start = time.perf_counter()

    while len(frames) < SEQUENCE_LENGTH:
        if time.perf_counter() - start > timeout:
            return None, None               # timed out; caller retries
        frame = cam.read()
        if frame is None:
            continue

        vector = extractor.process(frame)
        extractor.draw_overlay(frame)
        if vector is not None:
            frames.append(vector)
        else:
            dropped += 1

        banner(frame, header(fps_meter.tick()) + [
            (f"RECORDING {len(frames)}/{SEQUENCE_LENGTH}", GREEN),
            (f"skipped (no hand): {dropped}", AMBER if dropped else WHITE),
        ])
        progress_bar(frame, len(frames) / SEQUENCE_LENGTH)
        cv2.imshow(WINDOW, frame)

        signal = poll_key()
        if signal:
            return None, signal

    return np.stack(frames).astype(np.float32), None


def rest(cam: Camera, extractor: LandmarkExtractor, seconds: float,
         header, fps_meter: FPSMeter) -> Optional[str]:
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        frame = cam.read()
        if frame is None:
            continue
        extractor.process(frame)
        extractor.draw_overlay(frame)
        banner(frame, header(fps_meter.tick()) + [("saved - relax", GREEN)])
        cv2.imshow(WINDOW, frame)
        if poll_key() == QUIT:
            return QUIT
    return None


def main() -> None:
    args = parse_args()
    ensure_dirs()

    label = args.label.strip()
    subject = args.subject.strip()
    manifest = load_manifest()

    already = sum(1 for e in manifest["samples"]
                  if e["label"] == label and e["subject"] == subject)
    print(f"label={label} subject={subject} target={args.count} "
          f"(already on disk for this pair: {already})")
    if args.dry_run:
        print("DRY RUN - nothing will be written to disk")

    recorded = 0
    fps_meter = FPSMeter()

    with Camera(args.camera, args.width, args.height) as cam, \
            LandmarkExtractor() as extractor:
        while recorded < args.count:
            header = make_header(label, subject, recorded, args.count)

            signal = countdown(cam, extractor, args.countdown, header, fps_meter)
            if signal == QUIT:
                break
            if signal == RETRY:
                continue

            sequence, signal = capture_sequence(cam, extractor, args.timeout,
                                               header, fps_meter)
            if signal == QUIT:
                break
            if signal == RETRY:
                continue
            if sequence is None:
                print("  timed out waiting for a steady hand - retrying this sample")
                continue

            if args.dry_run:
                print(f"  [dry-run] would save shape={sequence.shape}")
            else:
                index = next_index(manifest, label, subject)
                path = add_sample(manifest, sequence, label, subject, index)
                # Written after every sample: a crash never costs more than one.
                save_manifest(manifest)
                print(f"  saved {path.name}  shape={sequence.shape}")
            recorded += 1

            if rest(cam, extractor, args.pause, header, fps_meter) == QUIT:
                break

    print(f"\nrecorded {recorded} sample(s) this run")
    print(summary(load_manifest()))


if __name__ == "__main__":
    main()
