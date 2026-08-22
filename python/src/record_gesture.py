"""Define a custom gesture: hold a shape, name it, map it to a phrase.

    python -m src.record_gesture --name thumbs_up --phrase "OK!"
    python -m src.record_gesture --list
    python -m src.record_gesture --delete thumbs_up
    python -m src.record_gesture --set-phrase thumbs_up --phrase "All good"

Recording flow: 3-second countdown, then 20 samples. Each sample is the median
of a rolling 30-frame window, exactly matching how letter samples are
aggregated - the KNN and the SVM must see the same kind of input.

Samples are taken every --stride frames rather than every frame. Consecutive
windows overlap by 29 of 30 frames, so sampling continuously produces 20
near-identical copies of a single moment: measured at 0.014 apart, against
2.8+ between genuinely different gestures. That fake tightness then poisons
the reject threshold derived from it.

Keep moving the gesture slightly throughout. The point of 20 samples is to
capture how the gesture varies, not to record the same instant 20 times.
"""
from __future__ import annotations

import argparse
import time
from typing import List, Optional

import cv2
import numpy as np

try:
    from .aggregator import RollingBuffer
    from .camera import Camera, FPSMeter
    from .config import CUSTOM_GESTURE_SAMPLES, ensure_dirs
    from .custom_gestures import GestureStore, KNNClassifier
    from .landmark_extractor import LandmarkExtractor
except ImportError:  # running as a plain script
    from aggregator import RollingBuffer
    from camera import Camera, FPSMeter
    from config import CUSTOM_GESTURE_SAMPLES, ensure_dirs
    from custom_gestures import GestureStore, KNNClassifier
    from landmark_extractor import LandmarkExtractor

GREEN = (80, 220, 100)
AMBER = (60, 190, 255)
RED = (60, 60, 235)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
WINDOW = "record custom gesture"

# How close a new gesture may sit to an existing one before we warn. Expressed
# as a fraction of the KNN's own reject threshold, so it tracks the feature
# scale rather than being a magic number.
CONFUSION_FRACTION = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a custom gesture")
    parser.add_argument("--name", help="gesture name, e.g. thumbs_up")
    parser.add_argument("--phrase", default="",
                        help='what it should output, e.g. "I need help"')
    parser.add_argument("--samples", type=int, default=CUSTOM_GESTURE_SAMPLES)
    parser.add_argument("--stride", type=int, default=6,
                        help="frames to skip between samples. Consecutive "
                             "windows overlap by 29 of 30 frames, so sampling "
                             "every frame yields 20 near-identical copies")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--list", action="store_true", help="list gestures and exit")
    parser.add_argument("--delete", metavar="NAME", help="delete a gesture and exit")
    parser.add_argument("--set-phrase", metavar="NAME",
                        help="change a gesture's phrase (use with --phrase)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing gesture without asking")
    return parser.parse_args()


def banner(frame: np.ndarray, lines: List[tuple], y0: int = 28) -> None:
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (10, y0 + i * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, colour, 2)


def progress_bar(frame: np.ndarray, fraction: float, colour=GREEN) -> None:
    height, width = frame.shape[:2]
    x0, x1, y0, y1 = 10, width - 10, height - 32, height - 14
    cv2.rectangle(frame, (x0, y0), (x1, y1), GREY, 1)
    filled = int((x1 - x0 - 2) * max(0.0, min(1.0, fraction)))
    if filled > 0:
        cv2.rectangle(frame, (x0 + 1, y0 + 1), (x0 + 1 + filled, y1 - 1),
                      colour, -1)


def capture(name: str, args: argparse.Namespace) -> Optional[np.ndarray]:
    """Run the countdown and collect samples. None if the user quit."""
    buffer = RollingBuffer()
    samples: List[np.ndarray] = []
    fps_meter = FPSMeter()
    phase = "countdown"
    since_last = 0
    started: Optional[float] = None
    capture_started: Optional[float] = None
    elapsed = 0.0

    with Camera(args.camera, args.width, args.height) as cam, \
            LandmarkExtractor() as extractor:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            vector = extractor.process(frame)
            extractor.draw_overlay(frame)
            buffer.push(vector)

            now = time.perf_counter()
            hands = extractor.hands_detected
            lines = [(f"gesture: {name}    fps {fps_meter.tick():4.1f}", WHITE)]

            if phase == "countdown":
                # The countdown only runs while a hand is visible, so it cannot
                # expire while the user is still getting into position.
                if not hands:
                    started = None
                    lines.append(("SHOW YOUR HAND", RED))
                    progress_bar(frame, 0.0, RED)
                else:
                    started = started if started is not None else now
                    remaining = args.countdown - (now - started)
                    if remaining <= 0:
                        phase = "capture"
                        capture_started = now
                    lines.append((f"get ready: {max(0.0, remaining):0.1f}s", AMBER))
                    progress_bar(frame, 1.0 - max(0.0, remaining) / args.countdown,
                                 AMBER)

            elif phase == "capture":
                if buffer.ready and vector is not None:
                    since_last += 1
                    if since_last >= args.stride or not samples:
                        samples.append(buffer.value())
                        since_last = 0
                lines.append((f"RECORDING {len(samples)}/{args.samples}  "
                              f"[{buffer.hands}h]", GREEN))
                lines.append(("keep moving it slightly - vary the angle", GREY))
                progress_bar(frame, len(samples) / args.samples)
                if not buffer.ready:
                    # The window restarts whenever the hand count changes, so a
                    # flickering second hand shows up here as a stalled fill.
                    lines.append((f"filling buffer {buffer.fill:.0%}  "
                                  f"keep {buffer.hands or 'both'} hand(s) steady",
                                  AMBER))
                if len(samples) >= args.samples:
                    elapsed = now - (capture_started or now)
                    break

            cv2.imshow(WINDOW, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                cv2.destroyAllWindows()
                return None

    cv2.destroyAllWindows()
    print(f"captured {len(samples)} samples in {elapsed:.2f}s")
    return np.stack(samples).astype(np.float32)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    store = GestureStore.load()

    if args.list:
        print(store.summary())
        return

    if args.delete:
        if store.remove(args.delete):
            store.save()
            print(f"deleted {args.delete!r}")
        else:
            print(f"no gesture named {args.delete!r}")
        print(store.summary())
        return

    if args.set_phrase:
        if store.set_phrase(args.set_phrase, args.phrase):
            store.save()
            print(f"{args.set_phrase!r} now outputs {args.phrase!r}")
        else:
            print(f"no gesture named {args.set_phrase!r}")
        return

    if not args.name:
        raise SystemExit("--name is required (or use --list / --delete)")

    name = args.name.strip()
    if name in store.gestures and not args.force:
        raise SystemExit(f"{name!r} already exists. Use --force to overwrite, "
                         f"or --delete it first.")

    print(f"recording {name!r}" + (f" -> {args.phrase!r}" if args.phrase else ""))
    print("hold the gesture steady once the countdown starts. q to abort.")

    samples = capture(name, args)
    if samples is None:
        print("aborted - nothing saved")
        return

    # Confusion warning: is this too close to something already recorded?
    if len(store):
        classifier = KNNClassifier(store)
        nearest, distance = classifier.nearest_gesture(samples, exclude=name)
        # Compared against that gesture's own threshold, not a shared average:
        # a wide two-handed gesture tolerates more distance before two shapes
        # genuinely risk being confused.
        limit = classifier.threshold_for(nearest or "") * CONFUSION_FRACTION
        if nearest and distance < limit:
            print(f"\nWARNING: this looks a lot like {nearest!r} "
                  f"(distance {distance:.3f}, warning below {limit:.3f}).")
            print("They will be confused with each other. Consider re-recording "
                  "with a more distinct shape.")

    store.add(name, samples, phrase=args.phrase)
    store.save()

    classifier = KNNClassifier(store)
    seconds = classifier.fit(store)
    print(f"\nsaved. KNN rebuilt in {seconds * 1000:.1f} ms")
    print(classifier.describe())
    print()
    print(store.summary())


if __name__ == "__main__":
    main()
