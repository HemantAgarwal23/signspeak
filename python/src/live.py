"""M5 + M6 + custom gestures - the whole system in one window.

    python -m src.live
    python -m src.live --mode custom  # start in custom-gesture mode
    python -m src.live --raw          # no merger, see the flicker it fixes

Camera -> landmarks -> 30-frame rolling buffer -> median -> classifier ->
merger -> text.

Two classifiers share that pipeline and the same feature space:
    ASL mode     pretrained RBF SVM over 24 fingerspelled letters
    custom mode  KNN over gestures you recorded yourself, each mapped to a
                 phrase - record them with `python -m src.record_gesture`

Both feed the same merger, so a custom gesture has to be held just as
deliberately as a letter before it commits.

Keys (the video window must have focus, not the terminal):
    q / ESC   quit             m  switch mode (ASL <-> custom)
    u         undo             c  clear output
    r         reset merger     SPACE  add a space
"""
from __future__ import annotations

import argparse
from typing import List

import cv2
import numpy as np

try:
    from .aggregator import RollingBuffer
    from .camera import Camera, FPSMeter
    from .config import TAU
    from .custom_gestures import GestureStore, KNNClassifier
    from .landmark_extractor import LandmarkExtractor
    from .merger import LetterMerger, Stages
    from .predictor import Predictor
except ImportError:  # running as a plain script
    from aggregator import RollingBuffer
    from camera import Camera, FPSMeter
    from config import TAU
    from custom_gestures import GestureStore, KNNClassifier
    from landmark_extractor import LandmarkExtractor
    from merger import LetterMerger, Stages
    from predictor import Predictor

ASL, CUSTOM = "asl", "custom"

GREEN = (80, 220, 100)
AMBER = (60, 190, 255)
RED = (60, 60, 235)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
WINDOW = "M5/M6 - live fingerspelling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live fingerspelling")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--raw", action="store_true",
                        help="disable every merger stage - shows the raw "
                             "per-frame output the merger exists to fix")
    parser.add_argument("--top-k", type=int, default=3,
                        help="how many candidate letters to display")
    parser.add_argument("--mode", choices=[ASL, CUSTOM], default=ASL,
                        help="which classifier to start in")
    parser.add_argument("--debug", action="store_true",
                        help="in custom mode, show the nearest gesture and its "
                             "distance even when it is rejected")
    return parser.parse_args()


def draw_panel(frame: np.ndarray, lines: List[tuple], y0: int = 26,
               scale: float = 0.62, step: int = 26) -> None:
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (10, y0 + i * step),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


def draw_progress(frame: np.ndarray, fraction: float, colour) -> None:
    height, width = frame.shape[:2]
    x0, x1, y0, y1 = 10, width - 10, height - 58, height - 42
    cv2.rectangle(frame, (x0, y0), (x1, y1), GREY, 1)
    filled = int((x1 - x0 - 2) * max(0.0, min(1.0, fraction)))
    if filled > 0:
        cv2.rectangle(frame, (x0 + 1, y0 + 1), (x0 + 1 + filled, y1 - 1),
                      colour, -1)


def classify(mode: str, aggregated: np.ndarray, predictor: Predictor,
             knn: KNNClassifier, top_k: int) -> tuple:
    """Returns (label, confidence, candidates) for the active mode."""
    if mode == ASL:
        candidates = predictor.top_k(aggregated, top_k)
        return candidates[0][0], candidates[0][1], candidates[1:]

    label, confidence = knn.predict(aggregated)
    return label, confidence, []


def main() -> None:
    args = parse_args()

    predictor = Predictor()
    print(predictor.describe())
    if not predictor.has_probabilities:
        print("WARNING: model has no calibrated probabilities; the merger's "
              "tau threshold will not mean what it should.")

    store = GestureStore.load()
    knn = KNNClassifier(store)
    print(knn.describe())

    mode = args.mode
    if mode == CUSTOM and not knn.ready:
        print("no custom gestures recorded yet - starting in ASL mode.\n"
              "  record one with: python -m src.record_gesture --name thumbs_up "
              '--phrase "OK!"')
        mode = ASL

    buffer = RollingBuffer()
    merger = LetterMerger(stages=Stages.none() if args.raw else None)
    fps_meter = FPSMeter()
    sentence: List[str] = []

    with Camera(args.camera, args.width, args.height) as cam, \
            LandmarkExtractor() as extractor:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            vector = extractor.process(frame)
            extractor.draw_overlay(frame)
            buffer.push(vector)

            label, confidence, committed = None, 0.0, None
            candidates: List[tuple] = []

            aggregated = buffer.value()
            if aggregated is not None and vector is not None:
                label, confidence, candidates = classify(
                    mode, aggregated, predictor, knn, args.top_k)

            committed = merger.update(label, confidence)
            if committed:
                if mode == ASL:
                    sentence.append(committed)
                else:
                    # A custom gesture emits its mapped phrase, not its name,
                    # and phrases need separating where letters do not.
                    gesture = store.gestures.get(committed)
                    sentence.append(
                        (gesture.output if gesture else committed) + " ")

            state = merger.state
            accepted = confidence >= TAU
            lines = [
                (f"fps {fps_meter.tick():4.1f}   mode: "
                 + ("ASL letters" if mode == ASL
                    else f"custom ({len(store)} gestures)")
                 + ("   MERGER OFF (--raw)" if args.raw else ""),
                 AMBER if args.raw else GREY),
            ]
            if not buffer.ready:
                lines.append((f"filling buffer {buffer.fill:.0%}  "
                              f"({buffer.hands} hand"
                              f"{'s' if buffer.hands != 1 else ''} in window)",
                              AMBER))
            elif label is None:
                if vector is None:
                    lines.append(("no hand", GREY))
                else:
                    lines.append((f"no match  [{buffer.hands}h in window]", GREY))
                    if mode == CUSTOM and knn.ready:
                        expected = sorted({n: h for n, h in knn.hands.items()}.values())
                        if buffer.hands not in expected:
                            lines.append((f"recorded gestures use "
                                          f"{'/'.join(map(str, sorted(set(expected))))}"
                                          f" hand(s)", AMBER))
            else:
                shown = label
                if mode == CUSTOM:
                    gesture = store.gestures.get(label)
                    if gesture and gesture.phrase:
                        shown = f"{label} -> {gesture.phrase}"
                lines.append((f"{shown}  {confidence:.2f}"
                              + (f"   [{buffer.hands}h]" if mode == CUSTOM else ""),
                              GREEN if accepted else RED))
                others = "   ".join(f"{l} {c:.2f}" for l, c in candidates)
                if others:
                    lines.append((others, GREY))

            if state["label"]:
                lines.append((f"holding {state['label']}  "
                              f"{state['frames']} frames  "
                              f"{state['progress']:.0%}",
                              GREEN if state["committed"] else AMBER))

            if args.debug and mode == CUSTOM and aggregated is not None:
                near, distance, threshold = knn.nearest_info(aggregated)
                if near is not None:
                    lines.append((f"nearest {near} d={distance:.3f} "
                                  f"limit={threshold:.3f}"
                                  f" {'OK' if distance <= threshold else 'REJECTED'}",
                                  GREEN if distance <= threshold else RED))

            draw_panel(frame, lines)
            draw_progress(frame, state["progress"],
                          GREEN if state["committed"] else AMBER)

            text = "".join(sentence)[-28:]
            cv2.putText(frame, text or "-", (10, frame.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                sentence.clear()
                merger.reset()
            elif key == ord("u"):
                if sentence:
                    sentence.pop()
                merger.undo()
            elif key == ord("r"):
                merger.reset()
            elif key == ord(" "):
                sentence.append(" ")
            elif key == ord("m"):
                if not knn.ready:
                    print("no custom gestures recorded yet - "
                          "python -m src.record_gesture --name <name>")
                else:
                    mode = CUSTOM if mode == ASL else ASL
                    # Both classifiers share the merger, so its accumulated
                    # candidate belongs to the old mode and must go.
                    merger.reset()
                    print(f"mode: {mode}")

    print("\nfinal:", "".join(sentence))


if __name__ == "__main__":
    main()
