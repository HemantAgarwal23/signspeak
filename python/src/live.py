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
    s         speak the sentence aloud
    a         toggle auto-speak (custom mode)
    h or ?    show the in-app help overlay
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
    from .speech import Speaker
    from .ui import Dashboard, ViewState
except ImportError:  # running as a plain script
    from aggregator import RollingBuffer
    from camera import Camera, FPSMeter
    from config import TAU
    from custom_gestures import GestureStore, KNNClassifier
    from landmark_extractor import LandmarkExtractor
    from merger import LetterMerger, Stages
    from predictor import Predictor
    from speech import Speaker
    from ui import Dashboard, ViewState

ASL, CUSTOM = "asl", "custom"

GREEN = (80, 220, 100)
AMBER = (60, 190, 255)
RED = (60, 60, 235)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
WINDOW = "signspeak"


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
    parser.add_argument("--no-speech", action="store_true",
                        help="disable text to speech entirely")
    parser.add_argument("--auto-speak", action="store_true",
                        help="speak each phrase as it is confirmed "
                             "(custom mode only - speaking every letter is noise)")
    parser.add_argument("--voice", default=None,
                        help="substring of a voice name, e.g. Zira")
    parser.add_argument("--debug", action="store_true",
                        help="in custom mode, show the nearest gesture and its "
                             "distance even when it is rejected")
    return parser.parse_args()


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

    speaker = Speaker(voice=args.voice, enabled=not args.no_speech)
    auto_speak = args.auto_speak
    if not speaker.available and not args.no_speech:
        print("speech unavailable (pyttsx3 not installed) - continuing silently")

    dashboard = Dashboard()
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
                    phrase = gesture.output if gesture else committed
                    sentence.append(phrase + " ")
                    if auto_speak:
                        # Only custom mode: auto-speaking each letter of a
                        # word being spelled is unusable noise.
                        speaker.say(phrase)

            state = merger.state
            view = ViewState(
                mode=mode,
                fps=fps_meter.tick(),
                hands=buffer.hands,
                buffer_fill=buffer.fill,
                buffer_ready=buffer.ready,
                label=label,
                confidence=confidence,
                candidates=candidates,
                hold_label=state["label"],
                hold_progress=state["progress"],
                hold_frames=state["frames"],
                committed=state["committed"],
                sentence="".join(sentence),
                gesture_count=len(store),
                auto_speak=auto_speak,
                speech_available=speaker.available,
                merger_on=not args.raw,
            )

            if label is not None:
                gesture = store.gestures.get(label) if mode == CUSTOM else None
                view.output_text = gesture.output if gesture else label
            elif vector is None:
                view.note = "show your hand"
            elif buffer.ready:
                view.note = "no match"
                if mode == CUSTOM and knn.ready:
                    expected = sorted(set(knn.hands.values()))
                    if buffer.hands not in expected:
                        view.note = (f"no match - your gestures use "
                                     f"{'/'.join(map(str, expected))} hand(s)")

            if args.debug and mode == CUSTOM and aggregated is not None:
                near, distance, threshold = knn.nearest_info(aggregated)
                if near is not None:
                    view.debug = (f"nearest {near}  d={distance:.3f}  "
                                  f"limit={threshold:.3f}  "
                                  f"{'accept' if distance <= threshold else 'reject'}")

            frame = dashboard.render(frame, view)

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
            elif key == ord("s"):
                spoken = "".join(sentence).strip()
                if spoken:
                    speaker.say(spoken)
            elif key == ord("a"):
                auto_speak = not auto_speak
                print(f"auto-speak: {'on' if auto_speak else 'off'}")
            elif key in (ord("h"), ord("?")):
                dashboard.show_help = not dashboard.show_help
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
