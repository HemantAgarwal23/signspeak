"""Dashboard rendering: video on the left, a real panel on the right.

Drawing text straight onto the camera feed is fine for a debug view and poor
for anything else - it sits over the hands you are trying to watch, and it is
unreadable against a bright background. Here the frame is composited onto a
wider canvas with a solid panel beside it, so the video stays clean and the
panel stays legible.

Still OpenCV rather than a GUI toolkit: one window, one render path, and
nothing to keep in sync between a widget tree and a capture loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Panel palette. BGR, because OpenCV.
BG = (28, 26, 24)
BG_ALT = (40, 37, 34)
FG = (238, 238, 238)
MUTED = (150, 146, 142)
GREEN = (110, 210, 120)
AMBER = (70, 190, 250)
RED = (80, 80, 240)
BLUE = (220, 170, 90)

PANEL_WIDTH = 380
PAD = 18


@dataclass
class ViewState:
    """Everything the panel draws. Filled in fresh each frame."""

    mode: str = "asl"
    fps: float = 0.0
    hands: int = 0
    buffer_fill: float = 0.0
    buffer_ready: bool = False

    label: Optional[str] = None
    confidence: float = 0.0
    candidates: Sequence[Tuple[str, float]] = field(default_factory=tuple)
    output_text: str = ""

    hold_label: Optional[str] = None
    hold_progress: float = 0.0
    hold_frames: int = 0
    committed: bool = False

    word: str = ""
    sentence: str = ""

    gesture_count: int = 0
    auto_speak: bool = False
    speech_available: bool = True
    merger_on: bool = True
    note: str = ""
    debug: str = ""


KEYS: List[Tuple[str, str]] = [
    ("m", "switch mode (ASL / custom)"),
    ("s", "speak the sentence"),
    ("a", "toggle auto-speak"),
    ("SPACE", "insert a space"),
    ("u", "undo last"),
    ("c", "clear everything"),
    ("r", "reset the merger"),
    ("h or ?", "show / hide this help"),
    ("q or ESC", "quit"),
]


def text(canvas, string, origin, colour=FG, scale=0.52, weight=1):
    cv2.putText(canvas, string, origin, FONT, scale, colour, weight, cv2.LINE_AA)


def wrap(string: str, width_px: int, scale: float = 0.52,
         weight: int = 1) -> List[str]:
    """Greedy word wrap measured in real rendered pixels, not characters."""
    if not string:
        return []
    lines, current = [], ""
    for word in string.split(" "):
        candidate = f"{current} {word}".strip()
        (w, _), _ = cv2.getTextSize(candidate, FONT, scale, weight)
        if w <= width_px or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def bar(canvas, x, y, width, height, fraction, colour, track=BG_ALT):
    cv2.rectangle(canvas, (x, y), (x + width, y + height), track, -1)
    filled = int(width * max(0.0, min(1.0, fraction)))
    if filled > 0:
        cv2.rectangle(canvas, (x, y), (x + filled, y + height), colour, -1)


def badge(canvas, x, y, string, colour):
    (w, h), _ = cv2.getTextSize(string, FONT, 0.46, 1)
    cv2.rectangle(canvas, (x, y - h - 7), (x + w + 16, y + 7), colour, -1)
    text(canvas, string, (x + 8, y), BG, 0.46, 1)
    return x + w + 16


class Dashboard:
    """Composites a camera frame and a status panel into one canvas."""

    def __init__(self, panel_width: int = PANEL_WIDTH) -> None:
        self.panel_width = panel_width
        self.show_help = False

    def render(self, frame: np.ndarray, view: ViewState) -> np.ndarray:
        height, width = frame.shape[:2]
        canvas = np.full((height, width + self.panel_width, 3), BG, np.uint8)
        canvas[:, :width] = frame

        self._panel(canvas, width, height, view)
        if self.show_help:
            self._help(canvas)
        return canvas

    # -- panel -------------------------------------------------------------
    def _panel(self, canvas, x0: int, height: int, view: ViewState) -> None:  # noqa: C901
        x = x0 + PAD
        inner = self.panel_width - PAD * 2
        y = PAD + 14

        text(canvas, "signspeak", (x, y), FG, 0.78, 2)
        y += 26
        text(canvas, f"{view.fps:.0f} fps   {view.hands} hand"
                     f"{'' if view.hands == 1 else 's'}", (x, y), MUTED, 0.44)

        y += 26
        cursor = badge(canvas, x, y,
                       "ASL LETTERS" if view.mode == "asl"
                       else f"CUSTOM x{view.gesture_count}",
                       BLUE if view.mode == "asl" else GREEN)
        if view.auto_speak and view.speech_available:
            cursor = badge(canvas, cursor + 8, y, "AUTO-SPEAK", AMBER)
        if not view.merger_on:
            badge(canvas, cursor + 8, y, "RAW", RED)

        # -- current prediction
        y += 42
        cv2.line(canvas, (x, y), (x + inner, y), BG_ALT, 1)
        y += 30

        if not view.buffer_ready:
            text(canvas, "reading your hand...", (x, y), AMBER, 0.56)
            y += 16
            bar(canvas, x, y, inner, 6, view.buffer_fill, AMBER)
            y += 26
            text(canvas, "hold still while the window fills",
                 (x, y), MUTED, 0.42)
        elif view.label is None:
            text(canvas, view.note or "no match", (x, y), MUTED, 0.56)
        else:
            text(canvas, view.output_text or view.label, (x, y), FG, 0.72, 2)
            y += 18
            bar(canvas, x, y, inner, 6, view.confidence,
                GREEN if view.confidence >= 0.5 else AMBER)
            y += 20
            text(canvas, f"confidence {view.confidence:.0%}", (x, y), MUTED, 0.42)

            for name, score in list(view.candidates)[:3]:
                y += 20
                text(canvas, f"{name}", (x, y), MUTED, 0.44)
                bar(canvas, x + 40, y - 6, inner - 40, 5, score, BG_ALT if
                    score < 0.05 else MUTED)

        # -- progress toward committing
        y += 44
        cv2.line(canvas, (x, y), (x + inner, y), BG_ALT, 1)
        y += 26
        if view.hold_label:
            state = "committed" if view.committed else "hold to confirm"
            text(canvas, f"{view.hold_label}  -  {state}", (x, y),
                 GREEN if view.committed else AMBER, 0.5)
            y += 14
            bar(canvas, x, y, inner, 10,
                view.hold_progress, GREEN if view.committed else AMBER)
        else:
            text(canvas, "nothing held", (x, y), MUTED, 0.5)
            y += 14
            bar(canvas, x, y, inner, 10, 0.0, MUTED)

        # -- accumulated text
        y += 44
        cv2.line(canvas, (x, y), (x + inner, y), BG_ALT, 1)
        y += 26
        text(canvas, "OUTPUT", (x, y), MUTED, 0.42)
        y += 26

        # Keep the output block inside the space actually left, so a long
        # sentence scrolls instead of running under the footer.
        footer_y = height - PAD
        reserved = 34 if view.debug else 8
        room = max(1, (footer_y - reserved - y) // 26)

        body = view.sentence.strip() or "-"
        lines = wrap(body, inner, 0.6, 1)
        clipped = lines[-room:]
        if len(lines) > room:
            text(canvas, f"... +{len(lines) - room} earlier",
                 (x, y - 14), MUTED, 0.38)
        for line in clipped:
            text(canvas, line, (x, y), FG, 0.6, 1)
            y += 26

        if view.debug:
            for line in wrap(view.debug, inner, 0.4)[:1]:
                text(canvas, line, (x, footer_y - 20), MUTED, 0.4)

        # -- footer
        footer = height - PAD
        text(canvas, "h  help      s  speak      m  mode",
             (x, footer), MUTED, 0.42)

    # -- help overlay ------------------------------------------------------
    def _help(self, canvas) -> None:
        height, width = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (12, 11, 10), -1)
        cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0, canvas)

        notes = [
            "ASL mode spells one letter at a time - hold each until the bar fills.",
            "Custom mode recognises your own gestures and speaks their phrase.",
            'Record one:  python -m src.record_gesture --name X --phrase "..."',
        ]

        # Lay the overlay out from the space available rather than fixed
        # offsets, so it cannot run off the bottom of a short window.
        step = min(32, max(20, (height - 190) // max(1, len(KEYS))))
        x, y = 56, 62
        text(canvas, "Keyboard", (x, y), FG, 0.88, 2)
        y += 20
        text(canvas, "the video window must have focus, not the terminal",
             (x, y), MUTED, 0.44)
        y += 30

        for key, description in KEYS:
            text(canvas, key, (x, y), AMBER, 0.54, 1)
            text(canvas, description, (x + 130, y), FG, 0.5, 1)
            y += step

        y += 6
        cv2.line(canvas, (x, y), (width - 56, y), BG_ALT, 1)
        y += 24
        for line in notes:
            text(canvas, line, (x, y), MUTED, 0.44)
            y += 22
