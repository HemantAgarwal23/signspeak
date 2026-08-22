"""Text to speech, off the render thread.

pyttsx3's `runAndWait()` blocks until the phrase finishes. Called from the
capture loop that would freeze the video for the length of every utterance, so
speaking happens on a worker thread fed by a queue and the loop never waits.

The engine is also created *inside* that thread. On Windows the SAPI5 driver is
COM-based and objects are not safely shared across threads, so an engine built
on the main thread and driven from a worker misbehaves in ways that look like
random hangs.

    speaker = Speaker()
    speaker.say("I need help")     # returns immediately
    speaker.close()
"""
from __future__ import annotations

import queue
import threading
from typing import List, Optional

_STOP = object()


class Speaker:
    """Non-blocking text to speech. Degrades to silence if unavailable."""

    def __init__(self, rate: Optional[int] = 165, voice: Optional[str] = None,
                 enabled: bool = True) -> None:
        self.rate = rate
        self.voice = voice
        self.available = False
        self._queue: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._voices: List[str] = []

        if not enabled:
            return
        try:
            import pyttsx3                      # noqa: F401
        except ImportError:
            # Speech is a nicety, not a dependency. Everything else still runs.
            return

        self.available = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import pyttsx3

        try:
            engine = pyttsx3.init()
        except Exception:
            self.available = False
            return

        if self.rate is not None:
            engine.setProperty("rate", self.rate)

        voices = engine.getProperty("voices") or []
        self._voices = [v.name for v in voices]
        if self.voice:
            wanted = self.voice.lower()
            for v in voices:
                if wanted in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break

        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            try:
                engine.say(item)
                engine.runAndWait()
            except Exception:
                # A failed utterance must never take down the capture loop.
                pass

        try:
            engine.stop()
        except Exception:
            pass

    def say(self, text: str) -> bool:
        """Queue text to be spoken. Returns immediately."""
        text = (text or "").strip()
        if not text or not self.available:
            return False
        self._queue.put(text)
        return True

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def voices(self) -> List[str]:
        return list(self._voices)

    def close(self) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "Speaker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
