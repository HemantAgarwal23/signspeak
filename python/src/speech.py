"""Text to speech, off the render thread.

pyttsx3's `runAndWait()` blocks until the phrase finishes. Called from the
capture loop that would freeze the video for the length of every utterance, so
speaking happens on a worker thread fed by a queue and the loop never waits.

The engine is created *inside* that thread, and rebuilt for every utterance.
Both matter:

* On Windows the SAPI5 driver is COM-based and its objects are not safe to
  share across threads.
* A reused engine speaks exactly once. Later `runAndWait()` calls return
  immediately and silently produce nothing - measured at 5.49s for the first
  pass of a phrase and 0.4s for each one after. Rebuilding costs ~0.07s.

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
        self._voices = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            try:
                self._speak_once(item)
            except Exception:
                # A failed utterance must never take down the capture loop.
                pass

    def _speak_once(self, text: str) -> None:
        """Build an engine, say one thing, dispose of it.

        A single reused engine speaks exactly once. Every later runAndWait()
        returns immediately without producing sound: measured against a phrase
        that takes 5.5s to speak, the first pass took 5.49s and every
        subsequent one 0.4-0.5s. It fails silently, which is why the symptom
        was "it spoke the first phrase and then went quiet".

        Rebuilding per utterance costs about 0.07s and always speaks.
        """
        import gc

        import pyttsx3

        engine = pyttsx3.init()
        try:
            if self.rate is not None:
                engine.setProperty("rate", self.rate)

            voices = engine.getProperty("voices") or []
            if not self._voices:
                self._voices = [v.name for v in voices]
            if self.voice:
                wanted = self.voice.lower()
                for v in voices:
                    if wanted in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break

            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            del engine
            # pyttsx3 hands back a cached engine while one is still referenced,
            # so the next call would inherit the spent one.
            gc.collect()

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
