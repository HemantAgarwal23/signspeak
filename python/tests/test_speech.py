"""Speech tests. Verifies utterances actually play, by timing them.

A silent failure is the whole point here: pyttsx3 returns instantly and
successfully when it is not speaking, so correctness can only be checked
against how long a known phrase should take.

    cd python && .venv/Scripts/python.exe tests/test_speech.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speech import Speaker   # noqa: E402

fails = []
LONG = ("This is a deliberately long sentence that should take "
        "several seconds to speak aloud properly")
MIN_SECONDS = 2.0            # well under the ~5s it really takes


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name:<50} got {got!r}")
    if not ok:
        fails.append(name)


sp = Speaker()
if not sp.available:
    print("pyttsx3 unavailable - skipping audio checks")
    sys.exit(0)

t0 = time.perf_counter()
sp.say(LONG)
check("say() returns in under 50ms", (time.perf_counter() - t0) < 0.05, True)

# Three in a row: the regression that mattered. A dead engine drains these in
# well under a second because it returns without speaking.
t0 = time.perf_counter()
for _ in range(2):
    sp.say(LONG)
while sp.pending and time.perf_counter() - t0 < 60:
    time.sleep(0.1)
time.sleep(0.5)
elapsed = time.perf_counter() - t0
check(f"3 phrases take real time to speak (>{3 * MIN_SECONDS:.0f}s)",
      elapsed > 3 * MIN_SECONDS, True)
print(f"      (3 phrases took {elapsed:.1f}s)")

check("voices discovered", len(sp.voices) > 0, True)
sp.close()

off = Speaker(enabled=False)
check("disabled speaker reports unavailable", off.available, False)
check("disabled speaker ignores say()", off.say("nothing"), False)
check("empty text is ignored", Speaker(enabled=False).say("   "), False)
off.close()

print()
print("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
