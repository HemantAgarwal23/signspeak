"""Merger state-machine tests. Pure logic - no webcam, no model needed.

    cd python && .venv/Scripts/python.exe tests/test_merger.py

Exits non-zero on failure, so it works as a pre-commit check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merger import LetterMerger, Stages

FPS = 30.0
def feed(m, label, conf, frames, t0):
    """Feed `frames` frames at 30fps. Returns (commits, next_t)."""
    out = []
    for i in range(frames):
        t = t0 + i / FPS
        c = m.update(label, conf, now=t)
        if c: out.append(c)
    return out, t0 + frames / FPS

fails = []
def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name:<52} got {got!r}")
    if not ok:
        fails.append(f"{name}: got {got!r} want {want!r}")

# 1. steady hold commits exactly once
m = LetterMerger(); c,_ = feed(m, "A", 0.9, 60, 0.0)
check("hold 60 frames commits once", c, ["A"])

# 2. below tau never commits
m = LetterMerger(); c,_ = feed(m, "A", 0.2, 60, 0.0)
check("confidence below tau never commits", c, [])

# 3. too few frames: 10 frames < min_frames 12
m = LetterMerger(); c,_ = feed(m, "A", 0.9, 10, 0.0)
check("10 frames (<12) does not commit", c, [])

# 4. enough frames but not enough time (fast replay)
m = LetterMerger()
c = [x for i in range(20) if (x := m.update("A", 0.9, now=i*0.001))]
check("20 frames in 0.02s does not commit (hold-time)", c, [])

# 5. spelling a word
m = LetterMerger(); t = 0.0; got = []
for ch in "CAB":
    c, t = feed(m, ch, 0.9, 20, t)
    got += c
check("spell CAB", got, ["C","A","B"])

# 6. double letter needs a gap
m = LetterMerger(); t = 0.0; got = []
c, t = feed(m, "L", 0.9, 20, t); got += c
c, t = feed(m, "L", 0.9, 20, t); got += c
check("LL with no gap -> single L (stage 4)", got, ["L"])

m = LetterMerger(); t = 0.0; got = []
c, t = feed(m, "L", 0.9, 20, t); got += c
c, t = feed(m, None, 0.0, 20, t); got += c        # hand dropped
c, t = feed(m, "L", 0.9, 20, t); got += c
check("LL with a gap -> LL", got, ["L","L"])

# 7. HELLO end to end
m = LetterMerger(); t = 0.0
for ch in "HELLO":
    if m.committed and m.committed[-1] == ch:
        _, t = feed(m, None, 0.0, 20, t)
    _, t = feed(m, ch, 0.9, 20, t)
check("spell HELLO", m.text, "HELLO")

# 8. flicker rejection: 3 frames of X inside a hold of A
m = LetterMerger(); t = 0.0; got = []
c, t = feed(m, "A", 0.9, 8, t); got += c
c, t = feed(m, "X", 0.9, 3, t); got += c
c, t = feed(m, "A", 0.9, 20, t); got += c
check("3-frame flicker of X is rejected", got, ["A"])

# 9. ablation: no merger at all
m = LetterMerger(stages=Stages.none()); c,_ = feed(m, "A", 0.9, 5, 0.0)
check("Stages.none() commits every frame", c, ["A"]*5)

# 10. ablation: confidence filter only
m = LetterMerger(stages=Stages(True, False, False, False))
c, t = feed(m, "A", 0.2, 5, 0.0)
c2, _ = feed(m, "A", 0.9, 5, t)
check("filter-only drops low-conf, passes high-conf", (c, c2), ([], ["A"]*5))

# 11. undo
m = LetterMerger(); t = 0.0
for ch in "AB": _, t = feed(m, ch, 0.9, 20, t)
m.undo()
check("undo removes last letter", m.text, "A")

# 12. progress bar is monotone and capped
m = LetterMerger()
prev, bad = 0.0, False
for i in range(20):
    m.update("A", 0.9, now=i/FPS)
    p = m.state["progress"]
    if p < prev - 1e-9 or p > 1.0: bad = True
    prev = p
check("progress monotone and <= 1.0", not bad, True)

print()
print("ALL PASS" if not fails else "FAILURES:\n" + "\n".join(fails))
sys.exit(1 if fails else 0)
