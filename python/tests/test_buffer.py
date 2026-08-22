"""RollingBuffer tests, focused on the hand-count transition bug.

    cd python && .venv/Scripts/python.exe tests/test_buffer.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aggregator import RollingBuffer, aggregate   # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name:<54} got {got!r}")
    if not ok:
        fails.append(name)


def one_handed(v=0.5):
    x = np.zeros(126, np.float32); x[63:] = v; return x


def two_handed(v=0.5):
    x = np.full(126, v, np.float32); return x


buf = RollingBuffer(length=30)
check("empty buffer is not ready", buf.ready, False)
check("empty buffer values None", buf.value(), None)

for _ in range(29):
    buf.push(one_handed())
check("29 frames still not ready", buf.ready, False)
buf.push(one_handed())
check("30 frames ready", buf.ready, True)
check("one-handed window reports 1 hand", buf.hands, 1)
check("left slot stays zero", bool(np.all(buf.value()[:63] == 0)), True)

# the bug: switching to two hands must not blend
buf.push(two_handed())
check("hand-count change resets the window", buf.ready, False)
check("window now tracks 2 hands", buf.hands, 2)

for _ in range(29):
    buf.push(two_handed())
check("refills after the change", buf.ready, True)
value = buf.value()
check("no blending - both slots fully present",
      (bool(np.all(value[:63] > 0)), bool(np.all(value[63:] > 0))), (True, True))

# and back again
buf.push(one_handed())
check("dropping to one hand resets too", buf.ready, False)

# a dropped frame (no hand at all) must not corrupt the window
buf2 = RollingBuffer(length=10)
for _ in range(10):
    buf2.push(two_handed())
buf2.push(None)
check("None push is ignored, window intact", buf2.ready, True)
check("window still 2 hands after None", buf2.hands, 2)

# median still rejects a single outlier frame
buf3 = RollingBuffer(length=11)
for _ in range(10):
    buf3.push(one_handed(0.5))
buf3.push(one_handed(99.0))
check("single outlier frame does not move the median",
      bool(np.allclose(buf3.value()[63:], 0.5)), True)

check("clear() empties and forgets the mask",
      (buf3.clear(), buf3.ready, buf3.hands)[1:], (False, 0))

print()
print("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
