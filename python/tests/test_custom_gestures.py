"""Custom-gesture KNN tests. Pure logic - no webcam needed.

    cd python && .venv/Scripts/python.exe tests/test_custom_gestures.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from custom_gestures import GestureStore, KNNClassifier   # noqa: E402

RNG = np.random.default_rng(7)
fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name:<52} got {got!r}")
    if not ok:
        fails.append(name)


def fake_hand(seed, jitter=0.01, n=20, two_handed=False):
    """A plausible gesture: 21 landmarks per hand, small jitter between samples.

    two_handed fills both slots, which is what a real two-handed gesture looks
    like and what the per-gesture reject threshold exists to handle.
    """
    right = RNG.random((21, 3)).astype(np.float32)
    left = RNG.random((21, 3)).astype(np.float32)
    out = np.zeros((n, 126), np.float32)
    for i in range(n):
        out[i, 63:] = (right + RNG.normal(0, jitter, right.shape)).reshape(-1)
        if two_handed:
            out[i, :63] = (left + RNG.normal(0, jitter, left.shape)).reshape(-1)
    return out


store = GestureStore(path=Path(tempfile.mkdtemp()) / "gestures.json")
store.add("thumbs_up", fake_hand(1), phrase="OK!")
store.add("stop", fake_hand(2), phrase="Please stop")
store.add("help", fake_hand(3), phrase="I need help")

knn = KNNClassifier(store)
check("three gestures stored", len(store), 3)
check("phrase mapping used for output", store.gestures["thumbs_up"].output, "OK!")
check("name used when no phrase set",
      store.add("bare", fake_hand(9)).output, "bare")
store.remove("bare")
knn.fit(store)

# recognises each of its own gestures
correct = 0
for name in ("thumbs_up", "stop", "help"):
    sample = store.gestures[name].samples[0]
    label, conf = knn.predict(sample)
    correct += label == name
check("recognises all 3 stored gestures", correct, 3)

# a held-out-ish variation still matches
label, conf = knn.predict(store.gestures["stop"].samples[-1])
check("matches a different sample of the same gesture", label, "stop")
check("confidence is a real number in (0, 1]", 0 < conf <= 1.0, True)

# an unrelated hand must be rejected, not forced into a class
far = np.zeros((1, 126), np.float32)
far[0, 63:] = (RNG.random((21, 3)) * 8 + 20).reshape(-1)
label, conf = knn.predict(far[0])
check("unrelated pose is rejected (no match)", (label, conf), (None, 0.0))

# empty store answers nothing rather than crashing
empty = KNNClassifier(GestureStore(path=Path(tempfile.mkdtemp()) / "g.json"))
check("empty store is not ready", empty.ready, False)
check("empty store predicts None", empty.predict(far[0]), (None, 0.0))

# confusion warning fires for a near-duplicate, not for a distinct gesture
dupe = store.gestures["thumbs_up"].samples + RNG.normal(0, 0.002, (20, 126))
near, d_near = knn.nearest_gesture(dupe.astype(np.float32))
far_g, d_far = knn.nearest_gesture(fake_hand(42))
check("near-duplicate identifies the gesture it clones", near, "thumbs_up")
check("near-duplicate is closer than a distinct gesture", d_near < d_far, True)

# --- mixed one- and two-handed gestures -----------------------------------
mixed = GestureStore(path=Path(tempfile.mkdtemp()) / "mixed.json")
mixed.add("one_hand_tight", fake_hand(11, jitter=0.005), phrase="one")
mixed.add("two_hand_wide", fake_hand(12, jitter=0.02, two_handed=True), phrase="two")
mixed_knn = KNNClassifier(mixed)

check("hand count detected per gesture",
      (mixed_knn.hands["one_hand_tight"], mixed_knn.hands["two_hand_wide"]),
      (1, 2))
# The threshold tracks the distance to the nearest OTHER gesture, not just a
# gesture's own jitter - samples come from overlapping windows and understate
# real variation badly. With exactly two gestures both sit the same distance
# apart, so both thresholds are the margin-derived one.
from custom_gestures import RELATIVE_MARGIN   # noqa: E402
import features as _f                          # noqa: E402

ca = _f.transform(mixed.gestures["one_hand_tight"].samples).mean(0)
cb = _f.transform(mixed.gestures["two_hand_wide"].samples).mean(0)
separation = float(np.linalg.norm(ca - cb))
check("threshold scales with distance to the nearest other gesture",
      abs(mixed_knn.threshold_for("one_hand_tight")
          - separation * RELATIVE_MARGIN) < 1e-4, True)
check("threshold is far looser than raw sample spread would give",
      mixed_knn.threshold_for("two_hand_wide") > 0.35, True)

# A third gesture placed near one of the others must tighten only that pair.
crowded = GestureStore(path=Path(tempfile.mkdtemp()) / "crowded.json")
crowded.add("alpha", fake_hand(21))
crowded.add("beta", fake_hand(22, two_handed=True))
near_alpha = crowded.gestures["alpha"].samples + RNG.normal(0, 0.05, (20, 126))
crowded.add("alpha_ish", near_alpha.astype(np.float32))
crowded_knn = KNNClassifier(crowded)
check("thresholds differ once neighbours differ",
      len(set(round(v, 4) for v in crowded_knn.thresholds.values())) > 1, True)
check("the crowded pair gets the tighter threshold",
      crowded_knn.threshold_for("alpha") < crowded_knn.threshold_for("beta"), True)

both = 0
for name in ("one_hand_tight", "two_hand_wide"):
    for i in (0, 7, -1):
        label, conf = mixed_knn.predict(mixed.gestures[name].samples[i])
        both += label == name
check("both gestures recognised when mixed", both, 6)

# a one-handed query must not be swallowed by the wide two-handed threshold
stray = np.zeros(126, np.float32)
stray[63:] = (RNG.random((21, 3)) * 6 + 12).reshape(-1)
check("stray pose still rejected in a mixed store",
      mixed_knn.predict(stray), (None, 0.0))

# training is genuinely fast - the few-shot claim
seconds = knn.fit(store)
check("KNN rebuild well under 2s", seconds < 2.0, True)
print(f"      (rebuilt 60 samples in {seconds * 1000:.1f} ms)")

# round-trip through disk
store.save()
reloaded = GestureStore.load(store.path)
check("survives save/load", sorted(reloaded.gestures), ["help", "stop", "thumbs_up"])
check("samples survive round-trip",
      bool(np.allclose(reloaded.gestures["stop"].samples,
                       store.gestures["stop"].samples, atol=1e-6)), True)
check("phrases survive round-trip", reloaded.gestures["thumbs_up"].phrase, "OK!")
label, _ = KNNClassifier(reloaded).predict(store.gestures["help"].samples[2])
check("reloaded store still classifies", label, "help")

# rename keeps data intact
reloaded.rename("stop", "halt")
check("rename moves the gesture", sorted(reloaded.gestures),
      ["halt", "help", "thumbs_up"])

print()
print("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
