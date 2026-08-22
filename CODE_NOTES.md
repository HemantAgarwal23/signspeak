# CODE_NOTES — what exists and how it flows

Written for you, the human. Read this before touching code, and after any
milestone, to understand what each file does and why. Updated as milestones land.

Last updated: after **M1, M2, M4, M5, M6** plus custom gestures (brought forward from M19/M20). Working end to end.

---

## 1. The 30-second version

**Two sources feed one dataset.** Bulk letters come from a public image dataset
run through MediaPipe offline; your own recordings supply the held-out test set
and calibration data. Both end up as 126-D vectors in the same manifest.

```
public images ──> MediaPipe (IMAGE mode) ──> data/packs/{label}__public.npy
                                              [import_images.py]
webcam frame (BGR)
   │  mirrored once, at capture time            [camera.py]
   ▼
MediaPipe Hands  →  21 landmarks × (x,y,z) × 2 hands
   │                                            [landmark_extractor.py]
   ▼
126-D float32 vector  (missing hand = 63 zeros)
   │
   ├── RECORDING: 30 consecutive vectors → (30,126) .npy on disk
   │                                            [data_collector.py, dataset.py]
   │
   └── LIVE (M5+): rolling 30-frame buffer → per-coordinate median → 126-D
                   → StandardScaler → SVM → (label, confidence)
                   → 4-stage merger → committed letters → word → sentence
```

Every tunable number lives in `python/src/config.py`. Nothing else defines a
constant.

---

## 2. Files that exist right now

### `python/src/config.py`
Single source of truth. Feature geometry (126 = 21 × 3 × 2), sequence length
(30), MediaPipe thresholds, SVM hyperparameters, merger thresholds, the 24-letter
alphabet, and every filesystem path. `ensure_dirs()` creates the write targets.

Four flags worth knowing:
- `MIRROR_INPUT = True` — frames are flipped horizontally **once**, at capture.
- `FLIP_HANDEDNESS = True` — corrects MediaPipe's inverted handedness label.
- `FEATURE_WRIST_RELATIVE = True` and `FEATURE_SCALE_NORMALISE = True` — the
  feature representation, applied in `features.py` at train/inference time,
  **not** at record time. Changing either needs a retrain, never a re-record.
  See §5 for why both had to be turned on.

### `python/src/camera.py`
`Camera` — context-managed webcam. Opens with `CAP_DSHOW` on Windows (the
default MSMF backend stalls for seconds on startup), sets a 1-frame buffer so you
see *now* rather than a queued frame, and applies the mirror flip. Every capture
loop in the project reads through this, so mirroring is decided in exactly one
place.

`FPSMeter` — rolling FPS over the last 30 frames. Call `.tick()` once per frame.

### `python/src/download_models.py`
Fetches `hand_landmarker.task` (~7.8 MB) into `python/data/models/` and copies it
to `web/public/models/` so the browser loads a byte-identical graph. Run once:
`python -m src.download_models`.

### `python/src/landmark_extractor.py`
The heart of the feature pipeline. `LandmarkExtractor.process(frame)` returns a
`(126,)` float32 array, or `None` when no hand is visible.

**The vector layout is a contract.** The browser port (M15) must reproduce it
exactly or the exported model outputs garbage:

| index range | contents |
|---|---|
| `0 .. 62`   | hand slot 0 = the user's **real left** hand, 21 landmarks × (x, y, z) |
| `63 .. 125` | hand slot 1 = the user's **real right** hand, same layout |

Inside a hand block, landmark *k* occupies indices `base+3k`, `base+3k+1`,
`base+3k+2` — x, y, z in that order, MediaPipe's native landmark numbering
(0 = wrist, 4 = thumb tip, 8 = index tip, … 20 = pinky tip).

**Which MediaPipe API.** This uses the **Tasks** API (`HandLandmarker`), not the
old `mp.solutions.hands`. Two reasons: mediapipe 1.x deleted `mp.solutions`
entirely, and the browser build uses `@mediapipe/tasks-vision` HandLandmarker
with the *same* `hand_landmarker.task` file — so both sides run identical graphs
and the M16 parity check has a real chance of passing. Consequences:
- The landmarker runs in `RunningMode.VIDEO`, which requires strictly increasing
  timestamps. The extractor keeps its own monotonic counter for this.
- The Tasks API ships no drawing helper, so `draw_overlay()` is hand-rolled:
  scale normalised coords to pixels, stroke the 21 bone connections, dot the
  joints.
- There is no `model_complexity` knob any more; the model variant is chosen by
  which `.task` file you download (we use float16/1).

Three decisions that matter:

0. **MediaPipe's handedness label is inverted, and we correct it.** We mirror
   the frame for a natural selfie preview, but MediaPipe reports handedness for
   the *un-mirrored* camera view — so raising your real left hand made it say
   `"Right"`. Confirmed on this machine on 2026-08-22.
   `config.FLIP_HANDEDNESS = True` flips the label back, and slots are defined
   in terms of your **real** hands throughout. If a future MediaPipe release
   changes this behaviour, flip that flag — **never** reorder `HAND_SLOTS`, or
   every sample recorded before the change becomes silently mislabelled.
1. **Handedness picks the slot, not detection order.** If the slot were chosen by
   the order MediaPipe happened to return hands in, the same physical hand would
   jump between halves of the vector between frames and the model would never
   learn anything. Ties (two hands classified the same, which happens when hands
   overlap) fall back to whichever slot is free.
2. **The extractor is stateful.** MediaPipe tracks hands across frames, which is
   faster and steadier than detecting fresh each frame. One instance per capture
   loop, reused.
3. **Missing hand = zeros, never dropped.** The vector is always 126 long.

`describe_layout()` prints the index map — paste its output into the M15 browser
prompt so the JS port has the contract in writing.

### `python/src/features.py`
Turns a raw landmark vector into model input. Deliberately separate from
recording: `data/raw/` always stores **raw absolute** coordinates, so the
representation can change without throwing away a single recorded sample, and
both settings can be compared on the identical dataset — a free ablation row for
the report.

`transform(x)` accepts `(126,)`, `(n, 126)` or `(n, 30, 126)` and never changes
the shape. With the default config it is the identity.

`to_wrist_relative(x)` subtracts each hand's wrist from that hand's 21
landmarks, discarding where the hand sits in frame and keeping only its shape.
An absent hand stays all-zero rather than becoming a negative wrist offset —
that special case is why it is not a plain subtraction.

Whatever is set here must be reimplemented identically in `web/src/buffer.js`.

### `python/src/aggregator.py`
`aggregate(sequence)` — per-coordinate **median** over the time axis,
`(n_frames, 126) -> (126,)`. Median rather than mean because MediaPipe
occasionally emits one badly wrong frame; a single outlier drags a mean but
barely moves a median. This is the interface between "sequences on disk" and
"one row per sample" that every classifier consumes.

### `python/src/import_images.py`
Turns `{label}/{image}.jpg` folders into landmark samples. Four choices worth
knowing:
- **One hand only** (`max_hands=1`) — these are single-hand stills, so allowing
  two lets MediaPipe invent a second hand out of the background.
- **The hand slot is pinned, not inferred.** A third-party still gives no clue
  whether the photo was mirrored, so its handedness label cannot be trusted.
  Every detected hand goes to slot 1 (right) by default. Consistency is what
  matters, not which slot.
- **Even stride, not the first N.** Consecutive files in these datasets are
  near-identical; taking the first 600 can be almost one continuous shot.
- **Detection failures are counted and reported per class.** Fist-like shapes
  (M, N, S, T) and tight crops are the usual casualties.

### `python/src/trainer.py` — M4
`load_features` → `features.transform` → split → `StandardScaler` (fitted on
train only, or the test distribution leaks in) → `SVC(rbf, C=10, gamma=scale,
probability=True)` → report → save to `data/models/`.

`--holdout-subject s01` trains on everyone else and tests only on that subject.
`--fast` skips probability calibration (3.7s vs 22s) for quick experiments —
but the merger needs confidences, so never ship a `--fast` model.
`--max-per-class N` subsamples the training pool; M8's sample-efficiency
experiment builds on this.

`model_meta.json` records the feature flags the model was trained with, so
inference can refuse a mismatched model. A representation mismatch otherwise
fails silently and looks like a bad model rather than a config bug.

### `python/src/predictor.py` — M5
Loads the model, scaler and labels; `predict(raw_vector)` returns
`(label, confidence, probabilities)`.

Takes a **raw** vector and applies `features.transform` itself, so no caller can
forget to. It also refuses to load a model whose recorded feature flags
disagree with the current config — a representation mismatch is invisible at
runtime and looks exactly like a badly trained model, which is a miserable thing
to debug.

Without probability calibration it falls back to a softmax over decision-function
margins: fine for ranking, wrong for the merger's tau threshold. That is why
`--fast` models must not be shipped.

### `python/src/merger.py` — M6
The four-stage state machine, and the piece your report calls a contribution.

| stage | what it does |
|---|---|
| 1 confidence filtering | drop predictions below `tau` (0.35) |
| 2 buffer accumulation | count consecutive frames agreeing on a label, track running mean confidence |
| 3 hold-time validation | commit only when frames ≥ 12 **and** elapsed ≥ 0.4s **and** mean confidence ≥ tau — all three at once |
| 4 duplicate suppression | never commit the same letter twice running |

Two design decisions worth defending in a viva:

- **Time is injected, not read from the clock.** `update(label, conf, now=...)`
  lets M11's ablation replay recorded sequences at any speed and get identical
  results. A merger that calls `time.perf_counter()` internally cannot be
  tested deterministically.
- **Each stage toggles independently** (`Stages`), which is exactly what the
  M11 ablation measures. The point is to *quantify* what the merger buys, not
  assert it.

**One deviation from the spec, and it matters.** Stage 4 as written makes double
letters unspellable — "HELLO" comes out "HELO". `idle_frames_to_forget` (15
frames, ~0.5s) clears the duplicate guard after a gap, so dropping your hand
between the two Ls works, which is the natural gesture anyway. M11 should
measure it.

### `python/src/live.py` — M5 + M6 demo
Camera → landmarks → rolling buffer → median → SVM → merger → text, in one
window. `--raw` disables every merger stage so you can *see* the flicker the
merger exists to fix — useful for the demo video, and an honest way to show the
before state.

### `python/tests/test_merger.py`
13 tests, pure logic, no webcam or model needed:
`.venv/Scripts/python.exe tests/test_merger.py`. Covers steady holds, sub-tau
rejection, too-few-frames, too-fast replay, spelling `CAB` and `HELLO`, the
double-letter gap, 3-frame flicker rejection, both ablation modes, undo, and
progress-bar monotonicity. Exits non-zero on failure.

### `python/src/aggregator.py` also holds `RollingBuffer`
The live counterpart to a recorded sequence: last 30 *good* frames, collapsed
by the same median. Frames with no hand are not added, matching how
`data_collector` records — training and inference must aggregate identically or
the model sees a different kind of input than it learned on.

### `python/src/custom_gestures.py` — brought forward from M19/M20
Your own gestures, each mapped to a phrase. A KNN (k=3, Euclidean) over samples
you record — "training" is transforming and storing them, which is why it takes
under a millisecond and why the few-shot claim holds.

Shares the *same* feature space as the letter SVM, so both classifiers see
identical input and a gesture recorded today survives a retrain of the letter
model. It also means custom gestures inherit the same chirality assumption —
record with the hand you will use.

**The reject rule is the interesting part.** A KNN always returns something, so
without one an idle hand fires whichever gesture happens to be nearest. Each
gesture carries its own threshold, taken as the larger of:

- `2.5 ×` its own sample spread, and
- `0.35 ×` the distance to the nearest *other* gesture.

That second term is not decoration — see §5b for the measurement that forced it.
Per-gesture rather than global because a two-handed gesture fills twice as many
non-zero dimensions and so spreads wider; one shared number would loosen the
one-handed gestures and tighten the two-handed ones simultaneously.

`nearest_info()` reports the nearest gesture, its distance and the limit, which
is what `live.py --debug` prints. It distinguishes "nothing was close" from
"something was close but just over the line".

### `python/src/record_gesture.py`
Countdown, then 20 samples, then a confusion check against existing gestures.

Samples are taken every `--stride` frames (default 6), **not** every frame.
Consecutive 30-frame windows overlap by 29 frames, so sampling continuously
gives 20 near-identical copies of one instant — measured at 0.014 apart against
2.8+ between genuinely different gestures. See §5b.

`--list`, `--delete`, `--set-phrase`, `--force` manage what you have.

`--export FILE` and `--import FILE` move a whole vocabulary between machines
(M25). Imports are **validated, not trusted** — wrong `feature_dim`, malformed
or non-numeric samples are rejected with a reason rather than crashing later
inside a distance calculation. Name collisions are reported, never silently
resolved: `--force` overwrites, `--prefix` renames. 20 gestures is about 200 KB
of JSON.

### `python/src/speech.py` — M22
Text to speech that never stalls the video. `pyttsx3.runAndWait()` blocks until
the phrase finishes, so speaking happens on a worker thread fed by a queue and
`say()` returns in ~0.02 ms.

Two non-obvious details, both found the hard way:

**The engine is created inside that thread**, not passed into it. On Windows the
SAPI5 driver is COM-based and its objects are not safe to share across threads.

**The engine is rebuilt for every utterance.** A reused one speaks exactly once;
every later `runAndWait()` returns immediately and silently produces nothing.
Measured against a phrase that takes ~5.5s to speak:

| pass | reused engine | fresh engine |
|---|---|---|
| 1 | 5.49s | 5.78s |
| 2 | 0.50s (silent) | 4.97s |
| 3 | 0.38s (silent) | 4.94s |

Rebuilding costs ~0.07s. `gc.collect()` after disposal matters — pyttsx3 hands
back a cached engine while one is still referenced, so without it the next call
inherits the spent one.

This is why `tests/test_speech.py` asserts on **elapsed time** rather than on a
return value: the failure is silent and successful-looking, so the only evidence
that speech happened is that it took as long as speech takes.

If `pyttsx3` is missing the Speaker degrades to silence rather than raising —
speech is a nicety, not a dependency.

In `live.py`: `s` speaks the sentence, `a` toggles auto-speak. Auto-speak is
custom-mode only, because speaking every letter of a word being spelled is
unusable noise.

### `python/src/dataset.py`
Owns `data/manifest.json`, the `.npy` files under `data/raw/`, and the bulk
**packs** under `data/packs/`.

- Disk layout: `data/raw/{label}/{subject}_{NNN}.npy`, each `(30, 126)` float32.
- **Deviation from the spec**, deliberate: the spec says `sample_NNN.npy`, but the
  subject id is in the filename so two people recording letter `A` never collide.
  Cross-subject evaluation (M9) depends on this.
- `manifest["samples"]` is the source of truth — one entry per sample with label,
  subject, index, timestamp. `manifest["counts"]` is a derived convenience view,
  recomputed on every write.
- Writes are atomic (temp file, then `replace`), so a crash mid-write cannot
  corrupt the manifest.
- `load_sequences()` returns `(sequences, labels, subjects)` — this is what the
  trainer (M4) and every experiment will call.
- **Packs.** Bulk imports are stored as one `(n, 126)` file per
  `(label, subject)` rather than one file per sample. 87k tiny files inside a
  OneDrive folder would trigger a sync storm and make every manifest write
  crawl. Packs hold already-aggregated vectors; `manifest["packs"]` indexes them.
- `load_features()` is what trainers and experiments call: recorded sequences
  get median-aggregated, packs pass straight through, and you get one
  `(n_samples, 126)` matrix of **raw** coordinates. Apply `features.transform()`
  before training.
- `load_sequences()` still returns raw per-frame data, needed by the M11 merger
  ablation. Recorded samples only — packs have no frames.
- `python -m src.dataset` prints a dataset summary;
  `python -m src.dataset --remove-label TEST` deletes a label's samples from
  both disk and manifest (destructive).

### `python/src/preview.py` — M1 exit check
Live webcam, skeleton overlay, FPS counter. Prints the vector shape once so you
can confirm `(126,)`. Press `v` to dump the current vector, `q` to quit.

### `python/src/data_collector.py` — M2, the recording tool
Per sample: countdown → capture → save → rest.

- Captures exactly 30 frames **in which a hand was detected**. Frames with no
  hand are counted as skipped and retried, never zero-filled, so no stored
  sequence has holes in it.
- A sample that cannot gather 30 good frames within `--timeout` (default 10s) is
  abandoned and retried automatically — this is what stops you from silently
  recording garbage when your hand drifts out of frame.
- The manifest is saved after **every** sample, so a crash costs at most one.
- `--subject` is required from day one. Retrofitting subject tags onto an
  existing dataset is miserable; the spec warns about this and it is right.
- `--dry-run` exercises the whole flow without writing anything.

---

## 3. How to run things

```bash
cd python

# one-time
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m src.download_models     # hand_landmarker.task

# M1 check: does the skeleton render at ~30 FPS?
.venv/Scripts/python.exe -m src.preview

# M2 check: record 5 throwaway samples
.venv/Scripts/python.exe -m src.data_collector --label TEST --count 5 --subject s01

# what is in the dataset?
.venv/Scripts/python.exe -m src.dataset
```

On Windows, `.venv/Scripts/python.exe -m src.X` from inside `python/` is the
reliable invocation. Every module also works when run as a plain script — the
imports fall back from `from .config import` to `from config import`.

---

## 4. The hybrid dataset plan (M3)

Two sources, different jobs:

| source | volume | job |
|---|---|---|
| Kaggle ASL Alphabet, imported | ~600/letter | trains the SVM |
| your own recordings | 10–15/letter | held-out test set + M10/M18 calibration |

**Why not just download everything.** The Kaggle set is **one signer**, so a
model trained on it alone has never seen another hand and you would have no
honest way to measure whether it generalises. Your own recordings become the
unseen subject — "trained on public data, tested on a signer it never saw" is a
genuine cross-subject result, and it costs ~40 minutes instead of ~3 hours.

Import first, then record. Order matters: the import prints a per-class
detection rate, and if some letters come out badly you want to know before you
spend an evening recording their counterparts.

```bash
# after the Kaggle download finishes
.venv/Scripts/python.exe -m src.import_images --src data/downloads/asl_alphabet_train/asl_alphabet_train --dry-run
.venv/Scripts/python.exe -m src.import_images --src data/downloads/asl_alphabet_train/asl_alphabet_train

# then your own, ~12 per letter, subject s01
.venv/Scripts/python.exe -m src.data_collector --label A --count 12 --subject s01
```

Unknowns that only running it will settle: MediaPipe's detection rate on tight
200×200 crops (fist-like M, N, S, T are the likely failures), and whether the
dataset images are mirrored relative to your webcam. The second one is why
`--slot` pins the hand instead of trusting handedness, and why the first real
train/test split is the moment of truth.

### Recording advice for your own samples

- 24 letters × 12 samples = 288 samples, roughly 40 minutes.
- **Split across 2–3 sessions on different days.** Varied lighting and hand
  placement is what stops the model memorising your desk lamp.
- Use one subject id per person, e.g. `s01` for you.
- Sign with the **same hand every time**. One-handed ASL fills one slot and
  leaves the other 63 floats zero; switching hands mid-dataset produces two
  incompatible feature layouts for the same letter.
- Delete any throwaway `TEST` samples before training, or M4 will happily learn
  a `TEST` class: `rm -r python/data/raw/TEST` then re-record the manifest, or
  just avoid the label.
- Keep a printed ASL alphabet chart in view. Wrong handshapes poison the dataset
  and you will not notice until M9.
- If you record letter `A` in three sittings, just run the collector three times
  with `--count 20` — indices continue automatically, nothing is overwritten.

---

## 4b. Where the data actually stands

Imported 2026-08-23 from Kaggle: **14,400 samples, 600 per letter, all 24
balanced**, 82.5% of 17,448 images yielded landmarks in 21 minutes.

Detection rate varied more than expected:

| letters | detection |
|---|---|
| D, K, P, Y | 97–98% |
| most others | 83–96% |
| **M, N** | **40%, 42%** |

M and N tuck the thumb between the fingers, and MediaPipe often refuses. They
still reached 600 samples because `--target` keeps scanning rather than stopping
at a fixed image count — M needed 1,504 images to get there. Worth a sentence in
the report: the classes hardest to *detect* are not the ones hardest to
*classify*.

**First trained model: 99.79% validation accuracy. Ignore it.** It is one
signer, and consecutive frames are near-identical, so a random split puts
near-duplicates of the same moment on both sides. The model can score highly by
recognising the photo session. The number that means something comes from your
own recordings, which the model has never seen — that is the next step.

---

## 5b. Two bugs that real use found and synthetic tests did not

Both were in custom gestures, both surfaced by actually waving hands at the
camera, and both invisible to tests built on random synthetic hands. Worth
recording because the same class of mistake will recur in the browser port.

**The rolling buffer blended different hand counts.** A two-handed gesture kept
reporting "no match" while its stored samples classified 20/20 at confidence
1.00 — which located the fault at inference, not in the recording. The buffer
holds 30 frames and takes a median; raising a second hand leaves the window full
of one-handed history, and once past half the window that hand's 63 coordinates
median to zero. The gesture silently became one-handed:

| one-handed share of the window | distance from its own centre |
|---|---|
| 0–15 of 30 | 0.000 |
| 20+ of 30 | **2.745** (threshold 0.35) |

Fix: the window restarts when the hand count changes, so a median is only ever
taken over frames of the same shape.

**The reject threshold was ~8× too tight.** Measuring the three real gestures:

| | |
|---|---|
| distance between different gestures | 2.8 – 4.2 |
| spread within one gesture | 0.04 – 0.18 |
| threshold at the time | 0.35 |

Everything in that gap was rejected. The spread was itself an artefact of
overlapping sampling windows — it described one frozen moment, not how a gesture
varies between sessions, and building a threshold on it encoded the artefact.
Fix: scale the threshold against the distance to the nearest other gesture as
well, and decorrelate the samples at record time.

The lesson for the browser port: synthetic random hands cannot reproduce
temporal artefacts. Test with recorded sequences.

---

## 5. Known trade-offs, written down so they do not surprise you later

**Why absolute coordinates had to go.** Training data is now cropped 200×200
dataset images where the hand fills the frame; inference is a wide 640×480
webcam shot where it does not. In absolute coordinates those two occupy
completely different number ranges and nothing transfers — a model trained on
one would score near chance on the other. So `FEATURE_WRIST_RELATIVE` and
`FEATURE_SCALE_NORMALISE` are both **True**, which supersedes the build spec's
"normalised (x, y, z)". Note the reversal: scale normalisation measured as
*useless* within one camera (see the table below) and is *necessary* across two.
M8 keeps all three settings as ablation rows, so the decision stays defensible
rather than asserted.

**The original measurement — kept because it is still the evidence.** The vector
holds raw MediaPipe normalised coordinates, which are positions *in the image
frame*, so where your hand sits is part of the feature.

Your own 5 TEST samples quantify the cost. One handshape, held 5 times:

| representation | mean distance between the 5 samples |
|---|---|
| absolute (current default) | 0.334 |
| wrist-relative | 0.136 |
| wrist-relative + scale-normalised | 0.145 |

Within a single sample the vector moves by only 0.003, so almost all of that
0.334 is your hand being in a slightly different place, not a different shape.
Wrist-relative cuts it 2.5×. Scale normalisation adds nothing, so it is not
implemented.

That number alone does not settle it — a representation that shrinks
within-class spread is only useful if it does not shrink *between*-class
separation by as much, and you cannot measure that with one class. So: absolute
stays the default per the spec, `data/raw/` stores raw coordinates either way,
and M8 gets an extra ablation row comparing the two on the real 24-letter
dataset. If M9's cross-subject number disappoints, this is the first knob to
turn.

**sklearn deprecation, harmless for now.** `SVC(probability=True)` is
deprecated in scikit-learn 1.9 and goes away in 1.11; the suggested replacement
is `CalibratedClassifierCV(SVC(), ensemble=False)`. Left alone deliberately —
switching changes what M13 has to export to ONNX, so it is a decision to make
once, at export time, not twice.

**Measured speed.** 17.3 ms/frame at 640×480 on this machine with no hand in
shot — the worst case, since every frame runs full palm detection. That is ~58
FPS of headroom against the 30 FPS constraint. Warm-up matters: the first few
frames cost ~100 ms while XNNPACK initialises, so ignore the FPS reading for the
first second.

**Mirroring and handedness.** Two independent choices, and both must be
reproduced in the browser or slot 0 and slot 1 swap and predictions turn to
nonsense — the single most likely cause of a Python/browser mismatch at M16:
1. the frame is flipped horizontally before detection, and
2. MediaPipe's handedness label is then inverted back
   (`FLIP_HANDEDNESS = True`).
Net effect: slot 0 always holds the user's real left hand.

For one-handed ASL this matters more than it looks. If you sign with your right
hand, every sample lands in slot 1 and slot 0 stays zeros; a left-handed signer
produces the mirror-image feature layout, and the model will not transfer. Worth
noting in the report, and worth asking your M9 volunteers which hand they sign
with.

---

## 6. Not built yet

`aggregator.py` (M4), `trainer.py` (M4), `predictor.py` (M5), `merger.py` (M6),
`app.py` (M7), everything under `experiments/` (M8–M12), `export/` (M13), and the
entire `web/` build (M14–M26). `PROGRESS.md` tracks the running state.
