# PROGRESS — working state for Claude

Purpose: if context is lost, read this file plus `CODE_NOTES.md` and resume
without re-reading the codebase. Update at the end of every milestone.

**Project:** few-shot gesture communication system. Spec is `proj_plan.md`
(v2, build-from-scratch, M1–M27). Python desktop first, then browser port.

**Environment (verified 2026-08-22):** Python 3.12.10, Node v22.20.0, npm 10.9.3,
git 2.51.0, Windows 11. Installed in venv: mediapipe 1.0.1, opencv 4.11.0.86,
numpy 1.26.4, scikit-learn 1.9.0, onnxruntime 1.29.0, skl2onnx 1.20.0. Repo root: `c:\Users\hpi7c\OneDrive\Desktop\sign language`.
Virtualenv at `python/.venv`. Not a git repo yet.

---

## Milestone status

| M | Milestone | State |
|---|---|---|
| M1 | config + landmark extraction + preview | **DONE** - user confirmed skeleton + correct handedness on webcam 2026-08-22 |
| M2 | recording tool | **DONE** - user recorded 5 TEST samples successfully 2026-08-22 |
| M3 | dataset — hybrid | **public half DONE** (14,400 imported); user's ~12/letter still to record |
| M4 | aggregator + trainer | **DONE** - trained on public data, 99.79% val (meaningless, see below) |
| M5 | live predictor | **DONE** - predictor.py + live.py, verified 8/8 on known samples |
| M6 | letter merger | **DONE** - 13/13 tests pass in tests/test_merger.py |
| M7 | desktop app | not started |
| M8–M12 | experiments | not started |
| M13 | ONNX export | not started |
| M19–M20 | custom gestures + phrases | **DONE early** - user asked for these before the experiments |
| M21 | confusion warning | **DONE** - warns at record time |
| M22 | speech output | **DONE** - pyttsx3 on a worker thread |
| M25 | export/import gesture sets | **DONE** - validated JSON |
| M14–M18, M23–M24, M26 | web build, accounts, mobile | not started |
| M27 | user testing | not started |

---

## What was built (M1 + M2)

Files created, all under `python/src/`:

- `config.py` — every constant and path in the project. Single source of truth.
- `camera.py` — `Camera` (DSHOW backend, 1-frame buffer, mirrors once) and
  `FPSMeter`.
- `landmark_extractor.py` — MediaPipe **Tasks** `HandLandmarker` wrapper, frame →
  `(126,)` float32 or `None`. Handedness (not detection order) selects the hand
  slot. Hand-rolled skeleton drawing (Tasks API has no drawing helper).
  `describe_layout()` prints the index contract for the M15 browser port.
- `download_models.py` — fetches `hand_landmarker.task` (7.8 MB) into
  `data/models/` and copies it to `web/public/models/`.
- `features.py` — `transform()` (wrist-relative then scale-normalise; scale
  implies wrist-relative), `to_wrist_relative()`, `to_scale_normalised()`,
  `describe()`. Absent hands stay all-zero throughout.
- `aggregator.py` — `aggregate()`, per-coordinate median over the time axis.
- `import_images.py` — `{label}/{image}` folders -> landmark packs. `--src
  --subject --source --limit --classes --slot --mirror --min-visible
  --dry-run`. max_hands=1, static IMAGE mode, even-stride subsampling, pinned
  hand slot, per-class detection rate reported.
- `dataset.py` — manifest read/write (atomic), `next_index`, `add_sample`,
  `add_pack`, `load_sequences`, `load_features`, `summary`, `remove_label`.
  CLI: `python -m src.dataset [--remove-label LABEL]`.
  **Packs**: bulk imports go to `data/packs/{label}__{subject}.npy` as one
  `(n,126)` file per pair, NOT one file per sample - 87k tiny files in a
  OneDrive folder would cause a sync storm. `load_features()` is the trainer's
  entry point (median-aggregates sequences, passes packs through, returns raw
  coordinates).
- `data_collector.py` — countdown → capture 30 hand-present frames → save →
  rest. `--label --count --subject --countdown --pause --timeout --camera
  --dry-run`. Manifest saved after every sample.
- `preview.py` — M1 verification: live skeleton + FPS + vector shape.
- `python/requirements.txt` — mediapipe, opencv, numpy<2, sklearn, joblib, pandas,
  matplotlib, seaborn, skl2onnx, onnx, onnxruntime.

Directories created: `python/{src,experiments,export,data/raw,data/models}`,
`web/`, `docs/results/`.

## Decisions made (do not silently reverse these)

1. **Hand slot is chosen by MediaPipe handedness label**, not detection order.
   Slot 0 = `"Left"` (indices 0–62), slot 1 = `"Right"` (63–125). Without this
   the same hand would swap halves between frames.
2. **Frames are mirrored once, in `Camera.read()`**, AND MediaPipe's handedness
   label is then inverted (`config.FLIP_HANDEDNESS = True`). MediaPipe reports
   handedness for the un-mirrored view — user confirmed on 2026-08-22 that
   raising the real left hand reported `"Right"`. Slots are defined by the
   user's REAL hands: slot 0 = real left. If MediaPipe ever changes this, flip
   `FLIP_HANDEDNESS`; never reorder `HAND_SLOTS`, or previously recorded
   samples become mislabelled. The browser port must reproduce BOTH steps.
3a. **DATASET STRATEGY CHANGED 2026-08-23 (user's call, hybrid route).**
   Instead of the user recording 24x60=1440 samples, bulk training data comes
   from the Kaggle ASL Alphabet set (grassknoted/asl-alphabet, 87k images,
   200x200, 29 classes) imported via `src/import_images.py`, and the user
   records only ~12/letter as a held-out test set + calibration data.
   Rationale: that Kaggle set is ONE signer, so training on it alone gives no
   generalisation evidence; the user's own recordings become the unseen
   subject, which is a genuine cross-subject result and cuts recording from
   ~3h to ~40min. Rejected alternatives: Google ASL Fingerspelling (real
   landmarks, 100+ signers, but continuous phrase-level spelling, 543
   landmarks/frame - segmentation is its own project) and Sign Language MNIST
   (28x28 pre-cropped, MediaPipe cannot detect hands at that resolution).

3b. **Feature representation is applied at train/inference time, in
   `features.py` - NOT at record time.** `data/raw/` always stores raw absolute
   coordinates. Originally the wrist-relative transform lived in the extractor,
   which would have baked the choice into every recorded file; moved out before
   the dataset was recorded. Consequence: changing representation needs a
   retrain, never a re-record, and both settings are comparable on the identical
   dataset (planned M8 ablation row).
   **Both `FEATURE_WRIST_RELATIVE` and `FEATURE_SCALE_NORMALISE` are now
   True**, overriding the spec's "absolute (x, y, z)". Forced by the hybrid
   dataset: training images are cropped 200x200 with the hand filling the frame,
   inference is a wide 640x480 webcam shot, and in absolute coordinates the two
   occupy different ranges so nothing transfers. Note the reversal - scale
   normalisation measured as useless within one camera and is necessary across
   two. M8 keeps all three settings as ablation rows.
   Measured on the user's 5 TEST samples: mean within-class distance 0.334
   absolute vs 0.136 wrist-relative vs 0.145 wrist-relative+scale-normalised;
   within-sample jitter only 0.003. Scale normalisation is not implemented -
   it made things worse.
4. **MediaPipe Tasks API, not `mp.solutions`.** mediapipe resolved to 1.0.1,
   which deleted the legacy `mp.solutions` namespace entirely — the original
   implementation had to be rewritten. This is a net win: the browser uses
   `@mediapipe/tasks-vision` HandLandmarker with the byte-identical
   `hand_landmarker.task` (float16/1) asset, so both sides run the same graph and
   M16 parity is far more likely. Runs in `RunningMode.VIDEO` with a
   self-maintained monotonic timestamp counter. No `model_complexity` knob
   exists in Tasks; the variant is the `.task` file.
5. **Sample filenames are `{subject}_{NNN}.npy`**, not the spec's
   `sample_NNN.npy`. Subject in the filename prevents collisions between people.
   Cross-subject eval (M9) needs it.
6. **Venv at `python/.venv`**, invoked as `.venv/Scripts/python.exe -m src.X`
   from inside `python/`.
7. Modules import via `try: from .config … except ImportError: from config …`
   so they work both as `-m src.X` and as plain scripts.
8. **Extractor supports `static=True`** (RunningMode.IMAGE, `detect()` instead
   of `detect_for_video()`) for image ingestion, and `process(force_slot=N)` to
   pin which hand slot receives the detected hand. Third-party stills give no
   reliable clue whether they were mirrored, so handedness cannot be trusted
   there; consistency matters, not which slot.

## Import + first training result (2026-08-23)

- **Kaggle import complete**: 14,400 samples, 600/letter, 24 classes, subject
  `public`, source `grassknoted/asl-alphabet`. 82.5% of 17,448 images yielded
  landmarks, 21 minutes. Stored as packs in `data/packs/`.
- Detection rates: D/K/P/Y 97-98%, most 83-96%, **M 40% and N 42%** (thumb
  tucked between fingers). All still reached 600 because `--target` keeps
  scanning; M needed 1,504 images.
- **CHIRALITY: the Kaggle set MUST be imported with `--mirror`.** Measured with
  a pose-independent palm-normal sign: user's mirrored webcam +0.071, Kaggle
  as-is -0.050, Kaggle mirrored +0.052. Without `--mirror` every training sample
  is the mirror image of what the user signs - validates near-perfectly, fails
  completely live. Any re-import must keep this flag.
- **First model trained and saved** (`data/models/svm.joblib`, `scaler.joblib`,
  `labels.json`, `model_meta.json`): 99.79% validation, 22s with probability
  calibration, 1437 support vectors. **This number is meaningless** - single
  signer, near-duplicate frames, random split leaks. The honest number needs the
  user's own recordings as a held-out subject.
- Known deprecation: sklearn 1.9 deprecates `SVC(probability=True)` (removed in
  1.11). Left as-is deliberately; revisit at M13 since it changes the ONNX
  export path.

## M5 + M6 (2026-08-23)

- `predictor.py`: takes RAW vectors and applies `features.transform` internally
  so callers cannot forget. Refuses to load a model whose `model_meta.json`
  feature flags disagree with config - guard verified by simulating a mismatch.
  Falls back to softmax-over-margins when the model is uncalibrated.
- `merger.py`: four stages, each independently toggleable via `Stages` (this is
  what M11's ablation drives). **Time is injected** (`update(..., now=)`) so
  replays are deterministic.
  **Deviation from spec:** `idle_frames_to_forget=15` clears the stage-4
  duplicate guard after a gap, otherwise "HELLO" -> "HELO". M11 should measure
  this.
- `aggregator.RollingBuffer`: last 30 good frames, median-collapsed. Frames with
  no hand are not added, matching data_collector.
- `live.py`: full pipeline in one window. `--raw` disables all merger stages to
  show the flicker (good for the demo video).
- `tests/test_merger.py`: 13 tests, all pass, no webcam/model needed.
- Predictor sanity: 8/8 correct on known training samples, confidences ~0.99.

## Custom gestures (2026-08-23, brought forward from M19/M20)

- `custom_gestures.py` (GestureStore + KNNClassifier), `record_gesture.py`,
  mode toggle in `live.py` (`m` key), `tests/test_custom_gestures.py`,
  `tests/test_buffer.py`.
- Storage: `data/custom/gestures.json`, plain JSON so the browser can hold the
  same structure in IndexedDB and export/import stays a file copy.
- User recorded 3 gestures: `no` (1 hand), `yes` (1 hand), `thumbs_up`
  (2 hands). Working end to end as of this entry.
- KNN rebuild: 0.3-0.5 ms for 60 samples. The <2s few-shot claim is met by a
  wide margin and is now measurable.

### TWO BUGS FOUND BY REAL USE - do not reintroduce

1. **RollingBuffer blended hand counts.** Median over a window mixing one- and
   two-handed frames zeroes a hand's 63 coordinates once its absence passes
   half the window. Distance from centroid went 0.000 -> 2.745 against a 0.35
   threshold. Fix: window restarts when the hand mask changes (`_mask`).
2. **Reject threshold ~8x too tight.** Between-gesture distances 2.8-4.2,
   within-gesture spread 0.04-0.18, threshold 0.35. The spread was an artefact
   of overlapping sampling windows (consecutive samples 0.014 apart). Fix:
   threshold = max(2.5 x spread, 0.35 x distance to nearest other gesture);
   plus `--stride 6` at record time to decorrelate samples. Thresholds moved
   0.35 -> ~1.0.

Both were invisible to synthetic-random-hand tests. **The browser port must be
tested against recorded sequences, not synthetic vectors.**

## Polish pass (2026-08-23)

- `speech.py`: pyttsx3 on a daemon worker thread; `say()` returns in ~0.02 ms so
  the capture loop never blocks. **Engine is built inside the worker thread**
  (SAPI5 is COM-based, not thread-safe) **and rebuilt per utterance**.
  BUG FOUND BY USE: a reused engine speaks once, then every later runAndWait()
  returns instantly and silently. Measured on a ~5.5s phrase: 5.49s, then 0.50s,
  then 0.38s. Fix is a fresh engine per utterance plus gc.collect() (pyttsx3
  returns a cached engine while one is still referenced). Costs ~0.07s.
  tests/test_speech.py asserts on ELAPSED TIME because the failure is silent and
  looks successful. Degrades to silence if pyttsx3 is absent. Keys in live.py: `s` speak, `a` toggle auto-speak
  (custom mode only - per-letter speech is noise).
- Export/import (M25): `--export FILE`, `--import FILE`, `--prefix`, `--force`.
  Imports validated (feature_dim, shape, numeric, non-empty); collisions
  reported not silently resolved. ~200 KB per 20 gestures.
- requirements.txt gained pyttsx3.
- All 3 test files pass; test_custom_gestures.py now ~35 checks.

## Verified so far

- All modules byte-compile; every dependency imports in the venv.
- Headless smoke test passed: landmarker builds in 1.2 s, blank frame returns
  `None`, `draw_overlay` is safe with no result, manifest round-trip
  (add 3 → save → load → `(3, 30, 126)`) works, shape guard rejects `(5, 126)`.
- Benchmark: **17.3 ms/frame at 640×480** with no hand in shot (worst case, full
  palm detection every frame) = ~58 FPS. The 30 FPS constraint has headroom.
  First ~1 s is XNNPACK warm-up at ~100 ms/frame; ignore early FPS readings.
- **M2 verified end to end on real data.** User recorded 5 TEST samples:
  `(5, 30, 126)`, slot 1 filled / slot 0 all zeros (user signs **right-handed**),
  x,y within [0,1], z in [-0.145, 0], within-sample per-coordinate std 0.0027
  (very steady hold).
- `features.to_wrist_relative` verified on that real data: empty slot stays
  exactly zero, active wrist lands on the origin, shape and dtype preserved
  across `(126,)`, `(n,126)` and `(n,30,126)`, and `transform()` is the identity
  under the default config.
- Handedness: user ran `src.preview`, reported real left hand labelled
  `"Right"` -> `FLIP_HANDEDNESS = True` added, unit-checked, and **user
  re-confirmed on webcam that both hands now label correctly**. M1 exit met.
- **Still unverified: real-world FPS with a hand in shot** (user did not report
  a number), and the M2 recording flow end to end.

## Open items / next actions

- **Known-unfixed:** the FPS readout in the collector was measuring once per
  sample (showed 0.2); fixed by making the header a per-frame callable. User has
  not re-run it since, so the real capture FPS is still unknown.
- 5 TEST samples are still in the dataset. Purge before M4:
  `python -m src.dataset --remove-label TEST`.
- **User signs right-handed** - slot 1. Volunteers for M9 must be asked which
  hand they sign with; a left-handed signer produces a mirrored feature layout.
- **Kaggle UI changed**: the token page now offers named "API Tokens" (needs
  kaggle CLI >= 1.8.0) and "Legacy API Credentials". User was told to use
  **Create Legacy API Key** -> `kaggle.json` -> the `.kaggle` folder in their
  home directory.
- **Unverified until the user runs it:** MediaPipe detection rate on 200x200
  crops, and whether the Kaggle images are mirrored relative to the webcam
  (this is why `--slot` pins the hand rather than trusting handedness). The
  first real train/test split is the moment of truth.
- **Downloads live OUTSIDE the repo** at `~/gesture-data/downloads` - the repo
  is inside OneDrive and 87k unpacked images would sync every one. Only derived
  packs stay in `data/`.
- Next up: `python -m src.data_collector --label TEST --count 5
  --subject s01` (M2 exit).
- **User must record the dataset (M3).** Everything downstream is blocked on it.
- Once data exists: M4 = `aggregator.py` (per-coordinate median over the (30,126)
  sequence) + `trainer.py` (load → aggregate → 80/20 stratified split
  random_state=42 → StandardScaler fit on train only → SVC(rbf, C=10,
  gamma=scale, probability=True) → report + timing → save to `data/models/`).
- git repo not yet initialised; spec asks for a commit per milestone. Not done —
  waiting for the user's go-ahead on committing.
- The `sign language plan.pdf` in the repo root is a scanned image PDF with no
  extractable text. `proj_plan.md` supersedes it. Ignore the PDF.

## Notes to self

- User wants terse output, no fluff. Two docs to maintain: `CODE_NOTES.md`
  (explains code flow, for the human) and this file (working state, for me).
- Spec's "one milestone per session, wait for confirmation" is guidance the user
  gives when they want it; they asked to just build. M1+M2 were done together
  because M3 is a hard human-labour gate anyway.
