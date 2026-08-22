# signspeak

Real-time ASL fingerspelling recognition, plus user-defined custom gestures that
train from 20 samples in under two seconds. Python desktop system now, browser
port in progress.

Video never leaves the device. Only 126-dimensional landmark arrays are ever
stored or transmitted.

---

## How it works

```
webcam frame
   │  mirrored once, at capture time
   ▼
MediaPipe HandLandmarker  ──>  21 landmarks x (x, y, z) x 2 hands
   ▼
126-D vector   (absent hand = 63 zeros)
   ▼
rolling 30-frame buffer  ──>  per-coordinate median  ──>  126-D
   ▼
wrist-relative + scale-normalised  ──>  StandardScaler
   ▼
RBF SVM  ──>  (letter, confidence)
   ▼
four-stage merger  ──>  committed letters  ──>  word  ──>  sentence
```

The merger is what makes the output usable. A classifier firing 30 opinions a
second disagrees with itself constantly; four stages — confidence filtering,
buffer accumulation, hold-time validation and duplicate suppression — turn that
stream into letters a person would accept.

## Status

| | |
|---|---|
| Landmark extraction, recording tool | done |
| Dataset: 14,400 samples, 600 x 24 letters | done |
| SVM training, live prediction, merger | done |
| Experiments (sample efficiency, cross-subject, ablation) | next |
| Browser port, custom gestures, calibration | planned |

Letters A–I and K–Y. J and Z are excluded: both require motion.

## Quick start

```bash
cd python
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m src.download_models

.venv/Scripts/python.exe -m src.preview      # check the camera and skeleton
.venv/Scripts/python.exe -m src.live         # live fingerspelling
.venv/Scripts/python.exe -m src.live --raw   # merger off, to see what it fixes
```

### Building the dataset

```bash
# public data: any {label}/{image} folder layout
.venv/Scripts/python.exe -m src.import_images --src <asl_alphabet_train> --mirror

# your own samples
.venv/Scripts/python.exe -m src.data_collector --label A --count 12 --subject s01

.venv/Scripts/python.exe -m src.trainer
.venv/Scripts/python.exe -m src.trainer --holdout-subject s01   # honest number
```

### Tests

```bash
.venv/Scripts/python.exe tests/test_merger.py
```

## Three things that are easy to get wrong

**Chirality.** A left hand and a right hand signing the same letter are mirror
images, and neither wrist-relative nor scale normalisation removes that. Public
datasets are usually raw camera shots while a live selfie feed is mirrored, so
importing without `--mirror` trains on the mirror image of every handshape the
user will actually sign. It validates near-perfectly and fails completely live.

**Representation lives at train time, not record time.** `data/` always holds
raw absolute coordinates. Changing representation costs a retrain, never a
re-record, and every setting stays comparable on the identical dataset. The
model records which representation it was trained under and refuses to load
against a mismatched config.

**Validation accuracy on single-signer data is not a result.** Consecutive
frames are near-identical, so a random split puts near-duplicates of the same
moment on both sides. The number that means something comes from holding out a
signer the model has never seen.

## Layout

```
python/
  src/
    config.py             every tunable, one place
    landmark_extractor.py frame -> 126-D vector (the layout is contractual)
    data_collector.py     interactive recording
    import_images.py      public image datasets -> landmark packs
    features.py           wrist-relative, scale normalisation
    aggregator.py         median aggregation + rolling buffer
    trainer.py            SVM training
    predictor.py          inference with a feature-mismatch guard
    merger.py             four-stage state machine
    live.py               the whole pipeline in one window
  tests/
web/                      browser port (in progress)
docs/results/             figures and tables
```

`CODE_NOTES.md` explains each module and the trade-offs behind it.

## Built with

MediaPipe Tasks (hand landmarks) · scikit-learn (SVM) · OpenCV (camera and
drawing only, no recognition) · NumPy

The browser port loads the byte-identical `hand_landmarker.task` graph, so both
sides run the same feature pipeline.
