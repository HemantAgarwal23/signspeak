# Build Spec v2 — Few-Shot Gesture Communication System
### Building from scratch — no existing code, no existing dataset

**How to use this:** Hand one milestone at a time to Claude Code in VS Code. Verify the exit criteria before starting the next. Do not paste the whole document at once.

---

## PART A — WHAT WE ARE BUILDING

### Session primer (paste at the start of every Claude Code session)

> I'm building a hand gesture recognition system in two parts: a Python desktop version first, then a browser version. The pipeline is: webcam → MediaPipe Hands extracts 21 landmarks per hand (126 numbers per frame, second hand zero-padded) → rolling 30-frame buffer collapsed to one vector by per-coordinate median → classified by an RBF-kernel SVM (ASL letters) or a KNN over user-recorded samples (custom gestures) → a four-stage state machine converts noisy per-frame predictions into stable text. Users can define new gestures from 20 samples in under 2 seconds. In the web version everything runs client-side; only 126-D landmark arrays are stored, never video.

### Hard constraints

- No GPU. Must run on a mid-range laptop CPU at 30 FPS.
- Custom gesture training completes in under 2 seconds.
- Web version: all inference in-browser, video never transmitted.
- Python version stays working permanently as the fallback demo.

### Key parameters (use these everywhere, keep them in one config file)

```
LANDMARKS_PER_HAND   = 21
COORDS               = 3          # x, y, z
MAX_HANDS            = 2
FEATURE_DIM          = 126        # 21 * 3 * 2
SEQUENCE_LENGTH      = 30         # frames (~1s at 30fps)
AGGREGATION          = median     # per-coordinate
SVM: kernel=rbf, C=10, gamma=scale, probability=True
MERGER: tau=0.35, min_frames=12, min_hold_time=0.4s
MEDIAPIPE: detection_confidence=0.7, tracking_confidence=0.7
CUSTOM_GESTURE_SAMPLES = 20
CALIBRATION_SAMPLES    = 5
```

---

## PART B — REPO STRUCTURE

```
gesture-comm/
├── python/
│   ├── src/
│   │   ├── config.py
│   │   ├── landmark_extractor.py
│   │   ├── data_collector.py
│   │   ├── aggregator.py
│   │   ├── trainer.py
│   │   ├── predictor.py
│   │   ├── merger.py
│   │   └── app.py
│   ├── experiments/
│   ├── export/
│   └── data/
│       ├── raw/{label}/sample_NNN.npy
│       ├── manifest.json
│       └── models/
├── web/
│   ├── index.html
│   ├── src/
│   │   ├── main.js
│   │   ├── landmarks.js
│   │   ├── buffer.js
│   │   ├── classifier-svm.js
│   │   ├── classifier-knn.js
│   │   ├── merger.js
│   │   ├── storage.js
│   │   ├── phrases.js
│   │   └── ui/
│   └── public/models/
└── docs/results/
```

---

## PHASE 1 — PYTHON FOUNDATION (weeks 1–2)

---

### M1 — Config and landmark extraction

**Prompt:**
> Create a Python project with the structure above. In `src/config.py`, define all the parameters I'll paste (from Part A). In `src/landmark_extractor.py`, build a class wrapping MediaPipe Hands that takes a BGR frame and returns a 126-element float32 numpy array: for each of up to 2 hands, 21 landmarks as normalised (x, y, z), flattened and concatenated, with zeros for a missing second hand. Return None if no hand is detected. Include a method to draw the landmark skeleton on a frame for preview. Also write a small script that opens the webcam and displays the live skeleton overlay with the current FPS, so I can verify it works.

**Exit:** skeleton renders over your hand at ~30 FPS. Print the vector shape and confirm it's `(126,)`.

---

### M2 — Recording tool

**Prompt:**
> In `src/data_collector.py`, build an interactive recording tool. It takes a label (e.g. "A") and a sample count. For each sample: show a live preview with skeleton overlay, run a 3-second countdown, then capture exactly 30 consecutive frames where a hand is detected, save as a numpy array of shape (30, 126) to `data/raw/{label}/sample_NNN.npy`, then pause 1 second before the next. Skip and retry frames where no hand is detected so every saved sequence is complete. Maintain `data/manifest.json` recording per-label sample counts, a subject ID field, and timestamps. Show progress on screen throughout.

**Important:** include a `--subject` flag from the start. You'll need subject tags for cross-subject evaluation later, and retrofitting them is painful.

**Exit:** record 5 test samples, verify the `.npy` files load with the right shape.

---

### M3 — RECORD THE DATASET
*No code. This is you and a webcam.*

- 24 ASL letters (A–I, K–Y — skip J and Z, they need motion)
- 60 samples per letter = 1440 samples
- ~25 minutes of pure recording; realistically 2–3 hours with setup and retakes
- **Split across 3+ sessions on different days** — varied lighting makes the model more robust, and it's less exhausting

**Reference:** use a printed ASL alphabet chart. Getting the handshapes right matters.

**Exit:** `manifest.json` shows 60 samples for each of 24 letters.

---

### M4 — Training pipeline

**Prompt:**
> In `src/aggregator.py`, write a function collapsing a (30, 126) sequence to a single 126-D vector via per-coordinate median. In `src/trainer.py`, build the training pipeline: load all sequences from `data/raw/` with their labels and subject IDs, aggregate each, split 80/20 stratified with random_state=42, fit StandardScaler on the training split only, train sklearn SVC with the config hyperparameters, report train and validation accuracy plus a full per-class classification report, measure and print wall-clock training time, and save the model, scaler, and label list to `data/models/`.

**Exit:** a trained model. Expect high validation accuracy — treat it as suspicious, not celebratory. M9 will tell you the real number.

---

### M5 — Live prediction

**Prompt:**
> In `src/predictor.py`, build a class that loads the trained model and scaler and exposes a predict method taking a 126-D vector and returning (label, confidence, all_probabilities). Then write a script maintaining a rolling 30-frame buffer from the webcam, aggregating with the median, and displaying the live prediction and confidence on screen alongside the skeleton overlay.

**Exit:** live predictions on screen. They will flicker. That's the problem M6 solves.

---

### M6 — Letter Merger

**Prompt:**
> In `src/merger.py`, implement the four-stage temporal state machine.
> Stage 1 — Confidence filtering: discard predictions with confidence below tau (0.35).
> Stage 2 — Buffer accumulation: if the predicted label matches the current buffer's label, increment its frame count and update the running mean confidence; if it differs, validate the current buffer through Stage 3 then start a new buffer with the new label.
> Stage 3 — Hold-time validation: commit a letter only if frame_count >= 12 AND elapsed_time >= 0.4s AND mean_confidence >= tau, all three simultaneously.
> Stage 4 — Duplicate prevention: if the letter being committed equals the last committed letter, suppress it.
> Expose the current buffer state (label, frame count, progress toward confirmation) so a UI can display it. Include a reset method.

**Exit:** hold a letter, it commits once. Hold it longer, it doesn't repeat. Spell a word cleanly.

---

### M7 — Desktop app

**Prompt:**
> In `src/app.py`, build the desktop application. Layout: live video with skeleton overlay, current raw prediction with confidence, a progress bar toward letter confirmation, the current word, the accumulated sentence, and buttons for clear word, undo letter, clear sentence, and reset merger. Use a non-blocking queue between the capture loop and the UI render loop so frame drops don't freeze the interface. Use tkinter or FreeSimpleGUI.

**Exit:** **a complete working desktop system.** This is your guaranteed demo. Commit it and never break it.

---

## PHASE 2 — EXPERIMENTS (week 3)

*These generate every figure in your report. Do them before the web build.*

---

### M8 — Sample efficiency
*Your most important result.*

**Prompt:**
> Write `experiments/sample_efficiency.py`. For each N in [5, 10, 20, 40, 60]: subsample N samples per class from the training pool, train the full pipeline, evaluate on a fixed held-out test set that is never subsampled. Repeat each N across 5 random seeds. Report mean and standard deviation of accuracy, plus wall-clock training time. Output a CSV and a matplotlib plot with error bars to `docs/results/`.

**Exit:** the graph backing your central claim. If it looks bad, tell me — the framing may need to change.

---

### M9 — Cross-subject evaluation

First: **record 2–3 other people**, ~15 samples per letter each, using `--subject`. Budget an hour per person.

**Prompt:**
> Write `experiments/cross_subject.py` implementing leave-one-subject-out evaluation: for each subject, train on all others and test on the held-out one. Report per-subject accuracy, the mean, and a comparison against the within-subject random-split baseline. Output a table and bar chart.

**Exit:** your honest generalisation number. Expect a significant drop. That's the finding.

---

### M10 — Calibration

**Prompt:**
> Write `experiments/calibration.py`. For each held-out subject, take 5 samples per class from them and build a KNN over those samples. Combine the pretrained SVM prediction with this personal KNN via confidence-weighted voting. Compare accuracy on that subject's remaining samples before and after calibration. Output a before/after bar chart.

**Exit:** proof that calibration recovers accuracy. Weakness → contribution.

---

### M11 — Merger ablation

**Prompt:**
> Write `experiments/merger_ablation.py`. Record video sequences of known spelled words, then measure character error rate under: no merger (raw per-frame), confidence filtering only, plus frame accumulation, plus hold-time validation, and the full four-stage version. Output a comparison table.

**Exit:** your contribution, quantified.

---

### M12 — Confusion matrix

**Prompt:**
> Write `experiments/confusion_matrix.py` producing a normalised confusion matrix heatmap with seaborn, plus the top 10 most-confused class pairs as a sorted table.

---

## PHASE 3 — WEB BUILD (weeks 4–8)

---

### M13 — Model export

**Prompt:**
> Write `export/export_to_onnx.py`. Convert the trained SVM pipeline to ONNX using skl2onnx with input shape (1, 126). Verify the exported model produces identical predictions to sklearn on 100 test samples. Save to `web/public/models/asl_svm.onnx`. Separately export the StandardScaler mean and scale arrays as `scaler.json` and the label list as `labels.json`.

**Fallback if skl2onnx fights you:** export support vectors, dual coefficients, gamma, and intercepts as JSON; implement RBF prediction manually in JS (~30 lines).

**Exit:** ONNX predictions match sklearn exactly.

---

### M14 — Web skeleton

**Prompt:**
> Set up `web/` using Vite with vanilla JavaScript, no framework. Create index.html with a video element and canvas overlay, plus src/main.js. Install @mediapipe/tasks-vision and onnxruntime-web. Minimal — no styling, no build complexity.

---

### M15 — Landmarks in browser

**Prompt:**
> In `src/landmarks.js`, implement webcam capture and MediaPipe hand detection using @mediapipe/tasks-vision HandLandmarker. Draw the skeleton on the canvas overlay and expose a callback providing a 126-element Float32Array per frame, matching exactly the coordinate ordering in my Python landmark_extractor.py, which I'll paste now. Display live FPS.

**Paste `landmark_extractor.py` into this prompt.** Ordering mismatch is the single most common failure here.

---

### M16 — Live prediction in browser

**Prompt:**
> In `src/buffer.js`, implement the rolling 30-frame buffer with per-coordinate median. In `src/classifier-svm.js`, load asl_svm.onnx with onnxruntime-web plus scaler.json, apply standardisation, run inference, return label and probabilities. Wire into main.js and display prediction and confidence live.

**Verification — do not skip:** feed the same 126-D vector through Python and browser. Outputs must match. If they don't, it's coordinate ordering or the scaler.

---

### M17 — Merger port

**Prompt:**
> Port the Letter Merger to `src/merger.js`. I'll paste merger.py. Implement all four stages identically with the same thresholds. Expose buffer state for the UI.

**Paste `merger.py`.** Don't describe it.

**Exit:** **working web fingerspelling system. Safe stopping point.** Everything after is upside.

---

### M18 — UI
> Video with overlay on the left; right panel showing prediction with confidence, confirmation progress, current word, sentence, and controls for clear word / undo letter / clear sentence / reset. Clean minimal styling, no CSS framework.

### M19 — Custom gestures
> Recording UI: name input, 3-second countdown, capture 20 median-aggregated samples with a progress bar, store in IndexedDB via storage.js. Implement KNN (k=3, Euclidean) over stored gestures returning label and distance-derived confidence. Display elapsed training time.

### M20 — Modes and phrases
> Mode toggle between ASL (SVM) and custom (KNN). In phrases.js, map gesture labels to output phrases, editable via UI. On confirmation in custom mode, output the mapped phrase. Include a starter pack of 8 phrases.

### M21 — Confusion warning
> After recording a new gesture, compute mean Euclidean distance from its samples to each existing gesture's samples. If the nearest is below a threshold calibrated from inter-class distances in the ASL dataset, warn the user and offer to re-record.

### M22 — Speech output
> Browser SpeechSynthesis. Speak button for the sentence, auto-speak toggle for confirmed phrases in custom mode, voice selection if available.

### M23 — Accounts and sync
> Supabase email/password auth. Store per-user gesture samples as JSON arrays of 126-D vectors plus phrase mappings. Load into IndexedDB on login, sync on change, fall back to local when offline. No image or video data is ever transmitted.

### M24 — Browser calibration
> Guided flow recording 5 samples per letter, stored as a personal KNN, combined with the SVM via confidence-weighted voting exactly as validated in experiments/calibration.py. Show before/after accuracy on the user's held-out samples.

### M25 — Export/import
> Export a user's full gesture vocabulary plus phrase mappings as downloadable JSON; import with schema validation and collision prompts.

### M26 — Polish and mobile
> Responsive layout, mobile camera handling, vertical stacking on narrow screens, help overlay, loading states for model download.

---

## PHASE 4 — VALIDATION (week 9)

### M27 — User testing
*Not code. Do not skip.*

- 2–3 people outside your team
- Each defines 3 gestures and uses the system for a real task
- Watch without helping. Note every confusion.
- Write up what broke.

**This paragraph outweighs any feature in your report.**

---

## TIMELINE

| Week | Milestones | Output |
|---|---|---|
| 1 | M1–M3 | Recording tool + full dataset recorded |
| 2 | M4–M7 | **Working Python desktop system** |
| 3 | M8–M12 | All experiments, all report figures |
| 4 | M13–M15 | Model exported, landmarks in browser |
| 5 | M16–M17 | **Working web system** ← safe stop |
| 6 | M18–M20 | UI, custom gestures, phrases |
| 7 | M21–M23 | Warnings, speech, accounts |
| 8 | M24–M26 | Calibration, export, polish |
| 9 | M27 | User testing |
| 10+ | — | Report, paper, demo video |

**Compression points:** stop after week 2 and you have a working system. Stop after week 3 and you have a working system *with evidence* — that alone is a complete, defensible project. Weeks 4–8 make it distinct. Week 9 makes it credible.

---

## WORKING WITH CLAUDE CODE

**Every session starts with:** the session primer (Part A) + the key parameters + the specific milestone prompt.

**Rules:**
- One milestone per session. Don't chain.
- Paste actual Python files when porting (M15, M17). Never describe them.
- Verify numerically at M16 before proceeding.
- `git commit` after every milestone.
- Paste real errors, not "it doesn't work."
- Ask it to explain the code — you're defending this in a viva.

**Say this every session:**
> Before writing code, tell me your plan and which files you'll change. Wait for my confirmation.

---

## THINGS THAT WILL GO WRONG

| Problem | Fix |
|---|---|
| Browser ≠ Python predictions | Coordinate ordering. Print both vectors side by side. |
| ONNX export fails | Export SVM internals as JSON, hand-implement RBF (~30 lines) |
| FPS below 30 | Drop to 15 FPS, halve buffer to 15, retune hold-time |
| MediaPipe won't load in browser | Must be served over HTTP, not file:// |
| Camera blocked | HTTPS or localhost only. Deploy to Vercel to test on other devices. |
| Sample-efficiency curve looks bad | Tell me — the framing may need to change |
| Volunteers unavailable | Recruit in week 1, not week 3 |

**Golden rule:** the Python desktop version must stay working the entire time.

---

## DO THIS WEEK

1. Start M1 today — landmark extraction is the foundation
2. **Message 2–3 people about being cross-subject volunteers.** This is the slowest thing to arrange and you can't compress it later.
3. Print an ASL alphabet chart before M3