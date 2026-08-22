// Wires the whole browser pipeline together:
//   camera -> mirrored canvas -> landmarks -> rolling buffer -> median
//          -> SVM (letters) or KNN (custom gestures) -> merger -> text
import { CUSTOM_GESTURE_SAMPLES, SEQUENCE_LENGTH, TAU } from "./config.js";
import { RollingBuffer } from "./buffer.js";
import { LetterMerger } from "./merger.js";
import { LandmarkExtractor, drawMirrored, openCamera } from "./landmarks.js";
import { SvmClassifier } from "./classifier-svm.js";
import { GestureStore, KnnClassifier } from "./classifier-knn.js";

const ASL = "asl";
const CUSTOM = "custom";
const RECORD_STRIDE = 6;   // see record_gesture.py: overlapping windows

const el = (id) => document.getElementById(id);
const ui = {
  video: el("video"), canvas: el("canvas"), overlay: el("overlay"),
  overlayNote: el("overlay-note"), start: el("start"),
  fps: el("fps"), hands: el("hands"), status: el("status"),
  prediction: el("prediction"), confidenceBar: el("confidence-bar"),
  confidenceText: el("confidence-text"), candidates: el("candidates"),
  holdLabel: el("hold-label"), holdFrames: el("hold-frames"), holdBar: el("hold-bar"),
  sentence: el("sentence"), autoSpeak: el("auto-speak"),
  modeAsl: el("mode-asl"), modeCustom: el("mode-custom"),
  customPanel: el("custom-panel"), gestureList: el("gesture-list"),
  recorder: el("recorder"), recorderStatus: el("recorder-status"),
  recordBar: el("record-bar"), gestureName: el("gesture-name"),
  gesturePhrase: el("gesture-phrase"),
};

const ctx = ui.canvas.getContext("2d", { willReadFrequently: true });

const state = {
  mode: ASL,
  running: false,
  sentence: [],
  fps: 0,
  lastFrame: performance.now(),
  recording: null,          // { name, phrase, samples, sinceLast }
  busy: false,
};

const buffer = new RollingBuffer(SEQUENCE_LENGTH);
const merger = new LetterMerger();
const extractor = new LandmarkExtractor();
const svm = new SvmClassifier();
let store = GestureStore.load();
let knn = new KnnClassifier(store);

// ------------------------------------------------------------------ speech
function speak(text) {
  if (!text || !window.speechSynthesis) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

// -------------------------------------------------------------------- boot
async function boot() {
  ui.start.disabled = true;
  ui.overlayNote.textContent = "loading model…";
  try {
    await Promise.all([extractor.init(), svm.init()]);
    ui.status.textContent = svm.describe();
  } catch (err) {
    ui.overlayNote.textContent = `Could not load the model: ${err.message}`;
    ui.start.disabled = false;
    return;
  }

  try {
    await openCamera(ui.video);
  } catch (err) {
    // Camera failures are the most common first-run problem, and the browser's
    // own message ("Permission denied") does not say what to do about it.
    ui.overlayNote.textContent = err.name === "NotAllowedError"
      ? "Camera permission was refused. Allow it in the address bar, then reload."
      : `Camera unavailable: ${err.message}`;
    ui.start.disabled = false;
    return;
  }

  ui.canvas.width = ui.video.videoWidth || 640;
  ui.canvas.height = ui.video.videoHeight || 480;
  ui.overlay.classList.add("hidden");
  state.running = true;
  requestAnimationFrame(loop);
}

// -------------------------------------------------------------- main loop
async function loop(now) {
  if (!state.running) return;

  const delta = now - state.lastFrame;
  state.lastFrame = now;
  state.fps = state.fps ? state.fps * 0.9 + (1000 / delta) * 0.1 : 1000 / delta;

  const { width, height } = ui.canvas;
  drawMirrored(ctx, ui.video, width, height);

  const vector = extractor.detect(ui.canvas, now);
  extractor.drawOverlay(ctx, width, height);
  buffer.push(vector);

  // Inference is async; skipping frames while one is in flight keeps the
  // preview smooth instead of queueing work we cannot keep up with.
  if (!state.busy) {
    state.busy = true;
    try {
      await step(vector);
    } finally {
      state.busy = false;
    }
  }

  render();
  requestAnimationFrame(loop);
}

async function step(vector) {
  const aggregated = buffer.value();

  if (state.recording) {
    handleRecording(aggregated);
    return;
  }

  let label = null;
  let confidence = 0;
  let candidates = [];

  if (aggregated && vector) {
    if (state.mode === ASL) {
      const result = await svm.predict(aggregated);
      label = result.label;
      confidence = result.confidence;
      candidates = result.top.slice(1);
    } else {
      const result = knn.predict(aggregated);
      label = result.label;
      confidence = result.confidence;
    }
  }

  state.label = label;
  state.confidence = confidence;
  state.candidates = candidates;

  const committed = merger.update(label, confidence);
  if (committed) {
    if (state.mode === ASL) {
      state.sentence.push(committed);
    } else {
      const phrase = store.output(committed);
      state.sentence.push(`${phrase} `);
      if (ui.autoSpeak.checked) speak(phrase);
    }
  }
}

// --------------------------------------------------------------- recording
function handleRecording(aggregated) {
  const rec = state.recording;
  if (!aggregated) {
    ui.recorderStatus.textContent =
      `waiting for a steady hand (${Math.round(buffer.fill * 100)}%)…`;
    return;
  }
  // Samples are spaced out because consecutive windows overlap by 29 of 30
  // frames - sampling every frame gives 20 copies of one instant.
  rec.sinceLast += 1;
  if (rec.samples.length && rec.sinceLast < RECORD_STRIDE) return;
  rec.sinceLast = 0;
  rec.samples.push(aggregated);

  const progress = rec.samples.length / CUSTOM_GESTURE_SAMPLES;
  ui.recordBar.style.width = `${Math.min(100, progress * 100)}%`;
  ui.recorderStatus.textContent =
    `recording ${rec.samples.length}/${CUSTOM_GESTURE_SAMPLES} `
    + `(${buffer.hands} hand${buffer.hands === 1 ? "" : "s"}) `
    + `— keep moving it slightly`;

  if (rec.samples.length >= CUSTOM_GESTURE_SAMPLES) finishRecording();
}

function finishRecording() {
  const rec = state.recording;
  state.recording = null;

  const warning = knn.ready ? knn.nearestGesture(rec.samples, rec.name) : null;
  store.add(rec.name, rec.samples, rec.phrase);
  const millis = knn.fit(store);

  let note = `saved "${rec.name}" — trained in ${millis.toFixed(1)} ms`;
  if (warning && warning.distance < knn.thresholdFor(warning.name) * 0.8) {
    note += `. Warning: very similar to "${warning.name}" `
          + `(distance ${warning.distance.toFixed(2)}); they may be confused.`;
  }
  ui.status.textContent = note;
  ui.recorder.classList.add("hidden");
  renderGestures();
  setMode(CUSTOM);
}

// ------------------------------------------------------------------ render
function render() {
  ui.fps.textContent = `${state.fps.toFixed(0)} fps`;
  const n = buffer.hands;
  ui.hands.textContent = extractor.handsDetected
    ? `${n} hand${n === 1 ? "" : "s"}`
    : "no hand";

  if (state.recording) return;

  if (!buffer.ready) {
    ui.prediction.textContent = "—";
    ui.confidenceText.textContent =
      `reading your hand… ${Math.round(buffer.fill * 100)}%`;
    ui.confidenceBar.style.width = `${buffer.fill * 100}%`;
    ui.confidenceBar.className = "meter-fill amber";
    ui.candidates.innerHTML = "";
  } else if (!state.label) {
    ui.prediction.textContent = "—";
    ui.confidenceText.textContent = extractor.handsDetected
      ? (state.mode === CUSTOM && !knn.ready
          ? "no gestures recorded yet"
          : "no match")
      : "show your hand";
    ui.confidenceBar.style.width = "0%";
    ui.candidates.innerHTML = "";
  } else {
    ui.prediction.textContent = state.mode === CUSTOM
      ? store.output(state.label)
      : state.label;
    const pct = Math.round(state.confidence * 100);
    ui.confidenceBar.style.width = `${pct}%`;
    ui.confidenceBar.className =
      `meter-fill${state.confidence >= TAU ? "" : " red"}`;
    ui.confidenceText.textContent = `confidence ${pct}%`;
    ui.candidates.innerHTML = (state.candidates ?? [])
      .map(([name, score]) => `<li><span>${name}</span>`
        + `<span class="track"><span style="width:${score * 100}%"></span></span>`
        + `<span>${(score * 100).toFixed(0)}%</span></li>`)
      .join("");
  }

  const hold = merger.state;
  ui.holdLabel.textContent = hold.label
    ? `${hold.label} — ${hold.committed ? "committed" : "hold to confirm"}`
    : "nothing held";
  ui.holdFrames.textContent = hold.label ? `${hold.frames} frames` : "";
  ui.holdBar.style.width = `${hold.progress * 100}%`;
  ui.holdBar.className = `meter-fill${hold.committed ? "" : " amber"}`;

  ui.sentence.textContent = state.sentence.join("");
}

function renderGestures() {
  if (!store.size) {
    ui.gestureList.innerHTML =
      '<li class="muted small">No gestures yet. Record one above.</li>';
    return;
  }
  ui.gestureList.innerHTML = store.names.map((name) => {
    const hands = knn.hands[name] ?? 1;
    return `<li><span><strong>${name}</strong>`
      + `<span class="meta"> → ${store.output(name)} `
      + `· ${hands}h</span></span>`
      + `<button class="ghost danger" data-delete="${name}">Delete</button></li>`;
  }).join("");
}

function setMode(mode) {
  state.mode = mode;
  merger.reset();          // its candidate belongs to the old classifier
  buffer.clear();
  ui.modeAsl.classList.toggle("active", mode === ASL);
  ui.modeCustom.classList.toggle("active", mode === CUSTOM);
  ui.customPanel.classList.toggle("hidden", mode !== CUSTOM);
}

// ------------------------------------------------------------------ events
ui.start.addEventListener("click", boot);
ui.modeAsl.addEventListener("click", () => setMode(ASL));
ui.modeCustom.addEventListener("click", () => setMode(CUSTOM));

el("speak").addEventListener("click", () => speak(state.sentence.join("").trim()));
el("undo").addEventListener("click", () => { state.sentence.pop(); merger.undo(); });
el("space").addEventListener("click", () => state.sentence.push(" "));
el("clear").addEventListener("click", () => { state.sentence = []; merger.reset(); });

el("record").addEventListener("click", () => {
  ui.gestureName.value = "";
  ui.gesturePhrase.value = "";
  ui.recordBar.style.width = "0%";
  ui.recorderStatus.textContent =
    "Hold the shape once recording starts, and move it slightly.";
  ui.recorder.classList.remove("hidden");
});

el("record-cancel").addEventListener("click", () => {
  state.recording = null;
  ui.recorder.classList.add("hidden");
});

el("record-start").addEventListener("click", () => {
  const name = ui.gestureName.value.trim().replace(/\s+/g, "_");
  if (!name) { ui.recorderStatus.textContent = "Give it a name first."; return; }
  if (!state.running) { ui.recorderStatus.textContent = "Enable the camera first."; return; }
  state.recording = {
    name, phrase: ui.gesturePhrase.value.trim(), samples: [], sinceLast: 0,
  };
  buffer.clear();
  ui.recorderStatus.textContent = "get ready…";
});

ui.gestureList.addEventListener("click", (event) => {
  const name = event.target?.dataset?.delete;
  if (!name) return;
  store.remove(name);
  knn.fit(store);
  renderGestures();
});

el("export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(store.toJSON(), null, 2)],
                        { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "signspeak-gestures.json";
  link.click();
  URL.revokeObjectURL(link.href);
});

el("import").addEventListener("click", () => el("import-file").click());
el("import-file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    const { added, skipped } = store.importPayload(payload);
    knn.fit(store);
    renderGestures();
    ui.status.textContent = `imported ${added.length}`
      + (skipped.length ? `, skipped ${skipped.length} (name collision)` : "");
  } catch (err) {
    ui.status.textContent = `import failed: ${err.message}`;
  }
  event.target.value = "";
});

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT") return;
  if (event.key === "m") setMode(state.mode === ASL ? CUSTOM : ASL);
  if (event.key === "s") speak(state.sentence.join("").trim());
  if (event.key === "c") { state.sentence = []; merger.reset(); }
});

renderGestures();
ui.status.textContent = "ready — enable the camera to begin";
