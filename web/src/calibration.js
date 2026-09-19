// M24 - personal calibration, as validated in python/experiments/calibration.py.
//
// A few of the user's own samples per letter go into a KnnClassifier (the same
// class that recognises custom gestures), and its answer is combined with the
// SVM's by confidence-weighted voting. Measured on an unseen signer: 83.9% ->
// 96.3% with 5 samples per letter. The vote rule below must stay identical to
// vote() in that experiment, or the number no longer describes this code.
import { CALIBRATION_COUNTDOWN, CALIBRATION_HELD_OUT, CALIBRATION_SAMPLES,
         CALIBRATION_STRIDE } from "./config.js";
import { GestureStore, KnnClassifier } from "./classifier-knn.js";

const STORE_KEY = "signspeak.calibration.v1";
const RESULT_KEY = "signspeak.calibration.result.v1";

// Agreement wins outright; disagreement goes to the higher confidence; an
// abstaining KNN defers to the SVM. The Python experiment only needs the label;
// live use also needs a confidence for the merger, which is the winner's.
export function vote(svm, knn) {
  if (!knn.label) return { label: svm.label, confidence: svm.confidence, source: "svm" };
  if (knn.label === svm.label) {
    return { label: svm.label,
             confidence: Math.max(svm.confidence, knn.confidence), source: "both" };
  }
  return knn.confidence > svm.confidence
    ? { label: knn.label, confidence: knn.confidence, source: "knn" }
    : { label: svm.label, confidence: svm.confidence, source: "svm" };
}

export function loadCalibration() {
  const store = GestureStore.load(STORE_KEY);
  let result = null;
  try { result = JSON.parse(localStorage.getItem(RESULT_KEY) ?? "null"); } catch { /* none */ }
  return { store, knn: new KnnClassifier(store), result };
}

export function saveCalibration(samplesByLetter, result) {
  const store = new GestureStore({}, STORE_KEY);
  for (const [letter, samples] of Object.entries(samplesByLetter)) {
    store.gestures[letter] = { samples, phrase: "", createdAt: new Date().toISOString() };
  }
  store.save();
  localStorage.setItem(RESULT_KEY, JSON.stringify(result));
  return { store, knn: new KnnClassifier(store), result };
}

export function clearCalibration() {
  localStorage.removeItem(STORE_KEY);
  localStorage.removeItem(RESULT_KEY);
  const store = new GestureStore({}, STORE_KEY);
  return { store, knn: new KnnClassifier(store), result: null };
}

// Before/after accuracy on the samples held back from the KNN. Same session as
// the calibration samples, so the "after" figure is an upper bound - the UI
// says so rather than presenting it as a promise.
export async function evaluate(svm, knn, heldOut) {
  let before = 0;
  let after = 0;
  let total = 0;
  for (const [letter, samples] of Object.entries(heldOut)) {
    for (const sample of samples) {
      const s = await svm.predict(sample);
      const k = knn.predict(sample);
      if (s.label === letter) before += 1;
      if (vote(s, k).label === letter) after += 1;
      total += 1;
    }
  }
  return { before: total ? before / total : 0,
           after: total ? after / total : 0, total };
}

// Guided capture: for each letter, a countdown to get the shape ready, then
// CALIBRATION_SAMPLES + CALIBRATION_HELD_OUT samples spaced CALIBRATION_STRIDE
// frames apart. Time is injected, as in the merger, so it can be tested.
export class CalibrationSession {
  constructor(letters) {
    this.letters = letters;
    this.index = 0;
    this.calibration = {};
    this.heldOut = {};
    this.skipped = [];
    this.startCountdown(null);
  }

  get letter() { return this.letters[this.index]; }
  get done() { return this.index >= this.letters.length; }
  get perLetter() { return CALIBRATION_SAMPLES + CALIBRATION_HELD_OUT; }

  startCountdown(now) {
    this.phase = "countdown";
    this.phaseStarted = now;
    this.samples = [];
    this.sinceLast = 0;
  }

  secondsLeft(now) {
    if (this.phase !== "countdown" || this.phaseStarted === null) return CALIBRATION_COUNTDOWN;
    return Math.max(0, CALIBRATION_COUNTDOWN - (now - this.phaseStarted) / 1000);
  }

  // Called once per processed frame with the buffer's median (or null).
  // Returns true when this call finished a letter.
  update(aggregated, now) {
    if (this.done) return false;
    if (this.phase === "countdown") {
      if (this.phaseStarted === null) this.phaseStarted = now;
      if (this.secondsLeft(now) > 0) return false;
      this.phase = "capture";
    }
    if (!aggregated) return false;

    // Consecutive windows share 29 of 30 frames; spacing them out is what makes
    // the samples differ at all.
    this.sinceLast += 1;
    if (this.samples.length && this.sinceLast < CALIBRATION_STRIDE) return false;
    this.sinceLast = 0;
    this.samples.push(Float32Array.from(aggregated));
    if (this.samples.length < this.perLetter) return false;

    this.calibration[this.letter] = this.samples.slice(0, CALIBRATION_SAMPLES);
    this.heldOut[this.letter] = this.samples.slice(CALIBRATION_SAMPLES);
    this.advance(now);
    return true;
  }

  skip(now) {
    if (this.done) return;
    this.skipped.push(this.letter);
    this.advance(now);
  }

  advance(now) {
    this.index += 1;
    this.startCountdown(now);
  }

  get progress() {
    const within = this.phase === "capture" ? this.samples.length / this.perLetter : 0;
    return Math.min(1, (this.index + within) / this.letters.length);
  }
}
