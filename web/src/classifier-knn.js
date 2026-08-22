// Port of python/src/custom_gestures.py.
//
// Storage is the same JSON shape the desktop build writes, so a vocabulary
// exported there imports here unchanged - that was the point of choosing JSON
// over a pickle on the Python side.
import { FEATURE_DIM, KNN_K } from "./config.js";
import { transform } from "./features.js";

const STORAGE_KEY = "signspeak.gestures.v1";
const SCHEMA_VERSION = 1;

export const REJECT_SPREAD_MULTIPLE = 2.5;
export const REJECT_FLOOR = 0.35;
// Sample spread alone sets the threshold far too tight: the 20 samples come
// from overlapping windows, so they describe one frozen moment rather than how
// a gesture varies. The threshold is therefore also scaled against the distance
// to the nearest OTHER gesture. Measured on real data in the Python build:
// spread within a gesture 0.04-0.18, distance between gestures 2.8-4.2.
export const RELATIVE_MARGIN = 0.35;
export const LONE_GESTURE_THRESHOLD = 1.0;

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < FEATURE_DIM; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export class GestureStore {
  constructor(gestures = {}) { this.gestures = gestures; }

  static load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return new GestureStore();
      const payload = JSON.parse(raw);
      const gestures = {};
      for (const [name, entry] of Object.entries(payload.gestures ?? {})) {
        gestures[name] = {
          samples: entry.samples.map((s) => Float32Array.from(s)),
          phrase: entry.phrase ?? "",
          createdAt: entry.created_at ?? entry.createdAt ?? "",
        };
      }
      return new GestureStore(gestures);
    } catch {
      // A corrupt store must not stop the app loading; the letters still work.
      return new GestureStore();
    }
  }

  save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.toJSON()));
  }

  toJSON() {
    const gestures = {};
    for (const [name, g] of Object.entries(this.gestures)) {
      gestures[name] = {
        samples: g.samples.map((s) => Array.from(s)),
        phrase: g.phrase,
        created_at: g.createdAt,
      };
    }
    return { schema_version: SCHEMA_VERSION, feature_dim: FEATURE_DIM, gestures };
  }

  add(name, samples, phrase = "") {
    this.gestures[name] = {
      samples: samples.map((s) => Float32Array.from(s)),
      phrase,
      createdAt: new Date().toISOString(),
    };
    this.save();
    return this.gestures[name];
  }

  remove(name) {
    const existed = name in this.gestures;
    delete this.gestures[name];
    this.save();
    return existed;
  }

  setPhrase(name, phrase) {
    if (!(name in this.gestures)) return false;
    this.gestures[name].phrase = phrase;
    this.save();
    return true;
  }

  output(name) {
    const g = this.gestures[name];
    return g?.phrase || name;
  }

  get names() { return Object.keys(this.gestures).sort(); }
  get size() { return this.names.length; }

  importPayload(payload, { overwrite = false, prefix = "" } = {}) {
    if (payload.feature_dim && payload.feature_dim !== FEATURE_DIM) {
      throw new Error(`file uses feature_dim ${payload.feature_dim}, `
                      + `this build uses ${FEATURE_DIM}`);
    }
    const added = [];
    const skipped = [];
    for (const [name, entry] of Object.entries(payload.gestures ?? {})) {
      const target = `${prefix}${name}`;
      if (target in this.gestures && !overwrite) { skipped.push(target); continue; }
      const samples = (entry.samples ?? []).map((s) => Float32Array.from(s));
      if (!samples.length || samples[0].length !== FEATURE_DIM) {
        skipped.push(target);
        continue;
      }
      this.gestures[target] = {
        samples, phrase: entry.phrase ?? "", createdAt: entry.created_at ?? "",
      };
      added.push(target);
    }
    this.save();
    return { added, skipped };
  }
}

export class KnnClassifier {
  constructor(store, k = KNN_K) {
    this.k = k;
    this.fit(store);
  }

  // "Training" is transforming and storing the samples - which is the whole
  // point of the few-shot claim, and why this takes under a millisecond.
  fit(store) {
    const started = performance.now();
    this.samples = [];
    this.labels = [];
    for (const name of store.names) {
      for (const sample of store.gestures[name].samples) {
        this.samples.push(transform(sample));
        this.labels.push(name);
      }
    }
    this.thresholds = this.computeThresholds();
    this.hands = this.countHands();
    return performance.now() - started;
  }

  get ready() { return this.samples.length > 0; }

  centroid(name) {
    const rows = this.samples.filter((_, i) => this.labels[i] === name);
    const out = new Float32Array(FEATURE_DIM);
    for (const row of rows) for (let i = 0; i < FEATURE_DIM; i++) out[i] += row[i];
    for (let i = 0; i < FEATURE_DIM; i++) out[i] /= Math.max(rows.length, 1);
    return out;
  }

  computeThresholds() {
    const names = [...new Set(this.labels)];
    const centroids = Object.fromEntries(names.map((n) => [n, this.centroid(n)]));
    const thresholds = {};

    for (const name of names) {
      const rows = this.samples.filter((_, i) => this.labels[i] === name);
      let bySpread = REJECT_FLOOR;
      if (rows.length >= 2) {
        const mean = rows.reduce((acc, r) => acc + distance(r, centroids[name]), 0)
                     / rows.length;
        bySpread = Math.max(REJECT_FLOOR, mean * REJECT_SPREAD_MULTIPLE);
      }
      const others = names.filter((n) => n !== name)
        .map((n) => distance(centroids[name], centroids[n]));
      const byMargin = others.length
        ? Math.min(...others) * RELATIVE_MARGIN
        : LONE_GESTURE_THRESHOLD;
      thresholds[name] = Math.max(bySpread, byMargin);
    }
    return thresholds;
  }

  countHands() {
    const counts = {};
    for (const name of new Set(this.labels)) {
      const rows = this.samples.filter((_, i) => this.labels[i] === name);
      const half = FEATURE_DIM / 2;
      let total = 0;
      for (const row of rows) {
        let left = 0;
        let right = 0;
        for (let i = 0; i < half; i++) if (row[i] !== 0) { left = 1; break; }
        for (let i = half; i < FEATURE_DIM; i++) if (row[i] !== 0) { right = 1; break; }
        total += left + right;
      }
      counts[name] = Math.round(total / Math.max(rows.length, 1));
    }
    return counts;
  }

  thresholdFor(name) { return this.thresholds[name] ?? REJECT_FLOOR; }

  predict(rawVector) {
    if (!this.ready) return { label: null, confidence: 0 };
    const query = transform(rawVector);

    const distances = this.samples.map((s, i) => [distance(s, query), this.labels[i]]);
    distances.sort((a, b) => a[0] - b[0]);
    const nearest = distances[0];

    // A KNN always answers, so without a reject rule an idle hand keeps firing
    // whichever gesture happens to be closest. Judged against the nearest
    // gesture's own threshold, so a wide two-handed gesture is not held to a
    // tight one-handed gesture's standard.
    const limit = this.thresholdFor(nearest[1]);
    if (nearest[0] > limit) return { label: null, confidence: 0, nearest };

    const votes = {};
    for (const [d, label] of distances.slice(0, this.k)) {
      votes[label] = (votes[label] ?? 0) + 1 / (d + 1e-6);
    }
    const total = Object.values(votes).reduce((a, b) => a + b, 0);
    const [label, weight] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
    const share = total ? weight / total : 0;
    const closeness = 1 - Math.min(1, nearest[0] / this.thresholdFor(label));

    return { label, confidence: share * (0.5 + 0.5 * closeness), nearest };
  }

  nearestInfo(rawVector) {
    if (!this.ready) return null;
    const query = transform(rawVector);
    let best = null;
    for (let i = 0; i < this.samples.length; i++) {
      const d = distance(this.samples[i], query);
      if (!best || d < best.distance) best = { name: this.labels[i], distance: d };
    }
    if (best) best.threshold = this.thresholdFor(best.name);
    return best;
  }

  // Mean pairwise distance, not the single closest pair: one sloppy sample
  // should not decide whether two gestures are confusable.
  nearestGesture(samples, exclude = "") {
    if (!this.ready) return null;
    const queries = samples.map((s) => transform(s));
    let best = null;
    for (const name of new Set(this.labels)) {
      if (name === exclude) continue;
      const rows = this.samples.filter((_, i) => this.labels[i] === name);
      let sum = 0;
      for (const q of queries) for (const r of rows) sum += distance(q, r);
      const mean = sum / (queries.length * rows.length);
      if (!best || mean < best.distance) best = { name, distance: mean };
    }
    return best;
  }
}
