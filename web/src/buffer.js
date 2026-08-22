// Port of RollingBuffer from python/src/aggregator.py.
import { FEATURE_DIM, SEQUENCE_LENGTH } from "./config.js";

function handMask(vector) {
  const half = FEATURE_DIM / 2;
  let left = false;
  let right = false;
  for (let i = 0; i < half; i++) {
    if (vector[i] !== 0) { left = true; break; }
  }
  for (let i = half; i < FEATURE_DIM; i++) {
    if (vector[i] !== 0) { right = true; break; }
  }
  return (left ? 1 : 0) | (right ? 2 : 0);
}

export class RollingBuffer {
  constructor(length = SEQUENCE_LENGTH) {
    this.length = length;
    this.frames = [];
    this.mask = null;
  }

  push(vector) {
    if (!vector) return;
    // A window mixing one- and two-handed frames medians into a vector that
    // never occurred: once over half the frames lack a hand, that hand's 63
    // coordinates collapse to zero and a two-handed gesture silently becomes a
    // one-handed one. Changing hand count starts a fresh window instead. This
    // cost real debugging time in the Python build - do not remove it here.
    const mask = handMask(vector);
    if (mask !== this.mask) {
      this.frames.length = 0;
      this.mask = mask;
    }
    this.frames.push(vector);
    if (this.frames.length > this.length) this.frames.shift();
  }

  get ready() { return this.frames.length === this.length; }

  get fill() { return this.frames.length / this.length; }

  get hands() {
    if (this.mask === null) return 0;
    return (this.mask & 1 ? 1 : 0) + (this.mask & 2 ? 1 : 0);
  }

  // Per-coordinate median. Median rather than mean because MediaPipe
  // occasionally emits one badly wrong frame; an outlier drags a mean but
  // barely moves a median.
  value() {
    if (!this.ready) return null;
    const n = this.frames.length;
    const out = new Float32Array(FEATURE_DIM);
    const column = new Array(n);
    const mid = n >> 1;
    const even = n % 2 === 0;
    for (let d = 0; d < FEATURE_DIM; d++) {
      for (let f = 0; f < n; f++) column[f] = this.frames[f][d];
      column.sort((a, b) => a - b);
      out[d] = even ? (column[mid - 1] + column[mid]) / 2 : column[mid];
    }
    return out;
  }

  clear() { this.frames.length = 0; this.mask = null; }
}
