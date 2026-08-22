// Port of python/src/merger.py. All four stages, same thresholds.
//
//   1 confidence filtering   drop predictions below tau
//   2 buffer accumulation    count consecutive frames agreeing on a label
//   3 hold-time validation   frames AND elapsed AND mean confidence, together
//   4 duplicate suppression  never commit the same label twice running
import {
  IDLE_FRAMES_TO_FORGET, MIN_FRAMES, MIN_HOLD_TIME, TAU,
} from "./config.js";

export const ALL_STAGES = {
  confidenceFilter: true,
  accumulation: true,
  holdTime: true,
  duplicateSuppression: true,
};

export const NO_STAGES = {
  confidenceFilter: false,
  accumulation: false,
  holdTime: false,
  duplicateSuppression: false,
};

export class LetterMerger {
  constructor({
    tau = TAU,
    minFrames = MIN_FRAMES,
    minHoldTime = MIN_HOLD_TIME,
    stages = null,
    idleFramesToForget = IDLE_FRAMES_TO_FORGET,
  } = {}) {
    this.tau = tau;
    this.minFrames = minFrames;
    this.minHoldTime = minHoldTime;
    this.stages = stages ?? { ...ALL_STAGES };
    // Without this, stage 4 makes double letters unspellable: HELLO comes out
    // HELO. A gap clears the guard, so dropping the hand between the two Ls
    // works - which is the natural gesture anyway.
    this.idleFramesToForget = idleFramesToForget;
    this.reset();
  }

  // `now` is injected so recorded sequences can be replayed deterministically,
  // exactly as in the Python version.
  update(label, confidence, now = performance.now() / 1000) {
    // Stage 1 - confidence filtering.
    if (label == null
        || (this.stages.confidenceFilter && confidence < this.tau)) {
      this.onIdle();
      return null;
    }

    if (!this.stages.accumulation) return this.commit(label);
    this.idleFrames = 0;

    // Stage 2 - buffer accumulation.
    if (!this.candidate || this.candidate.label !== label) {
      this.candidate = {
        label, frames: 0, sum: 0, startedAt: now, committed: false,
      };
    }
    const c = this.candidate;
    c.frames += 1;
    c.sum += confidence;

    // Stage 3 - hold-time validation. All three conditions, simultaneously.
    if (c.committed) return null;
    if (this.stages.holdTime
        && !(c.frames >= this.minFrames
             && now - c.startedAt >= this.minHoldTime
             && c.sum / c.frames >= this.tau)) {
      return null;
    }
    c.committed = true;
    return this.commit(label);
  }

  commit(label) {
    // Stage 4 - duplicate suppression.
    if (this.stages.duplicateSuppression && label === this.lastCommitted) {
      return null;
    }
    this.committed.push(label);
    this.lastCommitted = label;
    return label;
  }

  onIdle() {
    this.idleFrames += 1;
    this.candidate = null;
    if (this.idleFrames >= this.idleFramesToForget) this.lastCommitted = null;
  }

  get state() {
    const c = this.candidate;
    if (!c) {
      return {
        label: null, frames: 0, progress: 0, meanConfidence: 0, committed: false,
      };
    }
    const now = performance.now() / 1000;
    const byFrames = this.minFrames ? c.frames / this.minFrames : 1;
    const byTime = this.minHoldTime ? (now - c.startedAt) / this.minHoldTime : 1;
    return {
      label: c.label,
      frames: c.frames,
      // The slower of the two: showing the faster would leave the bar sitting
      // full while nothing commits.
      progress: Math.min(1, Math.min(byFrames, byTime)),
      meanConfidence: c.frames ? c.sum / c.frames : 0,
      committed: c.committed,
    };
  }

  get text() { return this.committed.join(""); }

  undo() {
    const removed = this.committed.pop() ?? null;
    this.lastCommitted = this.committed.at(-1) ?? null;
    return removed;
  }

  reset() {
    this.committed = [];
    this.candidate = null;
    this.lastCommitted = null;
    this.idleFrames = 0;
  }
}
