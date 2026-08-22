// Port of python/src/features.py. Applied at inference time; the vector coming
// out of landmarks.js is always raw absolute coordinates.
import {
  COORDS, FEATURE_DIM, FEATURE_SCALE_NORMALISE, FEATURE_WRIST_RELATIVE,
  HAND_BLOCK, LANDMARKS_PER_HAND, MAX_HANDS,
} from "./config.js";

const EPS = 1e-8;

// An absent hand is 63 zeros and must stay that way. Without this check it
// would pick up a negative wrist offset and start looking like real data.
function handPresent(v, start) {
  for (let i = start; i < start + HAND_BLOCK; i++) {
    if (v[i] !== 0) return true;
  }
  return false;
}

export function toWristRelative(vector) {
  const out = Float32Array.from(vector);
  for (let hand = 0; hand < MAX_HANDS; hand++) {
    const start = hand * HAND_BLOCK;
    if (!handPresent(out, start)) continue;
    const wx = out[start];
    const wy = out[start + 1];
    const wz = out[start + 2];
    for (let k = 0; k < LANDMARKS_PER_HAND; k++) {
      const i = start + k * COORDS;
      out[i] -= wx;
      out[i + 1] -= wy;
      out[i + 2] -= wz;
    }
  }
  return out;
}

export function toScaleNormalised(vector) {
  const out = Float32Array.from(vector);
  for (let hand = 0; hand < MAX_HANDS; hand++) {
    const start = hand * HAND_BLOCK;
    if (!handPresent(out, start)) continue;
    let size = 0;
    for (let k = 0; k < LANDMARKS_PER_HAND; k++) {
      const i = start + k * COORDS;
      const d = Math.hypot(out[i], out[i + 1], out[i + 2]);
      if (d > size) size = d;
    }
    const divisor = Math.max(size, EPS);
    for (let i = start; i < start + HAND_BLOCK; i++) out[i] /= divisor;
  }
  return out;
}

export function transform(vector, {
  wristRelative = FEATURE_WRIST_RELATIVE,
  scaleNormalise = FEATURE_SCALE_NORMALISE,
} = {}) {
  if (vector.length !== FEATURE_DIM) {
    throw new Error(`expected ${FEATURE_DIM} values, got ${vector.length}`);
  }
  let out = vector;
  // Scale normalisation is meaningless on absolute coordinates, so it implies
  // wrist-relative - same rule as features.py.
  const wr = scaleNormalise || wristRelative;
  if (wr) out = toWristRelative(out);
  if (scaleNormalise) out = toScaleNormalised(out);
  return out;
}
