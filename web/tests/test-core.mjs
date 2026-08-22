// Core logic tests for the browser build, runnable under plain node:
//
//     cd web && node tests/test-core.mjs
//
// These mirror python/tests/test_merger.py and test_buffer.py case for case.
// The two implementations have to agree, so the tests should too - if a case
// exists on one side only, one of them is untested.
import { RollingBuffer } from "../src/buffer.js";
import { LetterMerger, NO_STAGES } from "../src/merger.js";
import { toScaleNormalised, toWristRelative, transform } from "../src/features.js";
import { FEATURE_DIM } from "../src/config.js";

let failures = 0;

function check(name, got, want) {
  const g = JSON.stringify(got);
  const w = JSON.stringify(want);
  const ok = g === w;
  if (!ok) failures += 1;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(52)} got ${g}`);
}

// ---------------------------------------------------------------- features
const oneHand = new Float32Array(FEATURE_DIM);
for (let i = 63; i < FEATURE_DIM; i++) oneHand[i] = 0.2 + (i % 7) * 0.05;

const t = transform(oneHand);
check("empty slot stays exactly zero",
      Array.from(t.slice(0, 63)).every((x) => x === 0), true);
check("active wrist lands on the origin",
      [t[63], t[64], t[65]], [0, 0, 0]);

let maxRadius = 0;
for (let k = 0; k < 21; k++) {
  const i = 63 + k * 3;
  maxRadius = Math.max(maxRadius, Math.hypot(t[i], t[i + 1], t[i + 2]));
}
check("scale normalisation puts max radius at 1",
      Math.abs(maxRadius - 1) < 1e-6, true);

check("transform does not mutate its input", oneHand[63] !== 0, true);
check("wrong length is rejected", (() => {
  try { transform(new Float32Array(10)); return false; } catch { return true; }
})(), true);

const twoHand = Float32Array.from(oneHand);
for (let i = 0; i < 63; i++) twoHand[i] = 0.5 + (i % 5) * 0.03;
const t2 = transform(twoHand);
check("both hands normalised independently",
      [t2[0], t2[1], t2[2], t2[63], t2[64], t2[65]], [0, 0, 0, 0, 0, 0]);

// ------------------------------------------------------------------ buffer
const buf = new RollingBuffer(30);
check("empty buffer is not ready", buf.ready, false);
check("empty buffer values null", buf.value(), null);

for (let i = 0; i < 29; i++) buf.push(oneHand);
check("29 frames still not ready", buf.ready, false);
buf.push(oneHand);
check("30 frames ready", buf.ready, true);
check("one-handed window reports 1 hand", buf.hands, 1);

buf.push(twoHand);
check("hand-count change resets the window", buf.ready, false);
check("window now tracks 2 hands", buf.hands, 2);
for (let i = 0; i < 29; i++) buf.push(twoHand);
check("refills after the change", buf.ready, true);
const merged = buf.value();
check("no blending - both slots present",
      merged[0] !== 0 && merged[63] !== 0, true);

const outlierBuf = new RollingBuffer(11);
const base = new Float32Array(FEATURE_DIM); base.fill(0.5);
const spike = new Float32Array(FEATURE_DIM); spike.fill(99);
for (let i = 0; i < 10; i++) outlierBuf.push(base);
outlierBuf.push(spike);
check("single outlier does not move the median",
      Math.abs(outlierBuf.value()[0] - 0.5) < 1e-6, true);

// ------------------------------------------------------------------ merger
const FPS = 30;
function feed(m, label, conf, frames, t0) {
  const out = [];
  for (let i = 0; i < frames; i++) {
    const c = m.update(label, conf, t0 + i / FPS);
    if (c) out.push(c);
  }
  return [out, t0 + frames / FPS];
}

let m = new LetterMerger();
check("hold 60 frames commits once", feed(m, "A", 0.9, 60, 0)[0], ["A"]);

m = new LetterMerger();
check("below tau never commits", feed(m, "A", 0.2, 60, 0)[0], []);

m = new LetterMerger();
check("10 frames (<12) does not commit", feed(m, "A", 0.9, 10, 0)[0], []);

m = new LetterMerger();
const fast = [];
for (let i = 0; i < 20; i++) {
  const c = m.update("A", 0.9, i * 0.001);
  if (c) fast.push(c);
}
check("20 frames in 0.02s does not commit (hold-time)", fast, []);

m = new LetterMerger();
let time = 0; let spelled = [];
for (const ch of "CAB") {
  const [out, next] = feed(m, ch, 0.9, 20, time);
  spelled = spelled.concat(out); time = next;
}
check("spell CAB", spelled, ["C", "A", "B"]);

m = new LetterMerger();
time = 0; spelled = [];
for (const step of [["L", 0.9], ["L", 0.9]]) {
  const [out, next] = feed(m, step[0], step[1], 20, time);
  spelled = spelled.concat(out); time = next;
}
check("LL with no gap -> single L", spelled, ["L"]);

m = new LetterMerger();
time = 0; spelled = [];
let r = feed(m, "L", 0.9, 20, time); spelled = spelled.concat(r[0]); time = r[1];
r = feed(m, null, 0, 20, time); spelled = spelled.concat(r[0]); time = r[1];
r = feed(m, "L", 0.9, 20, time); spelled = spelled.concat(r[0]);
check("LL with a gap -> LL", spelled, ["L", "L"]);

m = new LetterMerger();
time = 0; spelled = [];
r = feed(m, "A", 0.9, 8, time); spelled = spelled.concat(r[0]); time = r[1];
r = feed(m, "X", 0.9, 3, time); spelled = spelled.concat(r[0]); time = r[1];
r = feed(m, "A", 0.9, 20, time); spelled = spelled.concat(r[0]);
check("3-frame flicker of X is rejected", spelled, ["A"]);

m = new LetterMerger({ stages: { ...NO_STAGES } });
check("no-stages commits every frame", feed(m, "A", 0.9, 5, 0)[0],
      ["A", "A", "A", "A", "A"]);

m = new LetterMerger();
time = 0;
for (const ch of "AB") { time = feed(m, ch, 0.9, 20, time)[1]; }
m.undo();
check("undo removes the last letter", m.text, "A");

m = new LetterMerger();
let previous = 0; let monotone = true;
for (let i = 0; i < 20; i++) {
  m.update("A", 0.9, i / FPS);
  const p = m.state.progress;
  if (p < previous - 1e-9 || p > 1) monotone = false;
  previous = p;
}
check("progress monotone and <= 1", monotone, true);

console.log();
console.log(failures ? `${failures} FAILURE(S)` : "ALL PASS");
process.exit(failures ? 1 : 0);
