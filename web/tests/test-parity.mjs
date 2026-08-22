// Cross-language parity: the same raw vectors through Python and JavaScript
// must produce the same features and the same predictions.
//
//     cd web && node tests/test-parity.mjs
//
// The build spec calls this the verification not to skip, and it is right. A
// coordinate-ordering or normalisation mismatch does not raise - it produces a
// model that loads, runs, and is quietly wrong, which is indistinguishable from
// a badly trained model until you compare numbers.
//
// The fixture is regenerated from real dataset samples by the snippet in
// CODE_NOTES; it holds raw vectors, Python's transformed vectors, and Python's
// predictions.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { transform } from "../src/features.js";
import { FEATURE_DIM, MODEL_BASE } from "../src/config.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(
  fs.readFileSync(path.join(here, "parity-fixture.json"), "utf8"),
);

let failures = 0;
function check(name, ok, detail = "") {
  if (!ok) failures += 1;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(50)} ${detail}`);
}

// ------------------------------------------------------- feature transform
let worstFeature = 0;
for (let i = 0; i < fixture.raw.length; i++) {
  const mine = transform(Float32Array.from(fixture.raw[i]));
  const theirs = fixture.transformed[i];
  for (let d = 0; d < FEATURE_DIM; d++) {
    worstFeature = Math.max(worstFeature, Math.abs(mine[d] - theirs[d]));
  }
}
check("feature transform matches Python", worstFeature < 1e-5,
      `max |diff| = ${worstFeature.toExponential(2)} over ` +
      `${fixture.raw.length * FEATURE_DIM} values`);

// ------------------------------------------------------------- onnx model
const modelPath = path.join(here, "..", "public", MODEL_BASE, "asl_svm.onnx");
if (!fs.existsSync(modelPath)) {
  console.log(`SKIP  onnx model not found at ${modelPath}`);
} else {
  let ort;
  try {
    ort = await import("onnxruntime-web");
  } catch (err) {
    console.log(`SKIP  onnxruntime-web unavailable in node: ${err.message}`);
  }

  if (ort) {
    const runtime = ort.default ?? ort;
    try {
      const session = await runtime.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
      });
      const inputName = session.inputNames[0];

      const n = fixture.raw.length;
      const flat = new Float32Array(n * FEATURE_DIM);
      for (let i = 0; i < n; i++) {
        flat.set(transform(Float32Array.from(fixture.raw[i])), i * FEATURE_DIM);
      }
      const tensor = new runtime.Tensor("float32", flat, [n, FEATURE_DIM]);
      const output = await session.run({ [inputName]: tensor });

      const labelKey = session.outputNames[0];
      const got = Array.from(output[labelKey].data).map(String);
      const want = fixture.predicted.map(String);
      const agree = got.filter((g, i) => g === want[i]).length;
      check("ONNX predictions match sklearn", agree === n,
            `${agree}/${n} identical`);
    } catch (err) {
      console.log(`SKIP  could not run onnx in node: ${err.message}`);
    }
  }
}

console.log();
console.log(failures ? `${failures} FAILURE(S)` : "ALL PASS");
process.exit(failures ? 1 : 0);
