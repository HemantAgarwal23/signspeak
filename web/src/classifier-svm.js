// Port of python/src/predictor.py. Runs the exported ONNX graph.
//
// The StandardScaler is folded into the graph at export time, so this file
// applies the feature transform and nothing else. Two implementations of
// standardisation would be two places to get it subtly wrong, and that kind of
// mismatch does not raise - it just makes the model quietly worse.
// The "wasm" entry, not the default one. The default pulls in the WebGPU/JSEP
// build - a 26 MB binary we never execute, since this model is a small SVM that
// runs faster on CPU than it would cost to upload to the GPU. The wasm-only
// build is 13 MB and gzips to about a quarter of that.
import * as ort from "onnxruntime-web/wasm";

import { FEATURE_DIM, FEATURE_SCALE_NORMALISE, FEATURE_WRIST_RELATIVE,
         MODEL_BASE } from "./config.js";
import { transform } from "./features.js";

// Serve the runtime from public/ort/ rather than letting the bundler place it.
// Without this ORT requests its .wasm from the site root, the dev server's SPA
// fallback answers with index.html, and instantiation fails with
// "expected magic word 00 61 73 6d, found 3c 21 64 6f" - which is "<!do".
ort.env.wasm.wasmPaths = "ort/";

// Single-threaded on purpose. Threaded wasm needs SharedArrayBuffer, which
// browsers only expose to cross-origin-isolated pages (COOP + COEP headers).
// Requiring those to run a small SVM would break static hosting for no gain.
ort.env.wasm.numThreads = 1;

export class SvmClassifier {
  constructor() {
    this.session = null;
    this.labels = [];
    this.meta = {};
    this.inputName = null;
  }

  async init() {
    const [labels, meta] = await Promise.all([
      fetch(`${MODEL_BASE}/labels.json`).then((r) => r.json()),
      fetch(`${MODEL_BASE}/model_meta.json`).then((r) => r.json())
        .catch(() => ({})),
    ]);
    this.labels = labels;
    this.meta = meta;
    this.checkFeatures();

    this.session = await ort.InferenceSession.create(
      `${MODEL_BASE}/asl_svm.onnx`, { executionProviders: ["wasm"] },
    );
    this.inputName = this.session.inputNames[0];
    return this;
  }

  // Refuse a model trained under a different representation. Silent mismatch
  // here is the nastiest failure in the project: it loads, predicts, and is
  // confidently wrong.
  checkFeatures() {
    const wr = this.meta.feature_wrist_relative;
    const sn = this.meta.feature_scale_normalise;
    if (wr === undefined && sn === undefined) return;
    if (wr !== FEATURE_WRIST_RELATIVE || sn !== FEATURE_SCALE_NORMALISE) {
      throw new Error(
        `feature representation mismatch: model was trained with `
        + `wrist_relative=${wr} scale_normalise=${sn}, but config.js says `
        + `${FEATURE_WRIST_RELATIVE} / ${FEATURE_SCALE_NORMALISE}. `
        + `Re-export the model or fix config.js.`,
      );
    }
  }

  async predict(rawVector) {
    if (!this.session) return { label: null, confidence: 0, top: [] };

    const featureVector = transform(rawVector);
    const tensor = new ort.Tensor("float32", featureVector, [1, FEATURE_DIM]);
    const output = await this.session.run({ [this.inputName]: tensor });

    const names = this.session.outputNames;
    const labelData = output[names[0]]?.data;
    const probData = names[1] ? output[names[1]]?.data : null;

    if (!probData) {
      // No probability output: fall back to the label alone. The merger's tau
      // then has nothing meaningful to compare against, so say so loudly.
      const label = String(labelData?.[0] ?? "");
      return { label, confidence: 1, top: [[label, 1]], calibrated: false };
    }

    const scores = Array.from(probData);
    let best = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;

    const top = scores
      .map((score, i) => [this.labels[i] ?? String(i), score])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);

    return {
      label: this.labels[best] ?? String(labelData?.[0] ?? ""),
      confidence: scores[best],
      top,
      calibrated: true,
    };
  }

  describe() {
    const accuracy = this.meta.test_accuracy;
    return `${this.labels.length} letters`
      + (accuracy ? `, reported test accuracy ${(accuracy * 100).toFixed(1)}%` : "");
  }
}
