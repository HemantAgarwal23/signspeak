// Copies runtime assets out of node_modules into public/ so they are served
// as plain static files.
//
// Runs automatically after `npm install`.
//
// Why not let the bundler handle them: onnxruntime otherwise requests its .wasm
// from the site root, the dev server's SPA fallback answers with index.html,
// and instantiation dies with "expected magic word 00 61 73 6d, found
// 3c 21 64 6f" - the bytes of "<!do". Serving them ourselves and pointing
// ort.env.wasm.wasmPaths at the copy is the fix.
//
// They are gitignored because they are large and reproducible: 13 MB of
// onnxruntime and about 10 MB of MediaPipe. This script is what makes a fresh
// clone work.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(here, "..");
const modules = path.join(web, "node_modules");

const jobs = [
  {
    label: "MediaPipe tasks-vision wasm",
    from: path.join(modules, "@mediapipe", "tasks-vision", "wasm"),
    to: path.join(web, "public", "wasm"),
    all: true,
  },
  {
    label: "onnxruntime-web runtime",
    from: path.join(modules, "onnxruntime-web", "dist"),
    to: path.join(web, "public", "ort"),
    files: ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"],
  },
];

let missing = false;

for (const job of jobs) {
  if (!fs.existsSync(job.from)) {
    console.warn(`  skip  ${job.label}: ${job.from} not found`);
    missing = true;
    continue;
  }
  fs.mkdirSync(job.to, { recursive: true });

  const names = job.all ? fs.readdirSync(job.from) : job.files;
  let copied = 0;
  let bytes = 0;
  for (const name of names) {
    const source = path.join(job.from, name);
    if (!fs.existsSync(source) || fs.statSync(source).isDirectory()) continue;
    fs.copyFileSync(source, path.join(job.to, name));
    copied += 1;
    bytes += fs.statSync(source).size;
  }
  console.log(`  ok    ${job.label}: ${copied} file(s), `
              + `${(bytes / 1e6).toFixed(1)} MB -> ${path.relative(web, job.to)}`);
}

// The hand landmark model is gitignored (7.8 MB), so a fresh clone - and any
// host that builds from git, such as Vercel - has to fetch it. Same URL as
// python/src/config.py, and checked against the hash of the file the parity
// tests were run with: the two builds must run the byte-identical graph.
const HAND_TASK_URL = "https://storage.googleapis.com/mediapipe-models/"
  + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const HAND_TASK_SHA256 =
  "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1";

const modelDir = path.join(web, "public", "models");
const model = path.join(modelDir, "hand_landmarker.task");
if (!fs.existsSync(model)) {
  try {
    const response = await fetch(HAND_TASK_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = Buffer.from(await response.arrayBuffer());
    const digest = crypto.createHash("sha256").update(data).digest("hex");
    if (digest !== HAND_TASK_SHA256) {
      throw new Error(`checksum mismatch (got ${digest.slice(0, 12)}...)`);
    }
    fs.mkdirSync(modelDir, { recursive: true });
    fs.writeFileSync(model, data);
    console.log(`  ok    hand landmark model: downloaded `
                + `${(data.length / 1e6).toFixed(1)} MB -> public/models`);
  } catch (err) {
    missing = true;
    console.warn(`  fail  hand landmark model: ${err.message}`);
    console.warn("        Run, from the python/ directory:");
    console.warn("          python -m src.download_models");
  }
}

if (missing) {
  console.warn("\nSome assets were not copied. Run `npm install` first.");
}
