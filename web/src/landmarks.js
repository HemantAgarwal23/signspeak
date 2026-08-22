// Port of python/src/landmark_extractor.py + camera.py for the browser.
//
// VECTOR LAYOUT - contractual, and identical to the Python build:
//   index   0..62   hand slot 0 = the user's real LEFT hand,  21 x (x, y, z)
//   index  63..125  hand slot 1 = the user's real RIGHT hand, 21 x (x, y, z)
//
// MIRRORING, which is the single easiest thing to get wrong here:
// getUserMedia gives un-mirrored pixels, while the Python build mirrors every
// frame at capture. So the frame is drawn flipped into a canvas and the
// detector runs on *that*, matching Python exactly. MediaPipe then reports
// handedness for the un-mirrored view, so its label is the opposite of the
// user's real hand and gets flipped back - again, exactly as Python does.
//
// Get either of those wrong and slot 0 and slot 1 swap: the model still runs,
// still reports confident predictions, and is entirely wrong.
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

import {
  DETECTION_CONFIDENCE, FEATURE_DIM, FLIP_HANDEDNESS, HAND_BLOCK, HAND_SLOTS,
  LANDMARKS_PER_HAND, MAX_HANDS, MIRROR_INPUT, MODEL_BASE,
  PRESENCE_CONFIDENCE, TRACKING_CONFIDENCE,
} from "./config.js";

// (start, end) pairs for the skeleton overlay, MediaPipe's standard topology.
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export class LandmarkExtractor {
  constructor({ maxHands = MAX_HANDS, flipHandedness = FLIP_HANDEDNESS } = {}) {
    this.maxHands = maxHands;
    this.flipHandedness = flipHandedness;
    this.landmarker = null;
    this.result = null;
    this.lastTimestamp = -1;
  }

  async init() {
    const fileset = await FilesetResolver.forVisionTasks("wasm");
    this.landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: `${MODEL_BASE}/hand_landmarker.task`,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: this.maxHands,
      minHandDetectionConfidence: DETECTION_CONFIDENCE,
      minHandPresenceConfidence: PRESENCE_CONFIDENCE,
      minTrackingConfidence: TRACKING_CONFIDENCE,
    });
    return this;
  }

  // `source` must already be the mirrored canvas - see drawMirrored below.
  detect(source, timestampMs) {
    if (!this.landmarker) return null;
    // VIDEO mode requires strictly increasing timestamps, and rAF can deliver
    // two frames inside the same millisecond.
    let ts = Math.round(timestampMs);
    if (ts <= this.lastTimestamp) ts = this.lastTimestamp + 1;
    this.lastTimestamp = ts;

    this.result = this.landmarker.detectForVideo(source, ts);
    if (!this.result?.landmarks?.length) return null;
    return this.toVector(this.result);
  }

  userHand(rawLabel) {
    if (!this.flipHandedness) return rawLabel;
    return rawLabel === "Left" ? "Right" : "Left";
  }

  toVector(result) {
    const vector = new Float32Array(FEATURE_DIM);
    const filled = [false, false];
    const handedness = result.handedness ?? [];

    for (let i = 0; i < result.landmarks.length; i++) {
      const raw = handedness[i]?.[0]?.categoryName ?? HAND_SLOTS[0];
      const label = this.userHand(raw);
      let slot = HAND_SLOTS.indexOf(label);
      if (slot < 0) slot = 0;

      // Two hands classified the same way (overlapping hands): the second one
      // takes whichever slot is still free.
      if (filled[slot]) {
        const other = 1 - slot;
        if (other >= MAX_HANDS || filled[other]) continue;
        slot = other;
      }

      const start = slot * HAND_BLOCK;
      const points = result.landmarks[i];
      for (let k = 0; k < LANDMARKS_PER_HAND; k++) {
        const p = points[k];
        const at = start + k * 3;
        vector[at] = p.x;
        vector[at + 1] = p.y;
        vector[at + 2] = p.z;
      }
      filled[slot] = true;
    }
    return vector;
  }

  get handsDetected() { return this.result?.landmarks?.length ?? 0; }

  get handednessLabels() {
    return (this.result?.handedness ?? [])
      .map((h) => this.userHand(h?.[0]?.categoryName ?? ""))
      .filter(Boolean);
  }

  drawOverlay(ctx, width, height) {
    const landmarks = this.result?.landmarks;
    if (!landmarks?.length) return;

    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(110, 231, 143, 0.9)";
    ctx.fillStyle = "rgba(250, 204, 21, 0.95)";

    for (const points of landmarks) {
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        if (!points[a] || !points[b]) continue;
        ctx.moveTo(points[a].x * width, points[a].y * height);
        ctx.lineTo(points[b].x * width, points[b].y * height);
      }
      ctx.stroke();
      for (const p of points) {
        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  close() {
    this.landmarker?.close();
    this.landmarker = null;
  }
}

// One place owns the mirroring decision, as camera.py does in the Python build.
export function drawMirrored(ctx, video, width, height) {
  ctx.save();
  if (MIRROR_INPUT) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, width, height);
  ctx.restore();
}

export async function openCamera(video, { width = 640, height = 480 } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser has no camera API. Chrome, Edge, Firefox "
                    + "and Safari all support it over https or localhost.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: width }, height: { ideal: height },
             facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}
