// Mirror of python/src/config.py. These numbers are a contract between the two
// builds - change one side and predictions silently degrade rather than fail.
export const LANDMARKS_PER_HAND = 21;
export const COORDS = 3;
export const MAX_HANDS = 2;
export const HAND_BLOCK = LANDMARKS_PER_HAND * COORDS;   // 63
export const FEATURE_DIM = HAND_BLOCK * MAX_HANDS;       // 126

// Slot 0 is the user's real LEFT hand, slot 1 their real right.
export const HAND_SLOTS = ["Left", "Right"];

export const SEQUENCE_LENGTH = 30;

export const DETECTION_CONFIDENCE = 0.7;
export const PRESENCE_CONFIDENCE = 0.7;
export const TRACKING_CONFIDENCE = 0.7;

// The camera image is mirrored before detection so the preview reads as a
// mirror, and MediaPipe reports handedness for the UN-mirrored view - so its
// label is the opposite of the user's real hand and must be flipped back.
// Both must match python/src/{camera,landmark_extractor}.py exactly.
export const MIRROR_INPUT = true;
export const FLIP_HANDEDNESS = true;

// Feature representation. Must equal what the model was trained with; the
// browser checks this against model_meta.json at startup.
export const FEATURE_WRIST_RELATIVE = true;
export const FEATURE_SCALE_NORMALISE = true;

// Letter merger.
export const TAU = 0.35;
export const MIN_FRAMES = 12;
export const MIN_HOLD_TIME = 0.4;      // seconds
export const IDLE_FRAMES_TO_FORGET = 15;

export const CUSTOM_GESTURE_SAMPLES = 20;
export const KNN_K = 3;

export const MODEL_BASE = "models";
