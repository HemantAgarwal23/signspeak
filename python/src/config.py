"""Single source of truth for every tunable in the project.

Both the Python pipeline and the exported web assets derive their numbers from
here. If a constant changes, it changes here and nowhere else.
"""
from __future__ import annotations

from pathlib import Path

# --- Feature geometry -------------------------------------------------------
LANDMARKS_PER_HAND = 21
COORDS = 3                       # x, y, z
MAX_HANDS = 2
HAND_BLOCK = LANDMARKS_PER_HAND * COORDS      # 63 numbers per hand
FEATURE_DIM = HAND_BLOCK * MAX_HANDS          # 126

# Slot order inside the 126-D vector, in terms of the USER'S REAL HANDS.
# Slot 0 = real left hand (indices 0..62), slot 1 = real right hand (63..125).
HAND_SLOTS = ("Left", "Right")

# MediaPipe reports handedness for the UN-mirrored camera view. We mirror the
# frame before detection (MIRROR_INPUT), so its label is the opposite of the
# user's real hand and must be flipped.
# Verified empirically 2026-08-22: raising the real left hand reported "Right".
# If a future MediaPipe release changes this, flip this flag - do not change
# HAND_SLOTS, or every previously recorded sample becomes mislabelled.
FLIP_HANDEDNESS = True

# --- Sequence / aggregation -------------------------------------------------
SEQUENCE_LENGTH = 30             # frames per recorded sample (~1s at 30 FPS)
AGGREGATION = "median"           # per-coordinate median over the sequence

# --- MediaPipe --------------------------------------------------------------
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
PRESENCE_CONFIDENCE = 0.7        # Tasks API only: hand-presence score per frame
MIRROR_INPUT = True              # flip horizontally once, at capture time

# MediaPipe Tasks bundle. The browser build loads the byte-identical file from
# web/public/models/, so both sides run the same graph.
HAND_TASK_FILENAME = "hand_landmarker.task"
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Feature representation, applied in features.py at TRAIN/INFERENCE time - not
# at record time. data/raw/ always stores raw absolute coordinates, so changing
# this needs a retrain, never a re-record, and both settings can be compared on
# the identical dataset.
#
# WRIST_RELATIVE subtracts each hand's wrist, discarding where the hand sits in
# the frame and keeping only its shape. Measured on 5 real samples of one
# handshape (2026-08-22): within-class spread fell 0.334 -> 0.136, a 2.5x cut.
#
# SCALE_NORMALISE then divides each hand by its own size (max distance from the
# wrist), discarding how large the hand appears.
#
# Both default to True because training data now comes from cropped 200x200
# dataset images while inference runs on a wide webcam frame. Without these the
# two live in completely different coordinate ranges and nothing transfers. The
# build spec's "absolute (x, y, z)" is superseded by that requirement; M8 keeps
# the comparison as an ablation row. Scale normalisation implies wrist-relative
# and is applied after it.
#
# Whatever is set here must be mirrored in web/src/buffer.js.
FEATURE_WRIST_RELATIVE = True
FEATURE_SCALE_NORMALISE = True

# --- Classifier -------------------------------------------------------------
SVM_KERNEL = "rbf"
SVM_C = 10
SVM_GAMMA = "scale"
SVM_PROBABILITY = True
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Letter merger state machine -------------------------------------------
TAU = 0.35                       # confidence floor
MIN_FRAMES = 12                  # frames a label must persist
MIN_HOLD_TIME = 0.4              # seconds a label must persist

# --- Custom gestures / calibration -----------------------------------------
CUSTOM_GESTURE_SAMPLES = 20
CALIBRATION_SAMPLES = 5
KNN_K = 3

# --- Dataset ----------------------------------------------------------------
# 24 static ASL letters. J and Z are excluded: both require motion.
ASL_LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]
SAMPLES_PER_LETTER = 60

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
DATA_DIR = PYTHON_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PACKS_DIR = DATA_DIR / "packs"          # bulk imported samples, one file per class
# Third-party datasets live OUTSIDE the project. The repo sits in a OneDrive
# folder, and 87k unpacked image files would trigger a full sync of every one
# of them. Only the derived packs (a few dozen small files) stay in data/.
DOWNLOADS_DIR = Path.home() / "gesture-data" / "downloads"
MODELS_DIR = DATA_DIR / "models"
MANIFEST_PATH = DATA_DIR / "manifest.json"
DOCS_RESULTS = PROJECT_ROOT / "docs" / "results"
WEB_MODELS_DIR = PROJECT_ROOT / "web" / "public" / "models"

HAND_TASK_PATH = MODELS_DIR / HAND_TASK_FILENAME
MODEL_PATH = MODELS_DIR / "svm.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
LABELS_PATH = MODELS_DIR / "labels.json"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for path in (DATA_DIR, RAW_DIR, PACKS_DIR, MODELS_DIR, DOCS_RESULTS,
                 WEB_MODELS_DIR, DOWNLOADS_DIR):
        path.mkdir(parents=True, exist_ok=True)
