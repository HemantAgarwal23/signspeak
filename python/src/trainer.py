"""M4 - train the ASL letter classifier.

    python -m src.trainer                    # stratified random split
    python -m src.trainer --holdout-subject s01
    python -m src.trainer --fast             # skip probability calibration

A WORD ON THE VALIDATION NUMBER THIS PRINTS
-------------------------------------------
With a stratified random split over the imported Kaggle data, expect something
near-perfect - and do not believe it. That dataset is a single signer, and
consecutive frames are near-identical, so a random split puts near-duplicates of
the same moment on both sides of the divide. The model can score highly by
recognising the photo session, not the handshape.

The honest number comes from `--holdout-subject`: train on everyone else, test
on a signer the model has never seen. That is what M9 formalises, and it is the
figure worth reporting.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from . import features
    from .config import (FEATURE_SCALE_NORMALISE, FEATURE_WRIST_RELATIVE,
                         LABELS_PATH, MODEL_PATH, MODELS_DIR, RANDOM_STATE,
                         SCALER_PATH, SVM_C, SVM_GAMMA, SVM_KERNEL, TEST_SIZE,
                         ensure_dirs)
    from .dataset import load_features, load_manifest
except ImportError:  # running as a plain script
    import features
    from config import (FEATURE_SCALE_NORMALISE, FEATURE_WRIST_RELATIVE,
                        LABELS_PATH, MODEL_PATH, MODELS_DIR, RANDOM_STATE,
                        SCALER_PATH, SVM_C, SVM_GAMMA, SVM_KERNEL, TEST_SIZE,
                        ensure_dirs)
    from dataset import load_features, load_manifest

METADATA_PATH = MODELS_DIR / "model_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ASL letter SVM")
    parser.add_argument("--holdout-subject", default=None,
                        help="train on everyone else, test only on this subject")
    parser.add_argument("--fast", action="store_true",
                        help="train without probability calibration - much "
                             "quicker, but the merger needs confidences, so "
                             "this is for experiments only")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="subsample the training pool (0 = use everything)")
    parser.add_argument("--no-save", action="store_true",
                        help="report only, write nothing to data/models/")
    return parser.parse_args()


def subsample(X: np.ndarray, y: np.ndarray, per_class: int,
              seed: int = RANDOM_STATE) -> tuple:
    if per_class <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    return X[keep], y[keep]


def split(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
          holdout: Optional[str]) -> tuple:
    """Returns (X_train, X_test, y_train, y_test, description)."""
    if holdout:
        test_mask = subjects == holdout
        if not test_mask.any():
            raise SystemExit(
                f"no samples for subject {holdout!r}. "
                f"Available: {', '.join(sorted(set(subjects)))}"
            )
        if test_mask.all():
            raise SystemExit(
                f"every sample belongs to {holdout!r}; nothing left to train on"
            )
        return (X[~test_mask], X[test_mask], y[~test_mask], y[test_mask],
                f"leave-one-subject-out, held out {holdout!r}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return (X_train, X_test, y_train, y_test,
            f"stratified random split, {int(TEST_SIZE * 100)}% test")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    X_raw, y, subjects = load_features(load_manifest())
    print(f"dataset: {X_raw.shape[0]} samples, {len(set(y))} classes, "
          f"subjects {sorted(set(subjects))}")
    print(features.describe())

    X = features.transform(X_raw)
    X_train, X_test, y_train, y_test, how = split(X, y, subjects,
                                                  args.holdout_subject)
    X_train, y_train = subsample(X_train, y_train, args.max_per_class)
    print(f"split: {how}")
    print(f"train {len(X_train)}  test {len(X_test)}")

    # Fitted on the training split only - fitting on everything would leak the
    # test set's distribution into the model.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                probability=not args.fast, random_state=RANDOM_STATE)

    print("training...")
    start = time.perf_counter()
    model.fit(X_train_s, y_train)
    elapsed = time.perf_counter() - start

    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    test_pred = model.predict(X_test_s)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\ntrained in {elapsed:.1f}s  ({len(model.support_)} support vectors)")
    print(f"train accuracy {train_acc:.4f}")
    print(f"test  accuracy {test_acc:.4f}")
    if not args.holdout_subject:
        print("  ^ single-signer data with near-duplicate frames: this number "
              "is optimistic.\n    The honest one needs --holdout-subject.")
    print()
    print(classification_report(y_test, test_pred, digits=3, zero_division=0))

    if args.no_save:
        print("--no-save: nothing written")
        return

    labels = sorted(set(y.tolist()))
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps({
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split": how,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_seconds": float(elapsed),
        "probability": bool(not args.fast),
        # Recorded so inference can refuse a model built on a different
        # representation - a mismatch here fails silently and looks like a bad
        # model rather than a config bug.
        "feature_wrist_relative": bool(FEATURE_WRIST_RELATIVE),
        "feature_scale_normalise": bool(FEATURE_SCALE_NORMALISE),
        "svm": {"kernel": SVM_KERNEL, "C": SVM_C, "gamma": SVM_GAMMA},
        "labels": labels,
        "subjects": sorted(set(subjects.tolist())),
    }, indent=2), encoding="utf-8")

    print(f"saved model  -> {MODEL_PATH}")
    print(f"saved scaler -> {SCALER_PATH}")
    print(f"saved labels -> {LABELS_PATH}")


if __name__ == "__main__":
    main()
