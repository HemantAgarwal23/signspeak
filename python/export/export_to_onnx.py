"""M13 - export the trained pipeline for the browser.

    python -m export.export_to_onnx
    python -m export.export_to_onnx --json-fallback

Writes to web/public/models/:
    asl_svm.onnx    scaler + SVM as one graph, input (1, 126)
    labels.json     class order
    model_meta.json feature flags, so the browser can refuse a mismatch

The scaler is folded into the ONNX graph rather than exported separately and
reapplied in JavaScript. Two places implementing standardisation is two places
to get it subtly wrong, and a silent mismatch there looks exactly like a bad
model. The browser sends raw wrist-relative, scale-normalised landmarks and the
graph does the rest.

Every export is verified against sklearn on real samples before it is accepted -
an export that "succeeded" but predicts differently is worse than one that
failed loudly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features                                                # noqa: E402
from config import (FEATURE_DIM, FEATURE_SCALE_NORMALISE,      # noqa: E402
                    FEATURE_WRIST_RELATIVE, LABELS_PATH, MODEL_PATH,
                    MODELS_DIR, SCALER_PATH, WEB_MODELS_DIR, ensure_dirs)
from dataset import load_features, load_manifest               # noqa: E402

ONNX_PATH = WEB_MODELS_DIR / "asl_svm.onnx"
VERIFY_SAMPLES = 200
TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the SVM for the web")
    parser.add_argument("--json-fallback", action="store_true",
                        help="skip ONNX and export SVM internals as JSON for a "
                             "hand-written RBF implementation in JS")
    parser.add_argument("--samples", type=int, default=VERIFY_SAMPLES)
    return parser.parse_args()


def load_pipeline():
    for path in (MODEL_PATH, SCALER_PATH, LABELS_PATH):
        if not path.exists():
            raise SystemExit(f"missing {path.name} - run python -m src.trainer")
    return (joblib.load(MODEL_PATH), joblib.load(SCALER_PATH),
            json.loads(LABELS_PATH.read_text(encoding="utf-8")))


def verification_data(n: int) -> np.ndarray:
    """Real samples, not random noise.

    Random vectors are not valid hand poses, so an export could agree with
    sklearn on nonsense and still disagree on anything the model will actually
    see. The transform is applied here because that is what the browser sends.
    """
    X_raw, _, _ = load_features(load_manifest())
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_raw), min(n, len(X_raw)), replace=False)
    return features.transform(X_raw[idx]).astype(np.float32)


def export_onnx(model, scaler, X: np.ndarray) -> Tuple[bool, str]:
    """Convert scaler+SVM to one ONNX graph and verify it. (ok, message)."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        return False, f"skl2onnx unavailable: {exc}"

    pipeline = make_pipeline(scaler, model)
    initial_type = [("input", FloatTensorType([None, FEATURE_DIM]))]

    try:
        onnx_model = convert_sklearn(
            pipeline, initial_types=initial_type,
            # zipmap off: it wraps probabilities in a list of dicts, which is
            # awkward in JS and buys nothing over a plain tensor.
            options={id(pipeline): {"zipmap": False}},
            target_opset=15,
        )
    except Exception as exc:                       # noqa: BLE001
        return False, f"conversion failed: {type(exc).__name__}: {exc}"

    WEB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())

    try:
        import onnxruntime as ort
    except ImportError as exc:
        return False, f"cannot verify, onnxruntime missing: {exc}"

    session = ort.InferenceSession(str(ONNX_PATH),
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X})

    onnx_labels = np.asarray(outputs[0]).ravel()
    sklearn_labels = pipeline.predict(X)
    agree = int(np.sum(onnx_labels.astype(str) == sklearn_labels.astype(str)))

    detail = f"{agree}/{len(X)} predictions identical"
    if agree != len(X):
        return False, f"MISMATCH - {detail}"

    if len(outputs) > 1:
        onnx_proba = np.asarray(outputs[1], dtype=np.float64)
        if onnx_proba.ndim == 2 and hasattr(pipeline, "predict_proba"):
            sk_proba = pipeline.predict_proba(X)
            drift = float(np.max(np.abs(onnx_proba - sk_proba)))
            detail += f", max probability drift {drift:.2e}"
            if drift > TOLERANCE:
                return False, f"probability drift too large: {drift:.2e}"

    return True, detail


def export_json_fallback(model, scaler, labels, X: np.ndarray) -> Tuple[bool, str]:
    """Export SVM internals so RBF prediction can be written by hand in JS.

    The spec's fallback for when skl2onnx fights back. Decision values only -
    no Platt calibration - so the browser would need its own softmax over
    margins, exactly as predictor.py does for uncalibrated models.
    """
    gamma = model._gamma if hasattr(model, "_gamma") else model.gamma
    payload = {
        "kernel": model.kernel,
        "gamma": float(gamma),
        "support_vectors": model.support_vectors_.astype(float).tolist(),
        "dual_coef": model.dual_coef_.astype(float).tolist(),
        "intercept": model.intercept_.astype(float).tolist(),
        "n_support": model.n_support_.astype(int).tolist(),
        "classes": [str(c) for c in model.classes_],
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "probA": model.probA_.astype(float).tolist() if hasattr(model, "probA_") else [],
        "probB": model.probB_.astype(float).tolist() if hasattr(model, "probB_") else [],
        "note": "one-vs-one decision values; see predictor.py for the softmax "
                "fallback used when probabilities are unavailable",
    }
    path = WEB_MODELS_DIR / "asl_svm.json"
    WEB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    size = path.stat().st_size / 1e6
    return True, (f"wrote {path.name} ({size:.1f} MB, "
                  f"{len(payload['support_vectors'])} support vectors)")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    model, scaler, labels = load_pipeline()
    X = verification_data(args.samples)
    print(f"model: {len(labels)} classes, {len(model.support_vectors_)} "
          f"support vectors")
    print(f"verifying against {len(X)} real samples\n")

    ok = False
    if not args.json_fallback:
        start = time.perf_counter()
        ok, message = export_onnx(model, scaler, X)
        print(f"ONNX: {'OK' if ok else 'FAILED'} - {message} "
              f"({time.perf_counter() - start:.1f}s)")
        if ok:
            size = ONNX_PATH.stat().st_size / 1e6
            print(f"      {ONNX_PATH.name} is {size:.1f} MB")

    if not ok:
        print("\nfalling back to JSON export of the SVM internals")
        ok, message = export_json_fallback(model, scaler, labels, X)
        print(f"JSON: {message}")

    (WEB_MODELS_DIR / "labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8")

    meta_source = MODELS_DIR / "model_meta.json"
    meta = json.loads(meta_source.read_text(encoding="utf-8")) if \
        meta_source.exists() else {}
    meta.update({
        "feature_dim": FEATURE_DIM,
        "feature_wrist_relative": FEATURE_WRIST_RELATIVE,
        "feature_scale_normalise": FEATURE_SCALE_NORMALISE,
        "scaler_folded_into_graph": bool(ok and not args.json_fallback),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    (WEB_MODELS_DIR / "model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nwrote labels.json and model_meta.json to {WEB_MODELS_DIR}")
    if not ok:
        raise SystemExit("export did not verify - do not ship this model")


if __name__ == "__main__":
    main()
