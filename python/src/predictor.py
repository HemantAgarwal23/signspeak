"""M5 - load the trained model and classify a landmark vector.

    from src.predictor import Predictor
    predictor = Predictor()
    label, confidence, probabilities = predictor.predict(raw_vector)

Input is always a **raw** 126-D vector straight from the extractor. The
predictor applies `features.transform` itself, so no caller can forget to - a
representation mismatch is invisible at runtime and looks exactly like a badly
trained model.

For the same reason the constructor refuses to load a model whose recorded
feature flags disagree with the current config.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

try:
    from . import features
    from .config import (FEATURE_SCALE_NORMALISE, FEATURE_WRIST_RELATIVE,
                         LABELS_PATH, MODEL_PATH, MODELS_DIR, SCALER_PATH)
except ImportError:  # running as a plain script
    import features
    from config import (FEATURE_SCALE_NORMALISE, FEATURE_WRIST_RELATIVE,
                        LABELS_PATH, MODEL_PATH, MODELS_DIR, SCALER_PATH)

METADATA_PATH = MODELS_DIR / "model_meta.json"


class Predictor:
    """Trained SVM + scaler, wrapped so inference matches training exactly."""

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH,
                 labels_path=LABELS_PATH, strict: bool = True) -> None:
        for path in (model_path, scaler_path, labels_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path.name} - train a model first:\n"
                    f"  python -m src.trainer"
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.labels: List[str] = json.loads(labels_path.read_text(encoding="utf-8"))
        self.meta: Dict = {}
        if METADATA_PATH.exists():
            self.meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if strict:
                self._check_features()

        self.has_probabilities = hasattr(self.model, "predict_proba") and \
            getattr(self.model, "probability", False)

    def _check_features(self) -> None:
        """Refuse a model trained under a different representation.

        Silent mismatch here is the nastiest failure mode in the project: the
        model loads, predicts, and is confidently wrong.
        """
        expected = (self.meta.get("feature_wrist_relative"),
                    self.meta.get("feature_scale_normalise"))
        actual = (FEATURE_WRIST_RELATIVE, FEATURE_SCALE_NORMALISE)
        if expected != (None, None) and expected != actual:
            raise RuntimeError(
                f"feature representation mismatch.\n"
                f"  model trained with wrist_relative={expected[0]} "
                f"scale_normalise={expected[1]}\n"
                f"  config now says   wrist_relative={actual[0]} "
                f"scale_normalise={actual[1]}\n"
                f"Retrain (python -m src.trainer) or restore the old config."
            )

    def predict(self, raw_vector: np.ndarray
                ) -> Tuple[str, float, Optional[np.ndarray]]:
        """Raw 126-D vector -> (label, confidence, probabilities or None).

        Without probability calibration the confidence falls back to a softmax
        over the decision-function margins. That is a monotone stand-in, not a
        calibrated probability - fine for ranking, wrong for the merger's tau
        threshold, which is why trainer's --fast models should not be shipped.
        """
        x = features.transform(np.asarray(raw_vector, dtype=np.float32))
        x = self.scaler.transform(x.reshape(1, -1))

        if self.has_probabilities:
            probabilities = self.model.predict_proba(x)[0]
        else:
            margins = self.model.decision_function(x)[0]
            shifted = np.exp(margins - margins.max())
            probabilities = shifted / shifted.sum()

        index = int(np.argmax(probabilities))
        label = self.model.classes_[index]
        return str(label), float(probabilities[index]), probabilities

    def top_k(self, raw_vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Best k guesses, most confident first. Useful for debugging."""
        _, _, probabilities = self.predict(raw_vector)
        order = np.argsort(probabilities)[::-1][:k]
        return [(str(self.model.classes_[i]), float(probabilities[i]))
                for i in order]

    def describe(self) -> str:
        trained = self.meta.get("trained_at", "unknown")
        accuracy = self.meta.get("test_accuracy")
        accuracy = f"{accuracy:.4f}" if accuracy is not None else "unknown"
        return (f"model: {len(self.labels)} classes, trained {trained}, "
                f"reported test accuracy {accuracy}, "
                f"calibrated={self.has_probabilities}")
