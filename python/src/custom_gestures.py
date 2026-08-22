"""Custom gestures: define your own sign, map it to a phrase.

The distinctive half of the project. Where the ASL letter model is trained once
on thousands of images, a custom gesture is learned from 20 samples in well
under a second - because a KNN has no training step worth the name, only
storage.

Storage is plain JSON (`data/custom/gestures.json`) rather than a pickle or
npz, deliberately: the browser build keeps the same structure in IndexedDB, and
export/import between the two should be a file copy, not a conversion.

    store = GestureStore.load()
    store.add("thumbs_up", samples, phrase="OK!")
    classifier = KNNClassifier(store)
    label, confidence = classifier.predict(raw_vector)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from . import features
    from .config import CUSTOM_GESTURE_SAMPLES, DATA_DIR, FEATURE_DIM, KNN_K
except ImportError:  # running as a plain script
    import features
    from config import CUSTOM_GESTURE_SAMPLES, DATA_DIR, FEATURE_DIM, KNN_K

CUSTOM_DIR = DATA_DIR / "custom"
GESTURES_PATH = CUSTOM_DIR / "gestures.json"
SCHEMA_VERSION = 1

# A query further than this from every stored sample is called "no match"
# rather than forced into the nearest class. Without a reject rule a KNN always
# answers, so an idle hand would keep firing whichever gesture happened to be
# closest. Calibrated as a multiple of how spread out a gesture's own samples
# are - see KNNClassifier._reject_threshold.
REJECT_SPREAD_MULTIPLE = 2.5
REJECT_FLOOR = 0.35


@dataclass
class Gesture:
    name: str
    samples: np.ndarray                  # (n, 126) raw coordinates
    phrase: str = ""
    created_at: str = ""
    note: str = ""

    @property
    def output(self) -> str:
        """What the user sees when this gesture fires."""
        return self.phrase or self.name


@dataclass
class GestureStore:
    """Every custom gesture the user has defined, with its phrase mapping."""

    gestures: Dict[str, Gesture] = field(default_factory=dict)
    path: Path = GESTURES_PATH

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: Path = GESTURES_PATH) -> "GestureStore":
        if not path.exists():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        gestures = {}
        for name, entry in payload.get("gestures", {}).items():
            gestures[name] = Gesture(
                name=name,
                samples=np.asarray(entry["samples"], dtype=np.float32),
                phrase=entry.get("phrase", ""),
                created_at=entry.get("created_at", ""),
                note=entry.get("note", ""),
            )
        return cls(gestures=gestures, path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM,
            "gestures": {
                name: {
                    "samples": gesture.samples.tolist(),
                    "phrase": gesture.phrase,
                    "created_at": gesture.created_at,
                    "note": gesture.note,
                }
                for name, gesture in self.gestures.items()
            },
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)             # atomic: never a half-written file

    # -- editing -----------------------------------------------------------
    def add(self, name: str, samples: np.ndarray, phrase: str = "",
            note: str = "") -> Gesture:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected (n, {FEATURE_DIM}), got {samples.shape}")
        gesture = Gesture(name=name, samples=samples, phrase=phrase, note=note,
                          created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.gestures[name] = gesture
        return gesture

    def remove(self, name: str) -> bool:
        return self.gestures.pop(name, None) is not None

    def set_phrase(self, name: str, phrase: str) -> bool:
        if name not in self.gestures:
            return False
        self.gestures[name].phrase = phrase
        return True

    def rename(self, old: str, new: str) -> bool:
        if old not in self.gestures or new in self.gestures:
            return False
        gesture = self.gestures.pop(old)
        gesture.name = new
        self.gestures[new] = gesture
        return True

    # -- views -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.gestures)

    @property
    def names(self) -> List[str]:
        return sorted(self.gestures)

    def matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """All samples stacked, with their labels. Raw coordinates."""
        if not self.gestures:
            return (np.zeros((0, FEATURE_DIM), np.float32), np.array([], dtype=object))
        blocks, labels = [], []
        for name in self.names:
            samples = self.gestures[name].samples
            blocks.append(samples)
            labels.extend([name] * len(samples))
        return np.concatenate(blocks), np.array(labels, dtype=object)

    def summary(self) -> str:
        if not self.gestures:
            return "no custom gestures yet"
        lines = [f"{len(self.gestures)} custom gesture(s)"]
        for name in self.names:
            gesture = self.gestures[name]
            lines.append(f"  {name:<18} {len(gesture.samples):>3} samples  "
                         f"-> {gesture.output!r}")
        return "\n".join(lines)


class KNNClassifier:
    """k-nearest neighbours over stored gesture samples.

    Euclidean distance in the same feature space the SVM uses, so both
    classifiers see identical input and a gesture recorded today stays valid if
    the letter model is retrained.
    """

    def __init__(self, store: GestureStore, k: int = KNN_K) -> None:
        self.k = k
        self.labels: np.ndarray
        self.X: np.ndarray
        self.reject_threshold: float = REJECT_FLOOR
        self.fit(store)

    def fit(self, store: GestureStore) -> float:
        """(Re)build from the store. Returns elapsed seconds.

        There is no optimisation here - "training" is transforming and storing
        the samples, which is the whole point of the few-shot claim.
        """
        start = time.perf_counter()
        raw, self.labels = store.matrix()
        self.X = features.transform(raw) if len(raw) else raw
        self.reject_threshold = self._reject_threshold()
        return time.perf_counter() - start

    def _reject_threshold(self) -> float:
        """How far is too far, derived from how tight the gestures are.

        A gesture's own samples vary by some amount; anything several times
        further away than that is not the gesture. Calibrating from the data
        beats a hard-coded distance, which would depend on the feature scale.
        """
        if len(self.X) < 2:
            return REJECT_FLOOR
        spreads = []
        for name in np.unique(self.labels):
            block = self.X[self.labels == name]
            if len(block) < 2:
                continue
            centroid = block.mean(axis=0)
            spreads.append(np.linalg.norm(block - centroid, axis=1).mean())
        if not spreads:
            return REJECT_FLOOR
        return max(REJECT_FLOOR, float(np.mean(spreads)) * REJECT_SPREAD_MULTIPLE)

    @property
    def ready(self) -> bool:
        return len(self.X) > 0

    def predict(self, raw_vector: np.ndarray
                ) -> Tuple[Optional[str], float]:
        """Raw 126-D vector -> (label, confidence), or (None, 0.0) for no match."""
        if not self.ready:
            return None, 0.0

        query = features.transform(np.asarray(raw_vector, dtype=np.float32))
        distances = np.linalg.norm(self.X - query, axis=1)
        order = np.argsort(distances)[:min(self.k, len(distances))]

        if distances[order[0]] > self.reject_threshold:
            return None, 0.0

        # Vote among the k nearest, weighted by inverse distance, then express
        # confidence as the winner's share of that vote. A unanimous, very close
        # set of neighbours approaches 1.0; a split vote sits near 0.5.
        votes: Dict[str, float] = {}
        for i in order:
            weight = 1.0 / (distances[i] + 1e-6)
            votes[str(self.labels[i])] = votes.get(str(self.labels[i]), 0.0) + weight
        total = sum(votes.values())
        label, weight = max(votes.items(), key=lambda kv: kv[1])
        share = weight / total if total else 0.0

        # Temper the vote share by how close the nearest neighbour actually is,
        # so a lone distant-but-unanimous match cannot report high confidence.
        closeness = 1.0 - min(1.0, distances[order[0]] / self.reject_threshold)
        return label, float(share * (0.5 + 0.5 * closeness))

    def nearest_gesture(self, samples: np.ndarray, exclude: str = ""
                        ) -> Tuple[Optional[str], float]:
        """Closest existing gesture to a set of new samples, and the distance.

        Used to warn that a new gesture looks like one already recorded.
        """
        if not self.ready:
            return None, float("inf")
        query = features.transform(np.asarray(samples, dtype=np.float32))
        best_name, best_distance = None, float("inf")
        for name in np.unique(self.labels):
            if name == exclude:
                continue
            block = self.X[self.labels == name]
            # Mean over pairwise distances: less trigger-happy than the single
            # closest pair, which one sloppy sample could dominate.
            distance = float(np.linalg.norm(
                query[:, None, :] - block[None, :, :], axis=-1).mean())
            if distance < best_distance:
                best_name, best_distance = str(name), distance
        return best_name, best_distance

    def describe(self) -> str:
        if not self.ready:
            return "KNN: no gestures stored"
        return (f"KNN: {len(np.unique(self.labels))} gestures, "
                f"{len(self.X)} samples, k={self.k}, "
                f"reject beyond {self.reject_threshold:.3f}")


DEFAULT_PHRASES: List[Tuple[str, str]] = [
    ("thumbs_up", "OK!"),
    ("stop", "Please stop"),
    ("help", "I need help"),
    ("yes", "Yes"),
    ("no", "No"),
    ("thanks", "Thank you"),
    ("water", "I need water"),
    ("bathroom", "I need the bathroom"),
]
