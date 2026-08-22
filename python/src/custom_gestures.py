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
# closest.
#
# The threshold is per-gesture, not global, because gestures are not equally
# spread. A two-handed gesture fills twice as many non-zero dimensions as a
# one-handed one and so spreads wider in Euclidean distance; averaging the two
# into one number would loosen the one-handed gestures (easier to false-trigger)
# and tighten the two-handed ones (harder to recognise) at the same time.
# See KNNClassifier._compute_thresholds.
REJECT_SPREAD_MULTIPLE = 2.5
REJECT_FLOOR = 0.35

# Sample spread alone sets the threshold far too tight, and measurably so. The
# 20 samples of a gesture are medians of overlapping windows, so consecutive
# ones differ by ~0.014 while genuinely different gestures sit 2.8-4.2 apart.
# A threshold built only from that spread lands near 0.35 and rejects
# everything in the enormous gap between, even poses that are unambiguously
# closer to one gesture than to any other.
#
# So the threshold is also scaled against how far away the *nearest other*
# gesture is: accept within roughly a third of the way there, reject beyond.
# That tracks the real decision boundary rather than a frozen moment's jitter.
RELATIVE_MARGIN = 0.35

# With a single gesture stored there is no neighbour to measure against, so
# accept generously - there is nothing to confuse it with anyway.
LONE_GESTURE_THRESHOLD = 1.0


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

    def export_to(self, path: Path) -> int:
        """Write the whole vocabulary to a shareable JSON file."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return len(self.gestures)

    def import_from(self, path: Path, overwrite: bool = False,
                    prefix: str = "") -> Tuple[List[str], List[str]]:
        """Merge a gesture file in. Returns (added, skipped).

        Name collisions are reported rather than silently resolved: quietly
        overwriting someone's gesture, or quietly discarding the imported one,
        are both worse than saying which happened.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        problems = validate_payload(payload)
        if problems:
            raise ImportError_("; ".join(problems))

        added, skipped = [], []
        for name, entry in payload["gestures"].items():
            target = f"{prefix}{name}"
            if target in self.gestures and not overwrite:
                skipped.append(target)
                continue
            self.add(target,
                     np.asarray(entry["samples"], dtype=np.float32),
                     phrase=entry.get("phrase", ""),
                     note=entry.get("note", ""))
            added.append(target)
        return added, skipped

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
        self.thresholds: Dict[str, float] = {}
        self.hands: Dict[str, int] = {}
        self.fit(store)

    def fit(self, store: GestureStore) -> float:
        """(Re)build from the store. Returns elapsed seconds.

        There is no optimisation here - "training" is transforming and storing
        the samples, which is the whole point of the few-shot claim.
        """
        start = time.perf_counter()
        raw, self.labels = store.matrix()
        self.X = features.transform(raw) if len(raw) else raw
        self.thresholds = self._compute_thresholds()
        self.hands = self._count_hands()
        return time.perf_counter() - start

    def _compute_thresholds(self) -> Dict[str, float]:
        """Per-gesture "how far is too far".

        Two signals, and the larger wins:

        * how tightly the gesture's own samples cluster - but they come from
          overlapping windows, so this understates real variation badly;
        * how far away the nearest other gesture is - the actual boundary that
          matters, and the one that scales correctly with hand count, since a
          two-handed gesture is both wider *and* further from its neighbours.
        """
        names = [str(n) for n in np.unique(self.labels)]
        centroids = {name: self.X[self.labels == name].mean(axis=0)
                     for name in names}

        thresholds: Dict[str, float] = {}
        for name in names:
            block = self.X[self.labels == name]

            by_spread = REJECT_FLOOR
            if len(block) >= 2:
                spread = float(np.linalg.norm(block - centroids[name],
                                              axis=1).mean())
                by_spread = max(REJECT_FLOOR, spread * REJECT_SPREAD_MULTIPLE)

            others = [np.linalg.norm(centroids[name] - centroids[other])
                      for other in names if other != name]
            by_margin = (float(min(others)) * RELATIVE_MARGIN if others
                         else LONE_GESTURE_THRESHOLD)

            thresholds[name] = max(by_spread, by_margin)
        return thresholds

    def _count_hands(self) -> Dict[str, int]:
        """How many hands each gesture uses. Reported, not enforced."""
        counts: Dict[str, int] = {}
        for name in np.unique(self.labels):
            block = self.X[self.labels == name]
            left = np.abs(block[:, :FEATURE_DIM // 2]).sum(axis=1) > 0
            right = np.abs(block[:, FEATURE_DIM // 2:]).sum(axis=1) > 0
            counts[str(name)] = int(round(float((left.astype(int)
                                                 + right.astype(int)).mean())))
        return counts

    def threshold_for(self, name: str) -> float:
        return self.thresholds.get(name, REJECT_FLOOR)

    @property
    def reject_threshold(self) -> float:
        """Mean across gestures. For display only - predict() is per-gesture."""
        if not self.thresholds:
            return REJECT_FLOOR
        return float(np.mean(list(self.thresholds.values())))

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

        # Judged against the nearest gesture's own threshold, so a wide-spread
        # two-handed gesture is not held to a tight one-handed gesture's
        # standard, or vice versa.
        nearest_label = str(self.labels[order[0]])
        threshold = self.threshold_for(nearest_label)
        if distances[order[0]] > threshold:
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
        # Measured against the winner's own threshold, keeping confidences
        # comparable between one- and two-handed gestures.
        closeness = 1.0 - min(1.0, distances[order[0]] /
                              self.threshold_for(label))
        return label, float(share * (0.5 + 0.5 * closeness))

    def nearest_info(self, raw_vector: np.ndarray) -> Tuple[Optional[str], float, float]:
        """(nearest gesture, its distance, that gesture's threshold).

        Reports what predict() decided and why, without the reject rule hiding
        it - the difference between "nothing was close" and "something was
        close but just over the line".
        """
        if not self.ready:
            return None, float("inf"), 0.0
        query = features.transform(np.asarray(raw_vector, dtype=np.float32))
        distances = np.linalg.norm(self.X - query, axis=1)
        i = int(np.argmin(distances))
        name = str(self.labels[i])
        return name, float(distances[i]), self.threshold_for(name)

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
        detail = ", ".join(
            f"{name} ({self.hands.get(name, 1)}h, <{self.thresholds[name]:.2f})"
            for name in sorted(self.thresholds)
        )
        return (f"KNN: {len(self.thresholds)} gestures, {len(self.X)} samples, "
                f"k={self.k}\n  {detail}")


class ImportError_(Exception):
    """Raised when an imported gesture file is not usable."""


def validate_payload(payload: dict) -> List[str]:
    """Check an imported file. Returns a list of problems; empty means fine.

    Validated rather than trusted: these files are meant to be shared between
    people and between the desktop and browser builds, so a malformed or
    mismatched one should say what is wrong rather than crash later inside a
    distance calculation.
    """
    problems: List[str] = []
    if not isinstance(payload, dict):
        return ["file is not a JSON object"]

    dim = payload.get("feature_dim")
    if dim is not None and dim != FEATURE_DIM:
        problems.append(f"feature_dim is {dim}, this build uses {FEATURE_DIM}")

    gestures = payload.get("gestures")
    if not isinstance(gestures, dict) or not gestures:
        problems.append("no gestures found")
        return problems

    for name, entry in gestures.items():
        if not isinstance(entry, dict) or "samples" not in entry:
            problems.append(f"{name!r}: missing samples")
            continue
        try:
            samples = np.asarray(entry["samples"], dtype=np.float32)
        except (ValueError, TypeError):
            problems.append(f"{name!r}: samples are not numeric")
            continue
        if samples.ndim != 2 or samples.shape[1] != FEATURE_DIM:
            problems.append(f"{name!r}: expected (n, {FEATURE_DIM}), "
                            f"got {tuple(samples.shape)}")
        elif len(samples) < 2:
            problems.append(f"{name!r}: only {len(samples)} sample(s)")
    return problems


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
