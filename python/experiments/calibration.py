"""M10 - can a handful of a new user's own samples recover the accuracy lost
to an unseen signer?

    python -m experiments.calibration

For each recorded subject: the SVM is trained on everyone else (as in M9), the
first N of that subject's samples per letter become a personal KNN, and the two
are combined by confidence-weighted voting. Accuracy is measured on the
subject's remaining samples, before and after.

THE COMBINATION RULE
--------------------
Two voters, each with (label, confidence):

* the SVM, confidence = its calibrated probability (probability=True, exactly
  as shipped, so the numbers mean what they mean live);
* the personal KNN - custom_gestures.KNNClassifier, the same class that
  recognises custom gestures - which may also abstain when the nearest sample
  is beyond its reject threshold.

Agreement wins outright; disagreement goes to the higher confidence; an
abstaining KNN defers to the SVM. Deliberately that simple, because M24 must
reproduce it in the browser, where classifier-knn.js already exists.

CAVEAT FOR THE REPORT
---------------------
Calibration samples are the *first* N per letter in capture order and the test
samples are the rest, so no test sample precedes a calibration one. They are
still the same session, though, so "after" is an upper bound: a calibration
done today and used next week would score lower.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from sklearn.preprocessing import StandardScaler              # noqa: E402
from sklearn.svm import SVC                                   # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features                                               # noqa: E402
from config import (CALIBRATION_SAMPLES, DOCS_RESULTS,        # noqa: E402
                    RANDOM_STATE, SVM_C, SVM_GAMMA, SVM_KERNEL, ensure_dirs)
from custom_gestures import Gesture, GestureStore, KNNClassifier  # noqa: E402
from dataset import load_features, load_manifest              # noqa: E402

from experiments.cross_subject import BASE_SUBJECT            # noqa: E402

# Personal samples per letter to try. The test set is fixed at everything after
# the largest count, so every row is scored on identical samples.
COUNTS = [1, 2, 3, CALIBRATION_SAMPLES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personal calibration")
    parser.add_argument("--out", type=Path, default=DOCS_RESULTS)
    return parser.parse_args()


def vote(svm: Tuple[str, float], knn: Tuple[Optional[str], float]) -> str:
    """Confidence-weighted vote between the SVM and the personal KNN."""
    svm_label, svm_conf = svm
    knn_label, knn_conf = knn
    if knn_label is None or knn_label == svm_label:
        return svm_label
    return knn_label if knn_conf > svm_conf else svm_label


def first_n(labels: np.ndarray, n: int) -> np.ndarray:
    """Positions of the first n samples of each letter, in capture order."""
    return np.concatenate([np.flatnonzero(labels == label)[:n]
                           for label in sorted(set(labels.tolist()))])


def evaluate_subject(X_raw: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                     subject: str) -> List[dict]:
    X = features.transform(X_raw)
    train = subjects != subject
    scaler = StandardScaler().fit(X[train])
    # probability=True: the vote compares SVM confidence with KNN confidence,
    # so it needs the same calibrated probabilities the live app uses.
    model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                probability=True, random_state=RANDOM_STATE)
    model.fit(scaler.transform(X[train]), y[train])

    own_raw = X_raw[subjects == subject]
    own_y = y[subjects == subject]
    test = np.setdiff1d(np.arange(len(own_y)), first_n(own_y, max(COUNTS)))

    probabilities = model.predict_proba(scaler.transform(
        features.transform(own_raw[test])))
    svm = [(str(model.classes_[i]), float(p[i]))
           for p, i in zip(probabilities, probabilities.argmax(axis=1))]
    truth = own_y[test]
    before = float(np.mean([label for label, _ in svm] == truth))

    rows = []
    for n in COUNTS:
        cal = first_n(own_y, n)
        store = GestureStore(gestures={
            label: Gesture(label, own_raw[cal][own_y[cal] == label])
            for label in sorted(set(own_y[cal].tolist()))})
        knn = KNNClassifier(store)
        knn_out = [knn.predict(v) for v in own_raw[test]]

        knn_only = float(np.mean([label == t for (label, _), t
                                  in zip(knn_out, truth)]))
        combined = float(np.mean([vote(s, k) == t for s, k, t
                                  in zip(svm, knn_out, truth)]))
        abstained = float(np.mean([label is None for label, _ in knn_out]))
        rows.append({
            "subject": subject, "calibration_per_letter": n,
            "n_test": int(len(test)), "svm_only": before,
            "knn_only": knn_only, "svm_plus_knn": combined,
            "knn_abstain_rate": abstained,
        })
        print(f"  N={n}  svm {before:.4f}   knn {knn_only:.4f} "
              f"(abstains {abstained:.0%})   combined {combined:.4f}")
    return rows


def plot(rows: List[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = sorted({r["subject"] for r in rows})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    at_full = [r for r in rows
               if r["calibration_per_letter"] == CALIBRATION_SAMPLES]
    width = 0.27
    x = np.arange(len(subjects))
    for offset, key, name, colour in [
            (-width, "svm_only", "before (SVM)", "#9ca3af"),
            (0, "knn_only", "personal KNN only", "#1d4ed8"),
            (width, "svm_plus_knn", "after (SVM + KNN)", "#c2410c")]:
        values = [next(r[key] for r in at_full if r["subject"] == s)
                  for s in subjects]
        bars = ax.bar(x + offset, values, width, label=name, color=colour)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values],
                     padding=2, fontsize=8)
    ax.set_xticks(x, subjects)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy on held-out samples")
    ax.set_title(f"Calibration with {CALIBRATION_SAMPLES} samples per letter")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    for subject in subjects:
        subset = [r for r in rows if r["subject"] == subject]
        xs = [r["calibration_per_letter"] for r in subset]
        ax2.plot(xs, [r["svm_plus_knn"] for r in subset], marker="o",
                 color="#c2410c", label=f"{subject} SVM + KNN")
        ax2.plot(xs, [r["knn_only"] for r in subset], marker="o",
                 linestyle="--", color="#1d4ed8", label=f"{subject} KNN only")
        ax2.axhline(subset[0]["svm_only"], color="#9ca3af", linestyle=":",
                    label=f"{subject} before")
    ax2.set_xticks(COUNTS)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("personal samples per letter")
    ax2.set_ylabel("accuracy")
    ax2.set_title("How many personal samples are needed")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    X_raw, y, subjects = load_features(load_manifest())
    held_out = sorted(set(subjects.tolist()) - {BASE_SUBJECT})
    if not held_out:
        raise SystemExit("no recorded subjects - record with --subject s01 first")

    rows: List[dict] = []
    for subject in held_out:
        print(f"{subject}: training SVM on everyone else "
              "(probability=True, ~30s)")
        rows += evaluate_subject(X_raw, y, subjects, subject)

    path = args.out / "calibration.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")
    (args.out / "calibration.json").write_text(json.dumps(rows, indent=2),
                                               encoding="utf-8")
    plot(rows, args.out / "calibration.png")


if __name__ == "__main__":
    main()
