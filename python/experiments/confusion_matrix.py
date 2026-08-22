"""M12 - which letters does the model actually confuse?

    python -m experiments.confusion_matrix
    python -m experiments.confusion_matrix --split random

Produces a row-normalised confusion matrix heatmap, a ranked table of the most
confused class pairs, and per-class recall - all to docs/results/.

Row-normalised, so each row sums to 1 and the diagonal reads as **recall**: of
all the times this letter was shown, what fraction was recognised. That answers
"which letters does the model fail on", which is the question worth asking.
Column normalisation would answer "when it says R, how often is it right", a
different and less useful question here since the classes are balanced.

The split defaults to `temporal` for the same reason as M8: a random split on
single-session data puts near-identical frames on both sides and produces a
confusion matrix that is almost perfectly diagonal and tells you nothing.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402
from sklearn.preprocessing import StandardScaler              # noqa: E402
from sklearn.svm import SVC                                   # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features                                               # noqa: E402
from config import (DOCS_RESULTS, RANDOM_STATE, SVM_C, SVM_GAMMA,  # noqa: E402
                    SVM_KERNEL, ensure_dirs)
from dataset import load_features, load_manifest              # noqa: E402

# Imported rather than copied: the temporal split must be identical to M8's or
# the two experiments are not describing the same model.
from experiments.sample_efficiency import split_indices       # noqa: E402

TOP_PAIRS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confusion matrix")
    parser.add_argument("--split", choices=["temporal", "random"],
                        default="temporal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DOCS_RESULTS)
    return parser.parse_args()


def confused_pairs(matrix: np.ndarray, labels: List[str]
                   ) -> List[Tuple[str, str, float, int]]:
    """Off-diagonal entries, worst first.

    Symmetric pairs are kept separate on purpose: "R shown, U predicted" and
    "U shown, R predicted" are different failures and often have very different
    rates. Collapsing them would hide that asymmetry.
    """
    pairs = []
    counts = matrix.sum(axis=1, keepdims=True)
    normalised = matrix / np.maximum(counts, 1)
    for i, true_label in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i == j or matrix[i, j] == 0:
                continue
            pairs.append((true_label, predicted, float(normalised[i, j]),
                          int(matrix[i, j])))
    return sorted(pairs, key=lambda p: p[2], reverse=True)


def plot_matrix(matrix: np.ndarray, labels: List[str], path: Path,
                accuracy: float, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(10, 8.4))
    sns.heatmap(
        normalised, ax=ax, cmap="rocket_r", vmin=0, vmax=1,
        xticklabels=labels, yticklabels=labels, square=True,
        linewidths=0.4, linecolor="#ffffff",
        cbar_kws={"label": "fraction of true class", "shrink": 0.75},
        # Only annotate cells worth reading: 24x24 fully annotated is a wall of
        # numbers, and the interesting entries are the errors.
        annot=np.where(normalised >= 0.02,
                       np.round(normalised, 2).astype(object), ""),
        fmt="", annot_kws={"size": 6.5},
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("shown")
    ax.set_title(f"ASL letter confusion  -  {split} split, "
                 f"overall accuracy {accuracy:.1%}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    X_raw, y, subjects = load_features(load_manifest())
    X = features.transform(X_raw)
    train_idx, test_idx = split_indices(y, args.split, args.seed)

    print(f"dataset: {len(y)} samples, {len(set(y))} classes")
    print(f"split: {args.split}   train {len(train_idx)}  test {len(test_idx)}")

    scaler = StandardScaler().fit(X[train_idx])
    model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                probability=False, random_state=RANDOM_STATE)
    model.fit(scaler.transform(X[train_idx]), y[train_idx])
    predicted = model.predict(scaler.transform(X[test_idx]))

    accuracy = accuracy_score(y[test_idx], predicted)
    labels = sorted(set(y.tolist()))
    matrix = confusion_matrix(y[test_idx], predicted, labels=labels)
    print(f"overall accuracy {accuracy:.4f}\n")

    # -- per-class recall
    recall = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    order = np.argsort(recall)
    print("weakest letters:")
    for i in order[:8]:
        worst_j = int(np.argmax(np.where(np.arange(len(labels)) == i,
                                         -1, matrix[i])))
        print(f"  {labels[i]}  recall {recall[i]:.3f}   "
              f"most often mistaken for {labels[worst_j]}")

    pairs = confused_pairs(matrix, labels)
    print(f"\ntop {TOP_PAIRS} confused pairs:")
    for true_label, wrong, rate, count in pairs[:TOP_PAIRS]:
        print(f"  {true_label} -> {wrong}   {rate:.1%}  ({count} samples)")

    # -- outputs
    plot_matrix(matrix, labels, args.out / "confusion_matrix.png",
                accuracy, args.split)

    pairs_path = args.out / "confused_pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shown", "predicted", "rate", "count"])
        for row in pairs[:50]:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", row[3]])
    print(f"wrote {pairs_path}")

    recall_path = args.out / "per_class_recall.csv"
    with recall_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["letter", "recall", "support"])
        for i, label in enumerate(labels):
            writer.writerow([label, f"{recall[i]:.4f}", int(matrix[i].sum())])
    print(f"wrote {recall_path}")

    matrix_path = args.out / "confusion_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + matrix[i].tolist())
    print(f"wrote {matrix_path}")


if __name__ == "__main__":
    main()
