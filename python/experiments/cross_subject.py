"""M9 - does the model work on someone it has never seen?

    python -m experiments.cross_subject

Leave-one-subject-out. For every recorded subject (anyone but `public`), train
on everybody else and test on that person's samples. With only `s01` recorded
this reduces to "train on the Kaggle signer, test on s01"; every volunteer added
with `--subject s02` etc. joins the loop automatically.

WHY THIS NUMBER MATTERS MORE THAN M8'S
--------------------------------------
Every other figure in docs/results/ trains and tests on the same signer, so it
measures how well the model recognises *that person's* handshapes. Here the
test hands, camera, lighting and day are all new. The gap between this and the
within-subject baseline is the generalisation cost, and it is the honest
headline accuracy for the report.

The baseline is the M8 temporal split on `public` at full size, reusing M8's
split_indices so both numbers describe the same training setup.

A letter a subject never recorded (s01 has no Q) is simply absent from that
subject's test set. The model still knows it, so it can still be predicted -
a wrong guess of Q counts as an error, as it would live.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from sklearn.metrics import accuracy_score                    # noqa: E402
from sklearn.preprocessing import StandardScaler              # noqa: E402
from sklearn.svm import SVC                                   # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features                                               # noqa: E402
from config import (DOCS_RESULTS, RANDOM_STATE, SVM_C, SVM_GAMMA,  # noqa: E402
                    SVM_KERNEL, ensure_dirs)
from dataset import load_features, load_manifest              # noqa: E402

from experiments.sample_efficiency import (split_indices,     # noqa: E402
                                           train_and_score)

BASE_SUBJECT = "public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-subject evaluation")
    parser.add_argument("--out", type=Path, default=DOCS_RESULTS)
    return parser.parse_args()


def held_out_predictions(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                         subject: str) -> np.ndarray:
    """Train on every subject except `subject`, predict that subject's rows."""
    train = subjects != subject
    test = subjects == subject
    scaler = StandardScaler().fit(X[train])
    model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                probability=False, random_state=RANDOM_STATE)
    model.fit(scaler.transform(X[train]), y[train])
    return model.predict(scaler.transform(X[test]))


def per_letter(y_true: np.ndarray, y_pred: np.ndarray, subject: str
               ) -> List[dict]:
    rows = []
    for label in sorted(set(y_true.tolist())):
        mask = y_true == label
        wrong = y_pred[mask][y_pred[mask] != label]
        if len(wrong):
            values, counts = np.unique(wrong, return_counts=True)
            mistaken_for = str(values[np.argmax(counts)])
        else:
            mistaken_for = ""
        rows.append({
            "subject": subject,
            "label": label,
            "n": int(mask.sum()),
            "recall": float(np.mean(y_pred[mask] == label)),
            "most_often_mistaken_for": mistaken_for,
        })
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot(summary: List[dict], letters: List[dict], baseline: float,
         path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 2.6]})

    names = ["within-subject\n(public, temporal)"] + [r["subject"] for r in summary]
    values = [baseline] + [r["accuracy"] for r in summary]
    colours = ["#9ca3af"] + ["#c2410c"] * len(summary)
    bars = ax.bar(names, values, color=colours)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.set_title("Unseen signer vs same signer")
    ax.grid(axis="y", alpha=0.3)

    # Recall per letter, pooled over every held-out subject: which shapes fail
    # to transfer, rather than which subject is hardest.
    pooled: Dict[str, List[float]] = {}
    for row in letters:
        pooled.setdefault(row["label"], []).extend(
            [row["recall"]] * row["n"])
    labels = sorted(pooled)
    recall = [float(np.mean(pooled[label])) for label in labels]
    colours2 = ["#c2410c" if r < 0.8 else "#1d4ed8" for r in recall]
    bars2 = ax2.bar(labels, recall, color=colours2)
    # Label only the failures; a 0% bar is otherwise invisible.
    ax2.bar_label(bars2, labels=[f"{r:.0%}" if r < 0.8 else "" for r in recall],
                  padding=2, fontsize=8, color="#c2410c")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("recall on held-out subjects")
    ax2.set_title("Per letter  (orange: below 80%)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    X_raw, y, subjects = load_features(load_manifest())
    X = features.transform(X_raw)
    held_out = sorted(set(subjects.tolist()) - {BASE_SUBJECT})
    if not held_out:
        raise SystemExit("no recorded subjects - record with --subject s01 first")

    print(f"dataset: {len(y)} samples, subjects {sorted(set(subjects))}")

    base = np.flatnonzero(subjects == BASE_SUBJECT)
    train_idx, test_idx = split_indices(y[base], "temporal", 0)
    baseline, _, _ = train_and_score(X[base], y[base], train_idx, test_idx)
    print(f"within-subject baseline ({BASE_SUBJECT}, temporal): {baseline:.4f}\n")

    summary: List[dict] = []
    letters: List[dict] = []
    for subject in held_out:
        predicted = held_out_predictions(X, y, subjects, subject)
        truth = y[subjects == subject]
        accuracy = accuracy_score(truth, predicted)
        rows = per_letter(truth, predicted, subject)
        summary.append({"subject": subject, "n": int(len(truth)),
                        "letters": len(rows), "accuracy": float(accuracy)})
        letters += rows

        print(f"{subject}: {accuracy:.4f} on {len(truth)} samples, "
              f"{len(rows)} letters")
        for row in sorted(rows, key=lambda r: r["recall"])[:6]:
            if row["recall"] < 1:
                print(f"  {row['label']}  recall {row['recall']:.2f}   "
                      f"mostly read as {row['most_often_mistaken_for']}")

    mean = float(np.mean([r["accuracy"] for r in summary]))
    print(f"\nmean over {len(summary)} held-out subject(s): {mean:.4f}   "
          f"(within-subject {baseline:.4f}, gap {baseline - mean:+.4f})")

    write_csv(summary, args.out / "cross_subject.csv")
    write_csv(letters, args.out / "cross_subject_per_letter.csv")
    plot(summary, letters, baseline, args.out / "cross_subject.png")
    (args.out / "cross_subject.json").write_text(json.dumps(
        {"baseline_within_subject": baseline, "mean_held_out": mean,
         "subjects": summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
