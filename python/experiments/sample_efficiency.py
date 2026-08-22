"""M8 - how much data does a letter actually need?

The project's central claim is that useful gesture recognition needs very few
samples per class. This measures it: for each N, train on N samples per class
and evaluate on a held-out test set that is never subsampled.

    python -m experiments.sample_efficiency
    python -m experiments.sample_efficiency --split random
    python -m experiments.sample_efficiency --representations

TWO SPLITS, AND THE DIFFERENCE IS THE POINT
-------------------------------------------
`random` is the split the build spec asks for: stratified, shuffled. On this
dataset it leaks badly. The imported samples come from one signer's continuous
recording session, and although they were drawn with an even stride, neighbours
remain highly correlated - a random split therefore puts near-identical moments
on both sides of the divide and the model scores well by recognising the
session rather than the handshape.

`temporal` (the default here) splits by position instead: the first 70% of each
class trains, the last 30% tests. Samples were imported in filename order,
which is capture order, so this keeps train and test genuinely apart in time.

Reporting both is more useful than reporting either. The gap between them
measures the leakage, which is a finding rather than an embarrassment - and it
is the honest way to present a 99.8% number that would otherwise look absurd.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# sklearn 1.9 warns about SVC(probability=...) regardless of its value, once per
# fit. This experiment fits hundreds of times, and the noise buries the results.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features                                            # noqa: E402
from config import (DOCS_RESULTS, RANDOM_STATE, SVM_C, SVM_GAMMA,  # noqa: E402
                    SVM_KERNEL, ensure_dirs)
from dataset import load_features, load_manifest           # noqa: E402

SAMPLE_COUNTS = [5, 10, 20, 40, 60, 120, 300]
SEEDS = [0, 1, 2, 3, 4]
TEST_FRACTION = 0.3

REPRESENTATIONS: Dict[str, Dict[str, bool]] = {
    "absolute": {"wrist_relative": False, "scale_normalise": False},
    "wrist-relative": {"wrist_relative": True, "scale_normalise": False},
    "wrist+scale": {"wrist_relative": True, "scale_normalise": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample-efficiency experiment")
    parser.add_argument("--split", choices=["temporal", "random", "both"],
                        default="both")
    parser.add_argument("--counts", default=",".join(map(str, SAMPLE_COUNTS)))
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--representations", action="store_true",
                        help="also compare feature representations at each N")
    parser.add_argument("--out", type=Path, default=DOCS_RESULTS)
    return parser.parse_args()


def split_indices(y: np.ndarray, mode: str, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class train/test indices.

    temporal: the last TEST_FRACTION of each class by position, which is
              capture order, so test frames are from later in the session.
    random:   a shuffled stratified split, as the spec asks.
    """
    train_idx, test_idx = [], []
    rng = np.random.default_rng(seed)

    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        cut = int(len(idx) * (1 - TEST_FRACTION))
        if mode == "temporal":
            # No shuffling: position carries the temporal information that
            # makes this split meaningful.
            train_idx.append(idx[:cut])
            test_idx.append(idx[cut:])
        else:
            shuffled = rng.permutation(idx)
            train_idx.append(shuffled[:cut])
            test_idx.append(shuffled[cut:])

    return np.concatenate(train_idx), np.concatenate(test_idx)


def subsample(y: np.ndarray, pool: np.ndarray, per_class: int,
              seed: int) -> np.ndarray:
    """Draw N indices per class from the training pool."""
    rng = np.random.default_rng(seed)
    picked = []
    for label in np.unique(y[pool]):
        idx = pool[y[pool] == label]
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        picked.append(idx)
    return np.concatenate(picked)


def train_and_score(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                    test_idx: np.ndarray) -> Tuple[float, float, int]:
    """Returns (accuracy, seconds, n_train).

    probability=False throughout: Platt calibration multiplies training time
    roughly fivefold and does not change argmax accuracy, which is all this
    experiment measures.
    """
    scaler = StandardScaler().fit(X[train_idx])
    model = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                probability=False, random_state=RANDOM_STATE)

    start = time.perf_counter()
    model.fit(scaler.transform(X[train_idx]), y[train_idx])
    elapsed = time.perf_counter() - start

    predicted = model.predict(scaler.transform(X[test_idx]))
    return accuracy_score(y[test_idx], predicted), elapsed, len(train_idx)


def run_curve(X: np.ndarray, y: np.ndarray, counts: List[int], seeds: List[int],
              mode: str, tag: str = "") -> List[dict]:
    rows: List[dict] = []
    for n in counts:
        accuracies, times, sizes = [], [], []
        for seed in seeds:
            train_pool, test_idx = split_indices(y, mode, seed)
            available = min(np.bincount(
                np.unique(y[train_pool], return_inverse=True)[1]).min(), n)
            if available < n:
                # Not enough data for this N; stop rather than silently
                # reporting a smaller training set under the wrong label.
                break
            train_idx = subsample(y, train_pool, n, seed)
            accuracy, elapsed, size = train_and_score(X, y, train_idx, test_idx)
            accuracies.append(accuracy)
            times.append(elapsed)
            sizes.append(size)
        if not accuracies:
            continue

        row = {
            "split": mode,
            "representation": tag,
            "n_per_class": n,
            "n_train": int(np.mean(sizes)),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "accuracy_min": float(np.min(accuracies)),
            "accuracy_max": float(np.max(accuracies)),
            # Minimum, not mean or median. Contention only ever adds time, so
            # the fastest of the repeats is the least contaminated estimate of
            # what the fit actually costs. Mean and median both produced curves
            # where training got *cheaper* with more data, which is impossible.
            "train_seconds_min": float(np.min(times)),
            "train_seconds_median": float(np.median(times)),
            "train_seconds_mean": float(np.mean(times)),
            "seeds": len(accuracies),
        }
        rows.append(row)
        label = f"{tag}/" if tag else ""
        print(f"  {label}{mode:<8} N={n:<4} "
              f"acc {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}  "
              f"train {row['train_seconds_min']:5.3f}s  "
              f"({row['n_train']} samples)")
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_curve(rows: List[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = sorted({r["split"] for r in rows})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = {"temporal": "#c2410c", "random": "#1d4ed8"}

    for split in splits:
        subset = [r for r in rows if r["split"] == split and not r["representation"]]
        if not subset:
            continue
        xs = [r["n_per_class"] for r in subset]
        ys = [r["accuracy_mean"] for r in subset]
        es = [r["accuracy_std"] for r in subset]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=split,
                    color=colours.get(split))
        ax2.plot(xs, [r["train_seconds_min"] for r in subset], marker="o",
                 label=split, color=colours.get(split))

    ax.set_xscale("log")
    ax.set_xlabel("training samples per class")
    ax.set_ylabel("test accuracy")
    ax.set_title("Sample efficiency")
    ax.grid(alpha=0.3)
    ax.legend(title="split")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("training samples per class")
    ax2.set_ylabel("training time (s), best of 5")
    ax2.set_title("Training cost")
    ax2.grid(alpha=0.3)
    ax2.legend(title="split")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def plot_representations(rows: List[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [r["representation"] for r in rows if r["representation"]]
    if not tags:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for tag in dict.fromkeys(tags):
        subset = [r for r in rows
                  if r["representation"] == tag and r["split"] == "temporal"]
        if not subset:
            continue
        ax.errorbar([r["n_per_class"] for r in subset],
                    [r["accuracy_mean"] for r in subset],
                    yerr=[r["accuracy_std"] for r in subset],
                    marker="o", capsize=4, label=tag)
    ax.set_xscale("log")
    ax.set_xlabel("training samples per class")
    ax.set_ylabel("test accuracy (temporal split)")
    ax.set_title("Feature representation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    counts = [int(c) for c in args.counts.split(",")]
    seeds = SEEDS[:args.seeds]
    modes = (["temporal", "random"] if args.split == "both" else [args.split])

    X_raw, y, subjects = load_features(load_manifest())
    print(f"dataset: {X_raw.shape[0]} samples, {len(set(y))} classes, "
          f"subjects {sorted(set(subjects))}")
    print(f"counts {counts}  seeds {len(seeds)}  splits {modes}")
    print(f"test fraction {TEST_FRACTION:.0%}, never subsampled\n")

    rows: List[dict] = []
    X = features.transform(X_raw)
    for mode in modes:
        rows += run_curve(X, y, counts, seeds, mode)

    if args.representations:
        print()
        for tag, kwargs in REPRESENTATIONS.items():
            Xr = features.transform(X_raw, **kwargs)
            rows += run_curve(Xr, y, counts, seeds, "temporal", tag=tag)

    write_csv(rows, args.out / "sample_efficiency.csv")
    plot_curve(rows, args.out / "sample_efficiency.png")
    if args.representations:
        plot_representations(rows, args.out / "representation_ablation.png")

    headline = [r for r in rows if not r["representation"]]
    summary = {r["split"]: {str(r["n_per_class"]): round(r["accuracy_mean"], 4)
                            for r in headline if r["split"] == r["split"]}
               for r in headline}
    (args.out / "sample_efficiency.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\nsummary:")
    for mode in modes:
        subset = [r for r in headline if r["split"] == mode]
        if subset:
            best = max(subset, key=lambda r: r["n_per_class"])
            first = min(subset, key=lambda r: r["n_per_class"])
            print(f"  {mode:<8} N={first['n_per_class']}: "
                  f"{first['accuracy_mean']:.4f}   ->   "
                  f"N={best['n_per_class']}: {best['accuracy_mean']:.4f}")


if __name__ == "__main__":
    main()
