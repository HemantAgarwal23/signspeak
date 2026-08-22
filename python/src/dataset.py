"""Dataset manifest: the index of every recorded sample.

On-disk layout:

    python/data/raw/{label}/{subject}_{NNN}.npy    # (30, 126) float32
    python/data/manifest.json

The manifest is the source of truth for what exists; `counts` is a derived
convenience view that is recomputed on every write. Deviation from the build
spec's `sample_NNN.npy`: the subject id is in the filename so that samples from
different people never collide, which cross-subject evaluation (M9) needs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from .aggregator import aggregate
    from .config import (DATA_DIR, FEATURE_DIM, MANIFEST_PATH, PACKS_DIR,
                         RAW_DIR, SEQUENCE_LENGTH, ensure_dirs)
except ImportError:  # running as a plain script
    from aggregator import aggregate
    from config import (DATA_DIR, FEATURE_DIM, MANIFEST_PATH, PACKS_DIR,
                        RAW_DIR, SEQUENCE_LENGTH, ensure_dirs)

MANIFEST_VERSION = 1


def empty_manifest() -> Dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "samples": [],
        "packs": [],
        "counts": {},
    }


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    if not path.exists():
        return empty_manifest()
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest.setdefault("samples", [])
    manifest.setdefault("packs", [])
    manifest.setdefault("counts", {})
    return manifest


def save_manifest(manifest: Dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    manifest["counts"] = recompute_counts(manifest)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    ensure_dirs()
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    tmp.replace(path)                       # atomic: never leave a half file


def recompute_counts(manifest: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in manifest["samples"]:
        counts[entry["label"]][entry["subject"]] += 1
    for pack in manifest.get("packs", []):
        counts[pack["label"]][pack["subject"]] += pack["n_samples"]
    return {label: dict(by_subject) for label, by_subject in sorted(counts.items())}


def next_index(manifest: Dict[str, Any], label: str, subject: str) -> int:
    """Lowest unused sample number for this (label, subject) pair."""
    used = {
        entry["index"] for entry in manifest["samples"]
        if entry["label"] == label and entry["subject"] == subject
    }
    index = 0
    while index in used:
        index += 1
    return index


def sample_path(label: str, subject: str, index: int) -> Path:
    return RAW_DIR / label / f"{subject}_{index:03d}.npy"


def add_sample(manifest: Dict[str, Any], sequence: np.ndarray, label: str,
               subject: str, index: int) -> Path:
    """Write one (30, 126) sequence to disk and register it in the manifest."""
    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
        raise ValueError(
            f"expected shape ({SEQUENCE_LENGTH}, {FEATURE_DIM}), "
            f"got {sequence.shape}"
        )
    path = sample_path(label, subject, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, sequence.astype(np.float32))
    manifest["samples"].append({
        "file": path.relative_to(RAW_DIR.parent).as_posix(),
        "label": label,
        "subject": subject,
        "index": index,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    })
    return path


def add_pack(manifest: Dict[str, Any], vectors: np.ndarray, label: str,
             subject: str, source: str) -> Path:
    """Store many already-aggregated 126-D vectors as one file.

    Bulk imports mean tens of thousands of samples. One .npy per sample would
    be unusable here - this project lives in a OneDrive folder, and 87k tiny
    files would trigger a sync storm and make every manifest write crawl. One
    pack per (label, subject) keeps it to a few dozen files.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected (n, {FEATURE_DIM}), got {vectors.shape}")

    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    path = PACKS_DIR / f"{label}__{subject}.npy"
    np.save(path, vectors)

    manifest["packs"] = [pk for pk in manifest.get("packs", [])
                         if not (pk["label"] == label and pk["subject"] == subject)]
    manifest["packs"].append({
        "file": path.relative_to(DATA_DIR).as_posix(),
        "label": label,
        "subject": subject,
        "n_samples": int(vectors.shape[0]),
        "source": source,
        "aggregated": True,
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    })
    return path


def load_features(manifest: Dict[str, Any] | None = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the whole dataset as one row per sample.

    Recorded sequences are median-aggregated; imported packs are already
    aggregated. Returns (X, labels, subjects) with X shaped (n_samples, 126) of
    RAW coordinates - apply features.transform() before training.
    """
    manifest = manifest if manifest is not None else load_manifest()
    rows: List[np.ndarray] = []
    labels: List[str] = []
    subjects: List[str] = []

    if manifest["samples"]:
        sequences, seq_labels, seq_subjects = load_sequences(manifest)
        rows.append(aggregate(sequences))
        labels.extend(seq_labels.tolist())
        subjects.extend(seq_subjects.tolist())

    for pack in manifest.get("packs", []):
        path = DATA_DIR / pack["file"]
        if not path.exists():
            raise FileNotFoundError(f"pack listed in manifest is missing: {path}")
        vectors = np.load(path).astype(np.float32)
        rows.append(vectors)
        labels.extend([pack["label"]] * len(vectors))
        subjects.extend([pack["subject"]] * len(vectors))

    if not rows:
        raise RuntimeError("Dataset is empty. Record or import samples first.")

    return np.concatenate(rows), np.array(labels), np.array(subjects)


def load_sequences(manifest: Dict[str, Any] | None = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the whole dataset.

    Returns (sequences, labels, subjects) with sequences shaped
    (n_samples, SEQUENCE_LENGTH, FEATURE_DIM).
    """
    manifest = manifest if manifest is not None else load_manifest()
    sequences: List[np.ndarray] = []
    labels: List[str] = []
    subjects: List[str] = []
    missing: List[str] = []

    for entry in manifest["samples"]:
        path = RAW_DIR.parent / entry["file"]
        if not path.exists():
            missing.append(entry["file"])
            continue
        sequences.append(np.load(path).astype(np.float32))
        labels.append(entry["label"])
        subjects.append(entry["subject"])

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} sample(s) listed in the manifest are missing on "
            f"disk, first: {missing[0]}"
        )
    if not sequences:
        raise RuntimeError("Dataset is empty. Record samples with src.data_collector first.")

    return np.stack(sequences), np.array(labels), np.array(subjects)


def summary(manifest: Dict[str, Any] | None = None) -> str:
    manifest = manifest if manifest is not None else load_manifest()
    counts = recompute_counts(manifest)
    if not counts:
        return "manifest empty - nothing recorded or imported yet"

    recorded = len(manifest["samples"])
    imported = sum(pk["n_samples"] for pk in manifest.get("packs", []))
    lines = [f"{recorded + imported} samples across {len(counts)} labels "
             f"({recorded} recorded, {imported} imported)"]
    for label, by_subject in counts.items():
        detail = ", ".join(f"{s}:{n}" for s, n in sorted(by_subject.items()))
        lines.append(f"  {label:<3} total={sum(by_subject.values()):<5} ({detail})")
    return "\n".join(lines)


def remove_label(label: str, manifest: Dict[str, Any] | None = None) -> int:
    """Delete every sample for a label, on disk and in the manifest."""
    manifest = manifest if manifest is not None else load_manifest()
    keep, drop = [], []
    for entry in manifest["samples"]:
        (drop if entry["label"] == label else keep).append(entry)

    for entry in drop:
        path = RAW_DIR.parent / entry["file"]
        path.unlink(missing_ok=True)
    folder = RAW_DIR / label
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()

    n_packs = 0
    for pack in list(manifest.get("packs", [])):
        if pack["label"] == label:
            (DATA_DIR / pack["file"]).unlink(missing_ok=True)
            manifest["packs"].remove(pack)
            n_packs += pack["n_samples"]

    manifest["samples"] = keep
    save_manifest(manifest)
    return len(drop) + n_packs


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Inspect or prune the dataset")
    parser.add_argument("--remove-label", metavar="LABEL",
                        help="delete every sample for this label (destructive)")
    args = parser.parse_args()

    if args.remove_label:
        removed = remove_label(args.remove_label)
        print(f"removed {removed} sample(s) for label {args.remove_label!r}")
    print(summary())


if __name__ == "__main__":
    _main()
