"""Turn a folder of labelled hand images into 126-D landmark samples.

Built for the Kaggle ASL Alphabet dataset (grassknoted/asl-alphabet), whose
training folder looks like:

    asl_alphabet_train/A/A1.jpg, A2.jpg, ...
    asl_alphabet_train/B/...

but any `{label}/{image}` layout works.

Run from the `python/` directory:

    python -m src.import_images --src data/downloads/asl_alphabet_train/asl_alphabet_train
    python -m src.import_images --src <dir> --limit 600 --subject public

Notes on the choices here, because they are not obvious:

* **One hand only.** These are single-hand fingerspelling stills, so
  `max_hands=1` avoids MediaPipe inventing a second hand out of background.
* **Slot is pinned, not inferred.** A third-party still gives no reliable clue
  whether the photo was mirrored, so its handedness label cannot be trusted.
  Every detected hand goes into a fixed slot instead - by default slot 1, the
  right hand, matching a right-handed signer. Consistency is what matters, not
  which slot.
* **Detection failures are expected and counted.** Fist-like handshapes
  (M, N, S, T) and tight crops are the usual casualties. The per-class
  detection rate is reported, and a class that comes out badly is worth
  knowing about before it quietly poisons training.
* **Samples land in packs, not one file each.** See dataset.add_pack.
* **--mirror is required for the Kaggle set.** Measured 2026-08-23 via a
  pose-independent chirality test (sign of the palm normal): the user's
  mirrored webcam gives +0.071, Kaggle as-is gives -0.050, Kaggle mirrored
  gives +0.052. Importing without --mirror trains the model on the mirror image
  of every handshape the user will actually sign - which validates perfectly and
  fails completely on the live camera.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from .config import ASL_LETTERS, FEATURE_DIM, ensure_dirs
    from .dataset import add_pack, load_manifest, save_manifest, summary
    from .landmark_extractor import LandmarkExtractor
except ImportError:  # running as a plain script
    from config import ASL_LETTERS, FEATURE_DIM, ensure_dirs
    from dataset import add_pack, load_manifest, save_manifest, summary
    from landmark_extractor import LandmarkExtractor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract landmarks from a folder of labelled hand images")
    parser.add_argument("--src", required=True, type=Path,
                        help="directory containing one subfolder per label")
    parser.add_argument("--subject", default="public",
                        help="subject id recorded for every imported sample")
    parser.add_argument("--source", default="grassknoted/asl-alphabet",
                        help="provenance string stored in the manifest")
    parser.add_argument("--target", type=int, default=600,
                        help="wanted samples per class. Scanning stops once this "
                             "many images have yielded landmarks, so classes with "
                             "poor detection still end up balanced")
    parser.add_argument("--limit", type=int, default=0,
                        help="max images to scan per class (0 = as many as "
                             "needed to reach --target)")
    parser.add_argument("--classes", default=None,
                        help="comma-separated labels to import "
                             "(default: the 24 static ASL letters)")
    parser.add_argument("--slot", choices=["left", "right"], default="right",
                        help="which hand slot every detected hand is written to")
    parser.add_argument("--mirror", action="store_true",
                        help="flip images horizontally before detection")
    parser.add_argument("--min-visible", type=float, default=0.0,
                        help="skip a class if its detection rate falls below this "
                             "(0-1); the class is reported but not imported")
    parser.add_argument("--dry-run", action="store_true",
                        help="report detection rates without writing anything")
    return parser.parse_args()


def list_images(folder: Path, limit: int) -> List[Path]:
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if limit > 0 and len(files) > limit:
        # Even stride rather than the first N: consecutive frames in these
        # datasets are near-identical, so the first 600 files can be almost one
        # continuous shot.
        step = len(files) / limit
        files = [files[int(i * step)] for i in range(limit)]
    return files


def import_class(extractor: LandmarkExtractor, folder: Path, limit: int,
                 slot: int, mirror: bool, target: int = 0) -> tuple:
    """Returns (vectors (n, 126), attempted, elapsed_seconds).

    Scanning stops early once `target` samples have been collected, so a class
    MediaPipe struggles with is given more images rather than fewer samples.
    """
    files = list_images(folder, limit)
    vectors: List[np.ndarray] = []
    attempted = 0
    start = time.perf_counter()

    for path in files:
        if target and len(vectors) >= target:
            break
        image = cv2.imread(str(path))
        if image is None:
            continue
        attempted += 1
        if mirror:
            image = cv2.flip(image, 1)
        vector = extractor.process(image, force_slot=slot)
        if vector is not None:
            vectors.append(vector)

    elapsed = time.perf_counter() - start
    stacked = (np.stack(vectors).astype(np.float32) if vectors
               else np.zeros((0, FEATURE_DIM), dtype=np.float32))
    return stacked, attempted, elapsed


def main() -> None:
    args = parse_args()
    ensure_dirs()

    src: Path = args.src
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")

    wanted = ([c.strip() for c in args.classes.split(",") if c.strip()]
              if args.classes else list(ASL_LETTERS))
    folders = {label: src / label for label in wanted}
    missing = [label for label, folder in folders.items() if not folder.is_dir()]
    if missing:
        print(f"warning: no folder for {', '.join(missing)} - skipping")
        folders = {k: v for k, v in folders.items() if k not in missing}
    if not folders:
        raise SystemExit(f"none of the requested classes exist under {src}")

    slot = 0 if args.slot == "left" else 1
    print(f"importing {len(folders)} classes from {src}")
    print(f"subject={args.subject} slot={args.slot} target={args.target} "
          f"limit={args.limit or 'auto'} mirror={args.mirror}")
    if args.dry_run:
        print("DRY RUN - nothing will be written")
    print()

    manifest = load_manifest()
    rates: Dict[str, float] = {}
    total_kept = total_seen = 0
    started = time.perf_counter()

    # One static-mode extractor for the whole run: no temporal tracking, so
    # every image is judged on its own.
    with LandmarkExtractor(max_hands=1, static=True) as extractor:
        for label, folder in folders.items():
            vectors, attempted, elapsed = import_class(
                extractor, folder, args.limit, slot, args.mirror, args.target)
            rate = len(vectors) / attempted if attempted else 0.0
            rates[label] = rate
            total_kept += len(vectors)
            total_seen += attempted

            flag = ""
            if rate < 0.5:
                flag = "  <-- LOW"
            elif rate < 0.8:
                flag = "  <-- check"
            print(f"  {label:<3} {len(vectors):>5}/{attempted:<5} "
                  f"detected {rate:6.1%}  {elapsed:5.1f}s{flag}")

            if args.dry_run or len(vectors) == 0:
                continue
            if rate < args.min_visible:
                print(f"      skipped: detection rate below "
                      f"--min-visible {args.min_visible:.0%}")
                continue
            add_pack(manifest, vectors, label, args.subject, args.source)
            save_manifest(manifest)

    overall = total_kept / total_seen if total_seen else 0.0
    print(f"\n{total_kept}/{total_seen} images yielded landmarks "
          f"({overall:.1%}) in {time.perf_counter() - started:.0f}s")

    worst = sorted(rates.items(), key=lambda kv: kv[1])[:5]
    print("worst classes: " + ", ".join(f"{k} {v:.0%}" for k, v in worst))

    if not args.dry_run:
        print()
        print(summary(load_manifest()))


if __name__ == "__main__":
    main()
