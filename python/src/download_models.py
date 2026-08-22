"""Fetch the MediaPipe hand_landmarker.task bundle.

The same file is copied into web/public/models/ so the browser build (M15)
loads a byte-identical graph. Run once after cloning:

    python -m src.download_models
    python -m src.download_models --force
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request

try:
    from .config import (HAND_TASK_FILENAME, HAND_TASK_PATH, HAND_TASK_URL,
                         WEB_MODELS_DIR, ensure_dirs)
except ImportError:  # running as a plain script
    from config import (HAND_TASK_FILENAME, HAND_TASK_PATH, HAND_TASK_URL,
                        WEB_MODELS_DIR, ensure_dirs)


def download(force: bool = False) -> None:
    ensure_dirs()

    if HAND_TASK_PATH.exists() and not force:
        size = HAND_TASK_PATH.stat().st_size
        print(f"already present: {HAND_TASK_PATH} ({size / 1e6:.1f} MB)")
    else:
        print(f"downloading {HAND_TASK_URL}")
        tmp = HAND_TASK_PATH.with_suffix(".task.tmp")
        urllib.request.urlretrieve(HAND_TASK_URL, tmp)
        tmp.replace(HAND_TASK_PATH)
        print(f"saved {HAND_TASK_PATH} "
              f"({HAND_TASK_PATH.stat().st_size / 1e6:.1f} MB)")

    web_copy = WEB_MODELS_DIR / HAND_TASK_FILENAME
    if force or not web_copy.exists():
        shutil.copyfile(HAND_TASK_PATH, web_copy)
        print(f"copied to {web_copy}")
    else:
        print(f"web copy already present: {web_copy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MediaPipe assets")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file already exists")
    download(parser.parse_args().force)


if __name__ == "__main__":
    main()
