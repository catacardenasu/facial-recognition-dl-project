"""
Downloads the Pins Face Recognition dataset from Kaggle using kagglehub.

Requires:
    pip install kagglehub
    A valid Kaggle token saved at ~/.kaggle/access_token
        (see earlier setup steps if you haven't done this yet)

Usage:
    python download_dataset.py
"""

import kagglehub

DATASET_SLUG = "hereisburak/pins-face-recognition"


def main():
    print(f"[INFO] downloading dataset '{DATASET_SLUG}'...")
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"[INFO] dataset downloaded/cached at:\n  {path}")
    print("[INFO] this path is OUTSIDE your project folder.")
    print("[INFO] next, run prepare_dataset.py to copy a subset into your project.")


if __name__ == "__main__":
    main()