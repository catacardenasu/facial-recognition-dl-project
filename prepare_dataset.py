"""
Filters the raw Pins Face Recognition dataset down to a chosen subset of
people, cleans up folder names, and holds out a few test images per person.

Run this AFTER download_dataset.py.

What it does:
    1. Finds the raw "pins_<Name>" folders wherever kagglehub cached them.
    2. Picks the N people with the most images (or a list you specify).
    3. Copies their images into ./dataset/<clean_name>/ (strips "pins_",
       lowercases, replaces spaces with underscores).
    4. Moves a few images per person into ./examples/ as a held-out test set.

Usage:
    python prepare_dataset.py --num-people 6 --test-per-person 3
    python prepare_dataset.py --people "Adriana Lima" "Adam Sandler" --test-per-person 3
"""

import argparse
import importlib
import os
import re
import shutil
from pathlib import Path

DATASET_SLUG = "hereisburak/pins-face-recognition"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def clean_name(raw_folder_name: str) -> str:
    """'pins_Adriana Lima' -> 'adriana_lima'"""
    name = raw_folder_name.replace("pins_", "", 1)
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def find_pins_folders(root: Path):
    """Recursively find every folder whose name starts with 'pins_'."""
    return [p for p in root.rglob("pins_*") if p.is_dir()]


def list_images(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def download_dataset(slug: str) -> Path:
    """Download a dataset, reporting a useful error if kagglehub is unavailable."""
    try:
        kagglehub = importlib.import_module("kagglehub")
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] kagglehub is required. Install it with: "
            "python -m pip install kagglehub"
        ) from exc
    return Path(kagglehub.dataset_download(slug))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-people", type=int, default=6,
                         help="How many people to include (picks those with the most images)")
    parser.add_argument("--people", nargs="*", default=None,
                         help="Specific people to include instead, e.g. --people \"Adriana Lima\" \"Adam Sandler\"")
    parser.add_argument("--test-per-person", type=int, default=3,
                         help="How many images per person to hold out into examples/")
    parser.add_argument("--project-dir", type=str, default=".",
                         help="Path to your project root (where dataset/ and examples/ will be created)")
    args = parser.parse_args()

    print(f"[INFO] locating cached dataset for '{DATASET_SLUG}'...")
    cache_path = download_dataset(DATASET_SLUG)

    pins_folders = find_pins_folders(cache_path)
    if not pins_folders:
        raise SystemExit(f"[ERROR] no 'pins_*' folders found under {cache_path}")

    print(f"[INFO] found {len(pins_folders)} people in raw dataset")

    if args.people:
        wanted = {p.lower() for p in args.people}
        selected = [
            f for f in pins_folders
            if clean_name(f.name).replace("_", " ") in wanted
            or f.name.replace("pins_", "").lower() in wanted
        ]
        if not selected:
            raise SystemExit("[ERROR] none of the requested --people were found. "
                              "Check spelling against the raw folder names.")
    else:
        # sort by image count, descending, take top N
        pins_folders.sort(key=lambda f: len(list_images(f)), reverse=True)
        selected = pins_folders[:args.num_people]

    project_dir = Path(args.project_dir)
    dataset_dir = project_dir / "dataset"
    examples_dir = project_dir / "examples"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] selected {len(selected)} people:")
    for folder in selected:
        print(f"  - {folder.name} ({len(list_images(folder))} images)")

    for folder in selected:
        name = clean_name(folder.name)
        images = list_images(folder)

        if len(images) <= args.test_per_person:
            print(f"[WARN] {name} has too few images ({len(images)}) to hold out "
                  f"{args.test_per_person} test images — skipping test split for this person")
            test_images, train_images = [], images
        else:
            test_images = images[:args.test_per_person]
            train_images = images[args.test_per_person:]

        person_dataset_dir = dataset_dir / name
        person_dataset_dir.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            shutil.copy2(img, person_dataset_dir / img.name)

        for img in test_images:
            dest_name = f"{name}_{img.name}"
            shutil.copy2(img, examples_dir / dest_name)

        print(f"[INFO] {name}: {len(train_images)} -> dataset/{name}/, "
              f"{len(test_images)} -> examples/")

    print("\n[INFO] done. Your project folder now has:")
    print(f"  {dataset_dir}/<person>/*.jpg  (training images)")
    print(f"  {examples_dir}/*.jpg          (held-out test images)")
    print("\n[INFO] next step: run encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog")


if __name__ == "__main__":
    main()