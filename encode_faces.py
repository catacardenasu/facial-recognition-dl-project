"""
Encodes all faces in the dataset/ folder into 128-d embeddings and saves
them to a pickle file for later recognition.

Usage:
    python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog

Use --detection-method hog on CPU (fast, less accurate).
Use --detection-method cnn only if you have a CUDA-enabled GPU (slow on CPU).
"""

import argparse
import os
import pickle

import cv2
import face_recognition
from imutils import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True,
                     help="path to input directory of faces + images")
    ap.add_argument("-e", "--encodings", required=True,
                     help="path to serialized db of facial encodings")
    ap.add_argument("-d", "--detection-method", type=str, default="hog",
                     choices=["hog", "cnn"],
                     help="face detection model to use: either 'hog' or 'cnn'")
    args = vars(ap.parse_args())

    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(args["dataset"]))

    if not image_paths:
        raise SystemExit(f"[ERROR] no images found under '{args['dataset']}'. "
                          "Did you run prepare_dataset.py first?")

    known_encodings = []
    known_names = []

    for (i, image_path) in enumerate(image_paths):
        print(f"[INFO] processing image {i + 1}/{len(image_paths)}")
        name = image_path.split(os.path.sep)[-2]

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] could not read {image_path}, skipping")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(
            rgb, model=args["detection_method"]
        )
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    print(f"[INFO] serializing {len(known_encodings)} encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(args["encodings"], "wb") as f:
        f.write(pickle.dumps(data))

    print(f"[INFO] done. Saved to {args['encodings']}")


if __name__ == "__main__":
    main()
