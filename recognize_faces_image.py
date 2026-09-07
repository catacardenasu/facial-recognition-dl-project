"""
Recognizes faces in a single image using previously computed encodings.

Usage:
    python recognize_faces_image.py --encodings encodings.pickle --image examples/leonardo_dicaprio_xxx.jpg --detection-method hog

Press any key to close the result window.
"""

import argparse
import pickle

import cv2
import face_recognition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
                     help="path to serialized db of facial encodings")
    ap.add_argument("-i", "--image", required=True,
                     help="path to input image")
    ap.add_argument("-d", "--detection-method", type=str, default="hog",
                     choices=["hog", "cnn"],
                     help="face detection model to use: either 'hog' or 'cnn'")
    args = vars(ap.parse_args())

    print("[INFO] loading encodings...")
    with open(args["encodings"], "rb") as f:
        data = pickle.loads(f.read())

    image = cv2.imread(args["image"])
    if image is None:
        raise SystemExit(f"[ERROR] could not read image '{args['image']}'")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matched_idxs:
                matched_name = data["names"][i]
                counts[matched_name] = counts.get(matched_name, 0) + 1
            name = max(counts, key=counts.get)

        names.append(name)
        print(f"[INFO] found: {name}")

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    cv2.imshow("Image", image)
    print("[INFO] press any key on the image window to close it...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
