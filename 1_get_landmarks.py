import argparse
import json
from pathlib import Path

import cv2
import face_alignment
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 68-point 2D landmarks and visualize them.")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--key_image_path", default="source_image_path")
    parser.add_argument("--key_landmark", default="initial_landmark")
    return parser.parse_args()


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_fa(device="cuda"):
    return face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)


def get_landmarks(fa, img_path, resize=(512, 512)):
    """Return 68x2 list; return None if no face detected."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize)
    if img is None:
        print(f"[WARN] Cannot read image {img_path}")
        return None
    preds = fa.get_landmarks(img)
    if preds is None or len(preds) == 0:
        print(f"[WARN] No face detected in {img_path}")
        return None
    return preds[0].tolist()


# ---------------------- Main pipeline ----------------------
def main():
    args = parse_args()

    data = read_json(args.input)
    fa = load_fa()

    for img_id, item in tqdm(data.items(), total=len(data)):
        src_path = item[args.key_image_path]
        ldmk = get_landmarks(fa, src_path)
        if ldmk is None:
            continue

        item[args.key_landmark] = ldmk

    # Write updated JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
