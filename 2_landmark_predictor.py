import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import json_repair
import numpy as np
import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# --------------------------------------------------------------------------- #
#                              Helper functions                               #
# --------------------------------------------------------------------------- #
def load_prompt_template(path: str) -> str:
    """Load the prompt template from a plain-text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def convert_landmarks_to_dict(landmarks: List[List[int]]) -> Dict[str, List[List[int]]]:
    """
    Convert a flat list of 68 landmarks into the 4-group format required by the prompt.
    """
    return {
        "JAW/BROWS": landmarks[0:27],
        "NOSE": landmarks[27:36],
        "EYES": landmarks[36:48],
        "MOUTH": landmarks[48:68],
    }


def extract_predicted_landmarks(vlm_output: str) -> List[List[int]]:
    """
    Extract the final JSON block after [Raw Predicted Landmarks].
    Returns a flat list of 68 [x, y] integer pairs.
    """
    match = re.findall(r"```json\s*(\{.*?\})\s*```", vlm_output, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in VLM output.")
    json_block = match[-1]
    parsed: Dict[str, List[List[int]]] = json_repair.loads(json_block)

    flat = parsed.get("JAW/BROWS", []) + parsed.get("NOSE", []) + parsed.get("EYES", []) + parsed.get("MOUTH", [])
    return [[int(p[0]), int(p[1])] for p in flat]


def create_landmark_image(landmarks, size=(512, 512)):
    """Create landmark visualization identical to code2."""
    landmarks = np.array(landmarks)
    img = Image.new("RGB", size, color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    FACIAL_LANDMARKS_INDEXES = {
        "Jaw": list(range(0, 17)),
        "Right_eyebrow": list(range(17, 22)),
        "Left_eyebrow": list(range(22, 27)),
        "Nose_bridge": list(range(27, 31)),
        "Nose_tip": list(range(31, 36)),
        "Right_eye": list(range(36, 42)),
        "Left_eye": list(range(42, 48)),
        "Outer_lip": list(range(48, 60)),
        "Inner_lip": list(range(60, 68)),
    }

    colors = {
        "Jaw": (255, 255, 255),
        "Right_eyebrow": (255, 200, 0),
        "Left_eyebrow": (255, 200, 0),
        "Nose_bridge": (0, 255, 0),
        "Nose_tip": (0, 255, 0),
        "Right_eye": (255, 0, 0),
        "Left_eye": (255, 0, 0),
        "Outer_lip": (0, 0, 255),
        "Inner_lip": (0, 0, 255),
    }

    for name, idxs in FACIAL_LANDMARKS_INDEXES.items():
        pts = landmarks[idxs]
        color = colors[name]
        for x, y in pts:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)

        if name in ["Right_eye", "Left_eye", "Outer_lip", "Inner_lip"]:
            pts = np.vstack((pts, pts[0]))
        for i in range(len(pts) - 1):
            draw.line([tuple(pts[i]), tuple(pts[i + 1])], fill=color, width=2)
    return img


# --------------------------------------------------------------------------- #
#                                  VLM call                                   #
# --------------------------------------------------------------------------- #
def run_single(
    model,
    processor,
    instruction: str,
    image_path: str,
    initial_landmarks: List[List[int]],
    prompt_template: str,
) -> List[List[int]]:
    """Generate new landmarks for a single image given the instruction."""
    norm_landmarks_json = json.dumps(convert_landmarks_to_dict(initial_landmarks), ensure_ascii=False)
    prompt = prompt_template.format(instruction=instruction, norm_landmarks_json=norm_landmarks_json)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.split("<image>")[0]},
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt.split("<image>")[1]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=[text], images=images, videos=videos, do_resize=False, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return extract_predicted_landmarks(output_text)


# --------------------------------------------------------------------------- #
#                                  Main loop                                  #
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace):
    print(f"Loading model from {args.lp_model_path} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.lp_model_path,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.lp_model_path)

    prompt_template = load_prompt_template(args.prompt_file)

    with open(args.input, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    for key, entry in data.items():
        instruction = entry["caption"]
        image_path = entry["source_image_path"]
        initial_landmarks = entry["initial_landmark"]

        print(f"Processing {key} -> {instruction}")
        try:
            predicted = run_single(model, processor, instruction, image_path, initial_landmarks, prompt_template)
            entry["landmark_value"] = predicted

            # Save visualization image
            vis_name = f"{key}.png"
            vis_path = os.path.join(args.target_landmark_image_path, vis_name)
            vis_img = create_landmark_image(predicted)
            vis_img.save(vis_path)
            entry["target_landmark_image_path"] = vis_path

        except Exception as e:
            print(f"Error: {e}")
            entry["landmark_value"] = None
            entry["target_landmark_image_path"] = None

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Done. Results saved to {args.output}")


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new facial landmarks with a Vision-Language Model.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file with keys: caption, landmark_image_path, initial_landmark",
    )
    parser.add_argument("--output", required=True, help="Output JSON file that adds 'landmark_value'")
    parser.add_argument(
        "--target_landmark_image_path",
        required=True,
        default="examples/results/target_landmark/",
        help="Directory to save landmark visualization images",
    )
    parser.add_argument(
        "--prompt_file",
        default="prompt_template.txt",
        help="Path to prompt template (default: prompt_template.txt)",
    )
    parser.add_argument(
        "--lp_model_path",
        default="/path/to/lp_model",
        help="Path to the VLM (local dir or HF hub name)",
    )
    args = parser.parse_args()
    main(args)
