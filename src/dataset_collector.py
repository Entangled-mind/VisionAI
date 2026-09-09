"""VisionAI - Everyday Objects Dataset Collector & Annotator

Aggregates, downloads, and generates annotated datasets for common everyday objects:
- Stationery: pen, pencil, notebook, book, scissors, ruler, stapler
- Personal: glasses, sunglasses, wristwatch, wallet, keys, backpack
- Electronics: smartphone, laptop, computer_mouse, keyboard, headphones, remote_control
- Kitchen: coffee_mug, water_bottle, plate, spoon, fork
- Furniture: chair, desk

Produces YOLO-format annotations (class_id x_center y_center width height)
with train/val splits and dark/shadow simulation variants.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Define canonical everyday classes (25 classes)
EVERYDAY_CLASSES: List[str] = [
    "pen",
    "pencil",
    "notebook",
    "book",
    "glasses",
    "sunglasses",
    "wristwatch",
    "wallet",
    "keys",
    "coffee_mug",
    "water_bottle",
    "scissors",
    "ruler",
    "stapler",
    "computer_mouse",
    "keyboard",
    "headphones",
    "smartphone",
    "laptop",
    "backpack",
    "shoes",
    "chair",
    "desk",
    "plate",
    "remote_control",
]

CLASS_TO_ID: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(EVERYDAY_CLASSES)}


def create_dataset_directories(base_dir: Path) -> Dict[str, Path]:
    """Creates directory structure for YOLO training."""
    dirs = {
        "train_images": base_dir / "images" / "train",
        "val_images": base_dir / "images" / "val",
        "train_labels": base_dir / "labels" / "train",
        "val_labels": base_dir / "labels" / "val",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def generate_synthetic_desktop_sample(
    output_img_path: Path,
    output_txt_path: Path,
    width: int = 640,
    height: int = 640,
    is_dark: bool = False,
) -> List[Tuple[int, float, float, float, float]]:
    """Synthesizes a realistic desk/room scene with everyday objects and YOLO annotations."""
    desk_type = random.choice(["wood", "white", "dark_slate", "office_gray"])
    if desk_type == "wood":
        base_color = np.array([random.randint(110, 140), random.randint(140, 180), random.randint(180, 220)], dtype=np.uint8)
    elif desk_type == "white":
        base_color = np.array([random.randint(220, 245), random.randint(220, 245), random.randint(220, 245)], dtype=np.uint8)
    elif desk_type == "dark_slate":
        base_color = np.array([random.randint(40, 60), random.randint(45, 65), random.randint(45, 65)], dtype=np.uint8)
    else:
        base_color = np.array([random.randint(180, 200), random.randint(185, 205), random.randint(190, 210)], dtype=np.uint8)

    canvas = np.full((height, width, 3), base_color, dtype=np.uint8)

    # Add subtle gradient / texture
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    annotations: List[Tuple[int, float, float, float, float]] = []

    # Place 2 to 4 everyday items per scene
    available_objects = ["notebook", "pen", "coffee_mug", "glasses", "smartphone", "computer_mouse"]
    chosen_objects = random.sample(available_objects, k=random.randint(2, 4))
    occupied_boxes = []

    for obj in chosen_objects:
        class_id = CLASS_TO_ID[obj]

        if obj == "notebook":
            ow = random.randint(int(width * 0.30), int(width * 0.44))
            oh = random.randint(int(height * 0.35), int(height * 0.48))
            color = (random.randint(40, 90), random.randint(80, 150), random.randint(160, 220))
        elif obj == "pen":
            ow = random.randint(int(width * 0.04), int(width * 0.07))
            oh = random.randint(int(height * 0.25), int(height * 0.38))
            color = (random.randint(180, 240), random.randint(40, 80), random.randint(20, 60))
        elif obj == "glasses":
            ow = random.randint(int(width * 0.22), int(width * 0.30))
            oh = random.randint(int(height * 0.12), int(height * 0.18))
            color = (random.randint(20, 50), random.randint(20, 50), random.randint(20, 50))
        elif obj == "coffee_mug":
            size = random.randint(int(width * 0.15), int(width * 0.22))
            ow, oh = size, size
            color = (random.randint(210, 250), random.randint(210, 250), random.randint(210, 250))
        elif obj == "smartphone":
            ow = random.randint(int(width * 0.14), int(width * 0.20))
            oh = random.randint(int(height * 0.26), int(height * 0.36))
            color = (random.randint(25, 45), random.randint(25, 45), random.randint(25, 45))
        elif obj == "computer_mouse":
            ow = random.randint(int(width * 0.12), int(width * 0.18))
            oh = random.randint(int(height * 0.18), int(height * 0.25))
            color = (random.randint(40, 70), random.randint(40, 70), random.randint(40, 70))
        else:
            ow, oh = 80, 80
            color = (128, 128, 128)

        # Find placement
        ox = random.randint(15, width - ow - 15)
        oy = random.randint(15, height - oh - 15)
        for _ in range(12):
            overlap = False
            for bx, by, bw, bh in occupied_boxes:
                if not (ox + ow < bx or ox > bx + bw or oy + oh < by or oy > by + bh):
                    overlap = True
                    break
            if not overlap:
                break
            ox = random.randint(15, width - ow - 15)
            oy = random.randint(15, height - oh - 15)

        occupied_boxes.append((ox, oy, ow, oh))

        # Render object
        if obj == "coffee_mug":
            cv2.circle(canvas, (ox + ow // 2, oy + oh // 2), ow // 2, color, -1)
            cv2.circle(canvas, (ox + ow // 2, oy + oh // 2), (ow // 2) - 6, (70, 45, 25), -1)
            cv2.ellipse(canvas, (ox + ow - 4, oy + oh // 2), (10, 16), 0, -90, 90, color, 4)
        elif obj == "glasses":
            lens_w = (ow - 16) // 2
            cv2.rectangle(canvas, (ox, oy), (ox + lens_w, oy + oh), color, 3)
            cv2.rectangle(canvas, (ox + lens_w + 16, oy), (ox + ow, oy + oh), color, 3)
            cv2.line(canvas, (ox + lens_w, oy + oh // 3), (ox + lens_w + 16, oy + oh // 3), color, 3)
        elif obj == "pen":
            cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), color, -1)
            cv2.fillPoly(canvas, [np.array([[ox, oy + oh], [ox + ow, oy + oh], [ox + ow // 2, oy + oh + 8]])], (180, 180, 180))
            cv2.line(canvas, (ox - 3, oy + 12), (ox - 3, oy + 35), (200, 200, 200), 2)
        elif obj == "notebook":
            cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), color, -1)
            for sy in range(oy + 8, oy + oh - 8, 14):
                cv2.circle(canvas, (ox + 6, sy), 4, (210, 210, 210), -1)
        elif obj == "smartphone":
            cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), color, -1)
            cv2.rectangle(canvas, (ox + 4, oy + 8), (ox + ow - 4, oy + oh - 8), (15, 15, 15), -1)
        else:
            cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), color, -1)

        # Normalize coordinates for YOLO format
        cx = (ox + ow / 2.0) / width
        cy = (oy + oh / 2.0) / height
        norm_w = ow / float(width)
        norm_h = oh / float(height)
        annotations.append((class_id, cx, cy, norm_w, norm_h))

    # Apply low light / night vision attenuation if requested
    if is_dark:
        dark_factor = random.uniform(0.12, 0.26)
        canvas = (canvas.astype(np.float32) * dark_factor).astype(np.uint8)
        noise = np.random.normal(0, 4, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(str(output_img_path), canvas)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for class_id, cx, cy, norm_w, norm_h in annotations:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    return annotations


def generate_everyday_dataset(base_dir: Path, num_train: int = 100, num_val: int = 25) -> Path:
    """Generates a complete YOLOv8 dataset with train and validation splits."""
    dirs = create_dataset_directories(base_dir)

    print(f"[DatasetCollector] Generating {num_train} train samples in {dirs['train_images']}...")
    for i in range(num_train):
        is_dark = (i % 3 == 0)
        img_name = f"everyday_train_{i:04d}.jpg"
        txt_name = f"everyday_train_{i:04d}.txt"
        generate_synthetic_desktop_sample(
            dirs["train_images"] / img_name,
            dirs["train_labels"] / txt_name,
            width=640,
            height=640,
            is_dark=is_dark,
        )

    print(f"[DatasetCollector] Generating {num_val} val samples in {dirs['val_images']}...")
    for i in range(num_val):
        is_dark = (i % 3 == 0)
        img_name = f"everyday_val_{i:04d}.jpg"
        txt_name = f"everyday_val_{i:04d}.txt"
        generate_synthetic_desktop_sample(
            dirs["val_images"] / img_name,
            dirs["val_labels"] / txt_name,
            width=640,
            height=640,
            is_dark=is_dark,
        )

    yaml_path = base_dir / "everyday_objects.yaml"
    yaml_content = f"""# VisionAI Everyday Objects Dataset
path: {base_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(EVERYDAY_CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[DatasetCollector] Generated dataset YAML configuration at: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate everyday objects dataset for YOLOv8")
    parser.add_argument("--output-dir", type=str, default="data/everyday_dataset", help="Output directory")
    parser.add_argument("--train-samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=25, help="Number of validation samples")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    generate_everyday_dataset(out_path, num_train=args.train_samples, num_val=args.val_samples)
