"""VisionAI - Everyday Objects Model Training & Augmentation Pipeline

Fine-tunes YOLOv8 on common everyday objects (pens, pencils, notebooks,
glasses, sunglasses, watches, keys, mugs, etc.) with advanced augmentations:
- Mosaic 4-image stitching (multi-scale desktop objects)
- MixUp alpha blending (occlusion invariance)
- HSV Color & Exposure Jitter (fluorescent, warm & outdoor lighting)
- Synthetic Darkness / Low-Light Simulation (Night Vision robustness)
- Random Cutout / Erasing (dense desk clutter)

Exports trained weights to PyTorch (.pt) and ONNX (.onnx) formats.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset_collector import generate_everyday_dataset, EVERYDAY_CLASSES


def train_everyday_yolo(
    data_yaml_path: Path,
    pretrained_weights: str = "yolov8n.pt",
    epochs: int = 5,
    imgsz: int = 640,
    batch_size: int = 8,
    output_dir: Path = Path("models/runs_everyday"),
    device: str = "cpu",
) -> YOLO:
    """Fine-tunes YOLOv8 with advanced augmentations for everyday objects."""
    print(f"\n========================================================")
    print(f"[VisionAI Trainer] Initializing Transfer Learning Pipeline")
    print(f"  Pretrained Base:     {pretrained_weights}")
    print(f"  Dataset Config:      {data_yaml_path}")
    print(f"  Number of Classes:   {len(EVERYDAY_CLASSES)}")
    print(f"  Epochs:              {epochs}")
    print(f"  Image Size:          {imgsz}x{imgsz}")
    print(f"  Device:              {device}")
    print(f"========================================================\n")

    # Load base model
    model = YOLO(pretrained_weights)

    # Train with extensive data augmentations:
    # - mosaic: 1.0 (forces small object feature retention)
    # - mixup: 0.15 (blends overlapping desk objects)
    # - hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4 (color & lighting invariance)
    # - degrees: 10.0 (slight rotation on flat surfaces)
    # - scale: 0.5 (multiscale zoom from 0.5x to 1.5x)
    # - fliplr: 0.5 (left-right reflection)
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(output_dir),
        name="everyday_yolo",
        exist_ok=True,
        # Augmentation hyperparams
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        scale=0.5,
        fliplr=0.5,
        verbose=True,
    )

    print("\n[VisionAI Trainer] Training completed successfully!")
    return model


def export_model_variants(model: YOLO, export_dir: Path, export_onnx: bool = False) -> Dict[str, Path]:
    """Exports trained YOLO model to ONNX for cross-platform deployment."""
    export_dir.mkdir(parents=True, exist_ok=True)
    exported_paths = {}

    # 1. Save PyTorch best weights
    best_pt_path = export_dir / "visionai_everyday_yolov8n.pt"
    try:
        model.save(str(best_pt_path))
        exported_paths["pytorch"] = best_pt_path
        print(f"[VisionAI Exporter] Saved PyTorch model: {best_pt_path}")
    except Exception as e:
        print(f"[VisionAI Exporter] PyTorch save note: {e}")

    # 2. Export to ONNX format (optional)
    if export_onnx:
        try:
            print("[VisionAI Exporter] Exporting to ONNX format...")
            onnx_file = model.export(format="onnx", dynamic=False, simplify=True)
            exported_paths["onnx"] = Path(onnx_file)
            print(f"[VisionAI Exporter] ONNX export complete: {onnx_file}")
        except Exception as e:
            print(f"[VisionAI Exporter] ONNX export skipped/warning: {e}")
    else:
        print("[VisionAI Exporter] ONNX export skipped (use --export-onnx to generate ONNX file).")

    return exported_paths


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Everyday Objects with Augmentation")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--dataset-dir", type=str, default="data/everyday_dataset", help="Dataset directory")
    parser.add_argument("--generate-data", action="store_true", default=True, help="Auto-generate synthetic dataset")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX format")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 step dry-run to verify pipeline")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    yaml_path = dataset_path / "everyday_objects.yaml"

    if args.generate_data or not yaml_path.exists():
        print(f"[VisionAI Trainer] Generating augmented everyday objects dataset...")
        num_train = 16 if args.dry_run else 80
        num_val = 6 if args.dry_run else 20
        yaml_path = generate_everyday_dataset(dataset_path, num_train=num_train, num_val=num_val)

    epochs = 1 if args.dry_run else args.epochs
    trained_model = train_everyday_yolo(
        data_yaml_path=yaml_path,
        epochs=epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device="cpu",
    )

    export_model_variants(trained_model, Path("models"), export_onnx=args.export_onnx)


if __name__ == "__main__":
    main()
