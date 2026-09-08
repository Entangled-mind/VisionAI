"""VisionAI - Milestone 4: Object & Animal Detection Module

Integrates pretrained YOLOv8 (You Only Look Once) deep neural network.
Detects all 80 COCO classes, including:
- Animals: dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe, teddy bear
- Electronics: laptop, cell phone, mouse, keyboard, tv, remote
- Household: cup, bottle, chair, couch, book, clock, scissors, backpack
- Vehicles: bicycle, car, motorcycle, bus, airplane, train, boat
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    DEFAULT_CAMERA_INDEX,
)
from src.utils import draw_styled_box


# Pre-defined distinct colors for common COCO classes (BGR)
CLASS_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    # Animals (Vibrant warm tones)
    "dog": (0, 165, 255),          # Bright Orange
    "cat": (0, 200, 255),          # Amber Gold
    "bird": (255, 255, 0),         # Cyan Sky
    "horse": (42, 110, 180),       # Saddle Brown
    "sheep": (240, 240, 240),      # Wool White
    "cow": (120, 120, 120),        # Slate Gray
    "elephant": (160, 160, 160),   # Ash Gray
    "bear": (30, 60, 100),         # Grizzly Brown
    "zebra": (200, 200, 200),      # Striped Silver
    "giraffe": (0, 215, 255),      # Savanna Gold
    "teddy bear": (180, 105, 255), # Plush Pink
    # People & Electronics
    "person": (255, 100, 50),      # Coral Blue
    "cell phone": (0, 220, 255),   # Golden Yellow
    "laptop": (255, 0, 180),        # Vivid Magenta
    "mouse": (200, 255, 0),         # Mint Green
    "keyboard": (128, 255, 128),    # Pale Lime
    "tv": (255, 128, 0),           # Deep Sky Blue
    # Common Household Items
    "bottle": (50, 220, 100),       # Emerald Green
    "cup": (0, 140, 255),           # Deep Amber
    "chair": (180, 105, 255),       # Lavender
    "book": (255, 200, 0),          # Azure
    "backpack": (100, 100, 255),    # Coral Red
    "clock": (0, 255, 255),         # Yellow
    "scissors": (255, 150, 150),    # Soft Lilac
}


class ObjectDetector:
    """YOLOv8 Object & Animal Detector for real-time multi-class detection."""

    def __init__(
        self,
        model_path: Path = YOLO_MODEL_PATH,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        filter_classes: Optional[List[str]] = None,  # None = Detect all 80 COCO classes including animals
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.filter_classes = set(filter_classes) if filter_classes else None

        print(f"[ObjectDetector] Loading YOLOv8 model from {model_path}...")
        model_target = str(model_path) if model_path.exists() else "yolov8n.pt"
        self.model = YOLO(model_target)
        print(f"[ObjectDetector] YOLOv8 loaded! Detecting {'ALL 80 COCO classes (including animals)' if not self.filter_classes else len(self.filter_classes)} classes.")

    def set_confidence_threshold(self, threshold: float) -> None:
        """Dynamically adjusts detection confidence threshold."""
        self.confidence_threshold = max(0.05, min(0.95, threshold))

    def set_filter_classes(self, classes: Optional[List[str]]) -> None:
        """Sets or clears class filter."""
        self.filter_classes = set(classes) if classes else None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Performs object and animal detection on an image or video frame."""
        if frame is None or frame.size == 0:
            return []

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        first_result = results[0]
        boxes = first_result.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0].item())

            # Apply class filtering only if explicitly set
            if self.filter_classes and class_name not in self.filter_classes:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            detections.append({
                "box": (x1, y1, w, h),
                "class_name": class_name,
                "class_id": cls_id,
                "confidence": conf,
            })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Renders bounding boxes and labels for all detected objects and animals."""
        annotated = frame.copy()

        for det in detections:
            box = det["box"]
            cls_name = det["class_name"]
            conf = det["confidence"]

            # Distinct color from palette or deterministic hash
            color = CLASS_COLOR_PALETTE.get(
                cls_name,
                ((hash(cls_name) * 45) % 256, (hash(cls_name) * 85) % 256, (hash(cls_name) * 125) % 256),
            )

            draw_styled_box(annotated, box=box, label=cls_name.upper(), color=color, confidence=conf)

        return annotated


def run_live_object_detection(camera_index: int = DEFAULT_CAMERA_INDEX, max_frames: Optional[int] = None) -> None:
    """Runs real-time webcam object and animal detection stream."""
    from src.webcam_test import initialize_camera, draw_hud

    detector = ObjectDetector(filter_classes=None)
    cap = initialize_camera(camera_index)

    print("\n[Starting Live Object & Animal Detection] Focus on window and press 'q' or 'ESC' to exit.")

    try:
        frame_count = 0
        prev_time = time.time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            cur_time = time.time()
            if cur_time - prev_time > 0.5:
                fps = frame_count / (cur_time - prev_time)
                frame_count = 0
                prev_time = cur_time

            detections = detector.detect(frame)
            annotated = detector.draw_detections(frame, detections)
            annotated = draw_hud(annotated, fps)

            cv2.imshow("VisionAI - YOLOv8 Object & Animal Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if max_frames and frame_count >= max_frames:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionAI - YOLOv8 Object & Animal Detection")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index for live detection")
    parser.add_argument("--conf", type=float, default=YOLO_CONFIDENCE_THRESHOLD, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--no-display", action="store_true", help="Run without opening GUI windows")
    args = parser.parse_args()

    detector = ObjectDetector(confidence_threshold=args.conf, filter_classes=None)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[Error] File not found: {image_path}", file=sys.stderr)
            sys.exit(1)

        img = cv2.imread(str(image_path))
        detections = detector.detect(img)
        print(f"\n[Detection Result] Found {len(detections)} object/animal(s) in '{image_path}':")
        for i, det in enumerate(detections):
            print(f"  #{i+1}: Class='{det['class_name']}', Confidence={det['confidence']*100:.1f}%, Box={det['box']}")

        if not args.no_display:
            annotated = detector.draw_detections(img, detections)
            cv2.imshow("YOLOv8 Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        run_live_object_detection(camera_index=args.camera)


if __name__ == "__main__":
    main()
