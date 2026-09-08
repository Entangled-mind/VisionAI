"""VisionAI - Utilities Module

Provides helper functions for:
1. Downloading pretrained ONNX face models (YuNet and SFace).
2. Drawing clean, styled bounding boxes and labels on OpenCV frames.
3. Geometric utilities (Intersection over Union, bounding box checks).
4. Color palette generation.
"""

import urllib.request
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, YUNET_MODEL_PATH, SFACE_MODEL_PATH

# Model URLs from official OpenCV Zoo
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def download_file_if_missing(url: str, target_path: Path, description: str = "Model") -> bool:
    """Downloads a file from a URL if it does not already exist on disk.

    Args:
        url: Remote download URL.
        target_path: Destination local file path.
        description: Human-readable name for logging.

    Returns:
        bool: True if file is ready on disk, False if download failed.
    """
    if target_path.exists() and target_path.stat().st_size > 0:
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Download] Fetching {description} from:\n  {url}")
    print(f"  Destination: {target_path}...")

    try:
        urllib.request.urlretrieve(url, str(target_path))
        print(f"[Download] Successfully downloaded {description} ({target_path.stat().st_size / 1024:.1f} KB).")
        return True
    except Exception as exc:
        print(f"[Download Error] Could not download {description}: {exc}", file=sys.stderr)
        if target_path.exists():
            target_path.unlink()
        return False


def ensure_face_models() -> Tuple[bool, bool]:
    """Ensures YuNet detector and SFace recognizer weights are downloaded."""
    yunet_ok = download_file_if_missing(YUNET_URL, YUNET_MODEL_PATH, "YuNet Face Detector")
    sface_ok = download_file_if_missing(SFACE_URL, SFACE_MODEL_PATH, "SFace Face Recognizer")
    return yunet_ok, sface_ok


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes.

    Boxes are in format: (x, y, w, h).

    Args:
        boxA: First bounding box (x, y, w, h).
        boxB: Second bounding box (x, y, w, h).

    Returns:
        float: IoU value in range [0.0, 1.0].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    union_area = float(areaA + areaB - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def is_box_inside(inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int], margin: float = 0.1) -> bool:
    """Checks if an inner bounding box (e.g. face) is predominantly inside an outer box (e.g. person).

    Args:
        inner_box: (x, y, w, h) of potential inner box.
        outer_box: (x, y, w, h) of potential outer box.
        margin: Fractional expansion margin allowed.

    Returns:
        bool: True if inner_box center lies within outer_box.
    """
    ix, iy, iw, ih = inner_box
    ox, oy, ow, oh = outer_box

    center_x = ix + iw / 2
    center_y = iy + ih / 2

    return (ox <= center_x <= ox + ow) and (oy <= center_y <= oy + oh)


def draw_styled_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    confidence: Optional[float] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draws a professional bounding box with a high-contrast label pill badge.

    Args:
        image: Frame to draw on (in-place modification).
        box: (x, y, w, h) bounding box coordinates.
        label: Text string for label.
        color: (B, G, R) color tuple.
        confidence: Optional float confidence score (0.0 to 1.0).
        thickness: Line thickness for the box.

    Returns:
        np.ndarray: Annotated frame.
    """
    x, y, w, h = [int(v) for v in box]

    # Draw main bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    # Prepare label text
    if confidence is not None:
        display_text = f"{label} ({confidence * 100:.1f}%)"
    else:
        display_text = label

    # Font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thick = 1

    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, font_thick)
    pill_height = text_h + baseline + 8
    pill_width = text_w + 10

    # Ensure label doesn't render outside top of frame
    label_y1 = max(0, y - pill_height)
    label_y2 = label_y1 + pill_height
    label_x2 = min(image.shape[1], x + pill_width)

    # Draw solid pill background for text readability
    cv2.rectangle(image, (x, label_y1), (label_x2, label_y2), color, -1)

    # Determine contrasting text color (white on dark, black on light)
    brightness = color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)

    cv2.putText(
        image,
        display_text,
        (x + 5, label_y2 - baseline - 3),
        font,
        font_scale,
        text_color,
        font_thick,
        cv2.LINE_AA,
    )
    return image
