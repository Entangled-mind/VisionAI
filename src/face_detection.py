"""VisionAI - Milestone 2: Face Detection Module

Implements modern Deep Learning Face Detection via OpenCV YuNet (ONNX)
with 5-point facial landmarks (right eye, left eye, nose tip, right mouth, left mouth).

Provides:
- FaceDetector class with standardized output format:
    List[Dict]: [{"box": (x, y, w, h), "confidence": float, "landmarks": [...]}]
- Interactive webcam testing loop and static image testing.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    YUNET_MODEL_PATH,
    FACE_DETECTION_CONFIDENCE,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    TEST_IMAGES_DIR,
)
from src.utils import ensure_face_models, draw_styled_box


class FaceDetector:
    """Unified Deep Learning Face Detector powered by OpenCV YuNet."""

    def __init__(
        self,
        backend: str = "yunet",
        confidence_threshold: float = FACE_DETECTION_CONFIDENCE,
        nms_threshold: float = 0.3,
    ) -> None:
        """Initializes the Face Detector.

        Args:
            backend: Detector backend (default: 'yunet').
            confidence_threshold: Minimum detection confidence score (0.0 to 1.0).
            nms_threshold: Non-Maximum Suppression threshold to merge overlapping boxes.
        """
        self.backend = backend.lower()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.detector = None
        self._current_input_size = (320, 320)

        self._init_detector()

    def _init_detector(self) -> None:
        """Loads and initializes YuNet ONNX model."""
        ensure_face_models()
        if not YUNET_MODEL_PATH.exists():
            raise FileNotFoundError(f"YuNet ONNX model weights missing at: {YUNET_MODEL_PATH}")

        # In OpenCV 5.0, FaceDetectorYN_create or FaceDetectorYN.create initializes the model
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=str(YUNET_MODEL_PATH),
                config="",
                input_size=self._current_input_size,
                score_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
            )
        except Exception:
            self.detector = cv2.FaceDetectorYN_create(
                str(YUNET_MODEL_PATH),
                "",
                self._current_input_size,
                self.confidence_threshold,
                self.nms_threshold,
                5000,
            )
        print("[FaceDetector] Initialized Deep Learning backend: OpenCV YuNet (ONNX).")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detects faces in an image or video frame.

        Args:
            frame: Input BGR image array.

        Returns:
            List of detected faces, each represented as a dictionary:
            {
                "box": (x, y, w, h),
                "confidence": float,
                "landmarks": np.ndarray shape (5, 2) or None,
                "raw": original detector output vector
            }
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        # YuNet requires input_size to match current frame dimensions
        if self._current_input_size != (w, h):
            self._current_input_size = (w, h)
            self.detector.setInputSize((w, h))

        # detect() returns: retval, faces (faces is Nx15 array)
        # Layout: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        _, raw_faces = self.detector.detect(frame)

        if raw_faces is not None:
            for face in raw_faces:
                score = float(face[-1])
                if score < self.confidence_threshold:
                    continue

                # Bounding box coordinates
                x = max(0, int(face[0]))
                y = max(0, int(face[1]))
                bw = min(w - x, int(face[2]))
                bh = min(h - y, int(face[3]))

                # 5 Facial Landmarks: Right Eye, Left Eye, Nose Tip, Right Mouth, Left Mouth
                landmarks = face[4:14].reshape((5, 2)).astype(np.int32)

                detections.append({
                    "box": (x, y, bw, bh),
                    "confidence": score,
                    "landmarks": landmarks,
                    "raw": face,
                })

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        draw_landmarks: bool = True,
    ) -> np.ndarray:
        """Annotates frame with bounding boxes, confidence badges, and facial landmarks.

        Args:
            frame: Input frame to annotate.
            detections: List of detection dictionaries from detect().
            draw_landmarks: Whether to render 5-point facial landmarks.

        Returns:
            np.ndarray: Annotated frame.
        """
        annotated = frame.copy()

        for idx, det in enumerate(detections):
            box = det["box"]
            conf = det["confidence"]
            draw_styled_box(
                annotated,
                box=box,
                label=f"Face #{idx + 1}",
                color=(0, 255, 128),  # Vibrant mint green
                confidence=conf,
            )

            # Draw 5 landmarks if available
            if draw_landmarks and det["landmarks"] is not None:
                # Landmark colors: Eyes (Cyan), Nose (Yellow), Mouth (Orange)
                landmark_colors = [
                    (255, 255, 0),  # Right eye
                    (255, 255, 0),  # Left eye
                    (0, 255, 255),  # Nose
                    (0, 140, 255),  # Right mouth
                    (0, 140, 255),  # Left mouth
                ]
                for (lx, ly), col in zip(det["landmarks"], landmark_colors):
                    cv2.circle(annotated, (int(lx), int(ly)), 3, col, -1, cv2.LINE_AA)

        # Draw summary counter pill on top left
        count_text = f"Faces Detected: {len(detections)} | Backend: YUNET (Deep Learning)"
        cv2.putText(
            annotated,
            count_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated


def run_live_face_detection(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    max_frames: Optional[int] = None,
) -> None:
    """Runs real-time webcam face detection loop."""
    from src.webcam_test import initialize_camera, draw_hud

    print(f"\n[Starting Face Detection Webcam Stream using YuNet ONNX]...")
    detector = FaceDetector()
    cap = initialize_camera(camera_index)

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

            # Detect faces
            detections = detector.detect(frame)

            # Annotate frame
            annotated = detector.draw_detections(frame, detections)
            annotated = draw_hud(annotated, fps)

            cv2.imshow("VisionAI - Milestone 2: Face Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[User Action] 'q' or 'ESC' pressed. Exiting...")
                break

            if max_frames and frame_count >= max_frames:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[FaceDetector] Camera released and windows closed cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionAI - Milestone 2: Face Detection")
    parser.add_argument("--image", type=str, default=None, help="Optional image path to test on a static photo")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index for live detection")
    parser.add_argument("--no-display", action="store_true", help="Run without opening GUI windows")
    args = parser.parse_args()

    detector = FaceDetector()

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[Error] Image not found at: {image_path}", file=sys.stderr)
            sys.exit(1)
        img = cv2.imread(str(image_path))
        detections = detector.detect(img)
        print(f"\n[Detection Result] Found {len(detections)} face(s) in '{image_path}'.")
        for i, d in enumerate(detections):
            print(f"  Face #{i+1}: Box={d['box']}, Confidence={d['confidence']:.2f}")

        if not args.no_display:
            annotated = detector.draw_detections(img, detections)
            cv2.imshow("Face Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        run_live_face_detection(camera_index=args.camera)


if __name__ == "__main__":
    main()
