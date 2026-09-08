"""VisionAI - Interactive Multi-Shot Face Enrollment & Training Tool

Captures multiple diverse facial samples (frontal, slight angles, smiling, expressions)
from the webcam or from an image folder, generates augmented embeddings, and enrolls
the identity into the VisionAI recognition database.

Usage:
    python src/enroll_face.py --name "Priyanka"
    python src/enroll_face.py --name "Alex" --folder "path/to/photos"
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACES_DIR, DEFAULT_CAMERA_INDEX
from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognizer


def capture_webcam_samples(
    person_name: str,
    target_samples: int = 10,
    camera_index: int = DEFAULT_CAMERA_INDEX,
) -> Path:
    """Captures target_samples clear face snapshots from the webcam."""
    clean_name = person_name.strip()
    person_dir = FACES_DIR / clean_name
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}. Check camera connection and privacy permissions.")

    print("\n" + "=" * 60)
    print(f"  Enrolling Face for: '{clean_name}'")
    print("=" * 60)
    print(f"  * Target: Capture {target_samples} clear face angles.")
    print("  * Instructions:")
    print("      - Look straight at the camera.")
    print("      - Tilt your head slightly left, right, up, down, and smile.")
    print("      - Press 'c' to capture a sample, or 'q' to cancel.")
    print("=" * 60 + "\n")

    captured_count = 0
    last_auto_capture_time = time.time()

    try:
        while captured_count < target_samples:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            detections = detector.detect(frame)
            display_frame = frame.copy()

            # Status Banner
            status_text = f"Enrolling: {clean_name} | Samples: {captured_count}/{target_samples}"
            cv2.putText(display_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Press 'c' to capture or wait for auto-capture (every 1.5s)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            has_valid_face = False
            if detections:
                best_face = max(detections, key=lambda d: d["confidence"])
                bx, by, bw, bh = best_face["box"]
                # Green box if good size
                if bw >= 80 and bh >= 80:
                    has_valid_face = True
                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Good Face ({best_face['confidence']*100:.0f}%)", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
                    cv2.putText(display_frame, "Move Closer to Camera", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.imshow(f"VisionAI - Face Enrollment ({clean_name})", display_frame)
            key = cv2.waitKey(1) & 0xFF

            now = time.time()
            # Capture on key 'c' or auto-capture every 1.5s if face is good
            if (key == ord("c") or (has_valid_face and now - last_auto_capture_time >= 1.5)) and has_valid_face:
                captured_count += 1
                sample_path = person_dir / f"{clean_name.lower()}_sample_{captured_count:02d}.jpg"
                cv2.imwrite(str(sample_path), frame)
                print(f"  [Sample {captured_count}/{target_samples}] Saved -> {sample_path.name}")
                last_auto_capture_time = now

                # Visual flash feedback
                flash = np.full_like(display_frame, 255)
                cv2.imshow(f"VisionAI - Face Enrollment ({clean_name})", flash)
                cv2.waitKey(80)

            if key == ord("q") or key == 27:
                print("\n[Canceled] Enrollment aborted by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return person_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionAI - Multi-Shot Face Enrollment & Training")
    parser.add_argument("--name", type=str, required=True, help="Full name of the person to enroll (e.g. 'Priyanka')")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples to capture from camera (default: 8)")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera device index (default: 0)")
    parser.add_argument("--folder", type=str, default=None, help="Optional existing folder of photos instead of webcam")
    args = parser.parse_args()

    clean_name = args.name.strip()
    if not clean_name:
        print("[Error] Please provide a valid person name.", file=sys.stderr)
        sys.exit(1)

    recognizer = FaceRecognizer()

    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"[Error] Folder does not exist: {folder_path}", file=sys.stderr)
            sys.exit(1)
        target_dir = FACES_DIR / clean_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in folder_path.iterdir():
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img = cv2.imread(str(f))
                if img is not None:
                    recognizer.register_face(clean_name, img, use_augmentation=True)
    else:
        # Capture from webcam
        person_dir = capture_webcam_samples(clean_name, target_samples=args.samples, camera_index=args.camera)
        # Train on captured samples
        print(f"\n[Training] Extracting deep augmented embeddings for '{clean_name}'...")
        for sample_file in person_dir.glob("*.jpg"):
            img = cv2.imread(str(sample_file))
            if img is not None:
                recognizer.register_face(clean_name, img, use_augmentation=True)

    print("\n" + "=" * 60)
    print(f"  🎉 SUCCESS: Enrolled '{clean_name}' into VisionAI Database!")
    print(f"  Total embeddings in registry for {clean_name}: {len(recognizer.database.get(clean_name, []))}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
