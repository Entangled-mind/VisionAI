"""VisionAI - Milestone 3: Face Recognition & Registration System

Extends Face Detection from:
    "Is there a face?"
to:
    "Whose face is this?"

Key Engineering Capabilities:
1. Multi-Shot Registration & Data Augmentation:
   Enriches single or few-shot face images with synthetic variations (horizontal flips,
   brightness variations, slight rotations) to build robust embedding clusters in R^128.
2. Calibrated Cosine Similarity Engine:
   Uses calibrated threshold (default: 0.363 for SFace) to accurately distinguish
   known individuals from 'Unknown' faces across diverse webcam lighting and head poses.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FACES_DIR,
    REGISTERED_EMBEDDINGS_PATH,
    FACE_SIMILARITY_THRESHOLD,
    DEFAULT_CAMERA_INDEX,
)
from src.face_detection import FaceDetector
from src.embeddings import FaceEmbeddingExtractor
from src.utils import draw_styled_box


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes Cosine Similarity between two 1D normalized embedding vectors.

    Formula: (A . B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def generate_augmented_variations(image: np.ndarray) -> List[np.ndarray]:
    """Generates synthetic image augmentations to train robust face embeddings.

    Variations created:
    1. Original
    2. Horizontally mirrored (left-to-right symmetry)
    3. Brightness +25 (simulate sunlight/well-lit room)
    4. Brightness -25 (simulate dim room/shadows)
    5. High contrast (alpha=1.2)
    6. Slight clockwise rotation (+6 degrees)
    7. Slight counter-clockwise rotation (-6 degrees)

    Args:
        image: Original BGR image.

    Returns:
        List of augmented images.
    """
    variations = [image]
    h, w = image.shape[:2]

    # 1. Horizontal Flip
    variations.append(cv2.flip(image, 1))

    # 2. Brightness Variations
    bright = cv2.convertScaleAbs(image, alpha=1.0, beta=25)
    dark = cv2.convertScaleAbs(image, alpha=1.0, beta=-25)
    variations.extend([bright, dark])

    # 3. Contrast Variation
    contrast = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
    variations.append(contrast)

    # 4. Slight Rotations (+6 and -6 degrees)
    center = (w // 2, h // 2)
    for angle in [6, -6]:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
        variations.append(rotated)

    return variations


class FaceRecognizer:
    """Face Recognition system matching query faces against a registered database."""

    def __init__(
        self,
        similarity_threshold: float = FACE_SIMILARITY_THRESHOLD,
        db_path: Path = REGISTERED_EMBEDDINGS_PATH,
    ) -> None:
        """Initializes the Face Recognizer.

        Args:
            similarity_threshold: Minimum cosine similarity to accept a match (default: 0.363).
            db_path: Path to serialized pickle database of registered embeddings.
        """
        self.similarity_threshold = similarity_threshold
        self.db_path = db_path
        self.detector = FaceDetector()
        self.extractor = FaceEmbeddingExtractor()

        # Database dictionary: {person_name: List[np.ndarray]}
        self.database: Dict[str, List[np.ndarray]] = {}
        self.load_database()

    def load_database(self) -> None:
        """Loads registered face embeddings from disk if available."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    self.database = pickle.load(f)
                count = sum(len(v) for v in self.database.values())
                print(f"[FaceRecognizer] Loaded database with {len(self.database)} person(s) and {count} total embedding(s).")
            except Exception as exc:
                print(f"[FaceRecognizer Error] Could not read database file: {exc}", file=sys.stderr)
                self.database = {}
        else:
            print(f"[FaceRecognizer] No existing database at {self.db_path}. Initializing empty registry.")

    def save_database(self) -> None:
        """Saves registered face embeddings to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)
        print(f"[FaceRecognizer] Saved database to: {self.db_path}")

    def register_face(
        self,
        person_name: str,
        image: np.ndarray,
        use_augmentation: bool = True,
    ) -> bool:
        """Registers a face under person_name with optional data augmentation.

        Args:
            person_name: Identity name to enroll.
            image: BGR image containing the person's face.
            use_augmentation: Whether to generate 7+ augmented variations (recommended).

        Returns:
            bool: True if face was detected and registered.
        """
        clean_name = person_name.strip()
        if not clean_name:
            print("[Registration Error] Person name cannot be empty.", file=sys.stderr)
            return False

        # Generate variations for robust multi-shot embedding representation
        images_to_process = generate_augmented_variations(image) if use_augmentation else [image]

        valid_embeddings = []
        for img in images_to_process:
            detections = self.detector.detect(img)
            if detections:
                best_face = max(detections, key=lambda d: d["confidence"])
                try:
                    embedding, _ = self.extractor.extract_embedding(img, best_face["raw"])
                    valid_embeddings.append(embedding)
                except Exception as e:
                    continue

        if not valid_embeddings:
            print(f"[Registration Failed] No face detected in photo for '{clean_name}'.")
            return False

        if clean_name not in self.database:
            self.database[clean_name] = []

        self.database[clean_name].extend(valid_embeddings)
        self.save_database()
        print(f"[Registration Success] Enrolled '{clean_name}' with {len(valid_embeddings)} new embeddings (Total: {len(self.database[clean_name])}).")
        return True

    def build_database_from_directory(self, faces_root: Path = FACES_DIR) -> int:
        """Scans data/faces/<person_name>/* directories and trains/enrolls all identities."""
        faces_root.mkdir(parents=True, exist_ok=True)
        total_enrolled = 0

        for person_dir in sorted(faces_root.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            image_paths = [
                p for p in person_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
            ]

            print(f"\n[Training] Enrolling identity '{person_name}' from {len(image_paths)} photo(s)...")
            for img_path in image_paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                if self.register_face(person_name, img, use_augmentation=True):
                    total_enrolled += 1

        print(f"\n[Training Complete] Database contains {len(self.database)} identity(s) and {sum(len(v) for v in self.database.values())} embeddings.")
        return total_enrolled

    def identify_face(self, query_embedding: np.ndarray, top_k: int = 3) -> Tuple[str, float]:
        """Matches a query face embedding using Top-K Ensemble Cosine Similarity.

        Instead of comparing against a single vector, this computes similarity against
        all enrolled vectors for each identity and takes the average of the top-K highest
        similarities. This significantly reduces false alarms from outliers and improves
        live recognition consistency across varying lighting and angles.

        Args:
            query_embedding: 128-D normalized embedding vector.
            top_k: Number of highest similarity scores to average per identity.

        Returns:
            Tuple[str, float]: (matched_name, highest_similarity_score).
        """
        if not self.database:
            return "Unknown", 0.0

        best_name = "Unknown"
        max_similarity = -1.0

        for person_name, embeddings in self.database.items():
            if not embeddings:
                continue

            # Calculate cosine similarities against all enrolled samples for this person
            scores = [compute_cosine_similarity(query_embedding, reg_emb) for reg_emb in embeddings]
            scores.sort(reverse=True)

            # Take average of the top_k highest scores (or all available if < top_k)
            k = min(top_k, len(scores))
            avg_top_k = float(np.mean(scores[:k]))

            if avg_top_k > max_similarity:
                max_similarity = avg_top_k
                best_name = person_name

        # Enforce calibrated recognition threshold
        if max_similarity >= self.similarity_threshold:
            return best_name, max_similarity
        else:
            return "Unknown", max(0.0, max_similarity)

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detects and identifies all faces in a video frame."""
        detections = self.detector.detect(frame)
        results: List[Dict[str, Any]] = []

        for det in detections:
            raw_face = det["raw"]
            embedding, _ = self.extractor.extract_embedding(frame, raw_face)
            name, similarity = self.identify_face(embedding)

            results.append({
                "box": det["box"],
                "name": name,
                "similarity": similarity,
                "confidence": det["confidence"],
                "landmarks": det["landmarks"],
            })

        return results

    def draw_results(self, frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Renders recognized faces with high-contrast colored badges."""
        annotated = frame.copy()

        for res in results:
            box = res["box"]
            name = res["name"]
            sim = res["similarity"]

            if name != "Unknown":
                color = (0, 230, 70)  # Vivid Green for recognized identity
                label = f"{name} [{sim * 100:.1f}%]"
            else:
                color = (0, 100, 255)  # Orange-Red for Unknown
                label = f"Unknown [{sim * 100:.1f}%]"

            draw_styled_box(annotated, box, label=label, color=color, confidence=None)

            # Draw landmarks
            if res["landmarks"] is not None:
                for (lx, ly) in res["landmarks"]:
                    cv2.circle(annotated, (int(lx), int(ly)), 3, color, -1, cv2.LINE_AA)

        return annotated


def run_live_recognition(camera_index: int = DEFAULT_CAMERA_INDEX, max_frames: Optional[int] = None) -> None:
    """Runs real-time webcam face recognition loop."""
    from src.webcam_test import initialize_camera, draw_hud

    recognizer = FaceRecognizer()
    cap = initialize_camera(camera_index)

    print("\n[Starting Live Face Recognition] Focus on the window and press 'q' or 'ESC' to exit.")

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

            results = recognizer.process_frame(frame)
            annotated = recognizer.draw_results(frame, results)
            annotated = draw_hud(annotated, fps)

            cv2.imshow("VisionAI - Live Face Recognition", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if max_frames and frame_count >= max_frames:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionAI - Face Recognition & Training")
    parser.add_argument("--register-all", action="store_true", help="Scans data/faces/ and trains on all identities")
    parser.add_argument("--register-person", type=str, default=None, help="Name of person to register")
    parser.add_argument("--register-image", type=str, default=None, help="Image file path of person to register")
    parser.add_argument("--thresh", type=float, default=FACE_SIMILARITY_THRESHOLD, help="Cosine similarity threshold (default: 0.363)")
    parser.add_argument("--image", type=str, default=None, help="Run recognition on a static image file")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index for live recognition")
    parser.add_argument("--no-display", action="store_true", help="Run without opening GUI windows")
    args = parser.parse_args()

    recognizer = FaceRecognizer(similarity_threshold=args.thresh)

    if args.register_all:
        recognizer.build_database_from_directory()
        return

    if args.register_person and args.register_image:
        img = cv2.imread(args.register_image)
        if img is None:
            print(f"[Error] Could not load image: {args.register_image}", file=sys.stderr)
            sys.exit(1)
        recognizer.register_face(args.register_person, img, use_augmentation=True)
        return

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"[Error] Could not load image: {args.image}", file=sys.stderr)
            sys.exit(1)
        results = recognizer.process_frame(img)
        print(f"\n[Recognition Results] Found {len(results)} face(s):")
        for idx, res in enumerate(results):
            print(f"  Face #{idx + 1}: Name='{res['name']}', Similarity={res['similarity']*100:.1f}%, Box={res['box']}")

        if not args.no_display:
            annotated = recognizer.draw_results(img, results)
            cv2.imshow("Face Recognition Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        run_live_recognition(camera_index=args.camera)


if __name__ == "__main__":
    main()
