"""VisionAI - Milestone 5: Combined Vision Engine with Temporal Smoothing

Unifies Face Recognition and YOLOv8 Object Detection into a cohesive,
high-performance computer vision pipeline.

Key Stability Enhancements:
1. Temporal Identity Tracking & Smoothing:
   Eliminates video jitter and label flickering using Exponential Moving Average (EMA)
   filtering over consecutive frame embeddings.
2. Identity Hysteresis Latching:
   Maintains a recognized identity across blinks, rapid head turns, or momentary lighting drops.
3. Cross-Model Spatial Fusion:
   Associates YOLO 'person' bounding boxes with inner facial identities for unified labeling.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_CAMERA_INDEX, FACE_SIMILARITY_THRESHOLD
from src.face_recognition import FaceRecognizer
from src.object_detection import ObjectDetector, CLASS_COLOR_PALETTE
from src.utils import draw_styled_box, is_box_inside, compute_iou


@dataclass
class VisionResult:
    """Encapsulates all detections from a single processed frame."""
    faces: List[Dict[str, Any]] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    fused_entities: List[Dict[str, Any]] = field(default_factory=list)
    fps: float = 0.0


class FaceTrack:
    """Maintains temporal memory and smoothed embeddings for a single tracked face."""

    def __init__(self, box: Tuple[int, int, int, int], embedding: np.ndarray, name: str, similarity: float) -> None:
        self.box = box
        self.smoothed_embedding = embedding.copy()
        self.name = name
        self.similarity = similarity
        self.frames_unseen = 0
        self.hits = 1
        self.latched_name = name if name != "Unknown" else None
        self.latch_countdown = 8 if name != "Unknown" else 0

    def update(self, box: Tuple[int, int, int, int], raw_embedding: np.ndarray, current_name: str, current_sim: float) -> None:
        self.box = box
        self.frames_unseen = 0
        self.hits += 1

        # Exponential Moving Average (EMA) smoothing of 128-D embedding: 70% history, 30% new
        self.smoothed_embedding = 0.70 * self.smoothed_embedding + 0.30 * raw_embedding
        norm = np.linalg.norm(self.smoothed_embedding)
        if norm > 1e-6:
            self.smoothed_embedding /= norm

        # Update or maintain latch
        if current_name != "Unknown":
            self.name = current_name
            self.similarity = current_sim
            self.latched_name = current_name
            self.latch_countdown = 8  # Hold identity for at least 8 frames during brief drops
        elif self.latch_countdown > 0 and self.latched_name is not None:
            # Maintain latched identity during temporary motion blur or blink
            self.name = self.latched_name
            self.similarity = max(current_sim, 0.40)
            self.latch_countdown -= 1
        else:
            self.name = "Unknown"
            self.similarity = current_sim
            self.latched_name = None


class CombinedVisionEngine:
    """Unified Vision Engine with Temporal Identity Smoothing and YOLO Fusion."""

    def __init__(
        self,
        enable_faces: bool = True,
        enable_objects: bool = True,
        fuse_person_and_face: bool = True,
        similarity_threshold: float = FACE_SIMILARITY_THRESHOLD,
    ) -> None:
        self.enable_faces = enable_faces
        self.enable_objects = enable_objects
        self.fuse_person_and_face = fuse_person_and_face

        print("[VisionEngine] Initializing sub-models with Temporal Smoothing...")
        self.face_recognizer = FaceRecognizer(similarity_threshold=similarity_threshold) if enable_faces else None
        self.object_detector = ObjectDetector(filter_classes=None) if enable_objects else None

        # Temporal face tracking dictionary: {track_id: FaceTrack}
        self.tracks: Dict[int, FaceTrack] = {}
        self._next_track_id: int = 1

        print("[VisionEngine] Combined Vision Engine ready (Detects all 80 COCO objects & animals)!")

    def set_similarity_threshold(self, threshold: float) -> None:
        """Dynamically adjusts the face recognition similarity threshold."""
        if self.face_recognizer:
            self.face_recognizer.similarity_threshold = threshold

    def set_yolo_confidence(self, threshold: float) -> None:
        """Dynamically adjusts the YOLO object/animal confidence threshold."""
        if self.object_detector:
            self.object_detector.set_confidence_threshold(threshold)

    def _update_temporal_tracks(self, frame: np.ndarray, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Matches raw detections to temporal tracks to smooth embeddings and stabilize identities."""
        if not self.face_recognizer:
            return raw_detections

        # Increment unseen counter on all existing tracks
        for t in self.tracks.values():
            t.frames_unseen += 1

        smoothed_results = []
        unmatched_dets = []

        for det in raw_detections:
            raw_face = det["raw"]
            embedding, _ = self.face_recognizer.extractor.extract_embedding(frame, raw_face)
            box = det["box"]

            # Find best matching track via center distance or IoU
            best_track_id = None
            best_iou = 0.25  # Minimum IoU threshold to consider the same face across consecutive frames

            for t_id, track in self.tracks.items():
                iou = compute_iou(box, track.box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = t_id

            if best_track_id is not None:
                track = self.tracks[best_track_id]
                # Identify using the smoothed embedding
                name, sim = self.face_recognizer.identify_face(track.smoothed_embedding)
                track.update(box, embedding, name, sim)
                smoothed_results.append({
                    "box": box,
                    "name": track.name,
                    "similarity": track.similarity,
                    "confidence": det["confidence"],
                    "landmarks": det["landmarks"],
                })
            else:
                # Brand new face
                name, sim = self.face_recognizer.identify_face(embedding)
                new_track = FaceTrack(box, embedding, name, sim)
                self.tracks[self._next_track_id] = new_track
                self._next_track_id += 1
                smoothed_results.append({
                    "box": box,
                    "name": name,
                    "similarity": sim,
                    "confidence": det["confidence"],
                    "landmarks": det["landmarks"],
                })

        # Purge stale tracks unseen for more than 10 frames
        self.tracks = {k: v for k, v in self.tracks.items() if v.frames_unseen <= 10}

        return smoothed_results

    def process_frame(self, frame: np.ndarray) -> VisionResult:
        """Processes a single frame through both detection pipelines with temporal stabilization."""
        if frame is None or frame.size == 0:
            return VisionResult()

        faces = []
        objects = []

        if self.face_recognizer:
            raw_dets = self.face_recognizer.detector.detect(frame)
            faces = self._update_temporal_tracks(frame, raw_dets)

        if self.object_detector:
            objects = self.object_detector.detect(frame)

        fused_entities: List[Dict[str, Any]] = []
        used_face_indices = set()

        if self.fuse_person_and_face and faces and objects:
            for obj in objects:
                if obj["class_name"] == "person":
                    matched_face = None
                    matched_idx = -1

                    for f_idx, face in enumerate(faces):
                        if f_idx in used_face_indices:
                            continue
                        if is_box_inside(face["box"], obj["box"]):
                            matched_face = face
                            matched_idx = f_idx
                            break

                    if matched_face is not None:
                        used_face_indices.add(matched_idx)
                        is_known = matched_face["name"] != "Unknown"
                        fused_entities.append({
                            "type": "person",
                            "label": matched_face["name"] if is_known else "Unregistered Person",
                            "confidence": matched_face["similarity"] if is_known else matched_face["confidence"],
                            "box": obj["box"],
                            "sub_box": matched_face["box"],
                            "is_recognized_face": is_known,
                            "color": (0, 230, 70) if is_known else (0, 140, 255),
                        })
                    else:
                        fused_entities.append({
                            "type": "object",
                            "label": "person",
                            "confidence": obj["confidence"],
                            "box": obj["box"],
                            "sub_box": None,
                            "is_recognized_face": False,
                            "color": (255, 100, 50),
                        })
                else:
                    obj_color = CLASS_COLOR_PALETTE.get(
                        obj["class_name"],
                        ((hash(obj["class_name"]) * 45) % 256, (hash(obj["class_name"]) * 85) % 256, (hash(obj["class_name"]) * 125) % 256),
                    )
                    fused_entities.append({
                        "type": "object",
                        "label": obj["class_name"],
                        "confidence": obj["confidence"],
                        "box": obj["box"],
                        "sub_box": None,
                        "is_recognized_face": False,
                        "color": obj_color,
                    })

            for f_idx, face in enumerate(faces):
                if f_idx not in used_face_indices:
                    is_known = face["name"] != "Unknown"
                    fused_entities.append({
                        "type": "face",
                        "label": face["name"] if is_known else "Unknown Face",
                        "confidence": face["similarity"] if is_known else face["confidence"],
                        "box": face["box"],
                        "sub_box": None,
                        "is_recognized_face": is_known,
                        "color": (0, 230, 70) if is_known else (0, 100, 255),
                    })
        else:
            for face in faces:
                is_known = face["name"] != "Unknown"
                fused_entities.append({
                    "type": "face",
                    "label": face["name"] if is_known else "Unknown Face",
                    "confidence": face["similarity"] if is_known else face["confidence"],
                    "box": face["box"],
                    "sub_box": None,
                    "is_recognized_face": is_known,
                    "color": (0, 230, 70) if is_known else (0, 100, 255),
                })
            for obj in objects:
                obj_color = CLASS_COLOR_PALETTE.get(
                    obj["class_name"],
                    ((hash(obj["class_name"]) * 45) % 256, (hash(obj["class_name"]) * 85) % 256, (hash(obj["class_name"]) * 125) % 256),
                )
                fused_entities.append({
                    "type": "object",
                    "label": obj["class_name"],
                    "confidence": obj["confidence"],
                    "box": obj["box"],
                    "sub_box": None,
                    "is_recognized_face": False,
                    "color": obj_color,
                })

        return VisionResult(faces=faces, objects=objects, fused_entities=fused_entities)

    def draw_results(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """Renders detections with clear, high-contrast badges."""
        annotated = frame.copy()

        for entity in result.fused_entities:
            box = entity["box"]
            label = entity["label"]
            conf = entity["confidence"]
            color = entity["color"]

            if entity["is_recognized_face"]:
                display_label = f"* {label.upper()} [{conf*100:.0f}%]"
            elif "unknown" in label.lower() or "unregistered" in label.lower():
                display_label = f"{label} [{conf*100:.0f}%]"
            else:
                display_label = f"{label.upper()} [{conf*100:.0f}%]"

            draw_styled_box(annotated, box, label=display_label, color=color, confidence=None)

            if entity.get("sub_box") is not None:
                fx, fy, fw, fh = entity["sub_box"]
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (255, 255, 255), 1)

        return annotated


def run_live_vision_engine(camera_index: int = DEFAULT_CAMERA_INDEX, max_frames: Optional[int] = None) -> None:
    """Runs combined live vision stream."""
    from src.webcam_test import initialize_camera, draw_hud

    engine = CombinedVisionEngine()
    cap = initialize_camera(camera_index)

    print("\n[Starting Combined Vision Engine] Detecting faces, people, objects & animals.")
    print("Focus on the window and press 'q' or 'ESC' to exit.")

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

            result = engine.process_frame(frame)
            annotated = engine.draw_results(frame, result)
            annotated = draw_hud(annotated, fps)

            cv2.imshow("VisionAI - Combined Vision Engine", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if max_frames and frame_count >= max_frames:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionAI - Milestone 5: Combined Vision Engine")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index for live detection")
    parser.add_argument("--no-display", action="store_true", help="Run without opening GUI windows")
    args = parser.parse_args()

    engine = CombinedVisionEngine()

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[Error] File not found: {image_path}", file=sys.stderr)
            sys.exit(1)

        img = cv2.imread(str(image_path))
        result = engine.process_frame(img)
        print(f"\n[Combined Vision Results] Detected {len(result.fused_entities)} entity/entities in '{image_path}':")
        for i, ent in enumerate(result.fused_entities):
            print(f"  #{i+1}: Type={ent['type']}, Label='{ent['label']}', Conf={ent['confidence']*100:.1f}%, Box={ent['box']}")

        if not args.no_display:
            annotated = engine.draw_results(img, result)
            cv2.imshow("Combined Vision Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        run_live_vision_engine(camera_index=args.camera)


if __name__ == "__main__":
    main()
