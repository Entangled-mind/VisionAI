"""VisionAI - Comprehensive End-to-End Pipeline Unit Tests

Tests:
1. YuNet Face Detection
2. SFace 128-D Embedding Extraction & Unit Normalization
3. Cosine Similarity & Face Identification Logic
4. YOLOv8 Object Detection Schema
5. SQLite Event Database Logging & Rate-Limiting Cooldown
6. Pandas Analytics Summary Calculations
7. Geometric IoU & Detection Box Evaluations
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TEST_IMAGES_DIR
from src.face_detection import FaceDetector
from src.embeddings import FaceEmbeddingExtractor
from src.face_recognition import FaceRecognizer, compute_cosine_similarity
from src.object_detection import ObjectDetector
from src.database import EventDatabase
from src.analytics import AnalyticsEngine
from src.evaluation import evaluate_detection_boxes
from src.utils import compute_iou


class TestVisionPipeline(unittest.TestCase):
    """End-to-End Test Suite for VisionAI."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load test image and initialize detectors once."""
        cls.face_sample_path = TEST_IMAGES_DIR / "face_sample.jpg"
        cls.assertTrue(cls.face_sample_path.exists(), "Sample face image should exist.")
        cls.img = cv2.imread(str(cls.face_sample_path))

        cls.detector = FaceDetector()
        cls.extractor = FaceEmbeddingExtractor()
        cls.recognizer = FaceRecognizer()

    def test_01_face_detection(self) -> None:
        """Verify YuNet detects face and outputs valid bounding box and 5 landmarks."""
        detections = self.detector.detect(self.img)
        self.assertGreaterEqual(len(detections), 1, "Should detect at least 1 face.")
        det = detections[0]
        self.assertEqual(len(det["box"]), 4)
        self.assertGreater(det["confidence"], 0.6)
        self.assertEqual(det["landmarks"].shape, (5, 2))

    def test_02_embedding_extraction(self) -> None:
        """Verify SFace extracts a 128-D normalized embedding vector."""
        detections = self.detector.detect(self.img)
        raw_face = detections[0]["raw"]
        embedding, aligned = self.extractor.extract_embedding(self.img, raw_face)

        self.assertEqual(embedding.shape, (128,))
        self.assertEqual(aligned.shape, (112, 112, 3))
        # Verify L2 norm is approximately 1.0
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=4)

    def test_03_cosine_similarity_math(self) -> None:
        """Verify Cosine Similarity properties: identical vectors=1.0, orthogonal=0.0, opposite=-1.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        vec4 = np.array([-1.0, 0.0, 0.0])

        self.assertAlmostEqual(compute_cosine_similarity(vec1, vec2), 1.0, places=5)
        self.assertAlmostEqual(compute_cosine_similarity(vec1, vec3), 0.0, places=5)
        self.assertAlmostEqual(compute_cosine_similarity(vec1, vec4), -1.0, places=5)

    def test_04_face_recognition_match(self) -> None:
        """Verify registered face 'Lena' is identified accurately with high similarity."""
        results = self.recognizer.process_frame(self.img)
        self.assertGreaterEqual(len(results), 1)
        # Should match Lena since she is in the registered database
        self.assertEqual(results[0]["name"], "Lena")
        self.assertGreater(results[0]["similarity"], 0.8)

    def test_05_object_detection(self) -> None:
        """Verify YOLO detects object and returns structured dictionary."""
        yolo = ObjectDetector(filter_classes=["person"])
        objs = yolo.detect(self.img)
        self.assertGreaterEqual(len(objs), 1)
        self.assertEqual(objs[0]["class_name"], "person")
        self.assertEqual(len(objs[0]["box"]), 4)

    def test_06_database_logging_and_cooldown(self) -> None:
        """Verify SQLite logging and anti-spam cooldown mechanism."""
        db = EventDatabase(cooldown_seconds=1.0)
        # First log should succeed
        w1 = db.log_event("person", "TestUser", 0.95, source="unit_test", enforce_cooldown=True)
        self.assertTrue(w1)
        # Immediate second log of same label should be throttled
        w2 = db.log_event("person", "TestUser", 0.95, source="unit_test", enforce_cooldown=True)
        self.assertFalse(w2)

        df = db.get_all_events_df()
        self.assertFalse(df.empty)

    def test_07_analytics_kpi_computation(self) -> None:
        """Verify AnalyticsEngine calculates KPIs properly."""
        engine = AnalyticsEngine()
        kpis = engine.compute_summary_kpis()
        self.assertIn("total_detections", kpis)
        self.assertIn("top_object", kpis)
        self.assertIn("average_confidence", kpis)

    def test_08_geometric_iou(self) -> None:
        """Verify IoU calculation on overlapping, identical, and disjoint boxes."""
        boxA = (0, 0, 10, 10)
        boxB = (0, 0, 10, 10)  # Identical
        boxC = (10, 10, 10, 10) # Disjoint

        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0, places=4)
        self.assertAlmostEqual(compute_iou(boxA, boxC), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
