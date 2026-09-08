"""VisionAI - Milestone 3: Face Embedding Generator

Uses OpenCV's deep neural network SFace (ResNet-like architecture) to:
1. Align detected face crops using 5 facial landmarks (eye line rotation, scale, centering to 112x112).
2. Project the aligned face through the deep neural network into a 128-dimensional embedding space.
3. Normalize the 128-D vector to unit length (L2 norm = 1.0) so that Euclidean and Cosine distances
   can be computed via vector dot products.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SFACE_MODEL_PATH, EMBEDDING_DIMENSION
from src.utils import ensure_face_models


class FaceEmbeddingExtractor:
    """Extracts 128-dimensional deep feature embeddings from faces using SFace."""

    def __init__(self, model_path: Path = SFACE_MODEL_PATH) -> None:
        """Initializes the SFace embedding extractor.

        Args:
            model_path: Path to the SFace ONNX model file.
        """
        self.model_path = model_path
        self.recognizer = None
        self._init_model()

    def _init_model(self) -> None:
        """Loads and initializes the SFace model."""
        ensure_face_models()
        if not self.model_path.exists():
            raise FileNotFoundError(f"SFace ONNX model missing at: {self.model_path}")

        try:
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.model_path),
                config="",
            )
        except Exception:
            self.recognizer = cv2.FaceRecognizerSF_create(
                str(self.model_path),
                "",
            )
        print("[EmbeddingExtractor] SFace 128-D Deep Feature Extractor loaded successfully.")

    def extract_embedding(
        self,
        frame: np.ndarray,
        raw_face_detection: np.ndarray,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aligns, crops, and extracts a 128-D embedding vector from a detected face.

        Args:
            frame: Full BGR image containing the face.
            raw_face_detection: 15-element array from YuNet FaceDetector.
            normalize: Whether to normalize embedding vector to unit length (L2 norm = 1.0).

        Returns:
            Tuple of:
            - embedding: 128-dimensional 1D numpy array of float32.
            - aligned_face: (112, 112, 3) aligned and cropped face image.
        """
        if self.recognizer is None:
            raise RuntimeError("SFace recognizer is not initialized.")

        # SFace alignCrop aligns the face so eyes are horizontal, centered, 112x112
        aligned_face = self.recognizer.alignCrop(frame, raw_face_detection)

        # Extract 128-D feature representation
        raw_feature = self.recognizer.feature(aligned_face).flatten()

        if normalize:
            norm = np.linalg.norm(raw_feature)
            if norm > 1e-6:
                embedding = raw_feature / norm
            else:
                embedding = raw_feature
        else:
            embedding = raw_feature

        return embedding.astype(np.float32), aligned_face
