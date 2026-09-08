"""VisionAI - Central Configuration Module

Centralizes project-wide paths, model weights, database locations,
runtime parameters, and ML confidence/similarity thresholds.
"""

from pathlib import Path

# Project root directory (VisionAI/)
BASE_DIR: Path = Path(__file__).resolve().parent

# Data directories
DATA_DIR: Path = BASE_DIR / "data"
FACES_DIR: Path = DATA_DIR / "faces"
TEST_IMAGES_DIR: Path = DATA_DIR / "test_images"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ANALYTICS_DIR: Path = DATA_DIR / "analytics"

# Database path
DATABASE_PATH: Path = DATA_DIR / "events.db"

# Models directory
MODELS_DIR: Path = BASE_DIR / "models"
YOLO_MODEL_PATH: Path = MODELS_DIR / "yolov8n.pt"
YUNET_MODEL_PATH: Path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH: Path = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
REGISTERED_EMBEDDINGS_PATH: Path = MODELS_DIR / "registered_faces.pkl"

# Camera & Streaming Defaults
DEFAULT_CAMERA_INDEX: int = 0
DEFAULT_FRAME_WIDTH: int = 640
DEFAULT_FRAME_HEIGHT: int = 480
DEFAULT_FPS: int = 30
DEFAULT_WINDOW_NAME: str = "VisionAI - Live Feed"

# Face Detection Settings
FACE_DETECTION_CONFIDENCE: float = 0.65
FACE_DETECTOR_BACKEND: str = "yunet"  # 'yunet' or 'haar'

# Face Recognition Settings
FACE_SIMILARITY_THRESHOLD: float = 0.363  # Calibrated SFace cosine similarity threshold (0.363)
EMBEDDING_DIMENSION: int = 128

# Object Detection (YOLO) Settings
YOLO_CONFIDENCE_THRESHOLD: float = 0.25  # Standard YOLO confidence threshold (0.25)
YOLO_IOU_THRESHOLD: float = 0.45
DETECT_ALL_CLASSES: bool = True  # Detects all 80 COCO classes (animals, food, vehicles, indoor objects)
DEFAULT_DETECTED_CLASSES: list = [
    # Animals
    "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "teddy bear",
    # People & Electronics
    "person", "cell phone", "laptop", "mouse", "keyboard", "tv", "remote",
    # Everyday Items
    "bottle", "cup", "chair", "book", "backpack", "clock", "scissors"
]

# Database Logging Settings
LOG_COOLDOWN_SECONDS: float = 4.0  # Avoid duplicate entries within cooldown window


def ensure_directories() -> None:
    """Creates all required directories if they do not exist."""
    for directory in (
        DATA_DIR,
        FACES_DIR,
        TEST_IMAGES_DIR,
        PROCESSED_DIR,
        ANALYTICS_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# Automatically ensure directories exist upon import
ensure_directories()
