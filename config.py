"""
Configuration file for face identification system.
Contains all model paths, thresholds, and system parameters.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
RAW_IMAGES_DIR = DATASET_ROOT / "raw_images"
FACES_CROPPED_DIR = DATASET_ROOT / "faces_cropped"
EMBEDDINGS_DIR = DATASET_ROOT / "embeddings"

# YOLOv8 Face Detection Configuration
YOLO_MODEL_PATH = "yolov8n-face.pt"  # Will be downloaded automatically
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45

# Face Recognition Configuration
INSIGHTFACE_MODEL_NAME = "buffalo_l"  # High-quality pretrained model
FACE_RECOGNITION_THRESHOLD = 0.6  # Cosine similarity threshold (0-1)
# Higher threshold = stricter matching (fewer false positives)
# Lower threshold = looser matching (more false positives)

# Image preprocessing
TARGET_FACE_SIZE = (112, 112)  # Standard size for InsightFace models
MIN_FACE_SIZE = 20  # Minimum face size in pixels

# Visualization
BBOX_COLOR_KNOWN = (0, 255, 0)  # Green for known faces
BBOX_COLOR_UNKNOWN = (0, 0, 255)  # Red for unknown faces
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Video capture
VIDEO_SOURCE = 0  # 0 for default webcam, or path to video file
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30

# Embedding aggregation
EMBEDDING_AGGREGATION_METHOD = "mean"  # "mean" or "median"

# File formats
EMBEDDING_FILE_FORMAT = "npy"  # NumPy binary format
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
