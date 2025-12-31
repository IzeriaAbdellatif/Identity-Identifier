"""
Face detection module using YOLOv8.
Provides pretrained face detection without any training or fine-tuning.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics package not found. Install it with: pip install ultralytics"
    )

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    YOLOv8-based face detector for identifying face bounding boxes in images.
    Uses pretrained weights only - no training or fine-tuning.
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.YOLO_IOU_THRESHOLD,
    ):
        """
        Initialize the face detector.

        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for NMS
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        logger.info(f"Loading YOLOv8 face detection model from {model_path}")
        
        try:
            # Try to load the model (will download if not present)
            self.model = YOLO(model_path)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load {model_path}: {e}")
            logger.info("Falling back to standard YOLOv8n model")
            # Fallback to standard YOLOv8n if face-specific model not available
            self.model = YOLO("yolov8n.pt")

    def detect_faces(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of tuples (x1, y1, x2, y2, confidence) for each detected face
        """
        if image is None or image.size == 0:
            logger.warning("Empty or invalid image provided")
            return []

        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        faces = []
        
        # Extract bounding boxes
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Validate bounding box
                    if x2 > x1 and y2 > y1:
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Filter out very small faces
                        if width >= config.MIN_FACE_SIZE and height >= config.MIN_FACE_SIZE:
                            faces.append((x1, y1, x2, y2, float(conf)))

        logger.info(f"Detected {len(faces)} faces in image")
        return faces

    def crop_face(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.2,
    ) -> Optional[np.ndarray]:
        """
        Crop face from image with optional margin.

        Args:
            image: Input image
            bbox: Bounding box as (x1, y1, x2, y2)
            margin: Margin to add around face (as fraction of face size)

        Returns:
            Cropped face image or None if invalid
        """
        if image is None or image.size == 0:
            return None

        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Calculate margin
        face_width = x2 - x1
        face_height = y2 - y1
        margin_x = int(face_width * margin)
        margin_y = int(face_height * margin)

        # Apply margin with boundary checks
        x1_crop = max(0, x1 - margin_x)
        y1_crop = max(0, y1 - margin_y)
        x2_crop = min(w, x2 + margin_x)
        y2_crop = min(h, y2 + margin_y)

        # Crop face
        face_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]

        if face_crop.size == 0:
            logger.warning("Cropped face is empty")
            return None

        return face_crop

    def detect_and_crop_faces(
        self, image: np.ndarray, margin: float = 0.2
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detect faces and return cropped face images.

        Args:
            image: Input image
            margin: Margin around face crops

        Returns:
            List of tuples (cropped_face, bbox, confidence)
        """
        faces = self.detect_faces(image)
        results = []

        for *bbox, confidence in faces:
            face_crop = self.crop_face(image, tuple(bbox), margin)
            if face_crop is not None:
                results.append((face_crop, tuple(bbox), confidence))

        return results


if __name__ == "__main__":
    # Test the detector
    detector = FaceDetector()
    print("Face detector initialized successfully!")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print(f"IoU threshold: {detector.iou_threshold}")
