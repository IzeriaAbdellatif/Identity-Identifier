"""
Utility functions for the face identification system.
Handles file I/O, image processing, and embedding management.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load an image from disk.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: Path) -> bool:
    """
    Save an image to disk.

    Args:
        image: Image as numpy array
        output_path: Path to save image

    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return True
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False


def get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files from a directory.

    Args:
        directory: Directory path

    Returns:
        List of image file paths
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    image_files = []
    for ext in config.SUPPORTED_IMAGE_FORMATS:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    logger.info(f"Found {len(image_files)} images in {directory}")
    return sorted(image_files)


def save_embedding(embedding: np.ndarray, output_path: Path) -> bool:
    """
    Save an embedding vector to disk.

    Args:
        embedding: Embedding vector as numpy array
        output_path: Path to save embedding (.npy file)

    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), embedding)
        logger.info(f"Saved embedding to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving embedding to {output_path}: {e}")
        return False


def load_embedding(embedding_path: Path) -> Optional[np.ndarray]:
    """
    Load an embedding vector from disk.

    Args:
        embedding_path: Path to embedding file (.npy)

    Returns:
        Embedding vector or None if failed
    """
    try:
        embedding = np.load(str(embedding_path))
        logger.info(f"Loaded embedding from {embedding_path}")
        return embedding
    except Exception as e:
        logger.error(f"Error loading embedding from {embedding_path}: {e}")
        return None


def load_all_embeddings(embeddings_dir: Path = config.EMBEDDINGS_DIR) -> Dict[str, np.ndarray]:
    """
    Load all embeddings from the embeddings directory.

    Args:
        embeddings_dir: Directory containing embedding files

    Returns:
        Dictionary mapping person names to their embeddings
    """
    embeddings = {}

    if not embeddings_dir.exists():
        logger.warning(f"Embeddings directory does not exist: {embeddings_dir}")
        return embeddings

    for embedding_file in embeddings_dir.glob("*.npy"):
        person_name = embedding_file.stem
        embedding = load_embedding(embedding_file)
        if embedding is not None:
            embeddings[person_name] = embedding

    logger.info(f"Loaded {len(embeddings)} person embeddings")
    return embeddings


def get_person_folders(raw_images_dir: Path = config.RAW_IMAGES_DIR) -> List[Path]:
    """
    Get all person folders from raw images directory.

    Args:
        raw_images_dir: Root directory containing person folders

    Returns:
        List of person folder paths
    """
    if not raw_images_dir.exists():
        logger.warning(f"Raw images directory does not exist: {raw_images_dir}")
        return []

    person_folders = [d for d in raw_images_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(person_folders)} person folders")
    return sorted(person_folders)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to target size.

    Args:
        image: Input image
        target_size: Target size as (width, height)

    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def draw_face_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    is_known: bool = True,
) -> np.ndarray:
    """
    Draw bounding box and label on image.

    Args:
        image: Input image
        bbox: Bounding box as (x1, y1, x2, y2)
        label: Text label to display
        confidence: Confidence score
        is_known: Whether the person is known (affects color)

    Returns:
        Image with drawn annotations
    """
    x1, y1, x2, y2 = bbox
    
    # Choose color based on whether person is known
    color = config.BBOX_COLOR_KNOWN if is_known else config.BBOX_COLOR_UNKNOWN
    
    # Draw bounding box
    cv2.rectangle(
        image, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS
    )
    
    # Prepare label text
    text = f"{label}: {confidence:.2f}"
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        image,
        (x1, y1 - text_height - baseline - 10),
        (x1 + text_width + 10, y1),
        color,
        -1,
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        (255, 255, 255),
        config.FONT_THICKNESS,
    )
    
    return image


def validate_dataset_structure() -> bool:
    """
    Validate that the dataset directory structure exists.

    Returns:
        True if structure is valid, False otherwise
    """
    required_dirs = [
        config.RAW_IMAGES_DIR,
        config.FACES_CROPPED_DIR,
        config.EMBEDDINGS_DIR,
    ]
    
    all_exist = True
    for directory in required_dirs:
        if not directory.exists():
            logger.error(f"Required directory missing: {directory}")
            all_exist = False
        else:
            logger.info(f"Directory exists: {directory}")
    
    return all_exist


def create_dataset_structure() -> None:
    """
    Create the dataset directory structure if it doesn't exist.
    """
    directories = [
        config.RAW_IMAGES_DIR,
        config.FACES_CROPPED_DIR,
        config.EMBEDDINGS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")


def save_enrollment_log(log_path: Path, enrollment_data: Dict) -> bool:
    """
    Save enrollment log with metadata.

    Args:
        log_path: Path to log file
        enrollment_data: Dictionary containing enrollment metadata

    Returns:
        True if successful, False otherwise
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(enrollment_data, f, indent=2)
        logger.info(f"Saved enrollment log to {log_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving enrollment log: {e}")
        return False


def get_video_capture(source: int = config.VIDEO_SOURCE) -> Optional[cv2.VideoCapture]:
    """
    Initialize video capture from camera or video file.

    Args:
        source: Video source (0 for webcam, or path to video file)

    Returns:
        VideoCapture object or None if failed
    """
    try:
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return None
        
        # Set resolution if using webcam
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
        logger.info(f"Video capture initialized from source: {source}")
        return cap
    except Exception as e:
        logger.error(f"Error initializing video capture: {e}")
        return None


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    create_dataset_structure()
    is_valid = validate_dataset_structure()
    print(f"Dataset structure valid: {is_valid}")
    
    person_folders = get_person_folders()
    print(f"Found {len(person_folders)} person folders")
