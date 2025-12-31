"""
Face recognition module using InsightFace.
Generates embeddings and performs identity matching using cosine similarity.
"""
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError(
        "insightface package not found. Install it with: pip install insightface"
    )

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    InsightFace-based face recognizer for generating embeddings and matching identities.
    Uses pretrained ArcFace model - no training or fine-tuning.
    """

    def __init__(
        self,
        model_name: str = config.INSIGHTFACE_MODEL_NAME,
        recognition_threshold: float = config.FACE_RECOGNITION_THRESHOLD,
    ):
        """
        Initialize the face recognizer.

        Args:
            model_name: Name of pretrained InsightFace model
            recognition_threshold: Cosine similarity threshold for matching
        """
        self.recognition_threshold = recognition_threshold
        self.known_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info(f"Loading InsightFace model: {model_name}")
        
        try:
            # Initialize FaceAnalysis with the specified model
            self.app = FaceAnalysis(
                name=model_name,
                providers=['CPUExecutionProvider']  # Can use 'CUDAExecutionProvider' if GPU available
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding vector for a face image.

        Args:
            face_image: Face image as numpy array (BGR format)

        Returns:
            512-dimensional embedding vector or None if face not detected
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty or invalid face image provided")
            return None

        try:
            # Detect and analyze faces
            faces = self.app.get(face_image)
            
            if len(faces) == 0:
                logger.warning("No face detected in the provided image")
                return None
            
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected ({len(faces)}), using the first one")
            
            # Get embedding from the first detected face
            embedding = faces[0].embedding
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def compute_cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1, higher = more similar)
        """
        # Embeddings should already be normalized, but double-check
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)

    def load_known_embeddings(self, embeddings_dict: Dict[str, np.ndarray]) -> None:
        """
        Load dictionary of known person embeddings for identification.

        Args:
            embeddings_dict: Dictionary mapping person names to their embeddings
        """
        self.known_embeddings = embeddings_dict
        logger.info(f"Loaded {len(self.known_embeddings)} known identities")

    def identify_face(
        self, face_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Identify a face by comparing its embedding to known embeddings.

        Args:
            face_embedding: Embedding vector of the face to identify

        Returns:
            Tuple of (person_name, confidence_score) or (None, 0.0) if unknown
        """
        if len(self.known_embeddings) == 0:
            logger.warning("No known embeddings loaded")
            return None, 0.0

        best_match_name = None
        best_similarity = 0.0

        # Compare with all known embeddings
        for person_name, known_embedding in self.known_embeddings.items():
            similarity = self.compute_cosine_similarity(face_embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_name = person_name

        # Check if best match exceeds threshold
        if best_similarity >= self.recognition_threshold:
            logger.info(f"Identified as {best_match_name} (confidence: {best_similarity:.3f})")
            return best_match_name, best_similarity
        else:
            logger.info(f"Unknown person (best match: {best_similarity:.3f}, threshold: {self.recognition_threshold})")
            return None, best_similarity

    def aggregate_embeddings(
        self, embeddings: list[np.ndarray], method: str = config.EMBEDDING_AGGREGATION_METHOD
    ) -> Optional[np.ndarray]:
        """
        Aggregate multiple embeddings into a single representative embedding.

        Args:
            embeddings: List of embedding vectors
            method: Aggregation method ('mean' or 'median')

        Returns:
            Aggregated embedding vector or None if input is empty
        """
        if not embeddings:
            logger.warning("No embeddings provided for aggregation")
            return None

        embeddings_array = np.array(embeddings)

        if method == "mean":
            aggregated = np.mean(embeddings_array, axis=0)
        elif method == "median":
            aggregated = np.median(embeddings_array, axis=0)
        else:
            logger.error(f"Unknown aggregation method: {method}")
            return None

        # Normalize the aggregated embedding
        aggregated = aggregated / np.linalg.norm(aggregated)

        logger.info(f"Aggregated {len(embeddings)} embeddings using {method} method")
        return aggregated


if __name__ == "__main__":
    # Test the recognizer
    recognizer = FaceRecognizer()
    print("Face recognizer initialized successfully!")
    print(f"Recognition threshold: {recognizer.recognition_threshold}")
