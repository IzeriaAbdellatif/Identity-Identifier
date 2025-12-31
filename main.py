"""
Main application for real-time face identification system.
Provides CLI for enrollment and real-time identification.
"""
import cv2
import argparse
import logging
from pathlib import Path
from typing import Dict
import numpy as np
from datetime import datetime

import config
import utils
from detector import FaceDetector
from recognizer import FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceIdentificationSystem:
    """
    Complete face identification system with enrollment and recognition capabilities.
    """

    def __init__(self):
        """Initialize the face identification system."""
        logger.info("Initializing Face Identification System...")
        
        # Validate dataset structure
        if not utils.validate_dataset_structure():
            logger.info("Creating dataset structure...")
            utils.create_dataset_structure()
        
        # Initialize detector and recognizer
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        
        logger.info("System initialized successfully")

    def enroll_person(self, person_name: str) -> bool:
        """
        Enroll a person by processing their images and generating embeddings.

        Args:
            person_name: Name of the person to enroll

        Returns:
            True if enrollment successful, False otherwise
        """
        logger.info(f"Starting enrollment for: {person_name}")
        
        person_dir = config.RAW_IMAGES_DIR / person_name
        
        if not person_dir.exists():
            logger.error(f"Person directory does not exist: {person_dir}")
            return False
        
        # Get all images for this person
        image_files = utils.get_image_files(person_dir)
        
        if len(image_files) == 0:
            logger.error(f"No images found for {person_name}")
            return False
        
        logger.info(f"Found {len(image_files)} images for {person_name}")
        
        # Process each image and collect embeddings
        embeddings = []
        faces_cropped_dir = config.FACES_CROPPED_DIR / person_name
        faces_cropped_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, image_path in enumerate(image_files):
            logger.info(f"Processing image {idx + 1}/{len(image_files)}: {image_path.name}")
            
            # Load image
            image = utils.load_image(image_path)
            if image is None:
                continue
            
            # Detect faces
            faces = self.detector.detect_and_crop_faces(image)
            
            if len(faces) == 0:
                logger.warning(f"No face detected in {image_path.name}")
                continue
            
            if len(faces) > 1:
                logger.warning(f"Multiple faces detected in {image_path.name}, using the first one")
            
            # Get the first face
            face_crop, bbox, confidence = faces[0]
            
            # Save cropped face (optional, for inspection)
            crop_path = faces_cropped_dir / f"{image_path.stem}_face.jpg"
            utils.save_image(face_crop, crop_path)
            
            # Generate embedding
            embedding = self.recognizer.get_embedding(face_crop)
            
            if embedding is not None:
                embeddings.append(embedding)
                logger.info(f"Generated embedding from {image_path.name}")
            else:
                logger.warning(f"Failed to generate embedding from {image_path.name}")
        
        if len(embeddings) == 0:
            logger.error(f"No valid embeddings generated for {person_name}")
            return False
        
        logger.info(f"Generated {len(embeddings)} embeddings for {person_name}")
        
        # Aggregate embeddings
        aggregated_embedding = self.recognizer.aggregate_embeddings(embeddings)
        
        if aggregated_embedding is None:
            logger.error(f"Failed to aggregate embeddings for {person_name}")
            return False
        
        # Save aggregated embedding
        embedding_path = config.EMBEDDINGS_DIR / f"{person_name}.npy"
        success = utils.save_embedding(aggregated_embedding, embedding_path)
        
        if success:
            logger.info(f"Successfully enrolled {person_name}")
            
            # Save enrollment log
            enrollment_data = {
                "person_name": person_name,
                "enrollment_date": datetime.now().isoformat(),
                "num_images": len(image_files),
                "num_valid_embeddings": len(embeddings),
                "embedding_path": str(embedding_path)
            }
            log_path = config.EMBEDDINGS_DIR / f"{person_name}_log.json"
            utils.save_enrollment_log(log_path, enrollment_data)
            
            return True
        else:
            return False

    def enroll_all_persons(self) -> Dict[str, bool]:
        """
        Enroll all persons found in the raw_images directory.

        Returns:
            Dictionary mapping person names to enrollment status
        """
        logger.info("Starting batch enrollment...")
        
        person_folders = utils.get_person_folders()
        
        if len(person_folders) == 0:
            logger.error("No person folders found in raw_images directory")
            return {}
        
        results = {}
        
        for person_folder in person_folders:
            person_name = person_folder.name
            success = self.enroll_person(person_name)
            results[person_name] = success
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Enrollment complete: {successful}/{len(results)} persons enrolled successfully")
        
        return results

    def load_known_identities(self) -> int:
        """
        Load all known person embeddings from disk.

        Returns:
            Number of identities loaded
        """
        logger.info("Loading known identities...")
        embeddings = utils.load_all_embeddings()
        self.recognizer.load_known_embeddings(embeddings)
        return len(embeddings)

    def identify_from_webcam(self) -> None:
        """
        Run real-time face identification from webcam.
        Press 'q' to quit.
        """
        logger.info("Starting real-time identification from webcam...")
        
        # Load known identities
        num_identities = self.load_known_identities()
        
        if num_identities == 0:
            logger.error("No known identities loaded. Please run enrollment first.")
            return
        
        logger.info(f"Loaded {num_identities} known identities")
        
        # Initialize video capture
        cap = utils.get_video_capture()
        
        if cap is None:
            logger.error("Failed to initialize video capture")
            return
        
        logger.info("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Detect faces
                faces = self.detector.detect_and_crop_faces(frame)
                
                # Process each detected face
                for face_crop, bbox, det_confidence in faces:
                    # Generate embedding
                    embedding = self.recognizer.get_embedding(face_crop)
                    
                    if embedding is not None:
                        # Identify face
                        person_name, similarity = self.recognizer.identify_face(embedding)
                        
                        if person_name is not None:
                            # Known person
                            label = person_name
                            is_known = True
                        else:
                            # Unknown person
                            label = "Unknown"
                            is_known = False
                        
                        # Draw bounding box and label
                        frame = utils.draw_face_box(
                            frame, bbox, label, similarity, is_known
                        )
                
                # Display frame
                cv2.imshow("Face Identification System", frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quitting...")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released and windows closed")

    def identify_from_image(self, image_path: Path) -> None:
        """
        Identify faces in a single image.

        Args:
            image_path: Path to image file
        """
        logger.info(f"Identifying faces in image: {image_path}")
        
        # Load known identities
        num_identities = self.load_known_identities()
        
        if num_identities == 0:
            logger.error("No known identities loaded. Please run enrollment first.")
            return
        
        # Load image
        image = utils.load_image(image_path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Detect faces
        faces = self.detector.detect_and_crop_faces(image)
        
        if len(faces) == 0:
            logger.warning("No faces detected in image")
            return
        
        logger.info(f"Detected {len(faces)} faces")
        
        # Process each face
        for idx, (face_crop, bbox, det_confidence) in enumerate(faces):
            # Generate embedding
            embedding = self.recognizer.get_embedding(face_crop)
            
            if embedding is not None:
                # Identify face
                person_name, similarity = self.recognizer.identify_face(embedding)
                
                if person_name is not None:
                    label = person_name
                    is_known = True
                else:
                    label = "Unknown"
                    is_known = False
                
                # Draw bounding box and label
                image = utils.draw_face_box(
                    image, bbox, label, similarity, is_known
                )
                
                logger.info(f"Face {idx + 1}: {label} (confidence: {similarity:.3f})")
        
        # Display image
        cv2.imshow("Face Identification", image)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time Face Identification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll all persons from dataset/raw_images/
  python main.py --enroll-all
  
  # Enroll a specific person
  python main.py --enroll person_1
  
  # Run real-time identification from webcam
  python main.py --identify-webcam
  
  # Identify faces in an image
  python main.py --identify-image path/to/image.jpg
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--enroll",
        type=str,
        metavar="PERSON_NAME",
        help="Enroll a specific person by name"
    )
    group.add_argument(
        "--enroll-all",
        action="store_true",
        help="Enroll all persons from raw_images directory"
    )
    group.add_argument(
        "--identify-webcam",
        action="store_true",
        help="Run real-time identification from webcam"
    )
    group.add_argument(
        "--identify-image",
        type=str,
        metavar="IMAGE_PATH",
        help="Identify faces in a single image"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = FaceIdentificationSystem()
    
    # Execute requested operation
    if args.enroll:
        success = system.enroll_person(args.enroll)
        if success:
            print(f"\n✓ Successfully enrolled {args.enroll}")
        else:
            print(f"\n✗ Failed to enroll {args.enroll}")
    
    elif args.enroll_all:
        results = system.enroll_all_persons()
        print("\nEnrollment Results:")
        for person, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {person}")
    
    elif args.identify_webcam:
        system.identify_from_webcam()
    
    elif args.identify_image:
        image_path = Path(args.identify_image)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return
        system.identify_from_image(image_path)


if __name__ == "__main__":
    main()
