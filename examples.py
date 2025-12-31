"""
Example script demonstrating API usage.
Shows how to use the system programmatically without the CLI.
"""
import cv2
import numpy as np
from pathlib import Path

from detector import FaceDetector
from recognizer import FaceRecognizer
import utils
import config


def example_enrollment():
    """Example: Enroll a person programmatically."""
    print("=== Enrollment Example ===\n")
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Person to enroll
    person_name = "person_1"
    person_dir = config.RAW_IMAGES_DIR / person_name
    
    # Get images
    image_files = utils.get_image_files(person_dir)
    print(f"Found {len(image_files)} images for {person_name}")
    
    # Process images and collect embeddings
    embeddings = []
    
    for image_path in image_files:
        # Load image
        image = utils.load_image(image_path)
        if image is None:
            continue
        
        # Detect and crop faces
        faces = detector.detect_and_crop_faces(image)
        
        if len(faces) > 0:
            face_crop, bbox, confidence = faces[0]
            
            # Generate embedding
            embedding = recognizer.get_embedding(face_crop)
            
            if embedding is not None:
                embeddings.append(embedding)
                print(f"  ✓ Processed {image_path.name}")
    
    # Aggregate embeddings
    if embeddings:
        aggregated = recognizer.aggregate_embeddings(embeddings)
        
        # Save embedding
        embedding_path = config.EMBEDDINGS_DIR / f"{person_name}.npy"
        utils.save_embedding(aggregated, embedding_path)
        
        print(f"\n✓ Enrolled {person_name} with {len(embeddings)} embeddings")
    else:
        print(f"\n✗ No valid embeddings for {person_name}")


def example_identification_image():
    """Example: Identify faces in a single image."""
    print("\n=== Image Identification Example ===\n")
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Load known embeddings
    embeddings = utils.load_all_embeddings()
    recognizer.load_known_embeddings(embeddings)
    
    print(f"Loaded {len(embeddings)} known identities")
    
    # Load test image (replace with your image path)
    test_image_path = config.RAW_IMAGES_DIR / "person_1" / "test.jpg"
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        print("Please provide a valid image path")
        return
    
    image = utils.load_image(test_image_path)
    
    # Detect faces
    faces = detector.detect_and_crop_faces(image)
    print(f"Detected {len(faces)} faces")
    
    # Identify each face
    for idx, (face_crop, bbox, det_conf) in enumerate(faces):
        # Generate embedding
        embedding = recognizer.get_embedding(face_crop)
        
        if embedding is not None:
            # Identify
            person_name, similarity = recognizer.identify_face(embedding)
            
            if person_name:
                print(f"  Face {idx + 1}: {person_name} (similarity: {similarity:.3f})")
                is_known = True
            else:
                print(f"  Face {idx + 1}: Unknown (best: {similarity:.3f})")
                is_known = False
            
            # Draw on image
            label = person_name if person_name else "Unknown"
            image = utils.draw_face_box(image, bbox, label, similarity, is_known)
    
    # Show result
    cv2.imshow("Identification Result", image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_webcam_identification():
    """Example: Real-time identification from webcam."""
    print("\n=== Webcam Identification Example ===\n")
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Load known embeddings
    embeddings = utils.load_all_embeddings()
    recognizer.load_known_embeddings(embeddings)
    
    print(f"Loaded {len(embeddings)} known identities")
    
    # Initialize webcam
    cap = utils.get_video_capture()
    
    if cap is None:
        print("Failed to open webcam")
        return
    
    print("Press 'q' to quit\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process every frame (or skip frames for speed)
            if frame_count % 1 == 0:  # Process every frame
                # Detect faces
                faces = detector.detect_and_crop_faces(frame)
                
                # Identify each face
                for face_crop, bbox, det_conf in faces:
                    # Generate embedding
                    embedding = recognizer.get_embedding(face_crop)
                    
                    if embedding is not None:
                        # Identify
                        person_name, similarity = recognizer.identify_face(embedding)
                        
                        if person_name:
                            label = person_name
                            is_known = True
                        else:
                            label = "Unknown"
                            is_known = False
                        
                        # Draw on frame
                        frame = utils.draw_face_box(
                            frame, bbox, label, similarity, is_known
                        )
            
            # Display
            cv2.imshow("Webcam Identification", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")


def example_cosine_similarity():
    """Example: Direct embedding comparison."""
    print("\n=== Cosine Similarity Example ===\n")
    
    recognizer = FaceRecognizer()
    
    # Create two random normalized embeddings (for demonstration)
    embedding1 = np.random.randn(512)
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    
    embedding2 = np.random.randn(512)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Compute similarity
    similarity = recognizer.compute_cosine_similarity(embedding1, embedding2)
    
    print(f"Similarity between two random embeddings: {similarity:.4f}")
    
    # Self-similarity (should be 1.0)
    self_similarity = recognizer.compute_cosine_similarity(embedding1, embedding1)
    print(f"Self-similarity: {self_similarity:.4f} (should be 1.0)")


def main():
    """Run examples."""
    print("=" * 60)
    print("Face Identification System - API Examples")
    print("=" * 60)
    
    # Check if system is ready
    if not utils.validate_dataset_structure():
        print("\n⚠ Dataset structure not found. Creating...")
        utils.create_dataset_structure()
    
    # Run examples (comment out what you don't need)
    
    # Example 1: Enroll a person
    # example_enrollment()
    
    # Example 2: Identify from image
    # example_identification_image()
    
    # Example 3: Webcam identification
    # example_webcam_identification()
    
    # Example 4: Cosine similarity
    example_cosine_similarity()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    
    print("\nTo run specific examples, uncomment them in the main() function.")


if __name__ == "__main__":
    main()
