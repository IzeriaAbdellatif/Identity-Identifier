"""
Test script to verify system setup and basic functionality.
Run this to check if all components are working correctly.
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    try:
        import cv2
        print("  ✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"  ✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✓ NumPy imported successfully")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("  ✓ Ultralytics (YOLOv8) imported successfully")
    except ImportError as e:
        print(f"  ✗ Ultralytics import failed: {e}")
        print("  Install with: pip install ultralytics")
        return False
    
    try:
        import insightface
        print("  ✓ InsightFace imported successfully")
    except ImportError as e:
        print(f"  ✗ InsightFace import failed: {e}")
        print("  Install with: pip install insightface")
        return False
    
    try:
        import torch
        print(f"  ✓ PyTorch imported successfully (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ℹ CUDA not available, will use CPU")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    return True


def test_modules():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    try:
        import config
        print("  ✓ config.py imported successfully")
    except Exception as e:
        print(f"  ✗ config.py import failed: {e}")
        return False
    
    try:
        import utils
        print("  ✓ utils.py imported successfully")
    except Exception as e:
        print(f"  ✗ utils.py import failed: {e}")
        return False
    
    try:
        from detector import FaceDetector
        print("  ✓ detector.py imported successfully")
    except Exception as e:
        print(f"  ✗ detector.py import failed: {e}")
        return False
    
    try:
        from recognizer import FaceRecognizer
        print("  ✓ recognizer.py imported successfully")
    except Exception as e:
        print(f"  ✗ recognizer.py import failed: {e}")
        return False
    
    return True


def test_dataset_structure():
    """Test if dataset directory structure exists."""
    print("\nTesting dataset structure...")
    import config
    
    required_dirs = [
        config.RAW_IMAGES_DIR,
        config.FACES_CROPPED_DIR,
        config.EMBEDDINGS_DIR,
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists():
            print(f"  ✓ {directory.relative_to(config.PROJECT_ROOT)}")
        else:
            print(f"  ✗ {directory.relative_to(config.PROJECT_ROOT)} (missing)")
            all_exist = False
    
    # Check person folders
    person_folders = list(config.RAW_IMAGES_DIR.glob("person_*"))
    print(f"\n  Found {len(person_folders)} person folders:")
    for folder in person_folders:
        image_count = len(list(folder.glob("*.[jJ][pP][gG]"))) + \
                     len(list(folder.glob("*.[jJ][pP][eE][gG]"))) + \
                     len(list(folder.glob("*.[pP][nN][gG]")))
        print(f"    - {folder.name}: {image_count} images")
    
    return all_exist


def test_detector():
    """Test face detector initialization."""
    print("\nTesting face detector...")
    try:
        from detector import FaceDetector
        detector = FaceDetector()
        print("  ✓ Face detector initialized successfully")
        print(f"  ℹ Confidence threshold: {detector.confidence_threshold}")
        return True
    except Exception as e:
        print(f"  ✗ Face detector initialization failed: {e}")
        return False


def test_recognizer():
    """Test face recognizer initialization."""
    print("\nTesting face recognizer...")
    try:
        from recognizer import FaceRecognizer
        recognizer = FaceRecognizer()
        print("  ✓ Face recognizer initialized successfully")
        print(f"  ℹ Recognition threshold: {recognizer.recognition_threshold}")
        return True
    except Exception as e:
        print(f"  ✗ Face recognizer initialization failed: {e}")
        print("  Note: First run will download models (~200 MB)")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Face Identification System - Setup Test")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Package imports", test_imports()))
    
    # Test project modules
    results.append(("Project modules", test_modules()))
    
    # Test dataset structure
    results.append(("Dataset structure", test_dataset_structure()))
    
    # Test detector (will download model if needed)
    results.append(("Face detector", test_detector()))
    
    # Test recognizer (will download model if needed)
    results.append(("Face recognizer", test_recognizer()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Add photos to dataset/raw_images/person_*/")
        print("2. Run: python main.py --enroll-all")
        print("3. Run: python main.py --identify-webcam")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check internet connection (models download on first run)")
        print("- Verify dataset directory structure")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
