# Face Identification System

A complete real-time face identification system using pretrained deep learning models. This system can identify 3+ known people using YOLOv8 for face detection and InsightFace (ArcFace) for face recognition. **No training or fine-tuning required** - just add photos and run!

## Features

- ✅ **Pretrained Models Only**: Uses YOLOv8 for detection and InsightFace for recognition (no training needed)
- ✅ **Real-time Identification**: Webcam-based real-time face identification
- ✅ **Easy Person Enrollment**: Add new people by simply adding photos to folders
- ✅ **Cosine Similarity Matching**: Robust face matching using embedding similarity
- ✅ **Modular Architecture**: Clean, well-documented code with type hints
- ✅ **Confidence Scores**: Shows identification confidence for each detection
- ✅ **Unknown Face Detection**: Identifies unfamiliar faces as "Unknown"

## Project Structure

```
Identity-Identifier/
├── main.py                      # Main application with CLI
├── detector.py                  # YOLOv8 face detector
├── recognizer.py                # InsightFace recognition engine
├── utils.py                     # Utility functions
├── config.py                    # Configuration and parameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── dataset/
    ├── raw_images/              # Add person photos here!
    │   ├── person_1/           # Folder for person 1
    │   ├── person_2/           # Folder for person 2
    │   └── person_3/           # Folder for person 3
    ├── faces_cropped/          # Auto-generated cropped faces (cache)
    └── embeddings/             # Auto-generated embeddings (cache)
```

## Installation

### 1. Clone or Download Project

```bash
cd Identity-Identifier
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download pretrained models automatically:
- YOLOv8 face detection model (~6 MB)
- InsightFace buffalo_l model (~200 MB)

## Quick Start

### Step 1: Add Person Photos

Add 3-10 photos for each person you want to identify:

```bash
# Add photos for person_1
dataset/raw_images/person_1/
  ├── photo1.jpg
  ├── photo2.jpg
  └── photo3.jpg

# Add photos for person_2
dataset/raw_images/person_2/
  ├── image1.jpg
  └── image2.jpg

# Add photos for person_3
dataset/raw_images/person_3/
  ├── pic1.jpg
  ├── pic2.jpg
  └── pic3.jpg
```

**Photo Guidelines**:
- ✅ Use 3-10 photos per person (more is better)
- ✅ Include different angles, expressions, and lighting
- ✅ Ensure face is clearly visible
- ✅ Use good quality images (not blurry)
- ✅ Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Step 2: Enroll All Persons

Process all photos and generate embeddings:

```bash
python main.py --enroll-all
```

This will:
1. Load all images from `dataset/raw_images/`
2. Detect faces using YOLOv8
3. Generate embeddings using InsightFace
4. Aggregate embeddings per person (mean)
5. Save to `dataset/embeddings/`

### Step 3: Run Real-time Identification

Start webcam identification:

```bash
python main.py --identify-webcam
```

- Press **'q'** to quit
- Green box = Known person
- Red box = Unknown person
- Shows name and confidence score

## Usage Guide

### Enroll All Persons

Process all person folders:

```bash
python main.py --enroll-all
```

### Enroll Specific Person

Process a single person:

```bash
python main.py --enroll person_1
```

### Real-time Identification (Webcam)

Run live face identification:

```bash
python main.py --identify-webcam
```

### Identify Faces in Image

Process a single image file:

```bash
python main.py --identify-image path/to/image.jpg
```

## Adding New People

To add a new person to the system:

1. **Create folder** with person's name:
   ```bash
   mkdir dataset/raw_images/john_doe
   ```

2. **Add photos** (3-10 images recommended):
   ```bash
   # Copy photos to the folder
   cp photo*.jpg dataset/raw_images/john_doe/
   ```

3. **Enroll the person**:
   ```bash
   python main.py --enroll john_doe
   ```

That's it! The system will now recognize this person.

## Configuration

Edit `config.py` to customize system behavior:

### Detection Parameters

```python
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Face detection confidence (0-1)
MIN_FACE_SIZE = 20               # Minimum face size in pixels
```

### Recognition Parameters

```python
FACE_RECOGNITION_THRESHOLD = 0.6  # Similarity threshold (0-1)
# Higher = stricter (fewer false positives)
# Lower = looser (more false positives)
```

### Video Settings

```python
VIDEO_SOURCE = 0          # 0 = default webcam
VIDEO_WIDTH = 1280        # Frame width
VIDEO_HEIGHT = 720        # Frame height
```

## How It Works

### 1. Face Detection (YOLOv8)
- Detects faces in images/video frames
- Returns bounding boxes for each face
- Pretrained on face detection datasets

### 2. Face Recognition (InsightFace)
- Generates 512-dimensional embeddings for each face
- Uses ArcFace model (state-of-the-art face recognition)
- Pretrained on millions of face images

### 3. Identification (Cosine Similarity)
- Compares query embedding with known embeddings
- Uses cosine similarity (dot product of normalized vectors)
- Returns best match if similarity > threshold
- Otherwise, labels as "Unknown"

### 4. Embedding Aggregation
- Multiple photos per person → multiple embeddings
- Aggregates using mean (can change to median in config)
- Creates robust representative embedding per person

## Troubleshooting

### Issue: "No face detected"
- Ensure face is clearly visible and well-lit
- Try images with larger, frontal faces
- Check if face is too small (< 20 pixels)

### Issue: "Failed to load model"
- Check internet connection (models download automatically)
- Verify dependencies installed: `pip install -r requirements.txt`
- Try deleting `~/.insightface` and re-running

### Issue: Wrong person identified
- Lower `FACE_RECOGNITION_THRESHOLD` in `config.py`
- Add more diverse photos for correct person
- Ensure photos are high quality and faces are clear

### Issue: No webcam detected
- Check webcam permissions
- Try different `VIDEO_SOURCE` in `config.py` (0, 1, 2, etc.)
- Test webcam with: `python -c "import cv2; cv2.VideoCapture(0).read()"`

### Issue: Too many "Unknown" faces
- Raise `FACE_RECOGNITION_THRESHOLD` in `config.py`
- Add more photos for known persons
- Check embedding quality (re-run enrollment)

## Technical Details

### Models Used

1. **YOLOv8n-face**: Pretrained face detector
   - Input: RGB image (any size)
   - Output: Bounding boxes with confidence scores
   - Inference: ~30 FPS on CPU

2. **InsightFace buffalo_l**: Pretrained face recognizer
   - Input: Aligned face (112x112)
   - Output: 512-D embedding vector
   - Backbone: ResNet-100

### Performance

- **Detection**: ~30 FPS on CPU, ~100+ FPS on GPU
- **Recognition**: ~50 FPS on CPU, ~200+ FPS on GPU
- **Memory**: ~500 MB (models loaded)

### File Formats

- **Images**: JPG, JPEG, PNG, BMP
- **Embeddings**: NumPy binary format (.npy)
- **Logs**: JSON format

## Code Architecture

### `detector.py`
- `FaceDetector`: YOLOv8-based face detection
- `detect_faces()`: Find faces in image
- `crop_face()`: Extract face regions

### `recognizer.py`
- `FaceRecognizer`: InsightFace-based recognition
- `get_embedding()`: Generate face embedding
- `identify_face()`: Match face to known identities
- `compute_cosine_similarity()`: Calculate similarity score

### `utils.py`
- Image I/O functions
- Embedding save/load functions
- Drawing and visualization utilities
- Dataset validation

### `main.py`
- `FaceIdentificationSystem`: Main application class
- CLI interface for all operations
- Enrollment and identification workflows

### `config.py`
- All configuration parameters
- Model paths and thresholds
- File paths and constants

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLOv8)
- InsightFace
- ONNX Runtime

See `requirements.txt` for specific versions.

## License

This project uses pretrained models:
- **YOLOv8**: AGPL-3.0 license
- **InsightFace**: MIT license

## Tips for Best Results

1. **Photo Quality**: Use clear, well-lit photos
2. **Variety**: Include different angles and expressions
3. **Quantity**: 5-10 photos per person is ideal
4. **Lighting**: Varied lighting conditions improve robustness
5. **Background**: Different backgrounds help generalization
6. **Resolution**: Higher resolution = better accuracy

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review `config.py` parameters
3. Verify dataset structure and photo quality
4. Check model download and installation

## Future Enhancements

Possible improvements (not implemented):
- Multi-face tracking across frames
- Face alignment preprocessing
- GPU acceleration toggle
- Video file input
- REST API interface
- Database storage for embeddings
- Real-time enrollment (add faces during runtime)

---

**Ready to use!** Add photos, run enrollment, and start identifying faces! 🎯
