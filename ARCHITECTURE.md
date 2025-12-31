# System Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Face Identification System                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                            main.py                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │         FaceIdentificationSystem (Main Class)              │    │
│  │  - enroll_person()                                         │    │
│  │  - enroll_all_persons()                                    │    │
│  │  - identify_from_webcam()                                  │    │
│  │  - identify_from_image()                                   │    │
│  └────────────────────────────────────────────────────────────┘    │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            │ uses
            │
    ┌───────┴────────┬────────────────────┬────────────────┐
    │                │                    │                │
    ▼                ▼                    ▼                ▼
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│detector.│    │recognizer│    │   utils.py   │    │ config.  │
│   py    │    │   .py    │    │              │    │   py     │
└─────────┘    └──────────┘    └──────────────┘    └──────────┘
│              │                │                   │
│ YOLOv8      │ InsightFace    │ Helper           │ Settings
│ Detection   │ Recognition    │ Functions        │ Parameters
└─────────────┴────────────────┴──────────────────┴───────────┘
```

## Data Flow: Enrollment Phase

```
┌──────────────┐
│ Raw Images   │  dataset/raw_images/person_1/*.jpg
│ (person_1)   │
└──────┬───────┘
       │
       │ load_image()
       ▼
┌──────────────┐
│ Image Array  │  numpy array (H, W, 3) BGR
└──────┬───────┘
       │
       │ detect_and_crop_faces()
       ▼
┌──────────────┐
│ Face Crops   │  List[(crop, bbox, conf)]
└──────┬───────┘
       │
       │ get_embedding()
       ▼
┌──────────────┐
│ Embeddings   │  List[512-d vectors]
└──────┬───────┘
       │
       │ aggregate_embeddings()
       ▼
┌──────────────┐
│ Mean Embed   │  Single 512-d vector
└──────┬───────┘
       │
       │ save_embedding()
       ▼
┌──────────────┐
│ Saved File   │  dataset/embeddings/person_1.npy
└──────────────┘
```

## Data Flow: Identification Phase

```
┌──────────────┐
│ Webcam Frame │  Video capture
└──────┬───────┘
       │
       │ FaceDetector.detect_faces()
       ▼
┌──────────────┐
│ Face BBoxes  │  [(x1,y1,x2,y2,conf), ...]
└──────┬───────┘
       │
       │ crop_face()
       ▼
┌──────────────┐
│ Face Crop    │  numpy array
└──────┬───────┘
       │
       │ FaceRecognizer.get_embedding()
       ▼
┌──────────────┐
│ Query Embed  │  512-d vector
└──────┬───────┘
       │
       │ identify_face()
       │ (compare with all known embeddings)
       ▼
┌──────────────┐
│ Cosine Sim   │  similarity scores
└──────┬───────┘
       │
       │ if max(sim) > threshold
       ▼
┌──────────────┐     ┌──────────────┐
│ Known Person │     │   Unknown    │
│ (name, conf) │     │ (no match)   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       │ draw_face_box()    │
       └────────┬───────────┘
                ▼
         ┌──────────────┐
         │ Annotated    │
         │ Frame        │
         └──────────────┘
```

## Module Interactions

```
┌───────────────────────────────────────────────────────────────┐
│                         config.py                              │
│  (All parameters, paths, thresholds, constants)               │
└───────────────┬───────────────────────────────────────────────┘
                │ imported by all modules
    ┌───────────┴───────────┬────────────────┬──────────────┐
    │                       │                │              │
┌───▼────────┐    ┌─────────▼─────┐   ┌─────▼──────┐   ┌──▼─────┐
│ detector.py│    │ recognizer.py │   │  utils.py  │   │main.py │
└────────────┘    └───────────────┘   └────────────┘   └────────┘
     │                    │                  │               │
     │ uses               │ uses             │ uses          │
     ▼                    ▼                  ▼               │
┌─────────┐         ┌──────────┐      ┌──────────┐         │
│ YOLOv8  │         │InsightFace│     │ OpenCV   │         │
│ Model   │         │  Model    │     │ NumPy    │         │
└─────────┘         └───────────┘     └──────────┘         │
                                                            │
        ┌───────────────────────────────────────────────────┘
        │
        │ orchestrates all components
        ▼
  ┌─────────────┐
  │   Output    │
  │ (webcam or  │
  │   image)    │
  └─────────────┘
```

## Class Hierarchy

```
detector.py
└── FaceDetector
    ├── __init__(model_path, conf_threshold, iou_threshold)
    ├── detect_faces(image) → List[bbox]
    ├── crop_face(image, bbox) → cropped_image
    └── detect_and_crop_faces(image) → List[(crop, bbox, conf)]

recognizer.py
└── FaceRecognizer
    ├── __init__(model_name, recognition_threshold)
    ├── get_embedding(face_image) → 512-d vector
    ├── compute_cosine_similarity(emb1, emb2) → float
    ├── load_known_embeddings(embeddings_dict)
    ├── identify_face(embedding) → (name, confidence)
    └── aggregate_embeddings(embeddings_list) → mean_embedding

main.py
└── FaceIdentificationSystem
    ├── __init__()
    ├── enroll_person(person_name) → bool
    ├── enroll_all_persons() → Dict[str, bool]
    ├── load_known_identities() → int
    ├── identify_from_webcam() → None
    └── identify_from_image(image_path) → None
```

## File System Layout

```
Identity-Identifier/
│
├── Source Code (Python)
│   ├── main.py          ← Entry point, CLI, orchestration
│   ├── detector.py      ← YOLOv8 face detection
│   ├── recognizer.py    ← InsightFace recognition
│   ├── utils.py         ← Helper functions
│   └── config.py        ← All configuration
│
├── Support Files
│   ├── test_setup.py    ← Verify installation
│   ├── examples.py      ← API usage examples
│   └── requirements.txt ← Dependencies
│
├── Documentation
│   ├── README.md        ← Complete guide
│   ├── QUICKSTART.md    ← Quick start
│   ├── PROJECT_SUMMARY.md
│   └── ARCHITECTURE.md  ← This file
│
└── Data (created automatically)
    └── dataset/
        ├── raw_images/     ← USER ADDS PHOTOS HERE
        │   ├── person_1/
        │   ├── person_2/
        │   └── person_3/
        ├── faces_cropped/  ← Auto: cropped faces
        └── embeddings/     ← Auto: .npy files
```

## Pipeline Architecture

### Enrollment Pipeline

```
Input: dataset/raw_images/person_name/*.jpg
  │
  ├─> Load images (utils.load_image)
  │
  ├─> Detect faces (FaceDetector.detect_faces)
  │     └─> YOLOv8 inference
  │
  ├─> Crop faces (FaceDetector.crop_face)
  │
  ├─> Generate embeddings (FaceRecognizer.get_embedding)
  │     └─> InsightFace inference
  │
  ├─> Aggregate embeddings (FaceRecognizer.aggregate_embeddings)
  │     └─> Mean or median
  │
  └─> Save (utils.save_embedding)
        └─> dataset/embeddings/person_name.npy
```

### Identification Pipeline

```
Input: Webcam frame or image file
  │
  ├─> Detect faces (FaceDetector.detect_faces)
  │     └─> YOLOv8 inference → bounding boxes
  │
  ├─> For each detected face:
  │   │
  │   ├─> Crop face (FaceDetector.crop_face)
  │   │
  │   ├─> Generate embedding (FaceRecognizer.get_embedding)
  │   │     └─> InsightFace inference → 512-d vector
  │   │
  │   ├─> Compare with known embeddings
  │   │     └─> Cosine similarity with all known faces
  │   │
  │   ├─> Identify (FaceRecognizer.identify_face)
  │   │     ├─> If max_similarity > threshold → Known person
  │   │     └─> Else → Unknown
  │   │
  │   └─> Draw bounding box and label (utils.draw_face_box)
  │
  └─> Display frame (cv2.imshow)
```

## Algorithmic Details

### Face Detection (YOLOv8)

```
Input: RGB image (any size)
  ↓
Resize to model input size (640×640)
  ↓
YOLOv8 backbone (CSPDarknet)
  ↓
Neck (PAN - Path Aggregation Network)
  ↓
Head (Detection head)
  ↓
Output: [class, x, y, w, h, confidence]
  ↓
Non-Maximum Suppression (NMS)
  ↓
Final bounding boxes
```

### Face Recognition (InsightFace/ArcFace)

```
Input: Face crop (112×112×3)
  ↓
Preprocessing (normalization, alignment)
  ↓
ResNet-100 backbone
  ↓
512-dimensional embedding layer
  ↓
L2 normalization
  ↓
Output: 512-d unit vector
```

### Cosine Similarity

```
Given two embeddings e1, e2 (both unit vectors):

similarity = e1 · e2
           = Σ(e1[i] × e2[i]) for i in [0, 511]

Range: [-1, 1]
  1.0  → Identical
  0.6+ → Same person (typical threshold)
  0.0  → Orthogonal (unrelated)
 -1.0  → Opposite (very rare)
```

## Performance Characteristics

```
Component          CPU Time    GPU Time    Memory
─────────────────────────────────────────────────
YOLOv8 Detection   ~30ms      ~10ms       ~200MB
Face Cropping      ~1ms       N/A         minimal
InsightFace        ~20ms      ~5ms        ~300MB
Cosine Similarity  ~0.1ms     N/A         minimal
Visualization      ~5ms       N/A         minimal
─────────────────────────────────────────────────
Total per frame    ~56ms      ~15ms       ~500MB
FPS (approx)       ~18 FPS    ~67 FPS
```

## Extension Points

Want to extend the system? Here's where to plug in:

1. **New detection model**: Modify `detector.py` → `FaceDetector` class
2. **New recognition model**: Modify `recognizer.py` → `FaceRecognizer` class
3. **New matching algorithm**: Modify `recognizer.py` → `identify_face()` method
4. **New aggregation method**: Modify `recognizer.py` → `aggregate_embeddings()`
5. **New input source**: Modify `main.py` → add new method to `FaceIdentificationSystem`
6. **New output format**: Modify `utils.py` → add new visualization function

---

This architecture provides:
- ✅ Clear separation of concerns
- ✅ Easy to understand and modify
- ✅ Modular and extensible
- ✅ Type-safe with hints
- ✅ Well-documented
