# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (5 minutes)

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**Note**: First run will download pretrained models (~200 MB). This is automatic.

---

### Step 2: Add Photos (2 minutes)

Add 3-10 photos for each person to their folder:

```
dataset/raw_images/
  ├── person_1/          👈 Add photos here
  │   ├── photo1.jpg
  │   ├── photo2.jpg
  │   └── photo3.jpg
  ├── person_2/          👈 Add photos here
  │   └── ...
  └── person_3/          👈 Add photos here
      └── ...
```

**Tips**:
- Use clear, front-facing photos
- Include different expressions and angles
- Ensure good lighting
- 5-10 photos per person = best results

---

### Step 3: Run the System (1 minute)

```bash
# Enroll all persons (creates embeddings)
python main.py --enroll-all

# Start real-time identification
python main.py --identify-webcam
```

Press **'q'** to quit the webcam view.

---

## 📝 Usage Examples

### Test Setup (Optional)
```bash
python test_setup.py
```

### Enroll Specific Person
```bash
python main.py --enroll person_1
```

### Identify from Image
```bash
python main.py --identify-image path/to/photo.jpg
```

---

## 🔧 Adding New People

1. Create folder:
   ```bash
   mkdir dataset/raw_images/alice
   ```

2. Add photos:
   ```bash
   cp photos/*.jpg dataset/raw_images/alice/
   ```

3. Enroll:
   ```bash
   python main.py --enroll alice
   ```

Done! System now recognizes Alice.

---

## ⚙️ Configuration

Edit `config.py` to adjust:

- **Recognition strictness**: `FACE_RECOGNITION_THRESHOLD`
  - Default: 0.6
  - Higher = stricter (fewer false matches)
  - Lower = looser (more false matches)

- **Detection confidence**: `YOLO_CONFIDENCE_THRESHOLD`
  - Default: 0.5
  - Higher = only detect clear faces

- **Video source**: `VIDEO_SOURCE`
  - 0 = default webcam
  - 1, 2, ... = other cameras
  - "/path/to/video.mp4" = video file

---

## 🎯 Expected Results

✅ **Known person detected**:
- Green bounding box
- Name displayed
- Confidence score shown

❌ **Unknown person detected**:
- Red bounding box
- "Unknown" label
- Similarity score shown

---

## 🐛 Common Issues

### "No face detected in image"
- Ensure face is clearly visible
- Use better lighting
- Try front-facing photos

### "Failed to load InsightFace model"
- Check internet connection
- Models download automatically on first run
- Wait for download to complete (~200 MB)

### "No webcam detected"
- Check webcam permissions
- Try `VIDEO_SOURCE = 1` in config.py
- Test: `python -c "import cv2; cv2.VideoCapture(0).read()"`

### Too many "Unknown" detections
- Lower `FACE_RECOGNITION_THRESHOLD` in config.py
- Add more photos for each person
- Use higher quality photos

---

## 📊 Performance

- **CPU**: ~30 FPS detection, ~50 FPS recognition
- **GPU**: ~100+ FPS detection, ~200+ FPS recognition
- **Memory**: ~500 MB with models loaded

---

## ℹ️ More Information

See `README.md` for complete documentation.

---

**Ready!** Add photos and start identifying faces! 🎯
