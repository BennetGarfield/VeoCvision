# 🧠 Player Re-identification Tracker

> A robust and modular Computer Vision pipeline for **multi-player detection, tracking, and re-identification** in video footage, powered by **YOLO object detection**, handcrafted **feature extraction**, and real-time **multi-object tracking**.

---

## 🚀 Key Highlights

* 🎯 **YOLOv8-Based Detection** — Leverages state-of-the-art YOLO architecture for high-precision person detection.
* 🔁 **Custom Re-identification Engine** — Feature-based identity recovery even after occlusion or frame exit.
* 📌 **Multi-object Tracking** — Maintains persistent identities with unique player IDs across frames.
* 📈 **Online Learning Phase** — Continuously updates appearance models to adapt to real-time visual variations.
* ♻️ **Lost Player Recovery** — Recovers previously tracked players using memory and similarity heuristics.
* 🎥 **Annotated Video Output** — Generates real-time visualizations with bounding boxes, IDs, and stats.

---

## ⚙️ Environment Configuration

### 📌 Python Version

* `Python >= 3.8`

### 🧰 Core Dependencies

```bash
pip install opencv-python numpy ultralytics
```

* `opencv-python` → Video frame operations
* `numpy` → Feature vector computations
* `ultralytics` → YOLO detection integration
* `collections` → Python-native data handling

### 💻 Hardware Recommendations

* **CPU**: 4-core minimum
* **RAM**: 8GB (16GB+ preferred)
* **GPU**: CUDA-enabled GPU for acceleration *(Optional but optimal)*

---

## 🔧 Setup Instructions

### Step 1: Clone & Navigate

```bash
git clone https://github.com/your-repo/Player-Reid-Tracker.git
cd CVPROJECT
```

### Step 2: Activate Virtual Environment

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3: Install Packages

```bash
pip install -r requirements.txt
# or manually
pip install opencv-python numpy ultralytics
```

### Step 4: Run the Pipeline

```bash
python playersdetc.py
```

---

## 🧠 How It Works – System Architecture

### 1️⃣ Detection Phase

* YOLO detects bounding boxes per frame
* Applies confidence filtering (`conf_thresh=0.4`)

### 2️⃣ Feature Extraction

* **Color Histogram** (RGB analysis)
* **Gradient Descriptors** (Sobel filtering)
* **L2 Normalization** (for cosine distance consistency)

### 3️⃣ Tracking & Re-identification

* Initial tracking via appearance & motion similarity
* Lost tracks matched against incoming detections using:

  * **Appearance Weight**: `0.7`
  * **Motion Weight**: `0.3`
* Tracks are terminated after `30` missed frames or re-identified within `300` frames.

---

## 🛠️ Configuration Parameters

Tweak the behavior of the pipeline inside the `PlayerReidentificationTracker` class:

```python
# Detection
self.conf_thresh = 0.4
self.max_distance = 150
self.max_age = 30

# Re-identification
self.reid_memory_frames = 300
self.learning_frames = 90
self.reid_threshold = 0.6

# Features
self.feature_dim = 512
self.appearance_weight = 0.7
self.motion_weight = 0.3
```

---

## 🧪 Input & Output

| Type   | File                          | Description                          |
| ------ | ----------------------------- | ------------------------------------ |
| Input  | `15sec_input_720p.mp4`        | Raw video footage (720p recommended) |
| Input  | `best.pt`                     | YOLOv8 model weights                 |
| Output | `output_reidentification.mp4` | Final video with tracking overlays   |

---

## 🎬 Output Video Annotations

* 🎯 **Bounding Boxes** with distinct player-specific colors
* 🔢 **Player IDs** (P1, P2, …)
* 📊 **Confidence Scores**
* 📡 **Learning Phase Indicator**
* 🧾 **Live Stats**: frame number, active/lost count, ETA

---

## 📉 Real-Time Metrics (CLI Output)

* ✅ Processed Frames Count
* 🔁 FPS (Frames Per Second)
* 👥 Total Players Detected
* ⏱️ Estimated Time Remaining
* 🔍 Lost vs Active Tracks

---

## 🧰 Troubleshooting Guide

| Issue                     | Resolution                                                |
| ------------------------- | --------------------------------------------------------- |
| `Could not open video`    | Ensure `15sec_input_720p.mp4` exists in root directory    |
| `best.pt not loading`     | Verify model weights are valid YOLOv8 format              |
| Slow or laggy performance | Lower video resolution / Use GPU acceleration             |
| `venv` activation fails   | Use platform-specific `activate` commands (see **Setup**) |

---

## 🧬 Advanced Details

### 🔄 Similarity Computation

* `appearance_similarity = cosine(feature_vector_a, feature_vector_b)`
* `motion_similarity = inverse_distance(bbox_center_a, bbox_center_b)`
* Final matching score is a weighted average of the above.

### 🧠 Memory Buffer

* **Active Tracks**: On-screen players
* **Lost Buffer**: Recently lost tracks for re-ID
* **Feature History**: Cached appearance embeddings per player

---

## 🧾 Future Improvements (Suggested)

* ✅ GPU-optimized re-ID model integration (e.g., Deep SORT + MobileNet)
* ✅ Configurable YAML/JSON for parameters
* ✅ UI Dashboard for real-time monitoring
* ✅ Dataset evaluation support for benchmarked results

---

## 📚 References

* [YOLO by Ultralytics](https://github.com/ultralytics/ultralytics)
* [OpenCV Python Docs](https://docs.opencv.org/master/)
* Research on Person Re-ID and Multi-Object Tracking

---

## 📢 Final Notes

* The system is auto-calibrated for **short-length, high-resolution sports or surveillance videos**.
* Re-identification is robust up to **300 frames of absence**.
* The feature space is **normalized and modular**, ensuring scalability across different datasets or player uniforms.

---

