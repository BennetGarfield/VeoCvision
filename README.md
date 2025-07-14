# Player Re-identification Tracker

A sophisticated computer vision system for tracking and re-identifying players in video footage using YOLO object detection and custom feature extraction algorithms.

## Features

- **Player Detection**: Uses YOLO model for accurate player detection
- **Re-identification**: Advanced feature-based re-identification system
- **Multi-object Tracking**: Tracks multiple players simultaneously
- **Learning Phase**: Adapts to player appearances over time
- **Lost Player Recovery**: Re-identifies players who temporarily leave the frame
- **Visual Output**: Annotated video with player IDs and confidence scores

## Project Structure

```
CVPROJECT/
├── venv/                           # Virtual environment
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── share/
├── .gitattributes
├── 15sec_input_720p.mp4          # Input video file
├── best.pt                        # YOLO model weights
├── output_reidentification.mp4    # Output video (generated)
├── playersdetc.py                 # Main tracking script
└── README.md                      # This file
```

## Dependencies and Environment Requirements

### Python Version
- Python 3.8 or higher

### Required Packages
- **OpenCV**: `cv2` - Computer vision operations
- **NumPy**: `numpy` - Numerical computations
- **Ultralytics YOLO**: `ultralytics` - Object detection model
- **Collections**: Built-in Python module

### Hardware Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: Minimum 8GB, 16GB recommended
- **GPU**: Optional but recommended for faster processing (CUDA-compatible)

## Setup and Installation

### 1. Clone or Download the Project
Ensure you have all the project files in the `CVPROJECT` directory.

### 2. Activate Virtual Environment
```bash
source venv/Scripts/activate
```

### 3. Install Dependencies
If not already installed in the virtual environment:
```bash
pip install opencv-python numpy ultralytics
```

### 4. Verify Required Files
Make sure you have:
- `best.pt` - YOLO model weights file
- `15sec_input_720p.mp4` - Input video file
- `playersdetc.py` - Main script

## How to Run the Code

### Basic Usage
```bash
python playersdetc.py
```

### Complete Step-by-Step Process
1. **Navigate to project directory**:
   ```bash
   cd CVPROJECT
   ```

2. **Activate virtual environment**:
   ```bash
   source venv/Scripts/activate
   ```

3. **Run the tracking script**:
   ```bash
   python playersdetc.py
   ```

### Expected Output
- The program will process the input video `15sec_input_720p.mp4`
- Progress updates will be displayed in the console
- Output video `output_reidentification.mp4` will be generated
- Console will show tracking statistics and performance metrics

## Configuration Options

The tracker can be customized by modifying these parameters in the `PlayerReidentificationTracker` class:

```python
# Detection parameters
self.conf_thresh = 0.4              # Confidence threshold
self.max_distance = 150             # Maximum tracking distance
self.max_age = 30                   # Maximum frames before track is lost

# Re-identification parameters
self.reid_memory_frames = 300       # Memory duration for lost tracks
self.learning_frames = 90           # Learning phase duration
self.reid_threshold = 0.6           # Re-identification threshold

# Feature extraction
self.feature_dim = 512              # Feature vector dimension
self.appearance_weight = 0.7        # Weight for appearance similarity
self.motion_weight = 0.3            # Weight for motion similarity
```

## Input/Output Files

### Input Files
- **`best.pt`**: Pre-trained YOLO model weights
- **`15sec_input_720p.mp4`**: Input video file (720p resolution)

### Output Files
- **`output_reidentification.mp4`**: Processed video with player tracking annotations

## Algorithm Overview

### Detection Phase
1. YOLO model detects players in each frame
2. Extracts bounding boxes and confidence scores
3. Filters detections based on confidence threshold

### Feature Extraction
1. **Color Histograms**: RGB color distribution analysis
2. **Gradient Features**: Edge and texture information using Sobel operators
3. **Feature Normalization**: L2 normalization for consistent comparison

### Tracking and Re-identification
1. **Initial Tracking**: Associates detections across frames using similarity metrics
2. **Learning Phase**: Builds appearance models for each player
3. **Re-identification**: Matches lost players with new detections using stored features
4. **Track Management**: Maintains active tracks and manages lost track memory

## Performance Metrics

The system provides real-time feedback including:
- Processing FPS
- Number of active tracks
- Number of lost tracks awaiting re-identification
- Progress percentage and ETA
- Total unique players detected

## Troubleshooting

### Common Issues

1. **"Could not open video" error**:
   - Verify `15sec_input_720p.mp4` exists in the project directory
   - Check video file format compatibility

2. **Model loading errors**:
   - Ensure `best.pt` is present and not corrupted
   - Verify YOLO model compatibility

3. **Performance issues**:
   - Reduce video resolution if processing is slow
   - Adjust confidence threshold for fewer detections
   - Consider using GPU acceleration if available

### Virtual Environment Issues
If the virtual environment doesn't activate:
```bash
# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

## Technical Details

### Similarity Calculation
The system uses a weighted combination of:
- **Appearance Similarity**: Cosine similarity between feature vectors
- **Motion Similarity**: Inverse distance-based similarity

### Memory Management
- **Active Tracks**: Currently visible players
- **Lost Tracks**: Recently lost players available for re-identification
- **Feature Memory**: Maintains feature history for each player

## Output Video Features

The generated video includes:
- **Colored Bounding Boxes**: Different colors for each player
- **Player IDs**: Unique identifiers (P1, P2, etc.)
- **Confidence Scores**: Detection confidence values
- **Status Information**: Frame count, active/lost track counts
- **Learning Phase Indicator**: Shows when the system is learning

## Notes

- The system automatically detects the target class (person/player) from the YOLO model
- Learning phase lasts for the first 90 frames to build robust appearance models
- Lost players can be re-identified within 300 frames of disappearance
- Feature vectors are normalized for consistent similarity calculations