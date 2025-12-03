# NFL Player Tracker

Real-time player and ball detection and tracking system for NFL football videos using YOLO11 and ByteTrack.

## Project Status

### ✅ Completed
- **Player Detection & Tracking:** Fully functional with 15,988 training images
- **Multi-GPU Training:** Optimized for 4-GPU single-node training (outperforms 8-GPU multi-node)
- **Real-time Inference:** Processes video at 20+ FPS with confidence scores
- **Bounding Box Visualization:** White boxes around detected players with track IDs

### 🚧 In Progress
- **Football Detection:** Currently unreliable; implementing rule-based filtering for better accuracy
- **Position-Specific Labeling:** Re-labeling data with player positions (QB, WR, etc.)

### 📋 Future Work
- Yard line detection for distance calculations
- Multi-camera integration for 3D tracking
- Ball trajectory prediction
- Real-time statistics generation

## Features

- **YOLO11 Detection:** Custom-trained on NFL dataset
- **ByteTrack:** Maintains consistent player IDs across frames
- **Batch Processing:** Efficient 20-frame batch inference
- **Confidence Scoring:** Displays detection confidence on output
- **MP4/AVI Output:** Annotated video with player tracking

## Installation

```bash
# Clone repository
git clone [https://github.com/yourusername/nfl-player-tracker.git](https://github.com/yourusername/nfl-player-tracker.git)
cd nfl-player-tracker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Basic inference
python3 main_nfl.py input_videos/game.mp4 --output_video output_videos/tracked.mp4

# Skip cached detections and recompute
python3 main_nfl.py input_videos/game.mp4 --output_video output_videos/tracked.mp4 --skip_stubs

# Custom output path
python3 main_nfl.py input_videos/game.mp4 --output_video output_videos/my_output.mp4

