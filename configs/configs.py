# NFL Project Configuration
# Roboflow YOLO11n weights trained on NFL dataset
NFL_WEIGHTS_PATH = '/Users/mansoorpoonawala/Downloads/train3/weights/best.pt'

# For now, using the same weights for both player and ball detection
# since the model was trained on the full NFL dataset
PLAYER_DETECTOR_PATH = NFL_WEIGHTS_PATH
BALL_DETECTOR_PATH = NFL_WEIGHTS_PATH

# Court keypoint detector (can be disabled for now)
COURT_KEYPOINT_DETECTOR_PATH = 'models/court_keypoint_detector.pt'

# Paths
STUBS_DEFAULT_PATH = 'stubs'
OUTPUT_VIDEO_PATH = 'output_videos/output_video.avi'