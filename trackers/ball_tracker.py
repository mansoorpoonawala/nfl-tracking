from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class BallTracker:
    """
    A class that handles ball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    Adapted to work with different model types (basketball or football).
    """
    def __init__(self, model_path, ball_class_name='ball'):
        self.model = YOLO(model_path)
        self.ball_class_name = ball_class_name.lower()
        
        # Get the class names from the model
        self.class_names = self.model.names
        self.class_names_inv = {v.lower(): k for k, v in self.class_names.items()}
        
        print(f"[BallTracker] Model classes: {self.class_names}")
        print(f"[BallTracker] Looking for ball class: '{self.ball_class_name}'") 

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks=[]

        for frame_num, detection in enumerate(detections):
            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                # Get class name from ID and compare (case-insensitive)
                class_name = self.class_names.get(int(cls_id), '').lower()
                
                if class_name == self.ball_class_name:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        save_stub(stub_path,tracks)
        
        return tracks

    def remove_wrong_detections(self,ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.

        Args:
            ball_positions (list): List of detected ball positions across frames.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        
        maximum_allowed_distance = 50  # Increased for football field size
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        # Extract bbox values, handling empty detections
        ball_positions_list = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        
        # Filter out empty lists to get valid detections
        valid_positions = [pos for pos in ball_positions_list if len(pos) > 0]
        
        # If no valid detections, return original
        if len(valid_positions) == 0:
            return ball_positions
        
        # Create DataFrame only with valid positions
        df_ball_positions = pd.DataFrame(valid_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.ffill()
        
        # Reconstruct with interpolated values
        interpolated_list = df_ball_positions.to_numpy().tolist()
        
        # Map back to original structure, filling gaps
        result = []
        valid_idx = 0
        for i, original_pos in enumerate(ball_positions_list):
            if len(original_pos) > 0:
                # Use interpolated value for valid positions
                if valid_idx < len(interpolated_list):
                    result.append({1: {"bbox": interpolated_list[valid_idx]}})
                    valid_idx += 1
                else:
                    result.append({1: {"bbox": original_pos}})
            else:
                # For empty positions, use last valid interpolated value if available
                if valid_idx > 0 and valid_idx <= len(interpolated_list):
                    result.append({1: {"bbox": interpolated_list[valid_idx - 1]}})
                else:
                    result.append({1: {"bbox": []}})
        
        return result