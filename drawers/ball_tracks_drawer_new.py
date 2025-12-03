from .utils import draw_ball
import numpy as np

class BallTracksDrawer:
    """
    Draws ball tracks with rule-based filtering for better accuracy.
    """
    def __init__(self, confidence_threshold=0.5, max_distance=100):
        """
        Args:
            confidence_threshold: Minimum confidence to draw ball (0.0-1.0)
            max_distance: Max pixels ball can move between frames
        """
        self.confidence_threshold = confidence_threshold
        self.max_distance = max_distance
        self.last_ball_pos = None

    def draw(self, video_frames, ball_tracks):
        """
        Draw ball tracks with rule-based filtering.
        
        Args:
            video_frames: List of video frames
            ball_tracks: List of ball detections per frame
            
        Returns:
            List of frames with ball drawn
        """
        output_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            ball_data = ball_tracks[frame_num].get(1, {})
            bbox = ball_data.get('bbox', [])
            confidence = ball_data.get('confidence', 0.0)
            
            # Rule 1: Only draw if confidence is high enough
            if confidence < self.confidence_threshold or len(bbox) == 0:
                self.last_ball_pos = None
                output_frames.append(frame)
                continue
            
            # Rule 2: Check if ball moved too far (likely false detection)
            current_pos = np.array(bbox[:2])
            if self.last_ball_pos is not None:
                distance = np.linalg.norm(current_pos - self.last_ball_pos)
                if distance > self.max_distance:
                    # Skip this detection, likely error
                    output_frames.append(frame)
                    continue
            
            # Draw the ball
            frame = draw_ball(frame, bbox, color=(0, 255, 0))
            self.last_ball_pos = current_pos
            
            output_frames.append(frame)
        
        return output_frames
