"""
A utility module providing functions for drawing shapes on video frames.

This module includes functions to draw triangles and ellipses on frames, which can be used
to represent various annotations such as player positions or ball locations in sports analysis.
"""

import cv2 
import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

def draw_traingle(frame,bbox,color):
    """
    Draws a filled triangle on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the triangle in BGR format.

    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """
    y= int(bbox[1])
    x,_ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x,y],
        [x-10,y-20],
        [x+10,y-20],
    ])
    cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

    return frame

def draw_ellipse(frame,bbox,color,track_id=None,confidence=None):
    """
    Draws an ellipse and an optional rectangle with a track ID and confidence score on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the ellipse.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the ellipse in BGR format.
        track_id (int, optional): The track ID to display inside a rectangle. Defaults to None.
        confidence (float, optional): The confidence score to display. Defaults to None.

    Returns:
        numpy.ndarray: The frame with the ellipse and optional track ID and confidence drawn on it.
    """
    y1 = int(bbox[1])
    x1 = int(bbox[0])
    y2 = int(bbox[3])
    x2 = int(bbox[2])

    # Draw white rectangle around player
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),  # White box
        thickness=2
    )

    # Draw label at top-left of bounding box
    if track_id is not None:
        # Prepare label text
        label_text = f"Player"
        if confidence is not None:
            label_text += f" {confidence:.2f}"
        
        # Get text size to create background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            (255, 255, 255),  # White background
            cv2.FILLED
        )
        
        # Draw text
        cv2.putText(
            frame,
            label_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text
            2
        )

    return frame


def draw_ball(frame, bbox, color=(0, 255, 0)):
    """
    Draws a box around the detected ball.

    Args:
        frame (numpy.ndarray): The frame on which to draw.
        bbox (tuple): Bounding box (x1, y1, x2, y2).
        color (tuple): Color in BGR format. Defaults to green.

    Returns:
        numpy.ndarray: The frame with ball box drawn.
    """
    if len(bbox) < 4:
        return frame
    
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    # Draw green box around ball
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness=2
    )
    
    # Add "Football" label
    cv2.putText(
        frame,
        "Football",
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

    return frame
