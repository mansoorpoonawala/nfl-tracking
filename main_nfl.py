import os
import argparse
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
# from team_assigner import TeamAssigner  # Commented out to avoid memory crash
from drawers import (
    PlayerTracksDrawer,
    FrameNumberDrawer
)

from drawers.ball_tracks_drawer_new import BallTracksDrawer

from configs import (
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH,
    PLAYER_CLASS_NAME,
    BALL_CLASS_NAME
)

def parse_args():
    parser = argparse.ArgumentParser(description='NFL Football Video Analysis - Ball & Player Tracking')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    parser.add_argument('--skip_stubs', action='store_true',
                        help='Skip reading from stubs and recompute everything')
    parser.add_argument('--player_class', type=str, default=PLAYER_CLASS_NAME,
                        help='Name of player class in the model')
    parser.add_argument('--ball_class', type=str, default=BALL_CLASS_NAME,
                        help='Name of ball class in the model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_video) or '.', exist_ok=True)
    os.makedirs(args.stub_path, exist_ok=True)
    
    print(f"\n=== NFL Football Video Analysis ===")
    print(f"Input video: {args.input_video}")
    print(f"Output video: {args.output_video}")
    print(f"Player detector: {PLAYER_DETECTOR_PATH}")
    print(f"Ball detector: {BALL_DETECTOR_PATH}")
    print(f"Player class: {args.player_class}")
    print(f"Ball class: {args.ball_class}")
    print(f"Skip stubs: {args.skip_stubs}\n")
    
    # Read video
    print("[1/7] Reading video...")
    video_frames = read_video(args.input_video)
    print(f"      Loaded {len(video_frames)} frames")
    
    # Initialize Trackers
    print(f"\n[2/7] Loading player detector from: {PLAYER_DETECTOR_PATH}")
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH, player_class_name=args.player_class)
    
    print(f"[3/7] Loading ball detector from: {BALL_DETECTOR_PATH}")
    ball_tracker = BallTracker(BALL_DETECTOR_PATH, ball_class_name=args.ball_class)
    
    # Run Player Detection and Tracking
    print(f"\n[4/7] Running player detection and tracking...")
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=not args.skip_stubs,
        stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl')
    )
    print(f"      Detected players in {len(player_tracks)} frames")
    
    # Run Ball Detection and Tracking
    print(f"[5/7] Running ball detection and tracking...")
    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub=not args.skip_stubs,
        stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl')
    )
    print(f"      Detected ball in {len(ball_tracks)} frames")
    
    # Filter and interpolate ball tracks
    print(f"[6/7] Filtering and interpolating ball positions...")
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    
    # Assign Player Teams (simplified - skip for now to avoid memory issues)
    print(f"      Assigning players to teams...")
    # Create a simple team assignment based on player ID (even IDs = team 1, odd = team 2)
    player_assignment = []
    for frame_num in range(len(player_tracks)):
        frame_assignment = {}
        for player_id in player_tracks[frame_num].keys():
            # Simple assignment: even player IDs to team 1, odd to team 2
            frame_assignment[player_id] = 1 if player_id % 2 == 0 else 2
        player_assignment.append(frame_assignment)
    
    # Draw output
    print(f"\n[7/7] Drawing annotations...")
    
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    frame_number_drawer = FrameNumberDrawer()
    
    # Draw Player Tracks
    output_video_frames = player_tracks_drawer.draw(
        video_frames, 
        player_tracks,
        player_assignment,
        {i: None for i in range(len(video_frames))}
    )
    
    # Draw Ball Tracks
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    
    # Draw Frame Number
    output_video_frames = frame_number_drawer.draw(output_video_frames)
    
    # Save video
    print(f"      Saving output video to: {args.output_video}")
    save_video(output_video_frames, args.output_video)
    print(f"\n✓ Done! Output saved to {args.output_video}\n")

if __name__ == '__main__':
    main()
