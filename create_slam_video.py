#!/usr/bin/env python3
"""
Create SLAM camera video from Aria VRS file
"""

import argparse
import cv2
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain

def main():
    parser = argparse.ArgumentParser(description="Create SLAM camera video from Aria VRS")
    parser.add_argument("vrs_path", help="Path to VRS file")
    parser.add_argument("--output", default="slam_cameras.mp4", help="Output video file")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds (default: 30)")
    args = parser.parse_args()
    
    # Create data provider
    provider = data_provider.create_vrs_data_provider(args.vrs_path)
    
    # SLAM camera stream IDs
    slam_left_id = StreamId("1201-1")
    slam_right_id = StreamId("1201-2")
    
    # Setup video writer
    fps = 10  # SLAM cameras run at 10 Hz
    frame_count = args.duration * fps
    
    # Get first frames to determine size
    left_frames = []
    right_frames = []
    
    print(f"Extracting {args.duration} seconds of SLAM camera footage...")
    
    # Collect frames
    for sensor_data in provider.deliver_queued_sensor_data():
        stream_id = sensor_data.stream_id()
        
        if stream_id == slam_left_id:
            image_data = sensor_data.image_data_and_record()
            if image_data:
                frame = image_data[0].to_numpy_array()
                # Rotate 90 degrees clockwise
                frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                left_frames.append(frame_rotated)
                
        elif stream_id == slam_right_id:
            image_data = sensor_data.image_data_and_record()
            if image_data:
                frame = image_data[0].to_numpy_array()
                # Rotate 90 degrees clockwise
                frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                right_frames.append(frame_rotated)
        
        # Stop when we have enough frames
        if len(left_frames) >= frame_count and len(right_frames) >= frame_count:
            break
    
    print(f"Collected {len(left_frames)} left frames and {len(right_frames)} right frames")
    
    if len(left_frames) == 0 or len(right_frames) == 0:
        print("No SLAM camera frames found!")
        return
    
    # Create video writer
    h, w = left_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w*2, h))
    
    # Process frames
    print("Creating side-by-side video...")
    min_frames = min(len(left_frames), len(right_frames))
    
    for i in range(min_frames):
        # Create side-by-side view
        combined = np.hstack([left_frames[i], right_frames[i]])
        
        # Convert grayscale to BGR for video
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        
        # Add labels and center lines
        cv2.putText(combined_bgr, "SLAM Left (1201-1)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_bgr, "SLAM Right (1201-2)", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw center lines to show overlap region
        cv2.line(combined_bgr, (w//2, 0), (w//2, h), (0, 0, 255), 1)
        cv2.line(combined_bgr, (w + w//2, 0), (w + w//2, h), (0, 0, 255), 1)
        
        # Add frame info
        cv2.putText(combined_bgr, f"Frame {i+1}/{min_frames}", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame
        out.write(combined_bgr)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{min_frames} frames...")
    
    # Release video writer
    out.release()
    
    print(f"\nVideo saved to: {args.output}")
    print(f"Resolution: {w*2}x{h} @ {fps} fps")
    print(f"Duration: {min_frames/fps:.1f} seconds")
    print("\nThe red vertical lines show the center of each camera's field of view.")
    print("The overlapping region in the middle is where stereo depth can be computed.")

if __name__ == "__main__":
    main()