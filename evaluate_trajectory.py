#!/usr/bin/env python3
"""
Evaluate ORB-SLAM3 trajectory against ground truth
"""

import numpy as np
import sys
import os

def read_trajectory_tum(filename):
    """Read trajectory in TUM format"""
    trajectory = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 8:  # timestamp x y z qx qy qz qw
                timestamp = float(parts[0])
                position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                orientation = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
                trajectory.append((timestamp, position, orientation))
    return trajectory

def compute_trajectory_length(trajectory):
    """Compute total length of trajectory"""
    if len(trajectory) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(trajectory)):
        pos1 = trajectory[i-1][1]
        pos2 = trajectory[i][1]
        total_length += np.linalg.norm(pos2 - pos1)
    
    return total_length

def analyze_trajectory(trajectory_file):
    """Analyze trajectory statistics"""
    print(f"\nAnalyzing trajectory: {trajectory_file}")
    print("=" * 50)
    
    trajectory = read_trajectory_tum(trajectory_file)
    
    if not trajectory:
        print("ERROR: Could not read trajectory")
        return
    
    # Basic statistics
    print(f"Number of poses: {len(trajectory)}")
    
    # Time analysis
    timestamps = [t[0] for t in trajectory]
    duration = (timestamps[-1] - timestamps[0]) / 1e9  # Convert from ns to seconds
    fps = len(trajectory) / duration if duration > 0 else 0
    
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    # Trajectory length
    length = compute_trajectory_length(trajectory)
    print(f"Total trajectory length: {length:.3f} meters")
    print(f"Average speed: {length/duration:.3f} m/s")
    
    # Position statistics
    positions = np.array([t[1] for t in trajectory])
    pos_mean = np.mean(positions, axis=0)
    pos_std = np.std(positions, axis=0)
    
    print(f"\nPosition statistics:")
    print(f"  Mean: X={pos_mean[0]:.3f}, Y={pos_mean[1]:.3f}, Z={pos_mean[2]:.3f}")
    print(f"  Std:  X={pos_std[0]:.3f}, Y={pos_std[1]:.3f}, Z={pos_std[2]:.3f}")
    print(f"  Range: X=[{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]")
    print(f"         Y=[{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]")
    print(f"         Z=[{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]")
    
    # Check for large jumps (potential tracking failures)
    max_jump = 0.0
    jump_threshold = 0.5  # meters
    large_jumps = 0
    
    for i in range(1, len(trajectory)):
        jump = np.linalg.norm(trajectory[i][1] - trajectory[i-1][1])
        max_jump = max(max_jump, jump)
        if jump > jump_threshold:
            large_jumps += 1
    
    print(f"\nJump analysis:")
    print(f"  Maximum jump: {max_jump:.3f} meters")
    print(f"  Large jumps (>{jump_threshold}m): {large_jumps}")
    
    # Tracking continuity
    time_gaps = []
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i-1]) / 1e9
        time_gaps.append(gap)
    
    if time_gaps:
        print(f"\nTiming analysis:")
        print(f"  Average frame gap: {np.mean(time_gaps)*1000:.1f} ms")
        print(f"  Max frame gap: {np.max(time_gaps)*1000:.1f} ms")
        print(f"  Frames with gaps > 200ms: {sum(1 for g in time_gaps if g > 0.2)}")

def compare_with_ground_truth(est_file, gt_file):
    """Compare estimated trajectory with ground truth"""
    # This would require the ground truth trajectory file
    # For now, just a placeholder
    print("\nGround truth comparison not yet implemented")
    print("To evaluate accuracy, you need:")
    print("1. Ground truth trajectory from Aria MPS")
    print("2. Time alignment between trajectories")
    print("3. Compute ATE (Absolute Trajectory Error)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_trajectory.py <trajectory_file> [ground_truth_file]")
        sys.exit(1)
    
    trajectory_file = sys.argv[1]
    
    if not os.path.exists(trajectory_file):
        print(f"ERROR: File not found: {trajectory_file}")
        sys.exit(1)
    
    analyze_trajectory(trajectory_file)
    
    if len(sys.argv) > 2:
        gt_file = sys.argv[2]
        if os.path.exists(gt_file):
            compare_with_ground_truth(trajectory_file, gt_file)
        else:
            print(f"\nWARNING: Ground truth file not found: {gt_file}")