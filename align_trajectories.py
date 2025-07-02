#!/usr/bin/env python3
"""
Align SLAM and MPS trajectories by finding the best time offset
"""
import sys
import numpy as np

def read_trajectory(filename):
    """Read TUM format trajectory file"""
    timestamps = []
    poses = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                timestamps.append(float(parts[0]))
                # Store position only for now
                poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return np.array(timestamps), np.array(poses)

def find_time_offset(slam_timestamps, mps_timestamps):
    """Find the time offset that best aligns the trajectories"""
    
    # Convert to seconds for easier handling
    slam_t = slam_timestamps / 1e9
    mps_t = mps_timestamps / 1e9
    
    # The MPS trajectory covers the full recording
    # The SLAM trajectory is a subset (30 seconds)
    # We need to find where in MPS the SLAM trajectory starts
    
    slam_duration = slam_t[-1] - slam_t[0]
    print(f"SLAM trajectory duration: {slam_duration:.2f} seconds")
    print(f"MPS trajectory duration: {mps_t[-1] - mps_t[0]:.2f} seconds")
    
    # SLAM starts from some offset relative to recording start
    # Let's check what offset makes sense
    slam_start_relative = slam_t[0]
    mps_start = mps_t[0]
    
    print(f"\nSLAM first timestamp: {slam_start_relative:.6f} s")
    print(f"MPS first timestamp: {mps_start:.6f} s")
    
    # The offset should be: MPS_time = SLAM_time + offset
    # Since SLAM uses relative timestamps from ~61s and MPS uses device time
    offset = mps_start - slam_start_relative
    
    return offset

def main():
    if len(sys.argv) < 3:
        print("Usage: python align_trajectories.py <slam_trajectory> <mps_trajectory>")
        sys.exit(1)
    
    slam_file = sys.argv[1]
    mps_file = sys.argv[2]
    
    # Read trajectories
    print("Reading trajectories...")
    slam_timestamps, slam_poses = read_trajectory(slam_file)
    mps_timestamps, mps_poses = read_trajectory(mps_file)
    
    print(f"SLAM: {len(slam_timestamps)} poses")
    print(f"MPS: {len(mps_timestamps)} poses")
    
    # Find time offset
    offset_s = find_time_offset(slam_timestamps, mps_timestamps)
    offset_ns = offset_s * 1e9
    
    print(f"\nTime offset: {offset_s:.6f} seconds ({offset_ns:.0f} ns)")
    print(f"To align: MPS_time = SLAM_time + {offset_s:.6f}")
    
    # Save the offset for use in evaluation
    with open("time_offset.txt", "w") as f:
        f.write(f"{offset_s}\n")
    
    print("\nTime offset saved to: time_offset.txt")
    print("Use this with evo tools: --t_offset $(cat time_offset.txt)")
    
    # Also create an aligned trajectory file
    output_file = slam_file.replace('.txt', '_aligned.txt')
    with open(slam_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.startswith('#'):
                f_out.write(line)
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                timestamp_ns = float(parts[0])
                aligned_timestamp_ns = timestamp_ns + offset_ns
                f_out.write(f"{aligned_timestamp_ns:.6f}")
                for part in parts[1:]:
                    f_out.write(f" {part}")
                f_out.write("\n")
    
    print(f"\nAligned trajectory saved to: {output_file}")

if __name__ == "__main__":
    main()