#!/usr/bin/env python3
"""
Extract ground truth trajectory from Aria MPS (Machine Perception Services) data.
MPS provides SLAM trajectories in two forms:
- closed_loop_trajectory.csv: With loop closure corrections
- open_loop_trajectory.csv: Without loop closure (odometry only)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_mps_trajectory(mps_dir, output_dir, use_closed_loop=True, vrs_start_time_ns=None):
    """
    Extract MPS trajectory data and convert to various formats.
    
    Args:
        mps_dir: Path to MPS directory containing trajectory CSV files
        output_dir: Output directory for converted trajectories
        use_closed_loop: Use closed loop (True) or open loop (False) trajectory
    """
    # Select trajectory file
    trajectory_type = "closed_loop" if use_closed_loop else "open_loop"
    trajectory_file = Path(mps_dir) / "slam" / f"{trajectory_type}_trajectory.csv"
    
    if not trajectory_file.exists():
        print(f"Error: Trajectory file not found: {trajectory_file}")
        return False
    
    print(f"Reading MPS trajectory from: {trajectory_file}")
    
    # Read CSV file
    df = pd.read_csv(trajectory_file)
    print(f"Loaded {len(df)} poses from {trajectory_type} trajectory")
    
    # Extract relevant columns based on trajectory type
    if use_closed_loop:
        # Closed loop format has world frame poses
        timestamp_col = 'tracking_timestamp_us'
        tx_col, ty_col, tz_col = 'tx_world_device', 'ty_world_device', 'tz_world_device'
        qx_col, qy_col, qz_col, qw_col = 'qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device'
    else:
        # Open loop format has odometry frame poses
        timestamp_col = 'tracking_timestamp_us'
        tx_col, ty_col, tz_col = 'tx_odometry_device', 'ty_odometry_device', 'tz_odometry_device'
        qx_col, qy_col, qz_col, qw_col = 'qx_odometry_device', 'qy_odometry_device', 'qz_odometry_device', 'qw_odometry_device'
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # If VRS start time provided, calculate offset
    time_offset_ns = 0
    if vrs_start_time_ns is not None:
        # Find MPS timestamp closest to VRS start time
        mps_start_us = df[timestamp_col].iloc[0]
        mps_start_ns = mps_start_us * 1000  # Convert to nanoseconds
        time_offset_ns = vrs_start_time_ns - mps_start_ns
        print(f"\nTime alignment:")
        print(f"  VRS start time: {vrs_start_time_ns} ns")
        print(f"  MPS start time: {mps_start_ns} ns")
        print(f"  Time offset: {time_offset_ns} ns ({time_offset_ns/1e9:.3f} seconds)")
    
    # Save in TUM format (timestamp tx ty tz qx qy qz qw)
    tum_file = output_path / f"mps_{trajectory_type}_tum.txt"
    with open(tum_file, 'w') as f:
        f.write("# MPS ground truth trajectory in TUM format\n")
        f.write("# timestamp [ns] tx ty tz qx qy qz qw\n")
        
        for _, row in df.iterrows():
            # Convert microseconds to nanoseconds
            timestamp_ns = row[timestamp_col] * 1000  # us to ns
            
            # Apply time offset to align with VRS/ORB-SLAM3 timestamps
            aligned_timestamp_ns = timestamp_ns + time_offset_ns
            
            tx, ty, tz = row[tx_col], row[ty_col], row[tz_col]
            qx, qy, qz, qw = row[qx_col], row[qy_col], row[qz_col], row[qw_col]
            
            # Write timestamp in nanoseconds (matching ORB-SLAM3 format)
            f.write(f"{aligned_timestamp_ns:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    print(f"Saved TUM format to: {tum_file}")
    
    # Save in EuRoC format (nanoseconds, same pose format)
    euroc_file = output_path / f"mps_{trajectory_type}_euroc.csv"
    with open(euroc_file, 'w') as f:
        f.write("#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n")
        
        for _, row in df.iterrows():
            timestamp_ns = row[timestamp_col] * 1000  # Convert microseconds to nanoseconds
            tx, ty, tz = row[tx_col], row[ty_col], row[tz_col]
            qx, qy, qz, qw = row[qx_col], row[qy_col], row[qz_col], row[qw_col]
            
            # EuRoC format has quaternion as w,x,y,z order
            f.write(f"{timestamp_ns},{tx:.6f},{ty:.6f},{tz:.6f},{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f}\n")
    
    print(f"Saved EuRoC format to: {euroc_file}")
    
    # Save in KITTI format (4x4 transformation matrices)
    kitti_file = output_path / f"mps_{trajectory_type}_kitti.txt"
    with open(kitti_file, 'w') as f:
        for _, row in df.iterrows():
            tx, ty, tz = row[tx_col], row[ty_col], row[tz_col]
            qx, qy, qz, qw = row[qx_col], row[qy_col], row[qz_col], row[qw_col]
            
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
            
            # Write as 3x4 transformation matrix (R | t)
            f.write(f"{R[0,0]:.6f} {R[0,1]:.6f} {R[0,2]:.6f} {tx:.6f} ")
            f.write(f"{R[1,0]:.6f} {R[1,1]:.6f} {R[1,2]:.6f} {ty:.6f} ")
            f.write(f"{R[2,0]:.6f} {R[2,1]:.6f} {R[2,2]:.6f} {tz:.6f}\n")
    
    print(f"Saved KITTI format to: {kitti_file}")
    
    # Calculate and print statistics
    positions = df[[tx_col, ty_col, tz_col]].values
    timestamps = df[timestamp_col].values
    
    duration = (timestamps[-1] - timestamps[0]) / 1e6  # seconds
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_distance = np.sum(distances)
    
    print(f"\nTrajectory Statistics:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Number of poses: {len(df)}")
    print(f"  Frequency: {len(df)/duration:.1f} Hz")
    print(f"  Total distance: {total_distance:.2f} meters")
    print(f"  Average speed: {total_distance/duration:.2f} m/s")
    print(f"  Start position: [{positions[0,0]:.3f}, {positions[0,1]:.3f}, {positions[0,2]:.3f}]")
    print(f"  End position: [{positions[-1,0]:.3f}, {positions[-1,1]:.3f}, {positions[-1,2]:.3f}]")
    
    # Check if velocities are available
    if 'device_linear_velocity_x_device' in df.columns:
        vx = df['device_linear_velocity_x_device'].values
        vy = df['device_linear_velocity_y_device'].values
        vz = df['device_linear_velocity_z_device'].values
        velocities = np.sqrt(vx**2 + vy**2 + vz**2)
        print(f"  Max velocity: {np.max(velocities):.2f} m/s")
        print(f"  Mean velocity: {np.mean(velocities):.2f} m/s")
    
    return True


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def find_mps_data(sequence_dir):
    """Find MPS data directory for a given sequence."""
    sequence_path = Path(sequence_dir)
    
    # Check for mps directory
    mps_dir = sequence_path / "mps"
    if mps_dir.exists():
        return mps_dir
    
    # Check for MPS zip files
    mps_files = list(sequence_path.glob("*mps*.zip"))
    if mps_files:
        print(f"Found {len(mps_files)} MPS zip files. Please extract them first.")
        for f in mps_files:
            print(f"  - {f.name}")
        print("\nTo extract: unzip *mps*.zip")
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Extract MPS ground truth trajectory from Aria dataset')
    parser.add_argument('sequence_dir', help='Path to Aria sequence directory (containing VRS and MPS data)')
    parser.add_argument('--output-dir', default='ground_truth', help='Output directory')
    parser.add_argument('--open-loop', action='store_true', help='Use open loop trajectory instead of closed loop')
    parser.add_argument('--vrs-start-time-ns', type=int, help='VRS start time in nanoseconds for timestamp alignment')
    parser.add_argument('--list-sequences', action='store_true', help='List all sequences with MPS data')
    
    args = parser.parse_args()
    
    if args.list_sequences:
        # Find all sequences with MPS data
        aria_dir = Path("/mnt/ssd_ext/incSeg-data/aria_everyday")
        sequences_with_mps = []
        
        for seq_dir in sorted(aria_dir.glob("loc*_script*_seq*_rec*")):
            if (seq_dir / "mps").exists():
                sequences_with_mps.append(seq_dir.name)
        
        print(f"Found {len(sequences_with_mps)} sequences with extracted MPS data:")
        for seq in sequences_with_mps:
            print(f"  - {seq}")
        return
    
    # Find MPS data
    mps_dir = find_mps_data(args.sequence_dir)
    if not mps_dir:
        print(f"Error: No MPS data found in {args.sequence_dir}")
        return
    
    # Extract trajectory
    success = extract_mps_trajectory(
        mps_dir, 
        args.output_dir, 
        use_closed_loop=not args.open_loop,
        vrs_start_time_ns=args.vrs_start_time_ns
    )
    
    if success:
        print(f"\nGround truth extraction complete!")
        print(f"Output files in: {args.output_dir}/")


if __name__ == "__main__":
    main()