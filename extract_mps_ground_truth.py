#!/usr/bin/env python3
"""
Extract MPS ground truth from Aria dataset with proper timestamp alignment
Uses tracking_timestamp_us which corresponds to device time (same as VRS)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix"""
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ])
    return R


def extract_mps_trajectory(sequence_dir, output_dir="ground_truth_data", 
                          trajectory_type="closed_loop", vrs_start_time_ns=None):
    """
    Extract MPS trajectory and save in multiple formats
    
    Args:
        sequence_dir: Path to sequence directory containing mps/slam/
        output_dir: Output directory for ground truth files
        trajectory_type: "closed_loop" or "open_loop"
        vrs_start_time_ns: VRS start time for alignment (if provided)
    """
    sequence_path = Path(sequence_dir)
    
    # Check if MPS data exists - it could be in mps/slam/ or just in the root
    trajectory_file = None
    
    # Try different possible locations
    possible_paths = [
        sequence_path / "mps" / "slam" / f"{trajectory_type}_trajectory.csv",
        sequence_path / "slam" / f"{trajectory_type}_trajectory.csv", 
        sequence_path / f"{trajectory_type}_trajectory.csv"
    ]
    
    for path in possible_paths:
        if path.exists():
            trajectory_file = path
            print(f"Found trajectory at: {path}")
            break
    
    if not trajectory_file:
        # Check for zip files
        zip_files = list(sequence_path.glob("*mps*.zip"))
        if zip_files:
            print(f"Found {len(zip_files)} MPS zip files. Please extract them first.")
            for zf in zip_files:
                print(f"  - {zf.name}")
            print(f"\nTo extract: unzip *mps*.zip")
        else:
            print(f"No {trajectory_type}_trajectory.csv found in {sequence_dir}")
        return False
    
    # Load trajectory CSV
    print(f"Reading MPS trajectory from: {trajectory_file}")
    df = pd.read_csv(trajectory_file)
    print(f"Loaded {len(df)} poses from {trajectory_type} trajectory")
    
    # Use tracking_timestamp_us which corresponds to device time (same timebase as VRS)
    timestamp_col = 'tracking_timestamp_us'
    
    # Position and orientation columns
    tx_col, ty_col, tz_col = 'tx_world_device', 'ty_world_device', 'tz_world_device'
    qx_col, qy_col, qz_col, qw_col = 'qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device'
    
    # Check if we have odometry columns (some datasets might use different names)
    if tx_col not in df.columns and 'tx_odometry_device' in df.columns:
        print("Using odometry columns instead of world columns")
        tx_col, ty_col, tz_col = 'tx_odometry_device', 'ty_odometry_device', 'tz_odometry_device'
        qx_col, qy_col, qz_col, qw_col = 'qx_odometry_device', 'qy_odometry_device', 'qz_odometry_device', 'qw_odometry_device'
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert tracking timestamps from microseconds to nanoseconds
    df['timestamp_ns'] = df[timestamp_col] * 1000
    
    # Check timestamp alignment
    if vrs_start_time_ns is not None:
        # Find the closest MPS timestamp to VRS start
        mps_timestamps_ns = df['timestamp_ns'].values
        closest_idx = np.argmin(np.abs(mps_timestamps_ns - vrs_start_time_ns))
        time_diff_ns = vrs_start_time_ns - mps_timestamps_ns[closest_idx]
        
        print(f"\nTimestamp alignment check:")
        print(f"  VRS start time: {vrs_start_time_ns} ns ({vrs_start_time_ns/1e9:.6f} s)")
        print(f"  Closest MPS time: {mps_timestamps_ns[closest_idx]} ns ({mps_timestamps_ns[closest_idx]/1e9:.6f} s)")
        print(f"  Time difference: {time_diff_ns} ns ({time_diff_ns/1e9:.6f} s)")
        
        if abs(time_diff_ns) > 1e8:  # More than 0.1 second difference
            if 0.5e9 < abs(time_diff_ns) < 1.5e9:  # Between 0.5 and 1.5 seconds
                print(f"  ℹ️  Note: MPS typically starts ~0.8s after VRS recording begins (normal SLAM initialization delay)")
            else:
                print(f"  ⚠️  WARNING: Unusually large time difference detected!")
    
    # Save in TUM format (timestamp tx ty tz qx qy qz qw)
    tum_file = output_path / f"mps_{trajectory_type}_tum.txt"
    with open(tum_file, 'w') as f:
        f.write("# MPS ground truth trajectory in TUM format\n")
        f.write("# timestamp [ns] tx ty tz qx qy qz qw\n")
        
        for _, row in df.iterrows():
            timestamp_ns = row['timestamp_ns']
            tx, ty, tz = row[tx_col], row[ty_col], row[tz_col]
            qx, qy, qz, qw = row[qx_col], row[qy_col], row[qz_col], row[qw_col]
            
            # Write timestamp in nanoseconds (matching VRS/ORB-SLAM3 format)
            f.write(f"{timestamp_ns:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    print(f"Saved TUM format to: {tum_file}")
    
    # Save in EuRoC format
    euroc_file = output_path / f"mps_{trajectory_type}_euroc.csv"
    with open(euroc_file, 'w') as f:
        f.write("#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n")
        
        for _, row in df.iterrows():
            timestamp_ns = row['timestamp_ns']
            tx, ty, tz = row[tx_col], row[ty_col], row[tz_col]
            qx, qy, qz, qw = row[qx_col], row[qy_col], row[qz_col], row[qw_col]
            
            # EuRoC format has quaternion as w,x,y,z order
            f.write(f"{timestamp_ns},{tx:.6f},{ty:.6f},{tz:.6f},{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f}\n")
    
    print(f"Saved EuRoC format to: {euroc_file}")
    
    # Calculate and print statistics
    positions = df[[tx_col, ty_col, tz_col]].values
    timestamps_ns = df['timestamp_ns'].values
    
    duration = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9  # seconds
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
    
    # Show first few timestamps for verification
    print(f"\nFirst 5 timestamps (device time):")
    for i in range(min(5, len(df))):
        ts_ns = df['timestamp_ns'].iloc[i]
        print(f"  {ts_ns} ns ({ts_ns/1e9:.6f} s)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract MPS ground truth with proper timestamp alignment')
    parser.add_argument('sequence_dir', help='Path to sequence directory')
    parser.add_argument('--output-dir', default='ground_truth_data', help='Output directory')
    parser.add_argument('--trajectory-type', choices=['closed_loop', 'open_loop'], 
                       default='closed_loop', help='Which trajectory to extract')
    parser.add_argument('--vrs-start-time-ns', type=float, default=None,
                       help='VRS start time in nanoseconds for verification')
    
    args = parser.parse_args()
    
    success = extract_mps_trajectory(
        args.sequence_dir,
        args.output_dir,
        args.trajectory_type,
        args.vrs_start_time_ns
    )
    
    if success:
        print(f"\nGround truth extraction complete!")
        print(f"Output files in: {args.output_dir}/")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()