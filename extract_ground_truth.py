#!/usr/bin/env python3
"""
Extract ground truth trajectory from Aria MPS files
"""

import os
import sys
import zipfile
import json
import numpy as np

def extract_mps_trajectory(sequence_path):
    """Extract MPS trajectory from Aria dataset"""
    # Find MPS trajectories zip
    mps_files = [f for f in os.listdir(sequence_path) if 'mps_slam_trajectories.zip' in f]
    
    if not mps_files:
        print("ERROR: No MPS trajectory file found")
        return None
    
    mps_zip = os.path.join(sequence_path, mps_files[0])
    print(f"Found MPS file: {mps_zip}")
    
    # Extract trajectory
    with zipfile.ZipFile(mps_zip, 'r') as z:
        # List contents
        files = z.namelist()
        print(f"Contents: {files}")
        
        # Look for trajectory file
        traj_files = [f for f in files if 'trajectory' in f.lower() or 'pose' in f.lower()]
        if traj_files:
            print(f"Extracting: {traj_files[0]}")
            z.extract(traj_files[0], '/tmp/')
            return f"/tmp/{traj_files[0]}"
    
    return None

def find_sequence_for_vrs(vrs_path):
    """Find the sequence directory for a given VRS file"""
    # Extract sequence name from VRS path
    vrs_name = os.path.basename(vrs_path)
    parts = vrs_name.split('_')
    
    # Look for loc, script, seq, rec
    loc = script = seq = rec = None
    for i, part in enumerate(parts):
        if part.startswith('loc'):
            loc = part
        elif part.startswith('script'):
            script = part
        elif part.startswith('seq'):
            seq = part
        elif part.startswith('rec'):
            rec = part
    
    if all([loc, script, seq, rec]):
        seq_name = f"{loc}_{script}_{seq}_{rec}"
        seq_path = f"/mnt/ssd_ext/incSeg-data/aria_everyday/{seq_name}"
        if os.path.exists(seq_path):
            return seq_path
    
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_ground_truth.py <vrs_file_or_sequence_name>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Determine sequence path
    if input_path.endswith('.vrs'):
        seq_path = find_sequence_for_vrs(input_path)
    else:
        # Assume it's a sequence name
        seq_path = f"/mnt/ssd_ext/incSeg-data/aria_everyday/{input_path}"
    
    if not seq_path or not os.path.exists(seq_path):
        print(f"ERROR: Could not find sequence directory for: {input_path}")
        sys.exit(1)
    
    print(f"Sequence directory: {seq_path}")
    
    # Extract MPS trajectory
    traj_file = extract_mps_trajectory(seq_path)
    if traj_file:
        print(f"\nGround truth trajectory extracted to: {traj_file}")
        print("\nTo compare with ORB-SLAM3 output:")
        print(f"python evaluate_trajectory.py results/f_my_trajectory.txt {traj_file}")
    else:
        print("ERROR: Could not extract ground truth trajectory")