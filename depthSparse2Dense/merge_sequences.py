#!/usr/bin/env python3
"""
Merge multiple sparse depth sequences into a single training dataset
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm


def merge_sequences(input_dirs, output_dir, train_ratio=0.9):
    """
    Merge multiple sequence directories into a single dataset
    
    Args:
        input_dirs: List of input sequence directories
        output_dir: Output directory for merged dataset
        train_ratio: Ratio of data to use for training (rest for validation)
    """
    output_path = Path(output_dir)
    
    # Create output directories
    for subdir in ['sparse_depth', 'rgb', 'poses', 'metadata']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    # Collect all frames from all sequences
    all_frames = []
    global_frame_id = 0
    sequence_info = {}
    
    print(f"Processing {len(input_dirs)} sequences...")
    
    for seq_idx, input_dir in enumerate(input_dirs):
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: {input_dir} does not exist, skipping...")
            continue
            
        seq_name = input_path.name.replace('sparse_', '')
        print(f"\nProcessing sequence {seq_idx + 1}/{len(input_dirs)}: {seq_name}")
        
        # Load sequence metadata
        metadata_file = input_path / 'metadata' / 'frames.json'
        if not metadata_file.exists():
            print(f"  Warning: No metadata found for {seq_name}, skipping...")
            continue
            
        with open(metadata_file, 'r') as f:
            frames = json.load(f)
        
        sequence_info[seq_name] = {
            'start_frame': global_frame_id,
            'num_frames': len(frames),
            'original_dir': str(input_path)
        }
        
        # Process each frame
        for frame in tqdm(frames, desc=f"  Copying {seq_name}"):
            original_frame_id = frame['frame_id']
            frame_str = f"{original_frame_id:06d}"
            new_frame_str = f"{global_frame_id:06d}"
            
            # Update frame info
            new_frame = frame.copy()
            new_frame['frame_id'] = global_frame_id
            new_frame['original_frame_id'] = original_frame_id
            new_frame['sequence'] = seq_name
            new_frame['sequence_idx'] = seq_idx
            
            # Copy sparse depth
            src_sparse = input_path / 'sparse_depth' / f'{frame_str}.npy'
            src_conf = input_path / 'sparse_depth' / f'{frame_str}_conf.npy'
            if src_sparse.exists():
                shutil.copy2(src_sparse, output_path / 'sparse_depth' / f'{new_frame_str}.npy')
                if src_conf.exists():
                    shutil.copy2(src_conf, output_path / 'sparse_depth' / f'{new_frame_str}_conf.npy')
            
            # Copy RGB
            src_rgb = input_path / 'rgb' / f'{frame_str}.png'
            if src_rgb.exists():
                shutil.copy2(src_rgb, output_path / 'rgb' / f'{new_frame_str}.png')
            
            # Copy pose
            src_pose = input_path / 'poses' / f'{frame_str}.npy'
            if src_pose.exists():
                shutil.copy2(src_pose, output_path / 'poses' / f'{new_frame_str}.npy')
            
            all_frames.append(new_frame)
            global_frame_id += 1
        
        sequence_info[seq_name]['end_frame'] = global_frame_id - 1
    
    # Save merged metadata
    metadata_output = output_path / 'metadata'
    
    # Save all frames
    with open(metadata_output / 'frames.json', 'w') as f:
        json.dump(all_frames, f, indent=2)
    
    # Save sequence info
    with open(metadata_output / 'sequences.json', 'w') as f:
        json.dump(sequence_info, f, indent=2)
    
    # Copy camera parameters from first sequence (assuming all use same camera)
    first_input = Path(input_dirs[0])
    camera_params_src = first_input / 'metadata' / 'camera_params.json'
    if camera_params_src.exists():
        shutil.copy2(camera_params_src, metadata_output / 'camera_params.json')
    
    intrinsics_src = first_input / 'metadata' / 'intrinsics.npy'
    if intrinsics_src.exists():
        shutil.copy2(intrinsics_src, metadata_output / 'intrinsics.npy')
    
    # Create train/val split info
    n_total = len(all_frames)
    n_train = int(n_total * train_ratio)
    
    # Shuffle frames for random split (with fixed seed for reproducibility)
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    
    train_indices = sorted(indices[:n_train].tolist())
    val_indices = sorted(indices[n_train:].tolist())
    
    split_info = {
        'train': train_indices,
        'val': val_indices,
        'train_ratio': train_ratio,
        'random_seed': 42
    }
    
    with open(metadata_output / 'split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Print summary
    print(f"\nMerge complete!")
    print(f"Total frames: {n_total}")
    print(f"Training frames: {n_train}")
    print(f"Validation frames: {n_total - n_train}")
    print(f"\nSequence summary:")
    for seq_name, info in sequence_info.items():
        print(f"  {seq_name}: {info['num_frames']} frames (IDs {info['start_frame']}-{info['end_frame']})")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Merge multiple sparse depth sequences')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='Input sequence directories (e.g., sparse_seq131 sparse_seq133 ...)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for merged dataset')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of data for training (default: 0.9)')
    parser.add_argument('--pattern', help='Pattern to find input directories (e.g., "sparse_*")')
    
    args = parser.parse_args()
    
    # Handle pattern-based input
    if args.pattern:
        from glob import glob
        input_dirs = sorted(glob(args.pattern))
        if not input_dirs:
            print(f"No directories found matching pattern: {args.pattern}")
            return
    else:
        input_dirs = args.input_dirs
    
    # Filter out non-existent directories
    input_dirs = [d for d in input_dirs if Path(d).exists()]
    
    if not input_dirs:
        print("No valid input directories found!")
        return
    
    print(f"Found {len(input_dirs)} input directories:")
    for d in input_dirs:
        print(f"  - {d}")
    
    merge_sequences(input_dirs, args.output_dir, args.train_ratio)


if __name__ == '__main__':
    main()