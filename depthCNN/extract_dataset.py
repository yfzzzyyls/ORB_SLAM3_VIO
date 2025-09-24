#!/usr/bin/env python3
"""
Extract and organize ADT dataset into train/val/test folders for easier access.
Extracts RGB and depth images from VRS files and saves them as PNG/NPZ files.
Also extracts gaze information from eyegaze.csv files.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
import pandas as pd

# Fix projectaria_tools import
sys.path.append('/home/external/.local/lib/python3.9/site-packages')
from projectaria_tools.core import data_provider


def load_gaze_data(eyegaze_csv_path: Path) -> pd.DataFrame:
    """
    Load gaze data from eyegaze.csv file.
    Returns DataFrame with columns: tracking_timestamp_us, yaw_rads_cpf, pitch_rads_cpf
    """
    if not eyegaze_csv_path.exists():
        return None
    
    # Load only the columns we need
    gaze_df = pd.read_csv(eyegaze_csv_path, usecols=[
        'tracking_timestamp_us', 
        'yaw_rads_cpf', 
        'pitch_rads_cpf'
    ])
    
    # Convert timestamp from microseconds to nanoseconds for consistency
    gaze_df['timestamp_ns'] = gaze_df['tracking_timestamp_us'] * 1000
    
    return gaze_df


def pitch_yaw_to_pixel_coords(pitch_rad: float, yaw_rad: float, 
                              width: int = 1408, height: int = 1408,
                              fov_degrees: float = 98.1) -> Tuple[int, int]:
    """
    Convert pitch/yaw angles to pixel coordinates for RGB camera.
    
    Args:
        pitch_rad: Pitch angle in radians (positive down)
        yaw_rad: Yaw angle in radians (positive right)
        width: Image width in pixels (1408 for ADT RGB)
        height: Image height in pixels (1408 for ADT RGB)
        fov_degrees: Field of view in degrees (98.1 for ADT RGB camera)
    
    Returns:
        (x, y) pixel coordinates, or (-1, -1) if outside image bounds
    """
    # Convert FOV to radians
    fov_rad = np.radians(fov_degrees)
    
    # Calculate focal length in pixels (assuming square pixels)
    focal_length = (width / 2) / np.tan(fov_rad / 2)
    
    # Project to image plane
    # Note: ADT uses different conventions, may need to adjust signs
    x = focal_length * np.tan(yaw_rad) + width / 2
    y = focal_length * np.tan(pitch_rad) + height / 2
    
    # Round to integer pixel coordinates
    x_pixel = int(round(x))
    y_pixel = int(round(y))
    
    # Check if within image bounds
    if 0 <= x_pixel < width and 0 <= y_pixel < height:
        return x_pixel, y_pixel
    else:
        return -1, -1


def find_nearest_gaze_point(rgb_timestamp_ns: int, gaze_df: pd.DataFrame,
                           tolerance_ns: int = 1_000_000) -> Optional[Dict]:
    """
    Find the nearest gaze point for a given RGB timestamp.
    Returns dict with gaze info or None if no match within tolerance.
    """
    if gaze_df is None or len(gaze_df) == 0:
        return None
    
    # Find nearest timestamp
    time_diffs = np.abs(gaze_df['timestamp_ns'] - rgb_timestamp_ns)
    min_idx = time_diffs.idxmin()
    min_diff = time_diffs[min_idx]
    
    if min_diff <= tolerance_ns:
        gaze_row = gaze_df.iloc[min_idx]
        
        # Convert pitch/yaw to pixel coordinates
        x_pixel, y_pixel = pitch_yaw_to_pixel_coords(
            gaze_row['pitch_rads_cpf'],
            gaze_row['yaw_rads_cpf']
        )
        
        return {
            'timestamp_us': int(gaze_row['tracking_timestamp_us']),
            'pitch_rad': float(gaze_row['pitch_rads_cpf']),
            'yaw_rad': float(gaze_row['yaw_rads_cpf']),
            'x_pixel': x_pixel,
            'y_pixel': y_pixel,
            'time_diff_ms': float(min_diff / 1e6)
        }
    
    return None


def find_nearest_depth_frame(rgb_timestamp_ns, depth_provider, depth_stream_id, 
                           tolerance_ns=1_000_000):  # 1ms tolerance
    """
    Find the nearest depth frame for a given RGB timestamp.
    Returns (depth_index, time_diff_ns) or (None, None) if no match within tolerance.
    """
    # Binary search would be more efficient, but for simplicity we'll search nearby frames
    # This assumes depth timestamps are monotonically increasing
    
    # Get a reasonable search range
    num_depth = depth_provider.get_num_data(depth_stream_id)
    
    # Start with a coarse search to find approximate location
    best_idx = None
    best_diff = float('inf')
    
    # Sample every 100 frames for initial search
    for i in range(0, num_depth, 100):
        depth_data = depth_provider.get_image_data_by_index(depth_stream_id, i)
        depth_ts = depth_data[1].capture_timestamp_ns
        diff = abs(rgb_timestamp_ns - depth_ts)
        
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    
    # Fine search around best index
    search_start = max(0, best_idx - 100)
    search_end = min(num_depth, best_idx + 100)
    
    best_idx = None
    best_diff = float('inf')
    
    for i in range(search_start, search_end):
        depth_data = depth_provider.get_image_data_by_index(depth_stream_id, i)
        depth_ts = depth_data[1].capture_timestamp_ns
        diff = abs(rgb_timestamp_ns - depth_ts)
        
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    
    # Check if within tolerance
    if best_diff <= tolerance_ns:
        return best_idx, best_diff
    else:
        return None, None


def extract_sequence(seq_info: dict) -> dict:
    """Extract one sequence using timestamp-based matching."""
    seq_name = seq_info['seq_name']
    seq_dir = seq_info['seq_dir']
    output_dir = seq_info['output_dir']
    subsample = seq_info['subsample']
    
    # Create output directory
    seq_output_dir = output_dir / seq_name
    rgb_dir = seq_output_dir / 'rgb'
    depth_dir = seq_output_dir / 'depth'
    gaze_dir = seq_output_dir / 'gaze'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    gaze_dir.mkdir(parents=True, exist_ok=True)
    
    # Find VRS files and gaze data
    rgb_vrs = None
    depth_vrs = None
    eyegaze_csv = None
    
    for file in os.listdir(seq_dir):
        if file.endswith('_main_recording.vrs'):
            rgb_vrs = seq_dir / file
        elif file == 'depth_images.vrs':
            depth_vrs = seq_dir / file
        elif file == 'eyegaze.csv':
            eyegaze_csv = seq_dir / file
    
    if not rgb_vrs or not depth_vrs:
        return {
            'sequence': seq_name,
            'status': 'failed',
            'error': f"Missing VRS files in {seq_name}"
        }
    
    try:
        # Create providers
        rgb_provider = data_provider.create_vrs_data_provider(str(rgb_vrs))
        depth_provider = data_provider.create_vrs_data_provider(str(depth_vrs))
        
        # Get RGB stream
        rgb_stream_id = rgb_provider.get_stream_id_from_label("camera-rgb")
        
        # Find RGB depth stream (1408x1408)
        depth_streams = depth_provider.get_all_streams()
        depth_stream_id = None
        
        for stream_id in depth_streams:
            try:
                test_frame = depth_provider.get_image_data_by_index(stream_id, 0)
                if test_frame and test_frame[0]:
                    shape = test_frame[0].to_numpy_array().shape
                    if shape[0] == 1408 and shape[1] == 1408:
                        depth_stream_id = stream_id
                        break
            except:
                pass
        
        if depth_stream_id is None:
            return {
                'sequence': seq_name,
                'status': 'failed',
                'error': "Could not find RGB depth stream (1408x1408)"
            }
        
        # Get frame counts
        num_rgb_frames = rgb_provider.get_num_data(rgb_stream_id)
        num_depth_frames = depth_provider.get_num_data(depth_stream_id)
        
        print(f"\n{seq_name}: RGB={num_rgb_frames}, Depth={num_depth_frames}")
        
        # Load gaze data if available
        gaze_df = None
        if eyegaze_csv:
            gaze_df = load_gaze_data(eyegaze_csv)
            if gaze_df is not None:
                print(f"  Loaded {len(gaze_df)} gaze samples")
        
        # Process RGB frames with subsampling
        extracted_count = 0
        matched_count = 0
        gaze_matched_count = 0
        frame_indices = range(0, num_rgb_frames, subsample)
        
        # Save metadata
        metadata = {
            'sequence': seq_name,
            'num_frames': 0,  # Will be updated
            'subsample': subsample,
            'rgb_shape': None,
            'depth_shape': None,
            'has_gaze': gaze_df is not None,
            'frames': []
        }
        
        for idx, rgb_idx in enumerate(tqdm(frame_indices, desc=f"Extracting {seq_name}")):
            try:
                # Get RGB frame and timestamp
                rgb_data = rgb_provider.get_image_data_by_index(rgb_stream_id, rgb_idx)
                rgb_image = rgb_data[0].to_numpy_array()
                rgb_timestamp_ns = rgb_data[1].capture_timestamp_ns
                
                # Find matching depth frame by timestamp
                depth_idx, time_diff = find_nearest_depth_frame(
                    rgb_timestamp_ns, depth_provider, depth_stream_id
                )
                
                if depth_idx is None:
                    continue  # Skip if no matching depth found
                
                matched_count += 1
                
                # Get depth frame
                depth_data = depth_provider.get_image_data_by_index(depth_stream_id, depth_idx)
                depth_image = depth_data[0].to_numpy_array()
                depth_timestamp_ns = depth_data[1].capture_timestamp_ns
                
                # Save metadata for first frame
                if metadata['rgb_shape'] is None:
                    metadata['rgb_shape'] = list(rgb_image.shape)
                    metadata['depth_shape'] = list(depth_image.shape)
                
                # Save RGB as PNG
                rgb_filename = f"frame_{extracted_count:06d}.png"
                rgb_path = rgb_dir / rgb_filename
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                
                # Save depth as compressed numpy array (millimeters as uint16)
                depth_filename = f"frame_{extracted_count:06d}.npz"
                depth_path = depth_dir / depth_filename
                np.savez_compressed(depth_path, depth=depth_image)
                
                # Find matching gaze point
                gaze_info = None
                gaze_filename = None
                if gaze_df is not None:
                    gaze_info = find_nearest_gaze_point(rgb_timestamp_ns, gaze_df)
                    if gaze_info:
                        gaze_matched_count += 1
                        # Save gaze info as JSON
                        gaze_filename = f"frame_{extracted_count:06d}.json"
                        gaze_path = gaze_dir / gaze_filename
                        with open(gaze_path, 'w') as f:
                            json.dump(gaze_info, f, indent=2)
                
                # Add to metadata
                frame_metadata = {
                    'index': extracted_count,
                    'rgb_index': rgb_idx,
                    'depth_index': depth_idx,
                    'rgb_timestamp_ns': int(rgb_timestamp_ns),
                    'depth_timestamp_ns': int(depth_timestamp_ns),
                    'time_diff_ms': float(time_diff / 1e6),
                    'rgb': rgb_filename,
                    'depth': depth_filename
                }
                
                if gaze_filename:
                    frame_metadata['gaze'] = gaze_filename
                    frame_metadata['has_gaze'] = True
                else:
                    frame_metadata['has_gaze'] = False
                
                metadata['frames'].append(frame_metadata)
                
                extracted_count += 1
                
            except Exception as e:
                # Only print first few errors
                if extracted_count < 10:
                    print(f"Error processing frame {rgb_idx}: {e}")
                continue
        
        # Update frame count
        metadata['num_frames'] = extracted_count
        
        # Save metadata
        metadata_path = seq_output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Matched {matched_count}/{len(frame_indices)} RGB frames to depth")
        if gaze_df is not None:
            print(f"  Matched {gaze_matched_count}/{extracted_count} frames to gaze")
        print(f"  Extracted {extracted_count} frame pairs")
        
        return {
            'sequence': seq_name,
            'status': 'success',
            'extracted_frames': extracted_count,
            'total_rgb_frames': len(frame_indices),
            'matched_frames': matched_count,
            'gaze_matched_frames': gaze_matched_count,
            'has_gaze': gaze_df is not None
        }
        
    except Exception as e:
        return {
            'sequence': seq_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Extract ADT dataset for training')
    parser.add_argument('--data-root', type=str, default='/mnt/ssd_ext/incSeg-data/adt',
                        help='Path to ADT dataset root')
    parser.add_argument('--output-dir', type=str, default='./processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Subsample factor (1=all frames, 10=every 10th frame)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--use-split-info', action='store_true',
                        help='Use split_info.json for sequence selection')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum sequences to process (for testing)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    
    # Create output directories
    train_dir = output_root / 'train'
    val_dir = output_root / 'val'
    test_dir = output_root / 'test'
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find sequences based on mode
    train_sequences = []
    val_sequences = []
    test_sequences = []
    
    if args.use_split_info:
        # Load sequences from split_info.json
        split_info_path = data_root / 'split_info.json'
        if not split_info_path.exists():
            print(f"Error: {split_info_path} not found!")
            print("Please run download_100_sequences.py first.")
            return
        
        with open(split_info_path, 'r') as f:
            split_data = json.load(f)
        
        train_sequences = split_data.get('train', [])
        val_sequences = split_data.get('val', [])
        test_sequences = split_data.get('test', [])
        
        print(f"Loaded from split_info.json: {split_data.get('total', 0)} sequences")
    else:
        # Original behavior - look for clean sequences
        # Check train directory
        train_path = data_root / 'train'
        if train_path.exists():
            train_seqs = sorted([
                d for d in os.listdir(train_path)
                if d.startswith("Apartment_release_clean_seq") and 
                os.path.isdir(os.path.join(train_path, d))
            ])
            # Use first 7 for train, next 1 for val
            train_sequences = train_seqs[:7]
            val_sequences = train_seqs[7:8] if len(train_seqs) > 7 else []
        
        # Check test directory
        test_path = data_root / 'test'
        if test_path.exists():
            test_sequences = sorted([
                d for d in os.listdir(test_path)
                if d.startswith("Apartment_release_clean_seq") and 
                os.path.isdir(os.path.join(test_path, d))
            ])[:2]  # Use first 2 test sequences
    
    # Apply max sequences limit if specified
    if args.max_sequences:
        total_seqs = len(train_sequences) + len(val_sequences) + len(test_sequences)
        if total_seqs > args.max_sequences:
            # Proportionally reduce each split
            train_ratio = len(train_sequences) / total_seqs
            val_ratio = len(val_sequences) / total_seqs
            test_ratio = len(test_sequences) / total_seqs
            
            train_max = int(args.max_sequences * train_ratio)
            val_max = int(args.max_sequences * val_ratio)
            test_max = args.max_sequences - train_max - val_max
            
            train_sequences = train_sequences[:train_max]
            val_sequences = val_sequences[:val_max]
            test_sequences = test_sequences[:test_max]
    
    all_sequences = train_sequences + val_sequences + test_sequences
    
    print(f"Found {len(all_sequences)} sequences")
    print(f"Train: {len(train_sequences)} sequences")
    print(f"Val: {len(val_sequences)} sequences")
    print(f"Test: {len(test_sequences)} sequences")
    
    # Prepare extraction tasks
    tasks = []
    
    for split_name, sequences, output_dir in [
        ('train', train_sequences, train_dir),
        ('val', val_sequences, val_dir),
        ('test', test_sequences, test_dir)
    ]:
        print(f"\n{split_name.upper()} sequences:")
        for seq in sequences:
            # Find sequence directory - try multiple locations
            seq_dir = None
            
            # Try split-specific directory first
            if args.use_split_info:
                # For new structure, sequences might be in their respective split dirs
                candidate = data_root / split_name / seq
                if candidate.exists():
                    seq_dir = candidate
                else:
                    # Try without split subdirectory
                    candidate = data_root / seq
                    if candidate.exists():
                        seq_dir = candidate
            else:
                # Original structure
                if split_name in ['train', 'val']:
                    seq_dir = data_root / 'train' / seq
                else:  # test
                    seq_dir = data_root / 'test' / seq
            
            if seq_dir and seq_dir.exists():
                print(f"  ✓ {seq}")
                tasks.append({
                    'seq_name': seq,
                    'seq_dir': seq_dir,
                    'output_dir': output_dir,
                    'subsample': args.subsample
                })
            else:
                print(f"  ✗ {seq} - Not found")
    
    # Extract sequences in parallel
    print(f"\nExtracting sequences with {args.num_workers} workers...")
    results = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(extract_sequence, task): task 
            for task in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✓ {result['sequence']}: {result['extracted_frames']} frames")
            else:
                print(f"✗ {result['sequence']}: {result['error']}")
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successful: {len(successful)}/{len(results)} sequences")
    print(f"Failed: {len(failed)}/{len(results)} sequences")
    
    if failed:
        print("\nFailed sequences:")
        for r in failed:
            print(f"  - {r['sequence']}: {r['error']}")
    
    # Create split info file
    split_info = {
        'train': train_sequences,
        'val': val_sequences,
        'test': test_sequences,
        'subsample': args.subsample,
        'extraction_results': results
    }
    
    split_info_path = output_root / 'split_info.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSplit info saved to: {split_info_path}")
    
    # Print dataset statistics
    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        total_frames = 0
        for seq_dir in split_dir.iterdir():
            if seq_dir.is_dir():
                metadata_path = seq_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        total_frames += metadata['num_frames']
        
        print(f"\n{split_name.upper()}: {total_frames} total frames")
    
    print(f"\nProcessed data saved to: {output_root}")
    print("\nTo use with training:")
    print(f"python train.py --data-root {output_root}")


if __name__ == "__main__":
    main()