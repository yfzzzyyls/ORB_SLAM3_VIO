#!/usr/bin/env python3
"""
Extract and organize ADT dataset using timestamp-based matching between RGB and depth.
This correctly handles the time offset between RGB and depth recordings.
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
from typing import List, Tuple

# Fix projectaria_tools import
sys.path.append('/home/external/.local/lib/python3.9/site-packages')
from projectaria_tools.core import data_provider


def find_nearest_depth_frame(rgb_timestamp_ns, depth_provider, depth_stream_id, 
                           tolerance_ns=50_000_000):  # 50ms tolerance
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


def extract_sequence_timestamp_based(seq_info: dict) -> dict:
    """Extract one sequence using timestamp-based matching."""
    seq_name = seq_info['seq_name']
    seq_dir = seq_info['seq_dir']
    output_dir = seq_info['output_dir']
    subsample = seq_info['subsample']
    
    # Create output directory
    seq_output_dir = output_dir / seq_name
    rgb_dir = seq_output_dir / 'rgb'
    depth_dir = seq_output_dir / 'depth'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Find VRS files
    rgb_vrs = None
    depth_vrs = None
    
    for file in os.listdir(seq_dir):
        if file.endswith('_main_recording.vrs'):
            rgb_vrs = seq_dir / file
        elif file == 'depth_images.vrs':
            depth_vrs = seq_dir / file
    
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
        
        # Process RGB frames with subsampling
        extracted_count = 0
        matched_count = 0
        frame_indices = range(0, num_rgb_frames, subsample)
        
        # Save metadata
        metadata = {
            'sequence': seq_name,
            'num_frames': 0,  # Will be updated
            'subsample': subsample,
            'rgb_shape': None,
            'depth_shape': None,
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
                
                # Add to metadata
                metadata['frames'].append({
                    'index': extracted_count,
                    'rgb_index': rgb_idx,
                    'depth_index': depth_idx,
                    'rgb_timestamp_ns': int(rgb_timestamp_ns),
                    'depth_timestamp_ns': int(depth_timestamp_ns),
                    'time_diff_ms': float(time_diff / 1e6),
                    'rgb': rgb_filename,
                    'depth': depth_filename
                })
                
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
        print(f"  Extracted {extracted_count} frame pairs")
        
        return {
            'sequence': seq_name,
            'status': 'success',
            'extracted_frames': extracted_count,
            'total_rgb_frames': len(frame_indices),
            'matched_frames': matched_count
        }
        
    except Exception as e:
        return {
            'sequence': seq_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Extract ADT dataset with timestamp-based RGB-depth matching')
    parser.add_argument('--data-root', type=str, default='/mnt/ssd_ext/incSeg-data/adt',
                        help='Path to ADT dataset root')
    parser.add_argument('--output-dir', type=str, default='./processed_data_timestamped',
                        help='Output directory for processed data')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Subsample factor (1=all frames, 10=every 10th frame)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define train/val/test splits
    train_sequences = [
        "Apartment_release_clean_seq131_M1292",
        "Apartment_release_clean_seq133_M1292",
        "Apartment_release_clean_seq134_M1292",
        "Apartment_release_clean_seq135_M1292",
        "Apartment_release_clean_seq136_M1292",
        "Apartment_release_clean_seq137_M1292",
        "Apartment_release_clean_seq138_M1292",
    ]
    
    val_sequences = [
        "Apartment_release_clean_seq140_M1292",
    ]
    
    test_sequences = [
        "Apartment_release_clean_seq141_M1292",
        "Apartment_release_clean_seq142_M1292",
    ]
    
    # Create split directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Save split info
    split_info = {
        'train': train_sequences,
        'val': val_sequences,
        'test': test_sequences
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Extracting ADT dataset with timestamp-based matching")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Subsample: {args.subsample}")
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
            print(f"  - {seq}")
            tasks.append({
                'seq_name': seq,
                'seq_dir': data_root / seq,
                'output_dir': output_dir,
                'subsample': args.subsample
            })
    
    # Extract sequences
    print(f"\nExtracting sequences with timestamp matching...")
    results = []
    
    # Process one at a time for clearer output
    for task in tasks:
        result = extract_sequence_timestamp_based(task)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✓ {result['sequence']}: {result['extracted_frames']} frames extracted")
        else:
            print(f"✗ {result['sequence']}: {result['error']}")
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successful: {len(successful)}/{len(results)} sequences")
    print(f"Failed: {len(failed)} sequences")
    
    if successful:
        total_frames = sum(r['extracted_frames'] for r in successful)
        print(f"Total frames extracted: {total_frames}")
    
    if failed:
        print("\nFailed sequences:")
        for r in failed:
            print(f"  - {r['sequence']}: {r['error']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()