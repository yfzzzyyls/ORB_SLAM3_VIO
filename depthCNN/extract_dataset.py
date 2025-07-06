#!/usr/bin/env python3
"""
Extract and organize ADT dataset into train/val/test folders for easier access.
Extracts RGB and depth images from VRS files and saves them as PNG/NPZ files.
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


def extract_sequence(seq_info: dict) -> dict:
    """Extract one sequence and save to processed folder."""
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
        
        # Get stream IDs
        rgb_stream_id = rgb_provider.get_stream_id_from_label("camera-rgb")
        # First stream in depth VRS is RGB camera depth
        depth_stream_id = depth_provider.get_all_streams()[0]
        
        # Get number of frames
        num_frames = rgb_provider.get_num_data(rgb_stream_id)
        
        # Extract frames with subsampling
        extracted_count = 0
        frame_indices = range(0, num_frames, subsample)
        
        # Save metadata
        metadata = {
            'sequence': seq_name,
            'num_frames': len(frame_indices),
            'subsample': subsample,
            'rgb_shape': None,
            'depth_shape': None,
            'frames': []
        }
        
        for idx, frame_idx in enumerate(tqdm(frame_indices, desc=f"Extracting {seq_name}")):
            try:
                # Get RGB image
                rgb_data = rgb_provider.get_image_data_by_index(rgb_stream_id, frame_idx)
                rgb_image = rgb_data[0].to_numpy_array()
                timestamp_ns = rgb_data[1].capture_timestamp_ns
                
                # Get depth image
                depth_data = depth_provider.get_image_data_by_index(depth_stream_id, frame_idx)
                depth_image = depth_data[0].to_numpy_array()
                
                # Save metadata for first frame
                if metadata['rgb_shape'] is None:
                    metadata['rgb_shape'] = list(rgb_image.shape)
                    metadata['depth_shape'] = list(depth_image.shape)
                
                # Save RGB as PNG
                rgb_filename = f"frame_{idx:06d}.png"
                rgb_path = rgb_dir / rgb_filename
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                
                # Save depth as compressed numpy array (millimeters as uint16)
                depth_filename = f"frame_{idx:06d}.npz"
                depth_path = depth_dir / depth_filename
                np.savez_compressed(depth_path, depth=depth_image)
                
                # Add to metadata
                metadata['frames'].append({
                    'index': idx,
                    'original_index': frame_idx,
                    'timestamp_ns': timestamp_ns,
                    'rgb': rgb_filename,
                    'depth': depth_filename
                })
                
                extracted_count += 1
                
            except Exception as e:
                print(f"Error extracting frame {frame_idx} from {seq_name}: {e}")
                continue
        
        # Save metadata
        metadata_path = seq_output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'sequence': seq_name,
            'status': 'success',
            'extracted_frames': extracted_count,
            'total_frames': len(frame_indices)
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
    
    # Find all sequences
    all_sequences = sorted([
        d for d in os.listdir(data_root)
        if d.startswith("Apartment_release_clean_seq") and 
        os.path.isdir(os.path.join(data_root, d))
    ])[:10]  # Use first 10 sequences
    
    # Split sequences: 7 train, 1 val, 2 test
    train_sequences = all_sequences[:7]
    val_sequences = all_sequences[7:8]
    test_sequences = all_sequences[8:10]
    
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
            print(f"  - {seq}")
            tasks.append({
                'seq_name': seq,
                'seq_dir': data_root / seq,
                'output_dir': output_dir,
                'subsample': args.subsample
            })
    
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