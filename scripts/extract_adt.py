#!/usr/bin/env python3
"""
Comprehensive ADT data extraction script with parallel processing.
Extracts all available data streams efficiently using multiple CPUs.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
import time

# Fix projectaria_tools import
sys.path.append('/home/external/.local/lib/python3.9/site-packages')
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


def load_gaze_data(eyegaze_csv_path: Path) -> pd.DataFrame:
    """Load gaze data from eyegaze.csv file."""
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


def find_nearest_timestamp(target_ns, timestamps, tolerance_ns=1_000_000):
    """Find the nearest timestamp using binary search."""
    left, right = 0, len(timestamps) - 1
    best_idx = None
    best_diff = float('inf')
    
    while left <= right:
        mid = (left + right) // 2
        diff = abs(target_ns - timestamps[mid])
        
        if diff < best_diff:
            best_diff = diff
            best_idx = mid
        
        if target_ns < timestamps[mid]:
            right = mid - 1
        else:
            left = mid + 1
    
    if best_diff <= tolerance_ns:
        return best_idx, best_diff
    else:
        return None, None


def process_camera_stream(provider, stream_id, output_dir, indices, stream_name="camera", show_progress=True):
    """Process and save camera frames."""
    saved_count = 0
    failed_count = 0
    
    iterator = tqdm(indices, desc=f"    {stream_name}") if show_progress else indices
    
    for idx in iterator:
        try:
            data = provider.get_image_data_by_index(stream_id, idx)
            if data and data[0]:
                img = data[0].to_numpy_array()
                
                # Save image
                filename = f"frame_{idx:06d}.png"
                output_path = output_dir / filename
                
                if 'depth' in stream_name or 'segmentation' in stream_name:
                    # Save as 16-bit PNG
                    success = cv2.imwrite(str(output_path), img.astype(np.uint16))
                else:
                    # Regular image (RGB or grayscale)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        # Convert RGB to BGR for OpenCV
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(str(output_path), img)
                
                if success:
                    saved_count += 1
                else:
                    failed_count += 1
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:  # Only print first few errors
                print(f"      Error processing {stream_name} frame {idx}: {e}")
    
    return saved_count, failed_count


def extract_calibration_data(device_calib):
    """Extract calibration data from device calibration object."""
    calibration_data = {
        'device_name': 'Aria',
        'cameras': {},
        'imus': {}
    }
    
    # Camera calibrations
    camera_labels = ['camera-rgb', 'camera-slam-left', 'camera-slam-right', 'camera-et']
    
    for label in camera_labels:
        try:
            calib = device_calib.get_camera_calib(label)
            cam_name = label.replace('camera-', '')
            
            calibration_data['cameras'][cam_name] = {
                'label': label,
                'model_name': str(calib.get_model_name()),
                'image_width': int(calib.get_image_size()[0]),
                'image_height': int(calib.get_image_size()[1]),
                'focal_lengths': [float(calib.get_focal_lengths()[0]), float(calib.get_focal_lengths()[1])],
                'principal_point': [float(calib.get_principal_point()[0]), float(calib.get_principal_point()[1])],
                'distortion_coeffs': list(calib.get_projection_params()[3:]) if hasattr(calib, 'get_projection_params') else [],
                'transform_device_camera': calib.get_transform_device_camera().to_matrix().tolist(),
                'projection_params': list(calib.get_projection_params()) if hasattr(calib, 'get_projection_params') else None
            }
        except Exception as e:
            # Skip if camera not available
            pass
    
    # IMU calibrations
    imu_labels = ['imu-right', 'imu-left']
    
    for label in imu_labels:
        try:
            calib = device_calib.get_imu_calib(label)
            imu_name = label.replace('imu-', '')
            
            calibration_data['imus'][imu_name] = {
                'label': label,
                'model': 'calibrated',
                'update_rate': 1000.0 if 'right' in label else 800.0,  # Right IMU is 1000Hz, left is ~800Hz
                'transform_device_imu': calib.get_transform_device_imu().to_matrix().tolist()
            }
        except Exception as e:
            # Skip if IMU not available
            pass
    
    return calibration_data


def extract_imu_data(provider, imu_dir, start_idx, end_idx, subsample=1):
    """Extract IMU data for the sequence."""
    # Get IMU stream IDs
    imu_right_id = provider.get_stream_id_from_label("imu-right")
    imu_left_id = provider.get_stream_id_from_label("imu-left")
    
    # Get camera timestamps for time range
    camera_stream_id = provider.get_stream_id_from_label("camera-rgb")
    start_data = provider.get_image_data_by_index(camera_stream_id, start_idx * subsample)
    end_data = provider.get_image_data_by_index(camera_stream_id, end_idx * subsample)
    
    if not start_data or not end_data:
        print("Warning: Could not get camera timestamps for IMU extraction")
        return
    
    start_ns = start_data[1].capture_timestamp_ns
    end_ns = end_data[1].capture_timestamp_ns
    
    # Extract IMU data for both IMUs
    for imu_id, imu_name in [(imu_right_id, 'imu_data.json'), (imu_left_id, 'imu_left_data.json')]:
        if imu_id is None:
            continue
            
        imu_samples = []
        num_samples = provider.get_num_data(imu_id)
        
        print(f"    Extracting {imu_name}: {num_samples} total samples")
        
        # Use binary search to find start/end indices
        start_imu_idx = 0
        end_imu_idx = num_samples - 1
        
        # Find first IMU sample after start time
        for i in range(min(1000, num_samples)):
            data = provider.get_imu_data_by_index(imu_id, i)
            if data.capture_timestamp_ns >= start_ns:
                start_imu_idx = i
                break
        
        # Find last IMU sample before end time
        for i in range(max(0, num_samples - 1000), num_samples):
            data = provider.get_imu_data_by_index(imu_id, i)
            if data.capture_timestamp_ns > end_ns:
                end_imu_idx = i - 1
                break
        
        # Extract IMU samples in range
        for i in range(start_imu_idx, min(end_imu_idx + 1, num_samples)):
            data = provider.get_imu_data_by_index(imu_id, i)
            
            # Use the correct attribute names for the current API
            if hasattr(data, 'gyro_radsec') and hasattr(data, 'accel_msec2'):
                imu_samples.append({
                    'timestamp_ns': int(data.capture_timestamp_ns),
                    'gyro': list(data.gyro_radsec),
                    'accel': list(data.accel_msec2)
                })
        
        # Save IMU data
        imu_path = imu_dir / imu_name
        with open(imu_path, 'w') as f:
            json.dump(imu_samples, f)
        
        print(f"      Saved {len(imu_samples)} IMU samples")


def extract_other_sensors(provider, output_dir, start_idx, end_idx, subsample=1):
    """Extract magnetometer, barometer, and other sensor data."""
    sensors_dir = output_dir / 'sensors'
    sensors_dir.mkdir(exist_ok=True)
    
    # Get camera timestamps for time range
    camera_stream_id = provider.get_stream_id_from_label("camera-rgb")
    start_data = provider.get_image_data_by_index(camera_stream_id, start_idx * subsample)
    end_data = provider.get_image_data_by_index(camera_stream_id, end_idx * subsample)
    
    if not start_data or not end_data:
        return
    
    start_ns = start_data[1].capture_timestamp_ns
    end_ns = end_data[1].capture_timestamp_ns
    
    # Extract magnetometer data
    try:
        mag_id = provider.get_stream_id_from_label("mag0")
        if mag_id:
            mag_samples = []
            num_samples = provider.get_num_data(mag_id)
            
            for i in range(num_samples):
                data = provider.get_magnetometer_data_by_index(mag_id, i)
                if start_ns <= data.capture_timestamp_ns <= end_ns:
                    mag_samples.append({
                        'timestamp_ns': int(data.capture_timestamp_ns),
                        'magnetic_field': list(data.mag_tesla)
                    })
            
            if mag_samples:
                with open(sensors_dir / 'magnetometer.json', 'w') as f:
                    json.dump(mag_samples, f)
                print(f"      Saved {len(mag_samples)} magnetometer samples")
    except:
        pass
    
    # Extract barometer data
    try:
        baro_id = provider.get_stream_id_from_label("baro0")
        if baro_id:
            baro_samples = []
            num_samples = provider.get_num_data(baro_id)
            
            for i in range(num_samples):
                data = provider.get_barometer_data_by_index(baro_id, i)
                if start_ns <= data.capture_timestamp_ns <= end_ns:
                    baro_samples.append({
                        'timestamp_ns': int(data.capture_timestamp_ns),
                        'pressure': float(data.pressure()),
                        'temperature': float(data.temperature())
                    })
            
            if baro_samples:
                with open(sensors_dir / 'barometer.json', 'w') as f:
                    json.dump(baro_samples, f)
                print(f"      Saved {len(baro_samples)} barometer samples")
    except:
        pass


def extract_sequence(seq_info: dict) -> dict:
    """Extract all data streams from one sequence."""
    seq_name = seq_info['seq_name']
    seq_dir = seq_info['seq_dir']
    output_dir = seq_info['output_dir']
    subsample = seq_info['subsample']
    
    print(f"\n{'='*60}")
    print(f"Processing: {seq_name}")
    print(f"{'='*60}")
    
    # VRS file paths
    main_vrs = seq_dir / f"ADT_{seq_name}_main_recording.vrs"
    depth_vrs = seq_dir / "depth_images.vrs"
    seg_vrs = seq_dir / "segmentations.vrs"
    eyegaze_csv = seq_dir / "eyegaze.csv"
    instances_json = seq_dir / "instances.json"
    
    # Check if VRS files exist
    if not main_vrs.exists():
        return {'sequence': seq_name, 'status': 'failed', 'error': 'Missing main VRS file'}
    
    try:
        # Create output directories
        seq_output_dir = output_dir / seq_name
        directories = {
            'rgb': seq_output_dir / 'rgb',
            'slam_left': seq_output_dir / 'slam_left',
            'slam_right': seq_output_dir / 'slam_right',
            'rgb_depth': seq_output_dir / 'rgb_depth',
            'slam_left_depth': seq_output_dir / 'slam_left_depth',
            'slam_right_depth': seq_output_dir / 'slam_right_depth',
            'rgb_segmentation': seq_output_dir / 'rgb_segmentation',
            'slam_left_segmentation': seq_output_dir / 'slam_left_segmentation',
            'slam_right_segmentation': seq_output_dir / 'slam_right_segmentation',
            'et': seq_output_dir / 'et',
            'imu': seq_output_dir / 'imu',
            'calibration': seq_output_dir / 'calibration',
            'gaze': seq_output_dir / 'gaze'
        }
        
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Open VRS providers to get stream information
        print("  Opening VRS files...")
        main_provider = data_provider.create_vrs_data_provider(str(main_vrs))
        depth_provider = data_provider.create_vrs_data_provider(str(depth_vrs)) if depth_vrs.exists() else None
        seg_provider = data_provider.create_vrs_data_provider(str(seg_vrs)) if seg_vrs.exists() else None
        
        # Extract and save calibration data
        print("  Extracting calibration data...")
        device_calib = main_provider.get_device_calibration()
        calibration_data = extract_calibration_data(device_calib)
        
        calib_path = directories['calibration'] / 'calibration.json'
        with open(calib_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"  Saved calibration for {len(calibration_data['cameras'])} cameras and {len(calibration_data['imus'])} IMUs")
        
        # Get stream IDs
        rgb_stream_id = main_provider.get_stream_id_from_label("camera-rgb")
        slam_left_stream_id = main_provider.get_stream_id_from_label("camera-slam-left")
        slam_right_stream_id = main_provider.get_stream_id_from_label("camera-slam-right")
        et_stream_id = main_provider.get_stream_id_from_label("camera-et")
        
        # Get number of frames
        num_rgb_frames = main_provider.get_num_data(rgb_stream_id)
        num_slam_frames = main_provider.get_num_data(slam_left_stream_id)
        num_et_frames = main_provider.get_num_data(et_stream_id) if et_stream_id else 0
        
        print(f"\n{seq_name}:")
        print(f"  RGB: {num_rgb_frames} frames")
        print(f"  SLAM cameras: {num_slam_frames} frames each")
        if num_et_frames > 0:
            print(f"  Eye tracking: {num_et_frames} frames")
        
        # Load gaze data if available
        gaze_df = None
        if eyegaze_csv.exists():
            gaze_df = load_gaze_data(eyegaze_csv)
            if gaze_df is not None:
                print(f"  Gaze: {len(gaze_df)} samples")
        
        # Find depth and segmentation streams
        depth_stream_map = {}
        seg_stream_map = {}
        
        if depth_provider:
            depth_streams = depth_provider.get_all_streams()
            for stream_id in depth_streams:
                try:
                    test_data = depth_provider.get_image_data_by_index(stream_id, 0)
                    if test_data and test_data[0]:
                        img = test_data[0].to_numpy_array()
                        size = (img.shape[1], img.shape[0])
                        
                        if size == (1408, 1408):
                            depth_stream_map['rgb'] = stream_id
                            print(f"  Found RGB depth stream (1408x1408)")
                        elif size == (640, 480):
                            if 'slam_left' not in depth_stream_map:
                                depth_stream_map['slam_left'] = stream_id
                                print(f"  Found SLAM left depth stream (640x480)")
                            else:
                                depth_stream_map['slam_right'] = stream_id
                                print(f"  Found SLAM right depth stream (640x480)")
                except:
                    pass
        
        if seg_provider:
            seg_streams = seg_provider.get_all_streams()
            for stream_id in seg_streams:
                try:
                    test_data = seg_provider.get_image_data_by_index(stream_id, 0)
                    if test_data and test_data[0]:
                        img = test_data[0].to_numpy_array()
                        size = (img.shape[1], img.shape[0])
                        
                        if size == (1408, 1408):
                            seg_stream_map['rgb'] = stream_id
                            print(f"  Found RGB segmentation stream (1408x1408)")
                        elif size == (640, 480):
                            if 'slam_left' not in seg_stream_map:
                                seg_stream_map['slam_left'] = stream_id
                                print(f"  Found SLAM left segmentation stream (640x480)")
                            else:
                                seg_stream_map['slam_right'] = stream_id
                                print(f"  Found SLAM right segmentation stream (640x480)")
                except:
                    pass
        
        # Process RGB frames in parallel
        print("\n  Processing RGB frames...")
        rgb_indices = list(range(0, num_rgb_frames, subsample))
        
        # Get RGB timestamps
        rgb_timestamps = []
        for idx in rgb_indices:
            data = main_provider.get_image_data_by_index(rgb_stream_id, idx)
            if data:
                rgb_timestamps.append(data[1].capture_timestamp_ns)
        
        # Save RGB timestamps
        with open(seq_output_dir / 'rgb_timestamps.json', 'w') as f:
            json.dump([{'idx': i, 'timestamp_ns': int(ts)} for i, ts in enumerate(rgb_timestamps)], f)
        
        # Process RGB frames
        rgb_count, rgb_failed = process_camera_stream(main_provider, rgb_stream_id, directories['rgb'], 
                                                     rgb_indices, "rgb")
        print(f"    Saved {rgb_count} RGB frames")
        
        # Process RGB depth if available
        if depth_provider and 'rgb' in depth_stream_map:
            depth_count, _ = process_camera_stream(depth_provider, depth_stream_map['rgb'], 
                                                  directories['rgb_depth'], rgb_indices, "rgb_depth")
            print(f"    Saved {depth_count} RGB depth frames")
        
        # Process RGB segmentation if available
        if seg_provider and 'rgb' in seg_stream_map:
            seg_count, _ = process_camera_stream(seg_provider, seg_stream_map['rgb'], 
                                               directories['rgb_segmentation'], rgb_indices, "rgb_segmentation")
            print(f"    Saved {seg_count} RGB segmentation frames")
        
        # Process SLAM frames in parallel
        print("\n  Processing SLAM frames...")
        slam_indices = list(range(0, num_slam_frames, subsample))
        
        # Get SLAM timestamps
        slam_timestamps = []
        for idx in slam_indices:
            data = main_provider.get_image_data_by_index(slam_left_stream_id, idx)
            if data:
                slam_timestamps.append(data[1].capture_timestamp_ns)
        
        # Save SLAM timestamps (both left and right use same timestamps)
        with open(seq_output_dir / 'slam_left_timestamps.json', 'w') as f:
            json.dump([{'idx': i, 'timestamp_ns': int(ts)} for i, ts in enumerate(slam_timestamps)], f)
        with open(seq_output_dir / 'slam_right_timestamps.json', 'w') as f:
            json.dump([{'idx': i, 'timestamp_ns': int(ts)} for i, ts in enumerate(slam_timestamps)], f)
        
        # Process SLAM left
        slam_left_count, _ = process_camera_stream(main_provider, slam_left_stream_id, 
                                                  directories['slam_left'], slam_indices, "slam_left")
        print(f"    Saved {slam_left_count} SLAM left frames")
        
        # Process SLAM right
        slam_right_count, _ = process_camera_stream(main_provider, slam_right_stream_id, 
                                                   directories['slam_right'], slam_indices, "slam_right")
        print(f"    Saved {slam_right_count} SLAM right frames")
        
        # Process SLAM depth if available
        if depth_provider:
            if 'slam_left' in depth_stream_map:
                depth_count, _ = process_camera_stream(depth_provider, depth_stream_map['slam_left'], 
                                                      directories['slam_left_depth'], slam_indices, "slam_left_depth")
                print(f"    Saved {depth_count} SLAM left depth frames")
            
            if 'slam_right' in depth_stream_map:
                depth_count, _ = process_camera_stream(depth_provider, depth_stream_map['slam_right'], 
                                                      directories['slam_right_depth'], slam_indices, "slam_right_depth")
                print(f"    Saved {depth_count} SLAM right depth frames")
        
        # Process SLAM segmentation if available
        if seg_provider:
            if 'slam_left' in seg_stream_map:
                seg_count, _ = process_camera_stream(seg_provider, seg_stream_map['slam_left'], 
                                                    directories['slam_left_segmentation'], slam_indices, "slam_left_segmentation")
                print(f"    Saved {seg_count} SLAM left segmentation frames")
            
            if 'slam_right' in seg_stream_map:
                seg_count, _ = process_camera_stream(seg_provider, seg_stream_map['slam_right'], 
                                                    directories['slam_right_segmentation'], slam_indices, "slam_right_segmentation")
                print(f"    Saved {seg_count} SLAM right segmentation frames")
        
        # Extract Eye Tracking if available
        et_count = 0
        if et_stream_id and num_et_frames > 0:
            print("\n  Processing eye tracking frames...")
            et_indices = list(range(0, num_et_frames, subsample))
            
            # Get ET timestamps
            et_timestamps = []
            for idx in et_indices:
                data = main_provider.get_image_data_by_index(et_stream_id, idx)
                if data:
                    et_timestamps.append(data[1].capture_timestamp_ns)
            
            et_count, _ = process_camera_stream(main_provider, et_stream_id, directories['et'], et_indices, "et")
            print(f"    Saved {et_count} eye tracking frames")
        
        # Extract IMU data (sequential, fast)
        print("\n  Extracting IMU data...")
        extract_imu_data(main_provider, directories['imu'], 0, len(slam_indices) - 1, subsample)
        
        # Extract other sensors
        print("\n  Extracting other sensors...")
        extract_other_sensors(main_provider, seq_output_dir, 0, len(slam_indices) - 1, subsample)
        
        # Save gaze data with RGB frame matching
        if gaze_df is not None and len(rgb_timestamps) > 0:
            print("\n  Saving gaze data...")
            gaze_data = []
            
            for i, ts in enumerate(rgb_timestamps):
                gaze_idx, time_diff = find_nearest_timestamp(ts, gaze_df['timestamp_ns'].values)
                
                if gaze_idx is not None:
                    gaze_row = gaze_df.iloc[gaze_idx]
                    gaze_data.append({
                        'frame_idx': i,
                        'timestamp_ns': int(ts),
                        'timestamp_us': int(gaze_row['tracking_timestamp_us']),
                        'pitch_rad': float(gaze_row['pitch_rads_cpf']),
                        'yaw_rad': float(gaze_row['yaw_rads_cpf']),
                        'time_diff_ms': float(time_diff / 1e6)
                    })
            
            if gaze_data:
                with open(directories['gaze'] / 'gaze_data.json', 'w') as f:
                    json.dump(gaze_data, f, indent=2)
                print(f"    Saved {len(gaze_data)} gaze samples")
        
        # Copy instances.json if exists
        if instances_json.exists():
            import shutil
            shutil.copy(instances_json, seq_output_dir / 'instances.json')
            print("  Copied instances.json")
        
        # Save metadata
        metadata = {
            'sequence': seq_name,
            'num_frames': {
                'rgb': rgb_count,
                'slam_left': slam_left_count,
                'slam_right': slam_right_count,
                'et': et_count if et_stream_id else 0
            },
            'subsample': subsample,
            'has_depth': depth_provider is not None,
            'has_segmentation': seg_provider is not None,
            'has_gaze': gaze_df is not None,
            'calibration': calibration_data,
            'streams_extracted': list(directories.keys())
        }
        
        metadata_path = seq_output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ {seq_name}: Extraction complete")
        print(f"  RGB frames: {rgb_count}")
        print(f"  SLAM frames: {slam_left_count}/{slam_right_count} (left/right)")
        
        return {
            'sequence': seq_name,
            'status': 'success',
            'frames_extracted': {
                'rgb': rgb_count,
                'slam_left': slam_left_count,
                'slam_right': slam_right_count
            }
        }
        
    except Exception as e:
        print(f"\n✗ {seq_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'sequence': seq_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extract all data streams from ADT sequences with parallel processing'
    )
    parser.add_argument('--data-root', type=str, 
                        default='/mnt/ssd_ext/incSeg-data/adt',
                        help='Root directory of ADT dataset')
    parser.add_argument('--output-dir', type=str, 
                        default='../processed_adt',
                        help='Output directory')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Subsample rate (1=all frames, 10=every 10th frame)')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequences to process')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes for parallel sequence processing')
    
    args = parser.parse_args()
    
    # No need to set multiprocessing start method here
    
    # Find sequences
    data_root = Path(args.data_root)
    train_dir = data_root / 'train'
    test_dir = data_root / 'test'
    
    sequences = []
    dataset_splits = {'train': [], 'test': []}
    
    # Find all sequences
    for split, split_dir in [('train', train_dir), ('test', test_dir)]:
        if split_dir.exists():
            for seq_dir in sorted(split_dir.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.startswith('Apartment'):
                    seq_name = seq_dir.name
                    print(f"  Found {split} sequence: {seq_name}")
                    
                    if args.sequences is None or seq_name in args.sequences:
                        sequences.append({
                            'seq_name': seq_name,
                            'seq_dir': seq_dir,
                            'split': split,
                            'output_dir': Path(args.output_dir),
                            'subsample': args.subsample
                        })
                    
                    dataset_splits[split].append(seq_name)
    
    print(f"\nFound {len(sequences)} sequences to process")
    print(f"  Train: {len(dataset_splits['train'])} sequences")
    print(f"  Test: {len(dataset_splits['test'])} sequences")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset splits
    with open(output_dir / 'dataset_splits.json', 'w') as f:
        json.dump(dataset_splits, f, indent=2)
    
    # Process sequences
    print(f"\nExtracting sequences...")
    start_time = time.time()
    
    if args.workers == 1 or len(sequences) == 1:
        # Sequential processing
        results = []
        for seq_info in sequences:
            result = extract_sequence(seq_info)
            results.append(result)
    else:
        # Parallel processing
        num_workers = args.workers or min(4, len(sequences))
        print(f"Using {num_workers} workers")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(extract_sequence, seq_info) for seq_info in sequences]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Sequence processing failed: {e}")
                    results.append({'status': 'failed', 'error': str(e)})
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successful: {len(successful)}/{len(results)} sequences")
    print(f"Failed: {len(failed)}/{len(results)} sequences")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    
    if failed:
        print("\nFailed sequences:")
        for r in failed:
            print(f"  - {r['sequence']}: {r['error']}")
    
    # Save summary
    summary = {
        'total_sequences': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'results': results,
        'elapsed_time_minutes': elapsed_time/60
    }
    
    with open(output_dir / 'extraction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Data saved to: {args.output_dir}")
    print(f"Summary saved to: {args.output_dir}/extraction_summary.json")
    print(f"Dataset splits saved to: {args.output_dir}/dataset_splits.json")


if __name__ == "__main__":
    main()