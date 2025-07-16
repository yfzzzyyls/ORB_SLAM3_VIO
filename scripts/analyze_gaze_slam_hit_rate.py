#!/usr/bin/env python3
"""Analyze how often gazed objects have at least one SLAM point.

Samples frames starting from frame 500, every 25 frames up to frame 3000.
This ensures SLAM has stabilized and provides denser temporal sampling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append('/home/external/.local/lib/python3.9/site-packages')
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
import json


def convert_gaze_to_slam(pitch_rad, yaw_rad):
    """Convert gaze angles to SLAM camera pixel coordinates (after rotation)."""
    # Original SLAM camera: 640x480 (before rotation), ~150° FOV
    orig_width, orig_height = 640, 480
    fov_rad = np.radians(150.0)
    focal_length = (orig_width / 2) / np.tan(fov_rad / 2)
    
    # Project to original image plane (before rotation)
    x_orig = focal_length * np.tan(yaw_rad) + orig_width / 2
    y_orig = focal_length * np.tan(pitch_rad) + orig_height / 2
    
    # Apply 90° clockwise rotation transformation
    x_rotated = y_orig
    y_rotated = orig_width - x_orig - 1
    
    # Check bounds (rotated image is 480x640)
    if 0 <= x_rotated < 480 and 0 <= y_rotated < 640:
        return int(x_rotated), int(y_rotated)
    else:
        return -1, -1


def load_slam_features(tracking_file):
    """Load SLAM features from tracking file."""
    slam_features = []
    if tracking_file.exists():
        try:
            with open(tracking_file, 'r') as f:
                lines = f.readlines()
                # Skip header lines (9 lines)
                for line in lines[9:]:
                    line = line.strip()
                    # Skip empty lines and comment lines
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            u, v, depth = float(parts[0]), float(parts[1]), float(parts[2])
                            # Only include features with valid depth
                            if depth > 0:
                                slam_features.append((int(u), int(v), depth))
                        except ValueError:
                            # Skip lines that can't be parsed as numbers
                            continue
        except Exception as e:
            pass
    return slam_features


def load_instance_categories(sequence_name, vrs_dir="/mnt/ssd_ext/incSeg-data/adt"):
    """Load instance ID to category mapping from instances.json."""
    seq_dir = None
    for subdir in ['train', 'test']:
        potential_dir = Path(vrs_dir) / subdir / sequence_name
        if potential_dir.exists():
            seq_dir = potential_dir
            break
    
    if seq_dir is None:
        return {}
        
    instances_file = seq_dir / "instances.json"
    if not instances_file.exists():
        return {}
    
    with open(instances_file, 'r') as f:
        instances = json.load(f)
    
    # Create ID to category mapping
    id_to_category = {}
    for instance_id, info in instances.items():
        if isinstance(info, dict) and 'category' in info:
            id_to_category[int(instance_id)] = info['category']
    
    return id_to_category


def analyze_hit_rate_sequence(sequence):
    """Analyze what percentage of frames have SLAM points on gazed object."""
    
    # Setup paths
    vrs_dir = Path(f"/mnt/ssd_ext/incSeg-data/adt/train/{sequence}")
    vrs_file = vrs_dir / f"ADT_{sequence}_main_recording.vrs"
    seg_vrs = vrs_dir / "segmentations.vrs"
    tracking_dir = Path(f"/home/external/ORB_SLAM3_VIO/results/tracking_data_{sequence}_trajectory")
    
    if not tracking_dir.exists():
        return None
    
    # Load VRS files
    main_provider = data_provider.create_vrs_data_provider(str(vrs_file))
    seg_provider = data_provider.create_vrs_data_provider(str(seg_vrs))
    
    # Load gaze data
    gaze_file = vrs_dir / "eyegaze.csv"
    gaze_points = []
    if gaze_file.exists():
        gaze_df = pd.read_csv(gaze_file)
        for _, row in gaze_df.iterrows():
            gaze_points.append({
                'timestamp_us': row['tracking_timestamp_us'],
                'pitch': row['pitch_rads_cpf'],
                'yaw': row['yaw_rads_cpf']
            })
    
    # Get SLAM stream
    slam_stream = main_provider.get_stream_id_from_label("camera-slam-left")
    num_frames = main_provider.get_num_data(slam_stream)
    
    # Sample frames - start from frame 500, every 25 frames
    sample_indices = range(500, min(3000, num_frames), 25)
    
    # Counters
    frames_analyzed = 0
    frames_with_slam_on_gaze = 0
    frames_no_slam_on_gaze = 0
    object_hit_stats = {}
    
    for slam_idx in sample_indices:
        # Get timestamp
        slam_data = main_provider.get_image_data_by_index(slam_stream, slam_idx)
        if slam_data is None:
            continue
        timestamp = slam_data[1].capture_timestamp_ns / 1e9
        
        # Load SLAM features
        tracking_file = tracking_dir / f"frame_{slam_idx:06d}.txt"
        slam_features = load_slam_features(tracking_file)
        
        if not slam_features:
            continue
            
        # Get segmentation using inline extraction
        try:
            # Get SLAM segmentation stream (400-2, index 1)
            seg_streams = seg_provider.get_all_streams()
            if len(seg_streams) < 2:
                continue
            slam_seg_stream_id = seg_streams[1]  # 400-2 is SLAM-left
            
            # Get SLAM frame timestamp
            slam_timestamp_ns = slam_data[1].capture_timestamp_ns
            
            # First attempt: Direct timestamp matching
            seg_data = seg_provider.get_image_data_by_time_ns(
                slam_seg_stream_id, 
                slam_timestamp_ns, 
                TimeDomain.RECORD_TIME, 
                TimeQueryOptions.CLOSEST
            )
            
            if seg_data is not None:
                # Check if the match is reasonable (within 100ms)
                seg_timestamp_ns = seg_data[1].capture_timestamp_ns
                time_diff_ms = abs(seg_timestamp_ns - slam_timestamp_ns) / 1e6
                
                if time_diff_ms > 100:  # If difference > 100ms, try offset approach
                    seg_data = None
            
            # If direct matching failed or gave poor result, estimate offset
            if seg_data is None:
                # Get first timestamps to estimate initial offset
                slam_data_0 = main_provider.get_image_data_by_index(slam_stream, 0)
                seg_data_0 = seg_provider.get_image_data_by_index(slam_seg_stream_id, 0)
                
                if slam_data_0 and seg_data_0:
                    slam_ts_0 = slam_data_0[1].capture_timestamp_ns
                    seg_ts_0 = seg_data_0[1].capture_timestamp_ns
                    initial_offset = seg_ts_0 - slam_ts_0
                    
                    # For early frames (< 500), use the initial offset
                    # For later frames, offset should be near 0
                    if slam_idx < 500:
                        offset_ns = initial_offset
                    else:
                        # Interpolate: gradually reduce offset
                        progress = min(1.0, (slam_idx - 500) / 500)
                        offset_ns = initial_offset * (1 - progress)
                    
                    # Try with estimated offset
                    adjusted_timestamp = slam_timestamp_ns + offset_ns
                    seg_data = seg_provider.get_image_data_by_time_ns(
                        slam_seg_stream_id, 
                        adjusted_timestamp, 
                        TimeDomain.RECORD_TIME, 
                        TimeQueryOptions.CLOSEST
                    )
            
            if seg_data is None:
                continue
            
            # Get segmentation mask and rotate 90° clockwise to match SLAM
            seg_mask = seg_data[0].to_numpy_array()
            seg_mask = np.rot90(seg_mask, k=3)
            
            # Load category mapping
            id_to_category = load_instance_categories(sequence)
            
        except Exception as e:
            continue
            
        # Find closest gaze
        timestamp_us = timestamp * 1e6
        closest_gaze = None
        min_diff = float('inf')
        
        for gaze in gaze_points:
            diff = abs(gaze['timestamp_us'] - timestamp_us)
            if diff < min_diff and diff < 33000:
                min_diff = diff
                closest_gaze = gaze
        
        if not closest_gaze:
            continue
            
        # Get gaze location
        gaze_x, gaze_y = convert_gaze_to_slam(closest_gaze['pitch'], closest_gaze['yaw'])
        if not (0 <= gaze_x < 480 and 0 <= gaze_y < 640):
            continue
            
        # Get gazed object
        gaze_object_id = seg_mask[gaze_y, gaze_x]
        gaze_category = id_to_category.get(gaze_object_id, "unknown") if gaze_object_id > 0 else "background"
        
        # Check if ANY SLAM features are on the gazed object
        has_slam_on_gaze = False
        for (x, y, depth) in slam_features:
            if 0 <= x < 480 and 0 <= y < 640:
                if seg_mask[y, x] == gaze_object_id:
                    has_slam_on_gaze = True
                    break
        
        frames_analyzed += 1
        
        if has_slam_on_gaze:
            frames_with_slam_on_gaze += 1
        else:
            frames_no_slam_on_gaze += 1
            
        # Track per-object statistics
        if gaze_category not in object_hit_stats:
            object_hit_stats[gaze_category] = {'hits': 0, 'total': 0}
        object_hit_stats[gaze_category]['total'] += 1
        if has_slam_on_gaze:
            object_hit_stats[gaze_category]['hits'] += 1
    
    return {
        'frames_analyzed': frames_analyzed,
        'frames_with_slam': frames_with_slam_on_gaze,
        'frames_no_slam': frames_no_slam_on_gaze,
        'hit_rate': (frames_with_slam_on_gaze / frames_analyzed * 100) if frames_analyzed > 0 else 0,
        'object_stats': object_hit_stats
    }


def main():
    """Analyze hit rate for all sequences."""
    
    sequences = [
        "Apartment_release_clean_seq131_M1292",
        "Apartment_release_clean_seq133_M1292",
        "Apartment_release_clean_seq134_M1292",
        "Apartment_release_clean_seq135_M1292",
        "Apartment_release_clean_seq136_M1292",
        "Apartment_release_clean_seq137_M1292",
        "Apartment_release_clean_seq138_M1292",
        "Apartment_release_clean_seq140_M1292"
    ]
    
    print("="*70)
    print("SLAM HIT RATE ANALYSIS")
    print("How often does the gazed object have at least 1 SLAM point?")
    print("Sampling: Frames 500-3000, every 25 frames")
    print("="*70)
    
    all_results = []
    total_frames = 0
    total_hits = 0
    all_object_stats = {}
    
    for seq in sequences:
        print(f"\nAnalyzing {seq.split('_')[-2]}...")
        result = analyze_hit_rate_sequence(seq)
        
        if result:
            all_results.append(result)
            total_frames += result['frames_analyzed']
            total_hits += result['frames_with_slam']
            
            print(f"  Frames analyzed: {result['frames_analyzed']}")
            print(f"  Hit rate: {result['hit_rate']:.1f}% ({result['frames_with_slam']}/{result['frames_analyzed']})")
            
            # Merge object stats
            for obj, stats in result['object_stats'].items():
                if obj not in all_object_stats:
                    all_object_stats[obj] = {'hits': 0, 'total': 0}
                all_object_stats[obj]['hits'] += stats['hits']
                all_object_stats[obj]['total'] += stats['total']
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    
    if total_frames > 0:
        overall_hit_rate = (total_hits / total_frames) * 100
        print(f"\nTotal frames analyzed: {total_frames}")
        print(f"Frames with SLAM on gazed object: {total_hits} ({overall_hit_rate:.1f}%)")
        print(f"Frames without SLAM on gazed object: {total_frames - total_hits} ({100-overall_hit_rate:.1f}%)")
        
        print(f"\n**ANSWER: {overall_hit_rate:.1f}% of the time, gazed objects have at least 1 SLAM point**")
    
    # Per-object hit rates
    print("\n" + "-"*70)
    print("Hit rate by object type:")
    print(f"{'Object Type':<25} {'Hit Rate':<15} {'Times Gazed'}")
    print("-"*70)
    
    sorted_objects = sorted(all_object_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for obj, stats in sorted_objects[:15]:  # Top 15 objects
        if stats['total'] > 0:
            hit_rate = (stats['hits'] / stats['total']) * 100
            print(f"{obj:<25} {hit_rate:>6.1f}%         {stats['total']:>4}")
    
    # Per-sequence hit rates
    print("\n" + "-"*70)
    print("Hit rate by sequence:")
    print(f"{'Sequence':<15} {'Hit Rate':<15} {'Frames Analyzed'}")
    print("-"*70)
    
    sequence_hit_rates = []
    for i, seq in enumerate(sequences):
        if i < len(all_results) and all_results[i]:
            result = all_results[i]
            seq_name = seq.split('_')[-2]  # Extract seq number
            hit_rate = result['hit_rate']
            sequence_hit_rates.append(hit_rate)
            print(f"{seq_name:<15} {hit_rate:>6.1f}%         {result['frames_analyzed']:>4}")
    
    if sequence_hit_rates:
        avg_hit_rate = sum(sequence_hit_rates) / len(sequence_hit_rates)
        print("-"*70)
        print(f"{'Average':<15} {avg_hit_rate:>6.1f}%")
        print("-"*70)


if __name__ == "__main__":
    main()