#!/usr/bin/env python3
"""Create 4-panel visualization using specified sequence."""

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd
import argparse

# Add projectaria_tools to path
sys.path.append('/home/external/.local/lib/python3.9/site-packages')
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


def convert_gaze_to_slam(pitch_rad, yaw_rad):
    """Convert gaze angles to SLAM camera pixel coordinates (unrotated 640x480)."""
    # SLAM camera: 640x480, ~150° FOV
    width, height = 640, 480
    fov_rad = np.radians(150.0)
    focal_length = (width / 2) / np.tan(fov_rad / 2)
    
    # Project to image plane
    x = focal_length * np.tan(yaw_rad) + width / 2
    y = focal_length * np.tan(pitch_rad) + height / 2
    
    # Check bounds
    if 0 <= x < width and 0 <= y < height:
        return int(x), int(y)
    else:
        return -1, -1


def create_semantic_colormap():
    """Create a colormap for semantic categories."""
    # Define colors for common ADT categories (RGB format)
    category_colors = {
        'wall': [200, 200, 200],
        'floor': [140, 140, 140],
        'door': [139, 69, 19],
        'chair': [255, 165, 0],
        'table': [165, 42, 42],
        'couch': [128, 0, 128],
        'bed': [255, 192, 203],
        'shelf': [210, 180, 140],
        'cabinet': [101, 67, 33],
        'part of a cabinet/wardrobe': [101, 67, 33],
        'book': [0, 128, 0],
        'wall artwork': [64, 224, 208],
        'wine rack': [139, 69, 19],
        'thermostat': [192, 192, 192],
        'fork': [192, 192, 192],
        'baking pan': [105, 105, 105],
        'pet bowl': [200, 50, 100],
        'vase': [255, 215, 0],
        'candle': [255, 255, 0],
        'notebook': [100, 200, 100],
        'decorative accessory': [255, 200, 200],
        'freestanding accessory': [200, 200, 255],
        'lamp': [255, 255, 0],
        'pillow': [255, 182, 193],
        'blanket': [188, 143, 143],
        'toy': [255, 105, 180],
        'clothing': [147, 112, 219],
        'towel': [176, 224, 230],
    }
    
    # Default colors for unknown categories
    default_colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
        [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128],
    ]
    
    return category_colors, default_colors


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


def create_4panel_frame(slam_idx, sequence, vrs_provider, seg_provider, gaze_points, timestamp, tracking_dir):
    """Create 4-panel visualization for a single frame."""
    
    print(f"Processing SLAM frame {slam_idx} at timestamp {timestamp:.6f}s", end="")
    
    # Get SLAM image by timestamp
    slam_stream_id = vrs_provider.get_stream_id_from_label("camera-slam-right")
    slam_timestamp_ns = int(timestamp * 1e9)
    slam_data = vrs_provider.get_image_data_by_time_ns(
        slam_stream_id, slam_timestamp_ns, TimeDomain.RECORD_TIME, TimeQueryOptions.CLOSEST
    )
    
    if slam_data is None:
        return None
        
    slam_image = slam_data[0].to_numpy_array()
    # Keep original orientation (640x480)
    
    # Convert to RGB
    if len(slam_image.shape) == 2:
        slam_image_rgb = cv2.cvtColor(slam_image, cv2.COLOR_GRAY2RGB)
    else:
        slam_image_rgb = slam_image.copy()
    
    # Extract segmentation inline
    try:
        # Get SLAM segmentation stream (400-3 for right SLAM camera)
        seg_streams = seg_provider.get_all_streams()
        if len(seg_streams) < 3:
            return None
        slam_seg_stream_id = seg_streams[2]  # 400-3 is SLAM-right
        
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
            # Get SLAM stream for first frame timestamp
            slam_stream_id_for_offset = vrs_provider.get_stream_id_from_label("camera-slam-right")
            slam_data_0 = vrs_provider.get_image_data_by_index(slam_stream_id_for_offset, 0)
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
            return None
        
        # Get segmentation mask (keep original orientation)
        seg_mask = seg_data[0].to_numpy_array()
        
        # Load category mapping
        id_to_category = load_instance_categories(sequence)
        
    except Exception as e:
        print(f"Error extracting segmentation: {e}")
        return None
    
    # Find closest gaze point
    timestamp_us = timestamp * 1e6
    closest_gaze = None
    min_diff = float('inf')
    
    for gaze in gaze_points:
        diff = abs(gaze['timestamp_us'] - timestamp_us)
        if diff < min_diff and diff < 33000:  # 33ms tolerance
            min_diff = diff
            closest_gaze = gaze
    
    # Create 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: SLAM features from actual tracking data
    panel1 = slam_image_rgb.copy()
    
    # Load SLAM tracking data
    tracking_file = tracking_dir / f"frame_{slam_idx:06d}.txt"
    
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
                                slam_features.append((int(u), int(v)))
                        except ValueError:
                            # Skip lines that can't be parsed as numbers
                            continue
        except Exception as e:
            print(f"Error reading tracking file: {e}")
    
    print(f" - {len(slam_features)} SLAM features with depth")
    
    # Draw SLAM features
    for (x, y) in slam_features:
        cv2.circle(panel1, (x, y), 3, (0, 255, 0), -1)  # Green dots
    
    axes[0, 0].imshow(panel1)
    axes[0, 0].set_title(f'SLAM Frame {slam_idx}', fontsize=14)
    axes[0, 0].axis('off')
    
    # Panel 2: Segmentation mask only
    category_colors, default_colors = create_semantic_colormap()
    seg_viz = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    
    unique_instances = np.unique(seg_mask)
    for idx, instance_id in enumerate(unique_instances):
        if instance_id == 0:
            continue
        
        category = id_to_category.get(instance_id, "unknown")
        if category in category_colors:
            color = category_colors[category]
        else:
            color_idx = hash(category) % len(default_colors)
            color = default_colors[color_idx]
        
        mask = seg_mask == instance_id
        seg_viz[mask] = color
    
    axes[0, 1].imshow(seg_viz)
    axes[0, 1].set_title('Segmentation Mask', fontsize=14)
    axes[0, 1].axis('off')
    
    # Panel 3: Gaze location only
    panel3 = slam_image_rgb.copy()
    gaze_text = "No gaze data"
    if closest_gaze:
        gaze_x, gaze_y = convert_gaze_to_slam(closest_gaze['pitch'], closest_gaze['yaw'])
        if 0 <= gaze_x < 640 and 0 <= gaze_y < 480:
            # Draw large crosshair
            cv2.line(panel3, (gaze_x - 30, gaze_y), (gaze_x + 30, gaze_y), (0, 255, 0), 3)
            cv2.line(panel3, (gaze_x, gaze_y - 30), (gaze_x, gaze_y + 30), (0, 255, 0), 3)
            cv2.circle(panel3, (gaze_x, gaze_y), 20, (0, 255, 0), 3)
            
            # Check what object gaze is on
            if 0 <= gaze_y < seg_mask.shape[0] and 0 <= gaze_x < seg_mask.shape[1]:
                object_id = seg_mask[gaze_y, gaze_x]
                if object_id > 0:
                    category = id_to_category.get(object_id, "unknown")
                    gaze_text = f"Gaze on: {category}"
                else:
                    gaze_text = "Gaze on: background"
    
    axes[1, 0].imshow(panel3)
    axes[1, 0].set_title(f'Gaze Location ({gaze_text})', fontsize=14)
    axes[1, 0].axis('off')
    
    # Panel 4: All overlapping
    panel4 = slam_image_rgb.copy()
    
    # Add segmentation overlay (40% opacity)
    panel4 = cv2.addWeighted(panel4, 0.6, seg_viz, 0.4, 0)
    
    # Add SLAM features
    for (x, y) in slam_features:
        cv2.circle(panel4, (x, y), 3, (0, 255, 0), -1)  # Green dots
    
    # Add gaze
    if closest_gaze:
        gaze_x, gaze_y = convert_gaze_to_slam(closest_gaze['pitch'], closest_gaze['yaw'])
        if 0 <= gaze_x < 640 and 0 <= gaze_y < 480:
            cv2.line(panel4, (gaze_x - 20, gaze_y), (gaze_x + 20, gaze_y), (0, 255, 0), 2)
            cv2.line(panel4, (gaze_x, gaze_y - 20), (gaze_x, gaze_y + 20), (0, 255, 0), 2)
            cv2.circle(panel4, (gaze_x, gaze_y), 15, (0, 255, 0), 2)
    
    axes[1, 1].imshow(panel4)
    axes[1, 1].set_title('All Overlapping', fontsize=14)
    axes[1, 1].axis('off')
    
    # Add main title
    fig.suptitle(f'Frame {slam_idx} (t={timestamp:.3f}s) - {sequence}', fontsize=16)
    plt.tight_layout()
    
    # Save the figure to a temporary buffer and reload to rotate
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # Read the image and rotate 90° clockwise for natural viewing
    img_array = plt.imread(buf)
    img_rotated = np.rot90(img_array, k=3)
    
    # Create new figure with rotated image
    fig_final = plt.figure(figsize=(10, 12))
    plt.imshow(img_rotated)
    plt.axis('off')
    plt.tight_layout()
    
    return fig_final


def main():
    """Create 4-panel visualizations for specified sequence."""
    
    parser = argparse.ArgumentParser(description='Visualize overlapping SLAM features, segmentation, and gaze')
    parser.add_argument('--seq', type=str, required=True,
                        help='Sequence name (e.g., Apartment_release_clean_seq135_M1292)')
    parser.add_argument('--frames', type=int, default=15,
                        help='Number of frames to visualize (default: 15)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: 4panel_[sequence])')
    
    args = parser.parse_args()
    
    sequence = args.seq
    
    # Find VRS files
    vrs_dir = None
    for subdir in ['train', 'test']:
        potential_dir = Path(f"/mnt/ssd_ext/incSeg-data/adt/{subdir}/{sequence}")
        if potential_dir.exists():
            vrs_dir = potential_dir
            break
    
    if vrs_dir is None:
        print(f"Error: Cannot find sequence {sequence} in ADT dataset")
        return
    
    vrs_file = vrs_dir / f"ADT_{sequence}_main_recording.vrs"
    seg_vrs = vrs_dir / "segmentations.vrs"
    
    print(f"Using sequence: {sequence}")
    
    # Check if SLAM tracking data exists
    tracking_dir = Path(f"../results/tracking_data_{sequence}_trajectory")
    if not tracking_dir.exists():
        print(f"Error: No tracking data found at {tracking_dir}")
        print("Please run SLAM on this sequence first")
        return
    
    # Load VRS files
    print(f"Loading VRS files...")
    main_provider = data_provider.create_vrs_data_provider(str(vrs_file))
    seg_provider = data_provider.create_vrs_data_provider(str(seg_vrs))
    
    # Load gaze data
    gaze_file = vrs_dir / "eyegaze.csv"
    gaze_points = []
    if gaze_file.exists():
        print(f"Loading gaze data...")
        gaze_df = pd.read_csv(gaze_file)
        for _, row in gaze_df.iterrows():
            gaze_points.append({
                'timestamp_us': row['tracking_timestamp_us'],
                'pitch': row['pitch_rads_cpf'],
                'yaw': row['yaw_rads_cpf']
            })
        print(f"Loaded {len(gaze_points)} gaze points")
    
    # Create output directory
    output_dir = Path(args.output_dir or f"4panel_{sequence}")
    output_dir.mkdir(exist_ok=True)
    
    # Get SLAM stream info
    slam_stream = main_provider.get_stream_id_from_label("camera-slam-right")
    num_frames = main_provider.get_num_data(slam_stream)
    
    print(f"\nTotal SLAM frames: {num_frames}")
    
    # Sample frames throughout the sequence
    # Take frames from different parts of the video
    sample_indices = [
        100,   # Early
        500,   # Early-mid
        1000,  # Mid
        1500,  # Mid
        2000,  # Mid-late
        2500,  # Late
        3000,  # Late
        3500,  # Very late
    ]
    
    # Add more samples in the middle range
    sample_indices.extend(range(1200, 1800, 50))
    sample_indices = sorted(list(set(sample_indices)))  # Remove duplicates and sort
    
    processed = 0
    
    for slam_idx in sample_indices:
        if slam_idx >= num_frames:
            continue
            
        # Get timestamp for this frame
        slam_data = main_provider.get_image_data_by_index(slam_stream, slam_idx)
        if slam_data is None:
            continue
            
        timestamp = slam_data[1].capture_timestamp_ns / 1e9
        
        fig = create_4panel_frame(slam_idx, sequence, main_provider, seg_provider, gaze_points, timestamp, tracking_dir)
        
        if fig is not None:
            # Save 4-panel visualization
            output_path = output_dir / f"frame_{processed:03d}_4panel.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            processed += 1
            if processed >= args.frames:
                break
    
    print(f"\nDone! Created {processed} 4-panel visualizations in {output_dir}")
    print("\nNote: Using actual SLAM tracking data")
    print("Green dots show SLAM features with valid depth estimates.")


if __name__ == "__main__":
    main()