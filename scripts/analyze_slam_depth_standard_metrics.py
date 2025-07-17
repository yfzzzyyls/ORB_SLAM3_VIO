#!/usr/bin/env python3
"""Calculate standard depth evaluation metrics for SLAM vs ground truth.

Uses the standard metrics from depth estimation literature:
- Abs Rel, Sq Rel, RMSE, RMSE log
- Threshold accuracies (δ < 1.25^n)
"""

import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.append('/home/external/.local/lib/python3.9/site-packages')
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


def load_slam_features(tracking_file):
    """Load SLAM features from tracking file."""
    slam_features = []
    if tracking_file.exists():
        try:
            with open(tracking_file, 'r') as f:
                lines = f.readlines()
                timestamp = float(lines[1].split()[1])
                
                for line in lines[9:]:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            u, v, depth = float(parts[0]), float(parts[1]), float(parts[2])
                            if depth > 0:
                                slam_features.append((int(u), int(v), depth))
                        except ValueError:
                            continue
            return slam_features, timestamp
        except Exception as e:
            pass
    return [], None


def get_gt_depth_at_timestamp(depth_provider, timestamp_s):
    """Get ground truth depth map at given timestamp."""
    depth_streams = depth_provider.get_all_streams()
    
    if len(depth_streams) < 3:
        return None, None
        
    depth_stream_id = depth_streams[2]  # Right SLAM depth
    timestamp_ns = int(timestamp_s * 1e9)
    
    depth_data = depth_provider.get_image_data_by_time_ns(
        depth_stream_id, timestamp_ns, TimeDomain.RECORD_TIME, TimeQueryOptions.CLOSEST
    )
    
    if depth_data is None:
        return None, None
        
    depth_map = depth_data[0].to_numpy_array()
    depth_map = depth_map.astype(np.float32) / 1000.0
    actual_timestamp_s = depth_data[1].capture_timestamp_ns / 1e9
    
    return depth_map, actual_timestamp_s


def compute_depth_metrics(pred_depths, gt_depths):
    """Compute standard depth evaluation metrics."""
    
    pred_depths = np.array(pred_depths)
    gt_depths = np.array(gt_depths)
    
    # Basic error metrics
    abs_diff = np.abs(pred_depths - gt_depths)
    abs_rel = np.mean(abs_diff / gt_depths)
    sq_rel = np.mean(((pred_depths - gt_depths) ** 2) / gt_depths)
    rmse = np.sqrt(np.mean((pred_depths - gt_depths) ** 2))
    
    # Log space error (add small epsilon to avoid log(0))
    log_diff = np.abs(np.log(pred_depths + 1e-6) - np.log(gt_depths + 1e-6))
    rmse_log = np.sqrt(np.mean(log_diff ** 2))
    
    # Threshold accuracies
    ratio = np.maximum(pred_depths / gt_depths, gt_depths / pred_depths)
    delta_1 = np.mean(ratio < 1.25) * 100
    delta_2 = np.mean(ratio < 1.25 ** 2) * 100
    delta_3 = np.mean(ratio < 1.25 ** 3) * 100
    
    # Also compute median versions
    abs_rel_median = np.median(abs_diff / gt_depths)
    
    return {
        'abs_rel': abs_rel,
        'abs_rel_median': abs_rel_median,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'delta_1.25': delta_1,
        'delta_1.25^2': delta_2,
        'delta_1.25^3': delta_3,
        'mean_pred': np.mean(pred_depths),
        'mean_gt': np.mean(gt_depths),
    }


def analyze_sequence(sequence, data_dir_suffix=""):
    """Analyze one sequence."""
    
    vrs_dir = None
    for subdir in ['train', 'test']:
        potential_dir = Path(f"/mnt/ssd_ext/incSeg-data/adt/{subdir}/{sequence}")
        if potential_dir.exists():
            vrs_dir = potential_dir
            break
    
    if vrs_dir is None:
        return None
        
    depth_vrs = vrs_dir / "depth_images.vrs"
    
    # Construct tracking directory
    # Default: tracking_data_{sequence}_trajectory (for default 2000 cap runs)
    # With suffix: tracking_data_{sequence}_{suffix} (e.g., cap1000_trajectory)
    if data_dir_suffix:
        tracking_dir = Path(f"/home/external/ORB_SLAM3_VIO/results/tracking_data_{sequence}_{data_dir_suffix}")
    else:
        tracking_dir = Path(f"/home/external/ORB_SLAM3_VIO/results/tracking_data_{sequence}_trajectory")
    
    if not tracking_dir.exists() or not depth_vrs.exists():
        return None
    
    depth_provider = data_provider.create_vrs_data_provider(str(depth_vrs))
    
    # Sample frames
    frame_files = sorted(tracking_dir.glob("frame_*.txt"))
    sample_indices = range(500, min(3000, len(frame_files)), 100)
    
    all_pred_depths = []
    all_gt_depths = []
    
    for idx in sample_indices:
        if idx < len(frame_files):
            frame_file = frame_files[idx]
            slam_features, timestamp = load_slam_features(frame_file)
            
            if not slam_features:
                continue
                
            gt_depth, gt_timestamp = get_gt_depth_at_timestamp(depth_provider, timestamp)
            
            if gt_depth is None:
                continue
                
            time_diff_ms = abs(gt_timestamp - timestamp) * 1000
            if time_diff_ms > 1:  # 1ms threshold
                continue
            
            for u, v, slam_depth in slam_features:
                if 0 <= u < 640 and 0 <= v < 480:
                    gt_depth_value = gt_depth[v, u]
                    if gt_depth_value > 0:
                        all_pred_depths.append(slam_depth)
                        all_gt_depths.append(gt_depth_value)
    
    if len(all_pred_depths) < 100:
        return None
    
    return compute_depth_metrics(all_pred_depths, all_gt_depths), len(all_pred_depths), all_pred_depths, all_gt_depths


def main():
    """Analyze all sequences."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze SLAM depth accuracy with standard metrics')
    parser.add_argument('--data-dir', type=str, default='', 
                       help='Suffix for tracking data directory (e.g., "cap1000_trajectory" for cap 1000). ' +
                            'If not specified, uses default tracking_data_*_trajectory directories.')
    args = parser.parse_args()
    
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
    print("STANDARD DEPTH METRICS FOR SLAM EVALUATION")
    if args.data_dir:
        print(f"Using tracking data suffix: {args.data_dir}")
    print("="*70)
    
    all_pred = []
    all_gt = []
    seq_results = []
    
    for seq in sequences:
        print(f"\nAnalyzing {seq.split('_')[-2]}...")
        result = analyze_sequence(seq, args.data_dir)
        
        if result:
            metrics, num_points, pred_depths, gt_depths = result
            seq_results.append((metrics, num_points))
            all_pred.extend(pred_depths)
            all_gt.extend(gt_depths)
            
            print(f"  Points: {num_points}")
            print(f"  Abs Rel: {metrics['abs_rel']:.3f} (median: {metrics['abs_rel_median']:.3f})")
            print(f"  RMSE: {metrics['rmse']:.3f}m")
            print(f"  δ < 1.25: {metrics['delta_1.25']:.1f}%")
    
    # Overall metrics
    if all_pred:
        print("\n" + "="*70)
        print("OVERALL METRICS (Standard Depth Evaluation)")
        print("="*70)
        
        # Recalculate on all data
        overall = compute_depth_metrics(all_pred, all_gt)
        
        print(f"\nError Metrics:")
        print(f"  Abs Rel: {overall['abs_rel']:.3f} (median: {overall['abs_rel_median']:.3f})")
        print(f"  Sq Rel: {overall['sq_rel']:.3f}")
        print(f"  RMSE: {overall['rmse']:.3f}m")
        print(f"  RMSE log: {overall['rmse_log']:.3f}")
        
        print(f"\nAccuracy Metrics:")
        print(f"  δ < 1.25: {overall['delta_1.25']:.1f}%")
        print(f"  δ < 1.25²: {overall['delta_1.25^2']:.1f}%")
        print(f"  δ < 1.25³: {overall['delta_1.25^3']:.1f}%")
        
        print(f"\nSummary:")
        print(f"  Mean SLAM depth: {overall['mean_pred']:.2f}m")
        print(f"  Mean GT depth: {overall['mean_gt']:.2f}m")
        print(f"  Scale ratio: {overall['mean_pred']/overall['mean_gt']:.3f}")


if __name__ == "__main__":
    main()