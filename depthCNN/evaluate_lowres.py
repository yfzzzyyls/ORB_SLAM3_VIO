#!/usr/bin/env python3
"""
Evaluation script for RT-MonoDepth-S on ADT test set with low-resolution support.
Computes standard depth estimation metrics at downsampled resolution.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import json
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import time
from typing import Dict, List, Optional

# Add project to path
sys.path.append(str(Path(__file__).parent))

from vrs_dataset import ADTVRSDataset
from processed_dataset import ProcessedADTDataset
from lowres_dataset import LowResADTDataset
from model_rtmonodepth import RTMonoDepthS, DepthMetrics
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


def custom_collate_fn(batch):
    """Custom collate function that handles None gaze values."""
    # Separate the batch into different components
    batch_dict = {
        'rgb': [],
        'depth': [],
        'valid_mask': [],
        'sequence': [],
        'frame_idx': [],
        'gaze': [],
        'scale_factor': [],
        'original_size': [],
        'lowres_size': []
    }
    
    for sample in batch:
        batch_dict['rgb'].append(sample['rgb'])
        batch_dict['depth'].append(sample['depth'])
        batch_dict['valid_mask'].append(sample['valid_mask'])
        batch_dict['sequence'].append(sample['sequence'])
        batch_dict['frame_idx'].append(sample['frame_idx'])
        batch_dict['gaze'].append(sample['gaze'])  # Can be None
        batch_dict['scale_factor'].append(sample['scale_factor'])
        batch_dict['original_size'].append(sample['original_size'])
        batch_dict['lowres_size'].append(sample['lowres_size'])
    
    # Use default_collate for tensor fields
    collated = {
        'rgb': default_collate(batch_dict['rgb']),
        'depth': default_collate(batch_dict['depth']),
        'valid_mask': default_collate(batch_dict['valid_mask']),
        'sequence': batch_dict['sequence'],  # List of strings
        'frame_idx': batch_dict['frame_idx'],  # List of ints
        'gaze': batch_dict['gaze'],  # List of dicts or None
        'scale_factor': batch_dict['scale_factor'][0],  # Same for all
        'original_size': batch_dict['original_size'][0],  # Same for all
        'lowres_size': batch_dict['lowres_size'][0]  # Same for all
    }
    
    return collated


def colorize_depth(depth_map, vmin=None, vmax=None, cmap='plasma'):
    """
    Colorize depth map for visualization.
    
    Args:
        depth_map: Depth map array [H, W]
        vmin: Minimum depth value for normalization
        vmax: Maximum depth value for normalization
        cmap: Matplotlib colormap name
        
    Returns:
        Colored depth map [H, W, 3] in BGR format
    """
    if vmin is None:
        vmin = depth_map.min()
    if vmax is None:
        vmax = depth_map.max()
    
    # Normalize to [0, 1]
    depth_norm = (depth_map - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    
    # Apply colormap
    cmap_func = plt.get_cmap(cmap)
    depth_colored = cmap_func(depth_norm)[:, :, :3]  # Remove alpha
    
    # Convert to BGR for OpenCV
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    
    return depth_colored


def save_qualitative_results(outputs_dir: Path, batch_idx: int, batch: Dict,
                           pred_depth: torch.Tensor, max_samples: int = 5):
    """Save qualitative results for visualization."""
    batch_size = min(batch['rgb'].size(0), max_samples)
    
    for i in range(batch_size):
        # Get data
        rgb = batch['rgb'][i].permute(1, 2, 0).cpu().numpy() * 255
        rgb = rgb.astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        gt_depth = batch['depth'][i, 0].cpu().numpy()
        pred = pred_depth[i, 0].cpu().numpy()
        valid_mask = batch['valid_mask'][i, 0].cpu().numpy()
        
        # Apply mask to ground truth for visualization
        gt_depth_vis = gt_depth.copy()
        gt_depth_vis[~valid_mask] = 0
        
        # Colorize depth maps
        vmax = gt_depth[valid_mask].max() if valid_mask.any() else 10.0
        gt_colored = colorize_depth(gt_depth_vis, vmin=0, vmax=vmax)
        pred_colored = colorize_depth(pred, vmin=0, vmax=vmax)
        
        # Compute error map
        error = np.abs(pred - gt_depth)
        error[~valid_mask] = 0
        error_colored = colorize_depth(error, vmin=0, vmax=1.0, cmap='hot')
        
        # Create comparison image
        h, w = rgb.shape[:2]
        comparison = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        comparison[:h, :w] = rgb
        comparison[:h, w:] = gt_colored
        comparison[h:, :w] = pred_colored
        comparison[h:, w:] = error_colored
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.3, h / 300)  # Scale font with image size
        cv2.putText(comparison, "RGB", (10, 30), font, font_scale, (255, 255, 255), 2)
        cv2.putText(comparison, "GT Depth", (w + 10, 30), font, font_scale, (255, 255, 255), 2)
        cv2.putText(comparison, "Pred Depth", (10, h + 30), font, font_scale, (255, 255, 255), 2)
        cv2.putText(comparison, "Error", (w + 10, h + 30), font, font_scale, (255, 255, 255), 2)
        
        # Add gaze point if available
        if 'gaze' in batch and batch['gaze'][i] is not None:
            gaze_x = int(batch['gaze'][i]['x'])
            gaze_y = int(batch['gaze'][i]['y'])
            
            # Draw gaze point on all quadrants
            for dx, dy in [(0, 0), (w, 0), (0, h), (w, h)]:
                cv2.circle(comparison, (dx + gaze_x, dy + gaze_y), 5, (0, 0, 255), -1)
                cv2.circle(comparison, (dx + gaze_x, dy + gaze_y), 6, (255, 255, 255), 2)
        
        # Save
        output_path = outputs_dir / f"batch_{batch_idx:04d}_sample_{i:02d}.png"
        cv2.imwrite(str(output_path), comparison)


def compute_gaze_metrics(batch: Dict, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> Dict[str, float]:
    """Compute metrics specifically at gaze locations."""
    gaze_metrics = {
        'gaze_mae': [],
        'gaze_rmse': [],
        'gaze_rel': []
    }
    
    batch_size = pred_depth.size(0)
    
    for i in range(batch_size):
        if 'gaze' not in batch or batch['gaze'][i] is None:
            continue
        
        # Get gaze coordinates (already scaled for low-res)
        gaze_x = int(batch['gaze'][i]['x'])
        gaze_y = int(batch['gaze'][i]['y'])
        
        # Ensure coordinates are within bounds
        h, w = pred_depth.shape[2:]
        if 0 <= gaze_x < w and 0 <= gaze_y < h:
            pred_at_gaze = pred_depth[i, 0, gaze_y, gaze_x].item()
            gt_at_gaze = gt_depth[i, 0, gaze_y, gaze_x].item()
            
            if gt_at_gaze > 0:  # Valid ground truth
                mae = abs(pred_at_gaze - gt_at_gaze)
                rmse = (pred_at_gaze - gt_at_gaze) ** 2
                rel = mae / gt_at_gaze
                
                gaze_metrics['gaze_mae'].append(mae)
                gaze_metrics['gaze_rmse'].append(rmse)
                gaze_metrics['gaze_rel'].append(rel)
    
    # Average metrics
    result = {}
    for key, values in gaze_metrics.items():
        if len(values) > 0:
            if key == 'gaze_rmse':
                result[key] = np.sqrt(np.mean(values))
            else:
                result[key] = np.mean(values)
        else:
            result[key] = 0.0
    
    return result


def evaluate_model(model, dataloader, device, output_dir: Optional[Path] = None,
                   max_vis_batches: int = 10, scale_factor: int = 1):
    """Evaluate model on dataset."""
    model.eval()
    
    # Metrics storage
    all_metrics = []
    all_gaze_metrics = []
    
    # Latency tracking
    forward_times = []
    
    # Create output directory for visualizations
    if output_dir:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(dataloader, desc='Evaluating')
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Get data
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            # Time the forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Forward pass
            outputs = model(rgb)
            pred_depth = outputs['depth'] if isinstance(outputs, dict) else outputs
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            # Record forward pass time (in milliseconds) - skip first few batches for warmup
            if batch_idx >= 5:  # Skip first 5 batches for GPU warmup
                forward_times.append((end_time - start_time) * 1000)
            
            # Resize prediction to match target size if needed
            if pred_depth.shape[2:] != depth.shape[2:]:
                pred_depth = F.interpolate(pred_depth, size=depth.shape[2:], mode='bilinear', align_corners=False)
            
            # Compute standard metrics
            batch_metrics = DepthMetrics.compute_metrics(
                pred_depth, depth, valid_mask
            )
            all_metrics.append(batch_metrics)
            
            # Compute gaze-specific metrics if available
            if 'gaze' in batch:
                gaze_metrics = compute_gaze_metrics(batch, pred_depth, depth)
                all_gaze_metrics.append(gaze_metrics)
            
            # Save visualizations for first few batches
            if output_dir and batch_idx < max_vis_batches:
                save_qualitative_results(vis_dir, batch_idx, batch, pred_depth)
            
            # Update progress bar
            pbar.set_postfix({
                'abs_rel': f"{batch_metrics['abs_rel']:.3f}",
                'a1': f"{batch_metrics['a1']:.3f}"
            })
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Average gaze metrics if available
    if all_gaze_metrics:
        gaze_keys = all_gaze_metrics[0].keys()
        for key in gaze_keys:
            values = [m[key] for m in all_gaze_metrics if m[key] > 0]
            if values:
                avg_metrics[key] = np.mean(values)
    
    # Add latency statistics
    if forward_times:
        avg_metrics['latency_mean_ms'] = np.mean(forward_times)
        avg_metrics['latency_std_ms'] = np.std(forward_times)
        avg_metrics['latency_median_ms'] = np.median(forward_times)
        avg_metrics['latency_min_ms'] = np.min(forward_times)
        avg_metrics['latency_max_ms'] = np.max(forward_times)
        avg_metrics['latency_p95_ms'] = np.percentile(forward_times, 95)
        avg_metrics['latency_p99_ms'] = np.percentile(forward_times, 99)
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate RT-MonoDepth-S on ADT test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='/mnt/ssd_ext/incSeg-data/adt',
                        help='Path to ADT dataset root')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-results', action='store_true',
                        help='Save qualitative results')
    parser.add_argument('--max-vis-batches', type=int, default=10,
                        help='Maximum number of batches to visualize')
    
    # Low-resolution evaluation
    parser.add_argument('--lowres-scale', type=int, default=1,
                        help='Downscale factor for low-resolution evaluation (1=full res, 16=1/16 res)')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    
    # Adjust output directory for low-res evaluation
    if args.lowres_scale > 1:
        output_dir = output_dir / f"lowres_{args.lowres_scale}x"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    print("Loading test dataset...")
    data_root = Path(args.data_root)
    
    if (data_root / 'test').exists():
        # Use processed dataset
        if args.lowres_scale > 1:
            # Low-resolution evaluation
            print(f"Low-resolution evaluation with scale factor {args.lowres_scale}")
            test_dataset = LowResADTDataset(
                data_root=args.data_root,
                split='test',
                scale_factor=args.lowres_scale,
                transform=None
            )
        else:
            # Full resolution evaluation
            test_dataset = ProcessedADTDataset(
                data_root=args.data_root,
                split='test',
                transform=None
            )
    else:
        # Use VRS dataset
        if args.lowres_scale > 1:
            raise ValueError("Low-resolution evaluation is only supported with pre-processed data. "
                           "Run extract_dataset.py first.")
        
        test_dataset = ADTVRSDataset(
            adt_root=args.data_root,
            split='test',
            transform=None,
            cache_dir=Path('./cache') / 'test',
            subsample_factor=1  # Use all frames for evaluation
        )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Load model
    print("Loading model...")
    model = RTMonoDepthS()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle DataParallel state dict
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Get training info if available
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
    if 'metrics' in checkpoint:
        print("Training metrics:")
        for k, v in checkpoint['metrics'].items():
            print(f"  {k}: {v:.3f}")
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(
        model, test_loader, device,
        output_dir=output_dir if args.save_results else None,
        max_vis_batches=args.max_vis_batches,
        scale_factor=args.lowres_scale
    )
    
    # Print results
    print("\nTest Results:")
    print("-" * 40)
    
    # Standard metrics
    standard_metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    for metric in standard_metrics:
        if metric in metrics:
            print(f"{metric:10s}: {metrics[metric]:.3f}")
    
    # Gaze metrics if available
    if 'gaze_mae' in metrics:
        print("\nGaze-specific Metrics:")
        print("-" * 40)
        print(f"{'gaze_mae':10s}: {metrics['gaze_mae']:.3f}")
        print(f"{'gaze_rmse':10s}: {metrics['gaze_rmse']:.3f}")
        print(f"{'gaze_rel':10s}: {metrics['gaze_rel']:.3f}")
    
    # Latency metrics
    if 'latency_mean_ms' in metrics:
        print("\nLatency Statistics (per frame):")
        print("-" * 40)
        print(f"Mean:       {metrics['latency_mean_ms']:.2f} ms")
        print(f"Std:        {metrics['latency_std_ms']:.2f} ms")
        print(f"Median:     {metrics['latency_median_ms']:.2f} ms")
        print(f"Min:        {metrics['latency_min_ms']:.2f} ms")
        print(f"Max:        {metrics['latency_max_ms']:.2f} ms")
        print(f"95th %ile:  {metrics['latency_p95_ms']:.2f} ms")
        print(f"99th %ile:  {metrics['latency_p99_ms']:.2f} ms")
        print(f"\nThroughput: {1000.0/metrics['latency_mean_ms']:.1f} FPS")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'dataset': args.data_root,
        'num_samples': len(test_dataset),
        'lowres_scale': args.lowres_scale,
        'resolution': f"{1408//args.lowres_scale}x{1408//args.lowres_scale}" if args.lowres_scale > 1 else "1408x1408",
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    if args.save_results:
        print(f"Visualizations saved to: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()