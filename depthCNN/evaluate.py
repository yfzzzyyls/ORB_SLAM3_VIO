#!/usr/bin/env python3
"""
Evaluation script for RT-MonoDepth-S on ADT test set.
Computes standard depth estimation metrics and saves qualitative results.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import json
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List

# Add project to path
sys.path.append(str(Path(__file__).parent))

from vrs_dataset import ADTVRSDataset
from model_rtmonodepth import RTMonoDepthS, DepthMetrics
from torch.utils.data import DataLoader


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
        cv2.putText(comparison, "RGB", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Ground Truth", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Prediction", (10, h + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Error", (w + 10, h + 30), font, 1, (255, 255, 255), 2)
        
        # Save
        seq_name = batch['sequence'][i]
        frame_idx = batch['frame_idx'][i].item()
        filename = f"{seq_name}_frame{frame_idx:06d}.png"
        cv2.imwrite(str(outputs_dir / filename), comparison)


def evaluate_model(model, test_loader, device, outputs_dir: Path, save_samples: int = 20):
    """Evaluate model on test set."""
    model.eval()
    
    all_metrics = []
    total_samples = 0
    saved_samples = 0
    
    with torch.no_grad():
        progress = tqdm(test_loader, desc="Evaluating")
        for batch_idx, batch in enumerate(progress):
            # Move to device
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            # Forward pass
            outputs = model(rgb)
            pred_depth = outputs['depth']
            
            # Compute metrics
            metrics = DepthMetrics.compute_metrics(pred_depth, depth, valid_mask)
            all_metrics.append(metrics)
            
            # Save qualitative results
            if saved_samples < save_samples:
                save_qualitative_results(
                    outputs_dir, batch_idx, batch, pred_depth,
                    max_samples=min(5, save_samples - saved_samples)
                )
                saved_samples += min(batch['rgb'].size(0), 5)
            
            # Update progress
            total_samples += batch['rgb'].size(0)
            progress.set_postfix({
                'samples': total_samples,
                'abs_rel': metrics['abs_rel']
            })
    
    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return avg_metrics, total_samples


def main():
    parser = argparse.ArgumentParser(description='Evaluate RT-MonoDepth-S on ADT test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='/mnt/ssd_ext/incSeg-data/adt',
                        help='Path to ADT dataset root')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Directory to cache extracted frames')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Subsample factor for frames (1=all frames, 10=every 10th frame)')
    parser.add_argument('--save-samples', type=int, default=20,
                        help='Number of qualitative samples to save')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth value in meters')
    parser.add_argument('--min-depth', type=float, default=0.1,
                        help='Minimum depth value in meters')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    qualitative_dir = output_dir / 'qualitative'
    qualitative_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = RTMonoDepthS(max_depth=args.max_depth, min_depth=args.min_depth)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = ADTVRSDataset(
        adt_root=args.data_root,
        split='test',
        transform=None,
        cache_dir=Path(args.cache_dir) / 'test',
        subsample_factor=args.subsample
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, total_samples = evaluate_model(
        model, test_loader, device, qualitative_dir, args.save_samples
    )
    
    # Print results
    print(f"\nEvaluation Results ({total_samples} samples):")
    print("-" * 60)
    
    metric_order = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    
    for metric in metric_order:
        stats = metrics[metric]
        print(f"{metric:>10}: {stats['mean']:>8.3f} ± {stats['std']:>6.3f} " +
              f"[{stats['min']:>8.3f}, {stats['max']:>8.3f}]")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'dataset': args.data_root,
        'test_samples': total_samples,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args)
    }
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Qualitative results saved to: {qualitative_dir}")
    
    # Create summary
    summary_lines = [
        f"RT-MonoDepth-S Evaluation Results",
        f"================================",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint: {Path(args.checkpoint).name}",
        f"Test samples: {total_samples}",
        f"",
        f"Metrics (mean ± std):",
        f"--------------------"
    ]
    
    for metric in metric_order:
        stats = metrics[metric]
        summary_lines.append(f"{metric:>10}: {stats['mean']:>8.3f} ± {stats['std']:>6.3f}")
    
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()