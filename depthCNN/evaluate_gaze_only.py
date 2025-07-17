#!/usr/bin/env python3
"""
Evaluation script for Gaze-Only RT-MonoDepth model.
Evaluates accuracy and latency specifically at gaze locations.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import cv2

# Add project to path
sys.path.append(str(Path(__file__).parent))

from lowres_dataset import LowResADTDataset
from gaze_only_rtmonodepth import GazeOnlyRTMonoDepth
from train_gaze_only import extract_gt_depth_at_gaze


def compute_gaze_specific_metrics(pred_depths: List[float], gt_depths: List[float], 
                                 gaze_positions: List[Tuple[float, float]]) -> Dict:
    """
    Compute metrics specific to gaze-based depth prediction.
    
    Args:
        pred_depths: List of predicted depths at gaze
        gt_depths: List of ground truth depths at gaze
        gaze_positions: List of (x, y) gaze positions
        
    Returns:
        Dictionary of metrics
    """
    pred_depths = np.array(pred_depths)
    gt_depths = np.array(gt_depths)
    
    # Basic metrics
    errors = np.abs(pred_depths - gt_depths)
    metrics = {
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean((pred_depths - gt_depths) ** 2)),
        'median_ae': np.median(errors),
        'std_ae': np.std(errors),
    }
    
    # Relative errors
    rel_errors = errors / (gt_depths + 1e-8)
    metrics['abs_rel'] = np.mean(rel_errors)
    metrics['median_rel'] = np.median(rel_errors)
    
    # Threshold accuracy
    ratio = np.maximum(pred_depths / gt_depths, gt_depths / pred_depths)
    metrics['delta_1.25'] = np.mean(ratio < 1.25)
    metrics['delta_1.25^2'] = np.mean(ratio < 1.25 ** 2)
    metrics['delta_1.25^3'] = np.mean(ratio < 1.25 ** 3)
    
    # Depth range analysis
    metrics['min_depth'] = np.min(gt_depths)
    metrics['max_depth'] = np.max(gt_depths)
    metrics['mean_depth'] = np.mean(gt_depths)
    metrics['median_depth'] = np.median(gt_depths)
    
    # Error by depth range
    near_mask = gt_depths < 2.0
    mid_mask = (gt_depths >= 2.0) & (gt_depths < 5.0)
    far_mask = gt_depths >= 5.0
    
    if np.any(near_mask):
        metrics['mae_near'] = np.mean(errors[near_mask])
        metrics['count_near'] = np.sum(near_mask)
    if np.any(mid_mask):
        metrics['mae_mid'] = np.mean(errors[mid_mask])
        metrics['count_mid'] = np.sum(mid_mask)
    if np.any(far_mask):
        metrics['mae_far'] = np.mean(errors[far_mask])
        metrics['count_far'] = np.sum(far_mask)
    
    # Error by image region (center vs peripheral)
    gaze_x = np.array([p[0] for p in gaze_positions])
    gaze_y = np.array([p[1] for p in gaze_positions])
    center_x, center_y = 44, 44  # Center of 88x88 image
    
    center_mask = ((gaze_x - center_x) ** 2 + (gaze_y - center_y) ** 2) < (22 ** 2)
    peripheral_mask = ~center_mask
    
    if np.any(center_mask):
        metrics['mae_center'] = np.mean(errors[center_mask])
        metrics['count_center'] = np.sum(center_mask)
    if np.any(peripheral_mask):
        metrics['mae_peripheral'] = np.mean(errors[peripheral_mask])
        metrics['count_peripheral'] = np.sum(peripheral_mask)
    
    return metrics


def measure_latency(model, device, input_size=(88, 88), num_warmup=50, num_iterations=200):
    """
    Measure inference latency of the model.
    
    Returns:
        Dictionary with latency statistics in milliseconds
    """
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    rgb = torch.randn(batch_size, 3, *input_size).to(device)
    gaze_x = torch.tensor([input_size[0] // 2], dtype=torch.float32).to(device)
    gaze_y = torch.tensor([input_size[1] // 2], dtype=torch.float32).to(device)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(rgb, gaze_x, gaze_y)
    
    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(rgb, gaze_x, gaze_y)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': np.mean(latencies),
        'median_ms': np.median(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'fps': 1000 / np.mean(latencies)
    }


def visualize_predictions(model, dataset, device, save_dir, num_samples=10):
    """Create visualizations of predictions vs ground truth."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx_num, idx in enumerate(indices):
        sample = dataset[idx]
        
        if sample['gaze'] is None or sample['gaze']['x'] < 0:
            continue
        
        # Prepare inputs
        rgb = sample['rgb'].unsqueeze(0).to(device)
        gaze_x = torch.tensor([sample['gaze']['x']], dtype=torch.float32).to(device)
        gaze_y = torch.tensor([sample['gaze']['y']], dtype=torch.float32).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(rgb, gaze_x, gaze_y)
            pred_depth = outputs['depth'].item()
        
        # Get ground truth
        if sample.get('gt_depth_at_gaze') is not None:
            gt_depth = sample['gt_depth_at_gaze']
        else:
            # Fallback to interpolation if exact GT not available
            gt_depth_map = sample['depth'].unsqueeze(0).to(device)
            gt_depth = extract_gt_depth_at_gaze(gt_depth_map, gaze_x, gaze_y).item()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB image with gaze point
        rgb_np = sample['rgb'].permute(1, 2, 0).numpy()
        axes[0].imshow(rgb_np)
        axes[0].scatter(sample['gaze']['x'], sample['gaze']['y'], 
                       c='red', s=100, marker='x', linewidths=3)
        axes[0].set_title(f'RGB with Gaze Point')
        axes[0].axis('off')
        
        # Ground truth depth map
        depth_np = sample['depth'].squeeze().numpy()
        im1 = axes[1].imshow(depth_np, cmap='viridis')
        axes[1].scatter(sample['gaze']['x'], sample['gaze']['y'], 
                       c='red', s=100, marker='x', linewidths=3)
        axes[1].set_title(f'GT Depth (at gaze: {gt_depth:.2f}m)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Error visualization
        error = abs(pred_depth - gt_depth)
        rel_error = error / gt_depth * 100
        
        # Create a simple bar chart showing predicted vs GT
        bars = axes[2].bar(['Ground Truth', 'Predicted'], [gt_depth, pred_depth], 
                          color=['green', 'blue'])
        axes[2].set_ylabel('Depth (m)')
        axes[2].set_title(f'Error: {error:.3f}m ({rel_error:.1f}%)')
        axes[2].set_ylim(0, max(gt_depth, pred_depth) * 1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars, [gt_depth, pred_depth]):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{value:.2f}m', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = save_dir / f'sample_{idx_num:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")


def evaluate_dataset(model, dataloader, device, save_results=False, output_dir=None):
    """Evaluate model on entire dataset."""
    model.eval()
    
    all_pred_depths = []
    all_gt_depths = []
    all_gaze_positions = []
    all_sequences = []
    
    print("Evaluating on dataset...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            if batch is None:
                continue
            
            # Get data
            rgb = batch['rgb'].to(device)
            gaze_x = batch['gaze_x'].to(device)
            gaze_y = batch['gaze_y'].to(device)
            
            # Get GT depth (either exact or interpolated)
            if 'gt_depth_at_gaze' in batch:
                gt_depth = batch['gt_depth_at_gaze'].to(device)
            else:
                # Fallback to interpolation
                depth_full = batch['depth'].to(device)
                gt_depth = extract_gt_depth_at_gaze(depth_full, gaze_x, gaze_y)
            
            # Get predictions
            outputs = model(rgb, gaze_x, gaze_y)
            pred_depth = outputs['depth']
            
            # Store results
            for i in range(pred_depth.shape[0]):
                all_pred_depths.append(pred_depth[i].item())
                all_gt_depths.append(gt_depth[i].item())
                all_gaze_positions.append((gaze_x[i].item(), gaze_y[i].item()))
    
    # Compute metrics
    metrics = compute_gaze_specific_metrics(all_pred_depths, all_gt_depths, all_gaze_positions)
    
    # Save detailed results if requested
    if save_results and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        results = {
            'predictions': all_pred_depths,
            'ground_truth': all_gt_depths,
            'gaze_positions': all_gaze_positions,
            'metrics': metrics
        }
        
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create error histogram
        errors = np.array(all_pred_depths) - np.array(all_gt_depths)
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Zero error')
        plt.xlabel('Depth Error (m)')
        plt.ylabel('Count')
        plt.title('Distribution of Depth Prediction Errors at Gaze')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'error_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(all_gt_depths, all_pred_depths, alpha=0.5, s=1)
        plt.plot([0, 10], [0, 10], 'r--', label='Perfect prediction')
        plt.xlabel('Ground Truth Depth (m)')
        plt.ylabel('Predicted Depth (m)')
        plt.title('Predicted vs Ground Truth Depth at Gaze')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(output_dir / 'scatter_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gaze-Only RT-MonoDepth')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Path to processed ADT dataset')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate on (train/val/test)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--lowres-scale', type=int, default=16,
                        help='Downscale factor (should match training)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save detailed results and visualizations')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results/gaze_only',
                        help='Directory to save results')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--visualize', action='store_true',
                        help='Create sample visualizations')
    parser.add_argument('--num-vis-samples', type=int, default=20,
                        help='Number of visualization samples')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = GazeOnlyRTMonoDepth()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle DataParallel
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Measure latency
    print("\nMeasuring inference latency...")
    latency_stats = measure_latency(model, device, input_size=(88, 88))
    print("Latency Statistics:")
    for key, value in latency_stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = LowResADTDataset(
        data_root=args.data_root,
        split=args.split,
        scale_factor=args.lowres_scale,
        transform=None
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create dataloader with custom collate function
    from train_gaze_only import custom_collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch)  # Use collate without augmentation
    )
    
    # Evaluate on dataset
    metrics = evaluate_dataset(model, dataloader, device, 
                             save_results=args.save_results,
                             output_dir=args.output_dir)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nCore Metrics:")
    print(f"  MAE: {metrics['mae']:.4f} m")
    print(f"  RMSE: {metrics['rmse']:.4f} m")
    print(f"  Abs Rel: {metrics['abs_rel']:.4f}")
    print(f"  δ < 1.25: {metrics['delta_1.25']:.3f}")
    print(f"  δ < 1.25²: {metrics['delta_1.25^2']:.3f}")
    print(f"  δ < 1.25³: {metrics['delta_1.25^3']:.3f}")
    
    print("\nDepth Range Analysis:")
    if 'mae_near' in metrics:
        print(f"  Near (<2m): {metrics['mae_near']:.4f} m (n={metrics['count_near']})")
    if 'mae_mid' in metrics:
        print(f"  Mid (2-5m): {metrics['mae_mid']:.4f} m (n={metrics['count_mid']})")
    if 'mae_far' in metrics:
        print(f"  Far (>5m): {metrics['mae_far']:.4f} m (n={metrics['count_far']})")
    
    print("\nSpatial Analysis:")
    if 'mae_center' in metrics:
        print(f"  Center: {metrics['mae_center']:.4f} m (n={metrics['count_center']})")
    if 'mae_peripheral' in metrics:
        print(f"  Peripheral: {metrics['mae_peripheral']:.4f} m (n={metrics['count_peripheral']})")
    
    print("\nLatency:")
    print(f"  Mean: {latency_stats['mean_ms']:.2f} ms")
    print(f"  FPS: {latency_stats['fps']:.1f}")
    
    # Create visualizations
    if args.visualize:
        print(f"\nCreating {args.num_vis_samples} sample visualizations...")
        vis_dir = Path(args.output_dir) / 'visualizations' if args.output_dir else Path('./visualizations')
        visualize_predictions(model, dataset, device, vis_dir, num_samples=args.num_vis_samples)
    
    # Save all results
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        complete_results = {
            'metrics': metrics,
            'latency': latency_stats,
            'checkpoint': args.checkpoint,
            'dataset_split': args.split,
            'lowres_scale': args.lowres_scale,
            'model_params': model.get_num_params()
        }
        
        with open(output_dir / 'complete_results.json', 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()