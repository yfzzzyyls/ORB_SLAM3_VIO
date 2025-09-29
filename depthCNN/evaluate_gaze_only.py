#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for Lightweight Gaze-Only Depth Prediction model.
Evaluates accuracy and latency specifically at gaze locations.
Supports both the original GazeOnlyRTMonoDepth and the lightweight architecture.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent))

from lowres_dataset import LowResADTDataset
from lightweight_gaze_encoder import LightweightGazeOnlyDepth
from gaze_only_rtmonodepth import GazeOnlyRTMonoDepth
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Lightweight Gaze-Only Depth Model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='lightweight',
                        choices=['lightweight', 'original'],
                        help='Model architecture to use')
    
    # Lightweight model specific args
    parser.add_argument('--encoder-levels', type=int, default=3,
                        help='Number of encoder levels (for lightweight model)')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='Base channels (for lightweight model)')
    
    # Dataset arguments
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Path to processed ADT dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--lowres-scale', type=int, default=16,
                        help='Downscale factor (should match training)')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-results', action='store_true',
                        help='Save detailed results and visualizations')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Create sample visualizations')
    parser.add_argument('--num-vis-samples', type=int, default=10,
                        help='Number of visualization samples')
    
    return parser.parse_args()


def compute_metrics(pred_depths: np.ndarray, gt_depths: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics for gaze-based depth prediction.
    
    Args:
        pred_depths: Predicted depths at gaze locations [N]
        gt_depths: Ground truth depths at gaze locations [N]
        
    Returns:
        Dictionary of metrics
    """
    # Remove any invalid predictions
    valid_mask = (gt_depths > 0.1) & (gt_depths < 10.0) & np.isfinite(pred_depths)
    pred_depths = pred_depths[valid_mask]
    gt_depths = gt_depths[valid_mask]
    
    if len(pred_depths) == 0:
        return {
            'mae': float('inf'),
            'rmse': float('inf'),
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'log_mae': float('inf'),
            'delta_1': 0.0,
            'delta_2': 0.0,
            'delta_3': 0.0,
        }
    
    # Basic metrics
    errors = np.abs(pred_depths - gt_depths)
    squared_errors = (pred_depths - gt_depths) ** 2
    
    # Relative errors
    rel_errors = errors / (gt_depths + 1e-6)
    sq_rel_errors = squared_errors / (gt_depths ** 2 + 1e-6)
    
    # Log errors
    log_errors = np.abs(np.log(pred_depths + 1e-6) - np.log(gt_depths + 1e-6))
    
    # Threshold accuracies
    ratio = np.maximum(pred_depths / gt_depths, gt_depths / pred_depths)
    delta_1 = np.mean(ratio < 1.25) * 100
    delta_2 = np.mean(ratio < 1.25 ** 2) * 100
    delta_3 = np.mean(ratio < 1.25 ** 3) * 100
    
    metrics = {
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean(squared_errors)),
        'abs_rel': np.mean(rel_errors),
        'sq_rel': np.mean(sq_rel_errors),
        'log_mae': np.mean(log_errors),
        'delta_1': delta_1,
        'delta_2': delta_2,
        'delta_3': delta_3,
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'min_error': np.min(errors),
        'max_error': np.max(errors),
    }
    
    return metrics


def create_model(args):
    """Create model based on architecture type."""
    if args.model == 'lightweight':
        model = LightweightGazeOnlyDepth(
            num_encoder_levels=args.encoder_levels,
            base_channels=args.base_channels,
            use_multi_scale_supervision=True  # Must match training
        )
    else:  # original
        model = GazeOnlyRTMonoDepth()
    
    return model


def custom_collate_fn(batch):
    """Custom collate function to handle None gaze values."""
    # Filter out samples with invalid gaze
    valid_samples = []
    for sample in batch:
        if (sample['gaze'] is not None and 
            sample['gaze']['x'] >= 0 and 
            sample['gt_depth_at_gaze'] is not None and
            sample['gt_depth_at_gaze'] > 0.1):
            valid_samples.append(sample)
    
    if len(valid_samples) == 0:
        return None
    
    # Standard collation for valid samples
    return torch.utils.data.default_collate(valid_samples)


def measure_latency(model, device, data_loader, num_runs=100):
    """Measure model inference latency using real data."""
    model.eval()
    
    # Get a real sample from the data loader
    real_batch = None
    for batch in data_loader:
        if batch is not None:
            real_batch = batch
            break
    
    if real_batch is None:
        print("Warning: Could not get real data for latency measurement")
        return 0.0
    
    # Use first sample from batch
    real_input = real_batch['rgb'][:1].to(device)
    real_gaze_x = real_batch['gaze']['x'][:1].float().to(device)
    real_gaze_y = real_batch['gaze']['y'][:1].float().to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(real_input, real_gaze_x, real_gaze_y)
    
    # Measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(real_input, real_gaze_x, real_gaze_y)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000  # ms
    return avg_latency


def visualize_predictions(model, data_loader, device, num_samples=10):
    """Create visualization of predictions vs ground truth."""
    model.eval()
    
    samples_collected = 0
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue
            
            images = batch['rgb'].to(device)
            gaze_x = batch['gaze']['x'].float().to(device)
            gaze_y = batch['gaze']['y'].float().to(device)
            gt_depths = batch['gt_depth_at_gaze'].to(device)
            
            outputs = model(images, gaze_x, gaze_y)
            pred_depths = outputs['depth'].squeeze()
            
            # Convert to numpy
            images_np = images.cpu().numpy()
            gt_depths_np = gt_depths.cpu().numpy()
            pred_depths_np = pred_depths.cpu().numpy()
            gaze_x_np = gaze_x.cpu().numpy()
            gaze_y_np = gaze_y.cpu().numpy()
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                if samples_collected >= num_samples:
                    break
                
                # Get individual sample
                img = images_np[i].transpose(1, 2, 0)
                gaze_x_val = float(gaze_x_np[i])
                gaze_y_val = float(gaze_y_np[i])
                gt_depth = float(gt_depths_np[i])
                pred_depth = float(pred_depths_np[i])
                
                row_idx = samples_collected
                
                # Plot image with gaze point
                axes[row_idx, 0].imshow(img)
                circle = patches.Circle((gaze_x_val, gaze_y_val), 3, 
                                      edgecolor='red', facecolor='none', linewidth=2)
                axes[row_idx, 0].add_patch(circle)
                axes[row_idx, 0].set_title(f'Input Image (Gaze at {gaze_x_val:.1f}, {gaze_y_val:.1f})')
                axes[row_idx, 0].axis('off')
                
                # Plot depth comparison
                depth_comparison = np.array([[gt_depth, pred_depth]])
                im = axes[row_idx, 1].imshow(depth_comparison, cmap='viridis', aspect='auto')
                axes[row_idx, 1].set_xticks([0, 1])
                axes[row_idx, 1].set_xticklabels(['GT', 'Pred'])
                axes[row_idx, 1].set_yticks([])
                axes[row_idx, 1].set_title(f'GT: {gt_depth:.2f}m, Pred: {pred_depth:.2f}m')
                plt.colorbar(im, ax=axes[row_idx, 1])
                
                # Plot error
                error = abs(pred_depth - gt_depth)
                axes[row_idx, 2].bar(['Error'], [error], color='red' if error > 0.5 else 'green')
                axes[row_idx, 2].set_ylim([0, max(1.0, error * 1.2)])
                axes[row_idx, 2].set_title(f'Error: {error:.3f}m ({error/gt_depth*100:.1f}%)')
                
                samples_collected += 1
            
            if samples_collected >= num_samples:
                break
    
    plt.tight_layout()
    return fig


def evaluate(args):
    """Main evaluation function."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if args.save_results:
        output_dir = Path(args.output_dir) / f"{args.model}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = LowResADTDataset(
        data_root=args.data_root,
        split=args.split,
        scale_factor=args.lowres_scale
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = create_model(args)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Display training metrics if available
    print(f"\nTraining Information:")
    print(f"  Trained for epochs: {checkpoint.get('epoch', 'N/A')}")
    if 'metrics' in checkpoint and isinstance(checkpoint['metrics'], dict):
        print(f"  Best validation metrics during training:")
        for k, v in checkpoint['metrics'].items():
            if isinstance(v, (int, float)):
                if k in ['mae', 'rmse', 'abs_rel', 'sq_rel', 'rmse_log']:
                    print(f"    {k}: {v:.4f}")
                elif k in ['a1', 'a2', 'a3']:
                    # Note: stored metrics are in decimal form (0.8167 = 81.67%)
                    print(f"    δ < 1.25^{k[-1]}: {v*100:.1f}%")
    
    # Handle DataParallel state dict
    state_dict = checkpoint['model_state_dict']
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Get model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Measure latency
    print("\nMeasuring inference latency...")
    avg_latency = measure_latency(model, device, data_loader)
    print(f"Average latency: {avg_latency:.2f}ms ({1000/avg_latency:.1f} FPS)")
    
    # Evaluate
    print("\nEvaluating model...")
    all_pred_depths = []
    all_gt_depths = []
    all_gaze_positions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if batch is None:
                continue
            
            images = batch['rgb'].to(device)
            gaze_x = batch['gaze']['x'].float().to(device)
            gaze_y = batch['gaze']['y'].float().to(device)
            gt_depths = batch['gt_depth_at_gaze'].to(device)
            
            # Forward pass
            outputs = model(images, gaze_x, gaze_y)
            pred_depths = outputs['depth'].squeeze()
            
            # Collect results
            all_pred_depths.extend(pred_depths.cpu().numpy().tolist())
            all_gt_depths.extend(gt_depths.cpu().numpy().tolist())
            
            # Collect gaze positions
            for i in range(len(gaze_x)):
                all_gaze_positions.append((
                    float(gaze_x[i]),
                    float(gaze_y[i])
                ))
    
    # Convert to numpy arrays
    all_pred_depths = np.array(all_pred_depths)
    all_gt_depths = np.array(all_gt_depths)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(all_pred_depths, all_gt_depths)
    
    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results - {args.model} Model")
    print("="*50)
    print(f"Checkpoint: {Path(args.checkpoint).name}")
    print(f"Dataset: {args.split} split ({len(all_pred_depths)} samples)")
    print(f"Model Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Inference Latency: {avg_latency:.2f}ms ({1000/avg_latency:.1f} FPS)")
    print("\nDepth Prediction Metrics:")
    print(f"  MAE:          {metrics['mae']:.4f}m")
    print(f"  RMSE:         {metrics['rmse']:.4f}m")
    print(f"  Abs Rel:      {metrics['abs_rel']:.4f}")
    print(f"  Sq Rel:       {metrics['sq_rel']:.4f}")
    print(f"  Log MAE:      {metrics['log_mae']:.4f}")
    print(f"  delta < 1.25:     {metrics['delta_1']:.1f}%")
    print(f"  delta < 1.25^2:   {metrics['delta_2']:.1f}%")
    print(f"  delta < 1.25^3:   {metrics['delta_3']:.1f}%")
    print(f"\nError Statistics:")
    print(f"  Median Error: {metrics['median_error']:.4f}m")
    print(f"  Std Error:    {metrics['std_error']:.4f}m")
    print(f"  Min Error:    {metrics['min_error']:.4f}m")
    print(f"  Max Error:    {metrics['max_error']:.4f}m")
    print("="*50)
    
    # Save results
    if args.save_results:
        # Save metrics
        results = {
            'model': args.model,
            'checkpoint': str(args.checkpoint),
            'dataset_split': args.split,
            'num_samples': len(all_pred_depths),
            'num_params': num_params,
            'latency_ms': avg_latency,
            'metrics': metrics,
            'args': vars(args)
        }
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions
        np.savez(
            output_dir / 'predictions.npz',
            pred_depths=all_pred_depths,
            gt_depths=all_gt_depths,
            gaze_positions=np.array(all_gaze_positions)
        )
        
        # Create visualizations
        if args.visualize:
            print("\nCreating visualizations...")
            fig = visualize_predictions(model, data_loader, device, args.num_vis_samples)
            fig.savefig(output_dir / 'visualizations.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save error distribution plot
        errors = np.abs(all_pred_depths - all_gt_depths)
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(metrics['mae'], color='red', linestyle='--', label=f'MAE: {metrics["mae"]:.3f}m')
        plt.axvline(metrics['median_error'], color='green', linestyle='--', label=f'Median: {metrics["median_error"]:.3f}m')
        plt.xlabel('Absolute Error (m)')
        plt.ylabel('Count')
        plt.title(f'Error Distribution - {args.model} Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to: {output_dir}")
    
    return metrics


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)