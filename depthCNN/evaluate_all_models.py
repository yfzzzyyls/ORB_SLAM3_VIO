#!/usr/bin/env python3
"""
Comprehensive evaluation script for comparing different model checkpoints.
Evaluates on test set and generates detailed metrics and visualizations.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from flexible_dataset import FlexibleResolutionDataset
from flexible_gaze_encoder import FlexibleGazeOnlyDepth, DualResolutionGazeDepth
from train_gaze_only import custom_collate_fn
from gaze_only_rtmonodepth import GazeDepthLoss


def evaluate_model(model, dataloader, device, model_name="Model"):
    """Evaluate a model and return comprehensive metrics."""
    model.eval()
    
    # Metrics storage
    all_errors = []
    all_rel_errors = []
    all_sq_rel_errors = []
    all_rmse = []
    all_rmse_log = []
    all_a1 = []
    all_a2 = []
    all_a3 = []
    
    # Timing
    inference_times = []
    
    print(f"\nEvaluating {model_name}...")
    pbar = tqdm(dataloader, desc=f'Evaluating {model_name}')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            # Get data based on model type
            if 'patch_rgb' in batch:
                # Dual-resolution model
                context_rgb = batch['rgb'].to(device)
                patch_rgb = batch['patch_rgb'].to(device)
                gaze_x = batch['gaze_x'].to(device)
                gaze_y = batch['gaze_y'].to(device)
                gt_depth = batch['gt_depth_at_gaze'].to(device)
                
                # Time inference
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                
                outputs = model(context_rgb, patch_rgb, gaze_x, gaze_y)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                inference_time = time.time() - start_time
                
            else:
                # Single-resolution model
                rgb = batch['rgb'].to(device)
                gaze_x = batch['gaze_x'].to(device)
                gaze_y = batch['gaze_y'].to(device)
                gt_depth = batch['gt_depth_at_gaze'].to(device)
                
                # Time inference
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                
                outputs = model(rgb, gaze_x, gaze_y)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                inference_time = time.time() - start_time
            
            pred_depth = outputs['depth']
            inference_times.append(inference_time)
            
            # Compute metrics for each sample
            for i in range(pred_depth.shape[0]):
                pred = pred_depth[i].item()
                gt = gt_depth[i].item()
                
                if gt > 0:  # Valid depth
                    # Absolute error
                    abs_err = abs(pred - gt)
                    all_errors.append(abs_err)
                    
                    # Relative error
                    rel_err = abs_err / gt
                    all_rel_errors.append(rel_err)
                    
                    # Squared relative error
                    sq_rel_err = ((pred - gt) ** 2) / gt
                    all_sq_rel_errors.append(sq_rel_err)
                    
                    # RMSE
                    all_rmse.append((pred - gt) ** 2)
                    
                    # RMSE log
                    if pred > 0:
                        all_rmse_log.append((np.log(pred) - np.log(gt)) ** 2)
                    
                    # Threshold accuracy
                    ratio = max(pred / gt, gt / pred)
                    all_a1.append(ratio < 1.25)
                    all_a2.append(ratio < 1.25 ** 2)
                    all_a3.append(ratio < 1.25 ** 3)
    
    # Compute final metrics
    metrics = {}
    if len(all_errors) > 0:
        metrics['mae'] = np.mean(all_errors)
        metrics['mae_std'] = np.std(all_errors)
        metrics['mae_median'] = np.median(all_errors)
        metrics['mae_max'] = np.max(all_errors)
        metrics['mae_percentile_95'] = np.percentile(all_errors, 95)
        
        metrics['abs_rel'] = np.mean(all_rel_errors)
        metrics['sq_rel'] = np.mean(all_sq_rel_errors)
        metrics['rmse'] = np.sqrt(np.mean(all_rmse))
        metrics['rmse_log'] = np.sqrt(np.mean(all_rmse_log)) if all_rmse_log else 0
        
        metrics['a1'] = np.mean(all_a1)
        metrics['a2'] = np.mean(all_a2)
        metrics['a3'] = np.mean(all_a3)
        
        # Timing metrics
        metrics['inference_time_mean'] = np.mean(inference_times[10:]) * 1000  # Skip warmup, convert to ms
        metrics['inference_time_std'] = np.std(inference_times[10:]) * 1000
        metrics['fps'] = 1.0 / np.mean(inference_times[10:]) if inference_times else 0
        
        metrics['num_samples'] = len(all_errors)
    
    return metrics


def load_checkpoint(checkpoint_path, device, model_type='flexible', image_size=88, use_dual_resolution=False):
    """Load a checkpoint and return the model."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model based on type
    if use_dual_resolution:
        model = DualResolutionGazeDepth(
            context_size=image_size,
            context_levels=3,
            context_channels=32,
            patch_size=96,
            patch_levels=3,
            patch_channels=32,  # Using 32 as per your training
            max_depth=10.0,
            min_depth=0.1,
            context_feature_dim=64,
            patch_feature_dim=192,
            use_attention_fusion=True,
            use_multi_scale_supervision=True
        )
    else:
        # Determine if it's multi-task based on checkpoint
        use_multi_task = 'multitask' in checkpoint_path.lower()
        
        model = FlexibleGazeOnlyDepth(
            num_encoder_levels=3,
            base_channels=32,
            gaze_feature_dim=64,
            image_size=image_size,
            max_depth=10.0,
            min_depth=0.1,
            use_multi_scale_supervision=True,
            use_multi_task=use_multi_task
        )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Get training metrics if available
    training_metrics = {}
    if 'metrics' in checkpoint:
        training_metrics = checkpoint['metrics']
    if 'epoch' in checkpoint:
        training_metrics['epoch'] = checkpoint['epoch']
    
    return model, training_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple model checkpoints')
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Path to processed dataset')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='List of checkpoint paths to evaluate')
    parser.add_argument('--names', nargs='+', required=True,
                        help='Names for each checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--test-split', type=str, default='test',
                        help='Which split to evaluate on (val or test)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.checkpoints) != len(args.names):
        raise ValueError("Number of checkpoints must match number of names")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = {}
    
    # Evaluate each checkpoint
    for checkpoint_path, model_name in zip(args.checkpoints, args.names):
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        # Determine model configuration based on name
        if '88x88' in model_name.lower() or '88' in model_name:
            image_size = 88
        elif '352' in model_name:
            image_size = 352
        else:
            image_size = 88  # Default
        
        use_dual_resolution = 'dual' in model_name.lower()
        
        # Create dataset
        dataset = FlexibleResolutionDataset(
            data_root=args.data_root,
            split=args.test_split,
            target_size=image_size,
            use_high_res_patch=use_dual_resolution,
            patch_size=96
        )
        
        print(f"Dataset: {len(dataset)} samples from {args.test_split} split")
        print(f"Image size: {image_size}×{image_size}")
        print(f"Dual-resolution: {use_dual_resolution}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # Load model
        model, training_metrics = load_checkpoint(
            checkpoint_path, 
            device, 
            image_size=image_size,
            use_dual_resolution=use_dual_resolution
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Evaluate
        metrics = evaluate_model(model, dataloader, device, model_name)
        
        # Store results
        all_results[model_name] = {
            'checkpoint': checkpoint_path,
            'image_size': image_size,
            'dual_resolution': use_dual_resolution,
            'num_params': num_params,
            'training_metrics': training_metrics,
            'test_metrics': metrics
        }
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"  MAE: {metrics['mae']:.4f}m (±{metrics['mae_std']:.4f})")
        print(f"  Median AE: {metrics['mae_median']:.4f}m")
        print(f"  95th percentile: {metrics['mae_percentile_95']:.4f}m")
        print(f"  abs_rel: {metrics['abs_rel']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}m")
        print(f"  a1 (δ<1.25): {metrics['a1']*100:.1f}%")
        print(f"  a2 (δ<1.25²): {metrics['a2']*100:.1f}%")
        print(f"  a3 (δ<1.25³): {metrics['a3']*100:.1f}%")
        print(f"  Inference: {metrics['inference_time_mean']:.2f}±{metrics['inference_time_std']:.2f}ms ({metrics['fps']:.1f} FPS)")
    
    # Save detailed results
    results_file = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create comparison table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"{'Model':<25} {'MAE (m)':<10} {'abs_rel':<10} {'RMSE (m)':<10} {'a1 (%)':<10} {'FPS':<10} {'Params':<12}")
    print(f"{'-'*100}")
    
    for name, results in all_results.items():
        metrics = results['test_metrics']
        print(f"{name:<25} {metrics['mae']:<10.4f} {metrics['abs_rel']:<10.4f} "
              f"{metrics['rmse']:<10.4f} {metrics['a1']*100:<10.1f} "
              f"{metrics['fps']:<10.1f} {results['num_params']:<12,}")
    
    # Find best model
    best_model = min(all_results.items(), key=lambda x: x[1]['test_metrics']['mae'])
    print(f"\n{'='*100}")
    print(f"BEST MODEL: {best_model[0]} with MAE={best_model[1]['test_metrics']['mae']:.4f}m")
    print(f"{'='*100}")
    
    # Save summary
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*100}\n\n")
        
        for name, results in all_results.items():
            metrics = results['test_metrics']
            f.write(f"{name}:\n")
            f.write(f"  Checkpoint: {results['checkpoint']}\n")
            f.write(f"  MAE: {metrics['mae']:.4f}m (±{metrics['mae_std']:.4f})\n")
            f.write(f"  abs_rel: {metrics['abs_rel']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}m\n")
            f.write(f"  a1: {metrics['a1']*100:.1f}%\n")
            f.write(f"  Inference: {metrics['inference_time_mean']:.2f}ms ({metrics['fps']:.1f} FPS)\n")
            f.write(f"  Parameters: {results['num_params']:,}\n\n")
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()