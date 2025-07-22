#!/usr/bin/env python3
"""
Flexible evaluation script that supports different model architectures and input sizes.
Can evaluate:
- 88x88 baseline models
- 88x88 multi-task models
- 352x352 models
- Dual-resolution models (88x88 context + 96x96 patch)
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader

# Add project to path
sys.path.append(str(Path(__file__).parent))

from flexible_dataset import FlexibleResolutionDataset
from flexible_gaze_encoder import FlexibleGazeOnlyDepth, DualResolutionGazeDepth
from train_gaze_only import custom_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Flexible Gaze Depth Models')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='auto',
                        choices=['auto', 'single', 'multitask', 'dual'],
                        help='Model type (auto-detect from checkpoint path)')
    
    # Model architecture arguments
    parser.add_argument('--image-size', type=int, default=88,
                        help='Input image size')
    parser.add_argument('--encoder-levels', type=int, default=3,
                        help='Number of encoder levels')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='Base channels for encoder')
    parser.add_argument('--gaze-feature-dim', type=int, default=64,
                        help='Dimension for gaze features')
    
    # Dual-resolution specific
    parser.add_argument('--patch-size', type=int, default=96,
                        help='High-res patch size for dual-resolution model')
    parser.add_argument('--patch-channels', type=int, default=32,
                        help='Base channels for patch encoder')
    
    # Dataset arguments
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Path to processed ADT dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-results', action='store_true',
                        help='Save detailed results')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def detect_model_type(checkpoint_path):
    """Auto-detect model type from checkpoint path."""
    path_lower = checkpoint_path.lower()
    
    if 'dual' in path_lower:
        return 'dual'
    elif 'multitask' in path_lower or 'multi_task' in path_lower:
        return 'multitask'
    else:
        return 'single'


def load_model(args, device):
    """Load model based on arguments."""
    # Auto-detect model type if needed
    if args.model_type == 'auto':
        model_type = detect_model_type(args.checkpoint)
        print(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type
    
    # Create model
    if model_type == 'dual':
        model = DualResolutionGazeDepth(
            context_size=args.image_size,
            context_levels=args.encoder_levels,
            context_channels=args.base_channels,
            patch_size=args.patch_size,
            patch_levels=args.encoder_levels,
            patch_channels=args.patch_channels,
            max_depth=10.0,
            min_depth=0.1,
            context_feature_dim=args.gaze_feature_dim,
            patch_feature_dim=192,
            use_attention_fusion=True,
            use_multi_scale_supervision=True
        )
        use_dual_resolution = True
    else:
        use_multi_task = (model_type == 'multitask')
        model = FlexibleGazeOnlyDepth(
            num_encoder_levels=args.encoder_levels,
            base_channels=args.base_channels,
            gaze_feature_dim=args.gaze_feature_dim,
            image_size=args.image_size,
            max_depth=10.0,
            min_depth=0.1,
            use_multi_scale_supervision=True,
            use_multi_task=use_multi_task
        )
        use_dual_resolution = False
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Get training info if available
    training_info = {}
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            training_info['epoch'] = checkpoint['epoch']
        if 'metrics' in checkpoint:
            training_info['val_metrics'] = checkpoint['metrics']
    
    return model, use_dual_resolution, training_info


def compute_metrics(pred_depths: np.ndarray, gt_depths: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics for depth prediction."""
    # Remove invalid predictions
    valid_mask = (gt_depths > 0.1) & (gt_depths < 10.0) & np.isfinite(pred_depths)
    pred_depths = pred_depths[valid_mask]
    gt_depths = gt_depths[valid_mask]
    
    if len(pred_depths) == 0:
        return {
            'mae': float('inf'),
            'rmse': float('inf'),
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'rmse_log': float('inf'),
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0,
        }
    
    # Basic metrics
    mae = np.mean(np.abs(pred_depths - gt_depths))
    rmse = np.sqrt(np.mean((pred_depths - gt_depths) ** 2))
    
    # Relative metrics
    abs_rel = np.mean(np.abs(pred_depths - gt_depths) / gt_depths)
    sq_rel = np.mean(((pred_depths - gt_depths) ** 2) / gt_depths)
    
    # Log metrics
    rmse_log = np.sqrt(np.mean((np.log(pred_depths) - np.log(gt_depths)) ** 2))
    
    # Threshold accuracy
    ratio = np.maximum(pred_depths / gt_depths, gt_depths / pred_depths)
    a1 = np.mean(ratio < 1.25)
    a2 = np.mean(ratio < 1.25 ** 2)
    a3 = np.mean(ratio < 1.25 ** 3)
    
    # Additional statistics
    metrics = {
        'mae': float(mae),
        'mae_std': float(np.std(np.abs(pred_depths - gt_depths))),
        'mae_median': float(np.median(np.abs(pred_depths - gt_depths))),
        'mae_max': float(np.max(np.abs(pred_depths - gt_depths))),
        'mae_percentile_95': float(np.percentile(np.abs(pred_depths - gt_depths), 95)),
        'rmse': float(rmse),
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse_log': float(rmse_log),
        'a1': float(a1),
        'a2': float(a2),
        'a3': float(a3),
        'num_samples': len(pred_depths)
    }
    
    return metrics


def evaluate_model(model, dataloader, device, use_dual_resolution):
    """Evaluate model on dataset."""
    all_preds = []
    all_gts = []
    inference_times = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch is None:
                continue
            
            # Get data
            if use_dual_resolution:
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
            
            # Skip first few batches for timing (warmup)
            if batch_idx > 5:
                inference_times.append(inference_time)
            
            # Collect predictions
            all_preds.extend(pred_depth.cpu().numpy().flatten())
            all_gts.extend(gt_depth.cpu().numpy().flatten())
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_gts)
    
    # Add timing metrics
    if inference_times:
        metrics['inference_time_mean'] = float(np.mean(inference_times) * 1000)  # ms
        metrics['inference_time_std'] = float(np.std(inference_times) * 1000)
        metrics['fps'] = float(1.0 / np.mean(inference_times))
    
    return metrics, all_preds, all_gts


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = FlexibleResolutionDataset(
        data_root=args.data_root,
        split=args.split,
        target_size=args.image_size,
        use_high_res_patch=(args.model_type == 'dual' or 'dual' in args.checkpoint.lower()),
        patch_size=args.patch_size
    )
    print(f"Dataset size: {len(dataset)} samples")
    
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
    model, use_dual_resolution, training_info = load_model(args, device)
    
    # Evaluate
    metrics, predictions, ground_truths = evaluate_model(
        model, dataloader, device, use_dual_resolution
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.split} split ({metrics['num_samples']} samples)")
    print(f"Image size: {args.image_size}×{args.image_size}")
    if use_dual_resolution:
        print(f"Patch size: {args.patch_size}×{args.patch_size}")
    
    if training_info:
        print(f"\nTraining info:")
        if 'epoch' in training_info:
            print(f"  Trained epochs: {training_info['epoch'] + 1}")
        if 'val_metrics' in training_info:
            val_mae = training_info['val_metrics'].get('mae', 'N/A')
            print(f"  Validation MAE: {val_mae}")
    
    print(f"\nTest Metrics:")
    print(f"  MAE: {metrics['mae']:.4f}m (±{metrics['mae_std']:.4f})")
    print(f"  Median AE: {metrics['mae_median']:.4f}m")
    print(f"  95th percentile: {metrics['mae_percentile_95']:.4f}m")
    print(f"  Max error: {metrics['mae_max']:.4f}m")
    print(f"  RMSE: {metrics['rmse']:.4f}m")
    print(f"  abs_rel: {metrics['abs_rel']:.4f}")
    print(f"  sq_rel: {metrics['sq_rel']:.4f}")
    print(f"  RMSE log: {metrics['rmse_log']:.4f}")
    print(f"  a1 (δ<1.25): {metrics['a1']*100:.1f}%")
    print(f"  a2 (δ<1.25²): {metrics['a2']*100:.1f}%")
    print(f"  a3 (δ<1.25³): {metrics['a3']*100:.1f}%")
    
    if 'inference_time_mean' in metrics:
        print(f"\nTiming:")
        print(f"  Inference: {metrics['inference_time_mean']:.2f}±{metrics['inference_time_std']:.2f}ms")
        print(f"  FPS: {metrics['fps']:.1f}")
    
    # Save results if requested
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(args.checkpoint).stem
        results_file = output_dir / f"results_{model_name}_{timestamp}.json"
        
        results = {
            'checkpoint': str(args.checkpoint),
            'args': vars(args),
            'metrics': metrics,
            'training_info': training_info
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save predictions
        predictions_file = output_dir / f"predictions_{model_name}_{timestamp}.npz"
        np.savez(predictions_file,
                 predictions=predictions,
                 ground_truths=ground_truths)
        print(f"Predictions saved to: {predictions_file}")
    
    print("="*60)


if __name__ == "__main__":
    main()