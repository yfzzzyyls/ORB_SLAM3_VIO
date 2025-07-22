#!/usr/bin/env python3
"""
Training script for Gaze-Only RT-MonoDepth model.
Trains on ADT dataset with gaze-specific depth prediction.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
import random

# Add project to path
sys.path.append(str(Path(__file__).parent))

from lowres_dataset import LowResADTDataset
from gaze_only_rtmonodepth import GazeOnlyRTMonoDepth, GazeDepthLoss
from model_rtmonodepth import DepthMetrics


class GazeAwareAugmentation:
    """Data augmentation that properly handles gaze coordinates."""
    
    def __init__(self, horizontal_flip_prob: float = 0.5, 
                 brightness_range: float = 0.2,
                 contrast_range: float = 0.2):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, rgb: torch.Tensor, depth: torch.Tensor, 
                 gaze_x: float, gaze_y: float, img_size: int = 88):
        """
        Apply augmentations while updating gaze coordinates.
        
        Returns:
            Augmented (rgb, depth, gaze_x, gaze_y)
        """
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            rgb = torch.flip(rgb, dims=[2])  # Flip width dimension
            depth = torch.flip(depth, dims=[2])
            gaze_x = img_size - 1 - gaze_x  # Flip gaze x coordinate
        
        # Brightness and contrast (only affects RGB, not depth or gaze)
        if self.brightness_range > 0 or self.contrast_range > 0:
            # Simple brightness adjustment
            brightness_factor = 1 + (random.random() * 2 - 1) * self.brightness_range
            rgb = torch.clamp(rgb * brightness_factor, 0, 1)
            
            # Simple contrast adjustment
            if self.contrast_range > 0:
                contrast_factor = 1 + (random.random() * 2 - 1) * self.contrast_range
                mean = rgb.mean(dim=[1, 2], keepdim=True)
                rgb = torch.clamp((rgb - mean) * contrast_factor + mean, 0, 1)
        
        return rgb, depth, gaze_x, gaze_y


def custom_collate_fn(batch):
    """Custom collate function that handles None gaze values and augmentation."""
    # Filter out samples with invalid gaze or missing GT depth
    valid_samples = []
    for sample in batch:
        if (sample['gaze'] is not None and 
            sample['gaze']['x'] >= 0 and 
            sample['gt_depth_at_gaze'] is not None):
            valid_samples.append(sample)
    
    if len(valid_samples) == 0:
        return None  # Skip this batch
    
    # Prepare batch tensors
    rgb_list = []
    depth_list = []
    gaze_x_list = []
    gaze_y_list = []
    valid_mask_list = []
    gt_depth_at_gaze_list = []
    depth_patch_stats_list = []
    high_res_patch_list = []
    patch_coords_list = []
    
    # Apply augmentation and collect samples
    augmentation = GazeAwareAugmentation()
    
    for sample in valid_samples:
        rgb = sample['rgb']
        depth = sample['depth']
        valid_mask = sample['valid_mask']
        gaze_x = sample['gaze']['x']
        gaze_y = sample['gaze']['y']
        gt_depth_at_gaze = sample['gt_depth_at_gaze']
        depth_patch_stats = sample.get('depth_patch_stats', None)
        high_res_patch = sample.get('high_res_patch', None)
        patch_coords = sample.get('patch_coords', None)
        
        # Apply augmentation
        # Get image size from the RGB tensor
        _, H, W = rgb.shape
        rgb, depth, gaze_x, gaze_y = augmentation(rgb, depth, gaze_x, gaze_y, img_size=W)
        
        # Validate gaze is still in bounds after augmentation
        # Get image size from the RGB tensor
        _, H, W = rgb.shape
        if 0 <= gaze_x < W and 0 <= gaze_y < H:
            rgb_list.append(rgb)
            depth_list.append(depth)
            gaze_x_list.append(gaze_x)
            gaze_y_list.append(gaze_y)
            valid_mask_list.append(valid_mask)
            gt_depth_at_gaze_list.append(gt_depth_at_gaze)
            
            # Note: depth_patch_stats are not augmented since they're ground truth
            # In a more sophisticated implementation, we might recompute stats after augmentation
            depth_patch_stats_list.append(depth_patch_stats)
            
            # Add high-res patch if available
            if high_res_patch is not None:
                high_res_patch_list.append(high_res_patch)
                patch_coords_list.append(patch_coords)
    
    if len(rgb_list) == 0:
        return None
    
    # Stack into tensors
    batch_dict = {
        'rgb': torch.stack(rgb_list),
        'depth': torch.stack(depth_list),
        'valid_mask': torch.stack(valid_mask_list),
        'gaze_x': torch.tensor(gaze_x_list, dtype=torch.float32),
        'gaze_y': torch.tensor(gaze_y_list, dtype=torch.float32),
        'gt_depth_at_gaze': torch.tensor(gt_depth_at_gaze_list, dtype=torch.float32).unsqueeze(1)
    }
    
    # Add high-res patches if available
    if len(high_res_patch_list) > 0:
        batch_dict['patch_rgb'] = torch.stack(high_res_patch_list)
        batch_dict['patch_coords'] = patch_coords_list
    
    # Add depth patch statistics if available
    if all(stats is not None for stats in depth_patch_stats_list):
        # Convert statistics to tensors
        stats_dict = {}
        stat_keys = depth_patch_stats_list[0].keys()
        
        for key in stat_keys:
            if key != 'depth_bin':  # Skip categorical for now
                values = [stats[key] for stats in depth_patch_stats_list]
                stats_dict[f'gt_{key}'] = torch.tensor(values, dtype=torch.float32)
        
        # Handle depth bin separately (categorical)
        depth_bins = [stats['depth_bin'] for stats in depth_patch_stats_list]
        stats_dict['gt_depth_bin'] = torch.tensor(depth_bins, dtype=torch.long)
        
        batch_dict.update(stats_dict)
    
    return batch_dict


def extract_gt_depth_at_gaze(depth_map: torch.Tensor, gaze_x: torch.Tensor, 
                            gaze_y: torch.Tensor) -> torch.Tensor:
    """
    Extract ground truth depth at gaze location using bilinear interpolation.
    Matches the method used for feature extraction.
    """
    B, C, H, W = depth_map.shape
    device = depth_map.device
    
    # Normalize gaze coordinates to [-1, 1]
    gaze_norm_x = 2.0 * gaze_x / (W - 1) - 1.0
    gaze_norm_y = 2.0 * gaze_y / (H - 1) - 1.0
    
    # Create sampling grid
    grid = torch.stack([gaze_norm_x, gaze_norm_y], dim=-1)
    grid = grid.view(B, 1, 1, 2)
    
    # Sample depth at gaze location
    sampled_depth = F.grid_sample(depth_map, grid, mode='bilinear', 
                                 padding_mode='border', align_corners=True)
    
    return sampled_depth.squeeze(2).squeeze(2)  # [B, 1]


def train_epoch(model, dataloader, optimizer, loss_fn, device, logger, 
                use_multi_scale_supervision=True):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    num_samples = 0  # Track samples instead of batches for proper averaging
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        if batch is None:
            continue
        
        # Get data
        rgb = batch['rgb'].to(device)
        gaze_x = batch['gaze_x'].to(device)
        gaze_y = batch['gaze_y'].to(device)
        gt_depth = batch['gt_depth_at_gaze'].to(device)  # Use exact GT depth
        
        batch_size = rgb.size(0)
        
        # Forward pass
        outputs = model(rgb, gaze_x, gaze_y)
        pred_depth = outputs['depth']
        
        # Main loss
        main_loss = loss_fn(pred_depth, gt_depth)
        
        # Multi-scale supervision
        aux_loss = 0
        if use_multi_scale_supervision and 'aux_depths' in outputs:
            for aux_depth in outputs['aux_depths']:
                aux_loss += 0.1 * loss_fn(aux_depth, gt_depth)
        
        # Total loss
        loss = main_loss + aux_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update stats - accumulate total loss weighted by batch size
        # This ensures proper averaging when batches have different sizes
        loss_value = loss.item()
        main_loss_value = main_loss.item()
        aux_loss_value = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0
        
        total_loss += loss_value * batch_size
        total_main_loss += main_loss_value * batch_size
        total_aux_loss += aux_loss_value * batch_size
        num_samples += batch_size
        
        # Update progress bar
        postfix_dict = {
            'loss': f'{loss_value:.4f}',
            'main': f'{main_loss_value:.4f}'
        }
        if aux_loss_value > 0:
            postfix_dict['aux'] = f'{aux_loss_value:.4f}'
        pbar.set_postfix(postfix_dict)
    
    # Average over all samples, not batches
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    avg_main_loss = total_main_loss / num_samples if num_samples > 0 else 0
    avg_aux_loss = total_aux_loss / num_samples if num_samples > 0 else 0
    
    return avg_loss, avg_main_loss, avg_aux_loss


def validate(model, dataloader, loss_fn, device, logger):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    # Metrics storage
    all_errors = []
    all_rel_errors = []
    all_sq_rel_errors = []
    all_rmse = []
    all_rmse_log = []
    all_a1 = []
    all_a2 = []
    all_a3 = []
    
    pbar = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            
            # Get data
            rgb = batch['rgb'].to(device)
            gaze_x = batch['gaze_x'].to(device)
            gaze_y = batch['gaze_y'].to(device)
            gt_depth = batch['gt_depth_at_gaze'].to(device)  # Use exact GT depth
            
            # Forward pass
            outputs = model(rgb, gaze_x, gaze_y)
            pred_depth = outputs['depth']
            
            # Compute loss
            loss = loss_fn(pred_depth, gt_depth)
            total_loss += loss.item()
            num_batches += 1
            
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
                    all_rmse_log.append((np.log(pred) - np.log(gt)) ** 2)
                    
                    # Threshold accuracy
                    ratio = max(pred / gt, gt / pred)
                    all_a1.append(ratio < 1.25)
                    all_a2.append(ratio < 1.25 ** 2)
                    all_a3.append(ratio < 1.25 ** 3)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Compute average metrics
    metrics = {}
    if len(all_errors) > 0:
        metrics['mae'] = np.mean(all_errors)
        metrics['abs_rel'] = np.mean(all_rel_errors)
        metrics['sq_rel'] = np.mean(all_sq_rel_errors)
        metrics['rmse'] = np.sqrt(np.mean(all_rmse))
        metrics['rmse_log'] = np.sqrt(np.mean(all_rmse_log))
        metrics['a1'] = np.mean(all_a1)
        metrics['a2'] = np.mean(all_a2)
        metrics['a3'] = np.mean(all_a3)
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle DataParallel
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
    
    # Always save latest
    latest_path = checkpoint_dir / 'checkpoint_latest.pth'
    torch.save(checkpoint, latest_path)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('GazeOnlyDepth')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main():
    parser = argparse.ArgumentParser(description='Train Gaze-Only RT-MonoDepth')
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Path to processed ADT dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/gaze_only',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs/gaze_only',
                        help='Directory to save logs')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (can be large for single-point prediction)')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Initial learning rate (scaled for batch 128)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth value in meters')
    parser.add_argument('--min-depth', type=float, default=0.1,
                        help='Minimum depth value in meters')
    parser.add_argument('--multi-scale-supervision', action='store_true', default=True,
                        help='Use multi-scale supervision')
    
    # Low-resolution parameters
    parser.add_argument('--lowres-scale', type=int, default=16,
                        help='Downscale factor (16 = 88x88 from 1408x1408)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Arguments: {args}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    logger.info(f"Using device: {device}")
    logger.info(f"Available GPUs: {n_gpus}")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    train_dataset = LowResADTDataset(
        data_root=args.data_root,
        split='train',
        scale_factor=args.lowres_scale,
        transform=None  # Augmentation handled in collate_fn
    )
    
    val_dataset = LowResADTDataset(
        data_root=args.data_root,
        split='val',
        scale_factor=args.lowres_scale,
        transform=None
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Create model
    logger.info("Creating model...")
    model = GazeOnlyRTMonoDepth(
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        use_multi_scale_supervision=args.multi_scale_supervision
    )
    
    # Move model to device
    model = model.to(device)
    
    # Multi-GPU support
    if n_gpus > 1:
        logger.info(f"Using DataParallel with {n_gpus} GPUs")
        model = nn.DataParallel(model)
    
    # Get number of parameters
    if hasattr(model, 'module'):
        num_params = model.module.get_num_params()
    else:
        num_params = model.get_num_params()
    logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Loss function
    loss_fn = GazeDepthLoss(alpha=0.85, grad_weight=0.1, rel_weight=0.1)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_mae = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle DataParallel when loading
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint and 'mae' in checkpoint['metrics']:
            best_mae = checkpoint['metrics']['mae']
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_loss, train_main_loss, train_aux_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, logger,
            use_multi_scale_supervision=args.multi_scale_supervision
        )
        
        logger.info(f"Train Loss: {train_loss:.4f} (main: {train_main_loss:.4f}, aux: {train_aux_loss:.4f})")
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, loss_fn, device, logger
        )
        
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info("Val Metrics:")
        for name, value in val_metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_metrics.get('mae', float('inf')) < best_mae
        if is_best:
            best_mae = val_metrics['mae']
            logger.info(f"New best model! MAE: {best_mae:.4f}")
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            checkpoint_dir, is_best=is_best
        )
        
        # Log to file
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_main_loss': train_main_loss,
            'train_aux_loss': train_aux_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'lr': scheduler.get_last_lr()[0]
        }
        
        log_file = log_dir / 'training_log.json'
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_data)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    logger.info("\nTraining complete!")
    logger.info(f"Best MAE: {best_mae:.4f}")


if __name__ == "__main__":
    main()