#!/usr/bin/env python3
"""
Training script for flexible resolution gaze-only depth model.
Supports training at different image sizes while using the same efficient architecture.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import logging

# Add project to path
sys.path.append(str(Path(__file__).parent))

from flexible_dataset import FlexibleResolutionDataset
from flexible_gaze_encoder import FlexibleGazeOnlyDepth, MultiTaskGazeLoss
from gaze_only_rtmonodepth import GazeDepthLoss
from train_gaze_only import custom_collate_fn, save_checkpoint, setup_logging, validate


def train_epoch_multitask(model, dataloader, optimizer, loss_fn, device, logger):
    """Train for one epoch with multi-task learning."""
    model.train()
    
    total_loss = 0
    task_losses = {}
    num_samples = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        if batch is None:
            continue
        
        # Get data
        rgb = batch['rgb'].to(device)
        gaze_x = batch['gaze_x'].to(device)
        gaze_y = batch['gaze_y'].to(device)
        
        batch_size = rgb.size(0)
        
        # Forward pass
        outputs = model(rgb, gaze_x, gaze_y)
        
        # Compute multi-task loss
        loss, loss_dict = loss_fn(outputs, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update stats
        loss_value = loss.item()
        total_loss += loss_value * batch_size
        num_samples += batch_size
        
        # Accumulate task-specific losses
        for task, task_loss in loss_dict.items():
            if task not in task_losses:
                task_losses[task] = 0
            task_losses[task] += task_loss.item() * batch_size
        
        # Update progress bar
        postfix_dict = {'loss': f'{loss_value:.4f}'}
        # Show main task loss
        if 'depth' in loss_dict:
            postfix_dict['depth'] = f'{loss_dict["depth"].item():.4f}'
        pbar.set_postfix(postfix_dict)
    
    # Average losses
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    avg_task_losses = {task: loss / num_samples for task, loss in task_losses.items()}
    
    return avg_loss, avg_task_losses


def train_epoch_dual(model, dataloader, optimizer, loss_fn, device, logger):
    """Train for one epoch with dual-resolution model."""
    model.train()
    
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        if batch is None:
            continue
        
        # Get data
        context_rgb = batch['rgb'].to(device)
        patch_rgb = batch['patch_rgb'].to(device)
        gaze_x = batch['gaze_x'].to(device)
        gaze_y = batch['gaze_y'].to(device)
        gt_depth = batch['gt_depth_at_gaze'].to(device)
        
        batch_size = context_rgb.size(0)
        
        # Forward pass
        outputs = model(context_rgb, patch_rgb, gaze_x, gaze_y)
        pred_depth = outputs['depth']
        
        # Compute main loss
        main_loss = loss_fn(pred_depth, gt_depth)
        
        # Compute auxiliary losses if available
        aux_loss = 0
        if 'aux_depths' in outputs:
            for aux_depth in outputs['aux_depths']:
                aux_loss += loss_fn(aux_depth, gt_depth)
            aux_loss = aux_loss / len(outputs['aux_depths'])
            
            # Total loss with auxiliary weight
            aux_weight = 0.2
            loss = main_loss + aux_weight * aux_loss
        else:
            loss = main_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update stats
        loss_value = loss.item()
        total_loss += loss_value * batch_size
        total_main_loss += main_loss.item() * batch_size
        if aux_loss != 0:
            total_aux_loss += aux_loss.item() * batch_size
        num_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'main': f'{main_loss.item():.4f}'
        })
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    avg_main_loss = total_main_loss / num_samples if num_samples > 0 else 0
    avg_aux_loss = total_aux_loss / num_samples if num_samples > 0 else 0
    
    return avg_loss, avg_main_loss, avg_aux_loss


def validate_dual(model, dataloader, loss_fn, device, logger):
    """Validate the dual-resolution model."""
    model.eval()
    
    total_loss = 0
    num_samples = 0
    
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
            context_rgb = batch['rgb'].to(device)
            patch_rgb = batch['patch_rgb'].to(device)
            gaze_x = batch['gaze_x'].to(device)
            gaze_y = batch['gaze_y'].to(device)
            gt_depth = batch['gt_depth_at_gaze'].to(device)
            
            batch_size = context_rgb.size(0)
            
            # Forward pass
            outputs = model(context_rgb, patch_rgb, gaze_x, gaze_y)
            pred_depth = outputs['depth']
            
            # Compute loss
            loss = loss_fn(pred_depth, gt_depth)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            # Compute metrics
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
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    
    return avg_loss, metrics


def validate_multitask(model, dataloader, loss_fn, device, logger):
    """Validate the model with multi-task outputs."""
    model.eval()
    
    total_loss = 0
    task_losses = {}
    num_samples = 0
    
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
            gt_depth = batch['gt_depth_at_gaze'].to(device)
            
            batch_size = rgb.size(0)
            
            # Forward pass
            outputs = model(rgb, gaze_x, gaze_y)
            pred_depth = outputs['depth']
            
            # Compute loss (multi-task loss during training, but only depth matters for metrics)
            if model.training:  # Should be False during validation
                loss, loss_dict = loss_fn(outputs, batch)
            else:
                # For validation, we only care about depth prediction accuracy
                from gaze_only_rtmonodepth import GazeDepthLoss
                simple_loss_fn = GazeDepthLoss(alpha=0.85)
                loss = simple_loss_fn(pred_depth, gt_depth)
                loss_dict = {'depth': loss}
            
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            # Accumulate task losses
            for task, task_loss in loss_dict.items():
                if task not in task_losses:
                    task_losses[task] = 0
                if isinstance(task_loss, torch.Tensor):
                    task_losses[task] += task_loss.item() * batch_size
                else:
                    task_losses[task] += task_loss * batch_size
            
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
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    
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
    
    # Add task losses to metrics for logging
    metrics['task_losses'] = {task: loss / num_samples for task, loss in task_losses.items()}
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train Flexible Resolution Gaze-Only Depth Model')
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Path to processed ADT dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/flexible_gaze',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs/flexible_gaze',
                        help='Directory to save logs')
    
    # Model architecture
    parser.add_argument('--encoder-levels', type=int, default=3, choices=[2, 3, 4, 5],
                        help='Number of encoder levels (2-5)')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='Base number of channels in encoder')
    parser.add_argument('--gaze-feature-dim', type=int, default=64,
                        help='Dimension for gaze features at each scale')
    parser.add_argument('--image-size', type=int, default=88,
                        help='Input image size (square images)')
    parser.add_argument('--use-multi-task', action='store_true',
                        help='Use multi-task learning with patch statistics')
    
    # Dual-resolution options
    parser.add_argument('--use-dual-resolution', action='store_true',
                        help='Use dual-resolution model with high-res patch')
    parser.add_argument('--patch-size', type=int, default=96,
                        help='Size of high-res patch (default: 96)')
    parser.add_argument('--patch-channels', type=int, default=48,
                        help='Base channels for patch encoder (default: 48)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Initial learning rate')
    parser.add_argument('--lr-scaling', type=str, default='none', 
                        choices=['none', 'linear', 'sqrt'],
                        help='Learning rate scaling method with batch size')
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
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup paths with image size in directory name
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir = checkpoint_dir / f"size{args.image_size}_level{args.encoder_levels}_ch{args.base_channels}"
    log_dir = Path(args.log_dir) / f"size{args.image_size}_level{args.encoder_levels}_ch{args.base_channels}"
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Using {args.encoder_levels}-level encoder with {args.image_size}×{args.image_size} input")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    logger.info(f"Using device: {device}")
    logger.info(f"Available GPUs: {n_gpus}")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    train_dataset = FlexibleResolutionDataset(
        data_root=args.data_root,
        split='train',
        target_size=args.image_size,
        transform=None,
        use_high_res_patch=args.use_dual_resolution,
        patch_size=args.patch_size
    )
    
    val_dataset = FlexibleResolutionDataset(
        data_root=args.data_root,
        split='val',
        target_size=args.image_size,
        transform=None,
        use_high_res_patch=args.use_dual_resolution,
        patch_size=args.patch_size
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
    if args.use_dual_resolution:
        logger.info(f"Creating dual-resolution model: {args.image_size}×{args.image_size} context + {args.patch_size}×{args.patch_size} patch")
        
        from flexible_gaze_encoder import DualResolutionGazeDepth
        
        model = DualResolutionGazeDepth(
            context_size=args.image_size,
            context_levels=args.encoder_levels,
            context_channels=args.base_channels,
            patch_size=args.patch_size,
            patch_levels=args.encoder_levels,
            patch_channels=args.patch_channels,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            context_feature_dim=args.gaze_feature_dim,
            patch_feature_dim=192,  # Fixed for balanced approach
            use_attention_fusion=True,
            use_multi_scale_supervision=args.multi_scale_supervision
        )
    else:
        logger.info(f"Creating flexible model for {args.image_size}×{args.image_size} images...")
        logger.info(f"Multi-task learning: {'Enabled' if args.use_multi_task else 'Disabled'}")
        
        model = FlexibleGazeOnlyDepth(
            num_encoder_levels=args.encoder_levels,
            base_channels=args.base_channels,
            gaze_feature_dim=args.gaze_feature_dim,
            image_size=args.image_size,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            use_multi_scale_supervision=args.multi_scale_supervision,
            use_multi_task=args.use_multi_task
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
    
    # Log architecture details
    if args.use_dual_resolution:
        # For dual-resolution model
        if hasattr(model, 'module'):
            context_params = sum(p.numel() for p in model.module.context_encoder.parameters())
            patch_params = sum(p.numel() for p in model.module.patch_encoder.parameters())
            fusion_params = sum(p.numel() for p in model.module.fusion.parameters())
            predictor_params = sum(p.numel() for p in model.module.depth_predictor.parameters())
        else:
            context_params = sum(p.numel() for p in model.context_encoder.parameters())
            patch_params = sum(p.numel() for p in model.patch_encoder.parameters())
            fusion_params = sum(p.numel() for p in model.fusion.parameters())
            predictor_params = sum(p.numel() for p in model.depth_predictor.parameters())
        
        logger.info(f"Architecture breakdown:")
        logger.info(f"  Context encoder: {context_params:,} params")
        logger.info(f"  Patch encoder: {patch_params:,} params")
        logger.info(f"  Feature fusion: {fusion_params:,} params")
        logger.info(f"  Depth predictor: {predictor_params:,} params")
        logger.info(f"  Total: {num_params:,} params")
    else:
        # For single-resolution model
        if hasattr(model, 'module'):
            encoder_params = sum(p.numel() for p in model.module.encoder.parameters())
            predictor_params = sum(p.numel() for p in model.module.depth_predictor.parameters())
        else:
            encoder_params = sum(p.numel() for p in model.encoder.parameters())
            predictor_params = sum(p.numel() for p in model.depth_predictor.parameters())
        
        logger.info(f"Architecture breakdown:")
        logger.info(f"  Encoder: {encoder_params:,} params")
        logger.info(f"  Predictor: {predictor_params:,} params")
        logger.info(f"  Efficiency: {(1 - num_params/1234161)*100:.1f}% reduction vs RT-MonoDepth")
    
    # Calculate theoretical receptive field
    receptive_field = 7  # Initial conv
    for level in range(args.encoder_levels):
        receptive_field = receptive_field * 2 + 2  # Stride 2 + 3x3 conv
    logger.info(f"  Theoretical receptive field: {receptive_field} pixels")
    
    # Loss function
    if args.use_multi_task:
        loss_fn = MultiTaskGazeLoss(alpha=0.85)
        logger.info("Using multi-task loss function")
    else:
        loss_fn = GazeDepthLoss(alpha=0.85, grad_weight=0.1, rel_weight=0.1)
        if args.use_dual_resolution:
            logger.info("Using standard gaze depth loss for dual-resolution model")
        else:
            logger.info("Using standard gaze depth loss")
    
    # Scale learning rate based on batch size
    base_batch_size = 32  # Base batch size for reference
    if args.lr_scaling == 'linear':
        scaled_lr = args.lr * (args.batch_size / base_batch_size)
        logger.info(f"Using linear LR scaling: {args.lr:.1e} -> {scaled_lr:.1e}")
    elif args.lr_scaling == 'sqrt':
        scaled_lr = args.lr * np.sqrt(args.batch_size / base_batch_size)
        logger.info(f"Using sqrt LR scaling: {args.lr:.1e} -> {scaled_lr:.1e}")
    else:
        scaled_lr = args.lr
        logger.info(f"No LR scaling, using: {scaled_lr:.1e}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
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
        
        # Restore scheduler state
        for _ in range(start_epoch):
            scheduler.step()
    
    # Training loop
    logger.info("Starting training...")
    
    # Training functions - we'll define multi-task versions inline below
    # Standard training functions from train_gaze_only are already imported
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        if args.use_dual_resolution:
            train_loss, train_main_loss, train_aux_loss = train_epoch_dual(
                model, train_loader, optimizer, loss_fn, device, logger
            )
            logger.info(f"Train Loss: {train_loss:.4f} (main: {train_main_loss:.4f}, aux: {train_aux_loss:.4f})")
        elif args.use_multi_task:
            train_loss, loss_dict = train_epoch_multitask(
                model, train_loader, optimizer, loss_fn, device, logger
            )
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info("Train Losses by Task:")
            for task, loss in loss_dict.items():
                logger.info(f"  {task}: {loss:.4f}")
        else:
            train_loss, train_main_loss, train_aux_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, device, logger,
                use_multi_scale_supervision=args.multi_scale_supervision
            )
            logger.info(f"Train Loss: {train_loss:.4f} (main: {train_main_loss:.4f}, aux: {train_aux_loss:.4f})")
        
        # Validate
        if args.use_dual_resolution:
            val_loss, val_metrics = validate_dual(
                model, val_loader, loss_fn, device, logger
            )
        elif args.use_multi_task:
            val_loss, val_metrics = validate_multitask(
                model, val_loader, loss_fn, device, logger
            )
        else:
            val_loss, val_metrics = validate(
                model, val_loader, loss_fn, device, logger
            )
        
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info("Val Metrics:")
        for name, value in val_metrics.items():
            if isinstance(value, dict):
                # Skip nested dictionaries like task_losses
                continue
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
        # Filter out nested dicts from val_metrics for JSON serialization
        clean_val_metrics = {k: v for k, v in val_metrics.items() if not isinstance(v, dict)}
        
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': clean_val_metrics,
            'lr': scheduler.get_last_lr()[0],
            'image_size': args.image_size
        }
        
        if args.use_multi_task:
            log_data['loss_breakdown'] = loss_dict
        else:
            log_data['train_main_loss'] = train_main_loss
            log_data['train_aux_loss'] = train_aux_loss
        
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
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info(f"Image size used: {args.image_size}×{args.image_size}")


if __name__ == "__main__":
    main()