#!/usr/bin/env python3
"""
Training script for RT-MonoDepth-S on ADT dataset with low-resolution support.
Trains a lightweight monocular depth estimation model on downsampled RGB-D pairs.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict

# Add project to path
sys.path.append(str(Path(__file__).parent))

from vrs_dataset import ADTVRSDataset, Compose, RandomCrop, RandomHorizontalFlip
from processed_dataset import ProcessedADTDataset
from lowres_dataset import LowResADTDataset
from model_rtmonodepth import RTMonoDepthS, SILogLoss, DepthMetrics


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


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('RTMonoDepthS')
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


def train_epoch(model, dataloader, optimizer, loss_fn, device, logger):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Get data
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        # Forward pass
        outputs = model(rgb)
        pred_depth = outputs['depth'] if isinstance(outputs, dict) else outputs
        
        # Resize prediction to match target size if needed
        if pred_depth.shape[2:] != depth.shape[2:]:
            pred_depth = F.interpolate(pred_depth, size=depth.shape[2:], mode='bilinear', align_corners=False)
        
        # Compute loss
        loss = loss_fn(pred_depth, depth, valid_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, dataloader, loss_fn, device, logger):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    # Depth metrics (DepthMetrics is a static class)
    all_metrics = []
    
    pbar = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for batch in pbar:
            # Get data
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            # Forward pass
            outputs = model(rgb)
            pred_depth = outputs['depth'] if isinstance(outputs, dict) else outputs
            
            # Resize prediction to match target size if needed
            if pred_depth.shape[2:] != depth.shape[2:]:
                pred_depth = F.interpolate(pred_depth, size=depth.shape[2:], mode='bilinear', align_corners=False)
            
            # Compute loss
            loss = loss_fn(pred_depth, depth, valid_mask)
            total_loss += loss.item()
            num_batches += 1
            
            # Compute metrics
            batch_metrics = DepthMetrics.compute_metrics(
                pred_depth, depth, valid_mask
            )
            all_metrics.append(batch_metrics)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Average loss
    avg_loss = total_loss / num_batches
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_loss, avg_metrics


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


def get_lr(optimizer):
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    parser = argparse.ArgumentParser(description='Train RT-MonoDepth-S on ADT dataset with low-resolution support')
    parser.add_argument('--data-root', type=str, default='/mnt/ssd_ext/incSeg-data/adt',
                        help='Path to ADT dataset root')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Directory to cache extracted frames')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory to save logs')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth value in meters')
    parser.add_argument('--min-depth', type=float, default=0.1,
                        help='Minimum depth value in meters')
    
    # Data parameters
    parser.add_argument('--subsample', type=int, default=1,
                        help='Subsample factor for frames (1=all frames, 10=every 10th frame)')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='Random crop size for training')
    
    # Low-resolution training
    parser.add_argument('--lowres-scale', type=int, default=1,
                        help='Downscale factor for low-resolution training (1=full res, 16=1/16 res)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    cache_dir = Path(args.cache_dir)
    
    # Adjust checkpoint directory for low-res training
    if args.lowres_scale > 1:
        checkpoint_dir = checkpoint_dir / f"lowres_{args.lowres_scale}x"
        log_dir = log_dir / f"lowres_{args.lowres_scale}x"
    
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
    
    # Check if we're using processed data or raw VRS
    data_root = Path(args.data_root)
    if (data_root / 'train').exists() and (data_root / 'val').exists():
        # Use processed dataset
        logger.info("Using pre-processed dataset")
        
        if args.lowres_scale > 1:
            # Low-resolution training
            logger.info(f"Low-resolution training with scale factor {args.lowres_scale}")
            
            # For low-res training, we don't need cropping if already at 88x88
            # Only apply horizontal flip augmentation
            train_transforms = Compose([
                RandomHorizontalFlip(p=0.5)
            ])
            
            train_dataset = LowResADTDataset(
                data_root=args.data_root,
                split='train',
                scale_factor=args.lowres_scale,
                transform=train_transforms
            )
            
            val_dataset = LowResADTDataset(
                data_root=args.data_root,
                split='val',
                scale_factor=args.lowres_scale,
                transform=None  # No augmentation for validation
            )
        else:
            # Full resolution training
            train_transforms = Compose([
                RandomCrop((args.crop_size, args.crop_size)),
                RandomHorizontalFlip(p=0.5)
            ])
            
            train_dataset = ProcessedADTDataset(
                data_root=args.data_root,
                split='train',
                transform=train_transforms
            )
            
            val_dataset = ProcessedADTDataset(
                data_root=args.data_root,
                split='val',
                transform=None  # No augmentation for validation
            )
    else:
        # Use VRS dataset (doesn't support low-res yet)
        if args.lowres_scale > 1:
            raise ValueError("Low-resolution training is only supported with pre-processed data. "
                           "Run extract_dataset.py first.")
        
        logger.info("Using VRS dataset (slower)")
        train_transforms = Compose([
            RandomCrop((args.crop_size, args.crop_size)),
            RandomHorizontalFlip(p=0.5)
        ])
        
        train_dataset = ADTVRSDataset(
            adt_root=args.data_root,
            split='train',
            transform=train_transforms,
            cache_dir=cache_dir / 'train',
            subsample_factor=args.subsample
        )
        
        val_dataset = ADTVRSDataset(
            adt_root=args.data_root,
            split='val',
            transform=None,  # No augmentation for validation
            cache_dir=cache_dir / 'val',
            subsample_factor=args.subsample * 2  # Less frequent sampling for validation
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
    model = RTMonoDepthS(max_depth=args.max_depth, min_depth=args.min_depth)
    
    # Move model to device first before DataParallel
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
    loss_fn = SILogLoss(lambda_weight=0.85)
    
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
    best_abs_rel = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Handle DataParallel when loading
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint and 'abs_rel' in checkpoint['metrics']:
            best_abs_rel = checkpoint['metrics']['abs_rel']
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, logger
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, loss_fn, device, logger
        )
        
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info("Val Metrics:")
        for name, value in val_metrics.items():
            logger.info(f"  {name}: {value:.3f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_metrics['abs_rel'] < best_abs_rel
        if is_best:
            best_abs_rel = val_metrics['abs_rel']
            logger.info(f"New best model! abs_rel: {best_abs_rel:.3f}")
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            checkpoint_dir, is_best=is_best
        )
        
        # Log to file
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'lr': get_lr(optimizer),
            'lowres_scale': args.lowres_scale
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
    logger.info(f"Best abs_rel: {best_abs_rel:.3f}")
    if args.lowres_scale > 1:
        logger.info(f"Trained at {args.lowres_scale}x downscale (resolution: {1408//args.lowres_scale}×{1408//args.lowres_scale})")


if __name__ == "__main__":
    main()