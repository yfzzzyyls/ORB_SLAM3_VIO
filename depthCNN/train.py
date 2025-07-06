#!/usr/bin/env python3
"""
Training script for RT-MonoDepth-S on ADT dataset.
Trains a lightweight monocular depth estimation model on 1408x1408 RGB-D pairs.
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
import time
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict

# Add project to path
sys.path.append(str(Path(__file__).parent))

from vrs_dataset import ADTVRSDataset, Compose, RandomCrop, RandomHorizontalFlip
from processed_dataset import ProcessedADTDataset
from model_rtmonodepth import RTMonoDepthS, SILogLoss, DepthMetrics


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
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_lr(optimizer):
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
    
    # Keep only last 5 checkpoints
    all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if len(all_checkpoints) > 5:
        for ckpt in all_checkpoints[:-5]:
            ckpt.unlink()


def train_epoch(model, train_loader, optimizer, loss_fn, device, logger):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_samples = 0
    
    progress = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(progress):
        # Move to device
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        # Forward pass
        outputs = model(rgb)
        pred_depth = outputs['depth']
        
        # Compute loss
        loss = loss_fn(pred_depth, depth, valid_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update stats
        batch_size = rgb.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        progress.set_postfix({
            'loss': loss.item(),
            'lr': get_lr(optimizer)
        })
    
    avg_loss = total_loss / total_samples
    return avg_loss


def validate(model, val_loader, loss_fn, device, logger):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    all_metrics = []
    
    with torch.no_grad():
        progress = tqdm(val_loader, desc="Validation")
        for batch in progress:
            # Move to device
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            # Forward pass
            outputs = model(rgb)
            pred_depth = outputs['depth']
            
            # Compute loss
            loss = loss_fn(pred_depth, depth, valid_mask)
            
            # Compute metrics
            metrics = DepthMetrics.compute_metrics(pred_depth, depth, valid_mask)
            all_metrics.append(metrics)
            
            # Update stats
            batch_size = rgb.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            progress.set_postfix({'loss': loss.item()})
    
    # Average loss
    avg_loss = total_loss / total_samples
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train RT-MonoDepth-S on ADT dataset')
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
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    cache_dir = Path(args.cache_dir)
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Arguments: {args}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    # Training transforms
    train_transforms = Compose([
        RandomCrop((args.crop_size, args.crop_size)),
        RandomHorizontalFlip(p=0.5)
    ])
    
    # Check if we're using processed data or raw VRS
    data_root = Path(args.data_root)
    if (data_root / 'train').exists() and (data_root / 'val').exists():
        # Use processed dataset
        logger.info("Using pre-processed dataset")
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
        # Use VRS dataset
        logger.info("Using VRS dataset (slower)")
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = RTMonoDepthS(max_depth=args.max_depth, min_depth=args.min_depth)
    model = model.to(device)
    
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
        checkpoint = torch.load(args.resume, map_location=device)
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
            'lr': get_lr(optimizer)
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


if __name__ == "__main__":
    main()