#!/usr/bin/env python3
"""
Training script for lightweight gaze-only depth model.
Uses efficient 2 or 3-level encoder designed for 88x88 input.
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

from lowres_dataset import LowResADTDataset
from lightweight_gaze_encoder import LightweightGazeOnlyDepth
from gaze_only_rtmonodepth import GazeDepthLoss
from train_gaze_only import custom_collate_fn, save_checkpoint, setup_logging, validate


def main():
    parser = argparse.ArgumentParser(description='Train Lightweight Gaze-Only Depth Model')
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Path to processed ADT dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/lightweight_gaze',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs/lightweight_gaze',
                        help='Directory to save logs')
    
    # Model architecture
    parser.add_argument('--encoder-levels', type=int, default=3, choices=[2, 3],
                        help='Number of encoder levels (2 or 3)')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='Base number of channels in encoder')
    parser.add_argument('--gaze-feature-dim', type=int, default=64,
                        help='Dimension for gaze features at each scale')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=4e-4,
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
    parser.add_argument('--multi-scale-supervision', action='store_true', default=True,
                        help='Use multi-scale supervision')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir = checkpoint_dir / f"level{args.encoder_levels}_ch{args.base_channels}"
    log_dir = Path(args.log_dir) / f"level{args.encoder_levels}_ch{args.base_channels}"
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Using {args.encoder_levels}-level lightweight encoder")
    
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
        scale_factor=16,  # 88x88
        transform=None
    )
    
    val_dataset = LowResADTDataset(
        data_root=args.data_root,
        split='val',
        scale_factor=16,
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
    logger.info(f"Creating {args.encoder_levels}-level lightweight model...")
    model = LightweightGazeOnlyDepth(
        num_encoder_levels=args.encoder_levels,
        base_channels=args.base_channels,
        gaze_feature_dim=args.gaze_feature_dim,
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
    
    # Log architecture details
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
        
        # Restore scheduler state
        for _ in range(start_epoch):
            scheduler.step()
    
    # Training loop
    logger.info("Starting training...")
    
    # Import train_epoch from original script
    from train_gaze_only import train_epoch
    
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
    logger.info(f"Model saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()