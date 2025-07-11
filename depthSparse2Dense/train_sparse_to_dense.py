#!/usr/bin/env python3
"""
Training script for sparse-to-dense depth completion
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import json

from slam_dataloader import create_dataloaders
from unet_depth import DepthCompletionUNet, DepthCompletionNet
from losses import SelfSupervisedDepthLoss, warp_frame
from metrics import compute_depth_metrics
from evaluate_full_ground_truth import load_ground_truth_depth


class DepthCompletionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check for multiple GPUs
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"Found {n_gpus} GPU(s)")
            for i in range(n_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        # Create model
        if config.model == 'unet':
            self.model = DepthCompletionUNet(n_channels=4).to(self.device)
        else:
            self.model = DepthCompletionNet().to(self.device)
        
        # Wrap model with DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        
        # Check if ground truth depth is available (moved here before dataloader creation)
        self.has_ground_truth = self._check_ground_truth_available()
        if self.has_ground_truth:
            print("Ground truth depth detected - will enable GT supervision")
        
        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            target_size=config.target_size if hasattr(config, 'target_size') else None,
            load_gt=self.has_ground_truth  # Enable GT loading if available
        )
        
        # Loss and optimizer
        # Use ground truth supervision if available
        gt_weight = getattr(config, 'gt_weight', 1.0) if self.has_ground_truth else 0.0
        self.criterion = SelfSupervisedDepthLoss(
            sparse_weight=config.sparse_weight,
            smooth_weight=config.smooth_weight,
            regularization_weight=getattr(config, 'regularization_weight', 0.1),
            gt_weight=gt_weight  # Enable GT supervision
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=5,
            factor=0.5
        )
        
        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.best_full_gt_metric = float('inf')  # Track best full GT performance
        
        # Full GT evaluation frequency
        self.full_gt_eval_freq = getattr(config, 'full_gt_eval_freq', 5)
        
        if self.has_ground_truth:
            print(f"Ground truth evaluation will be performed every {self.full_gt_eval_freq} epochs")
        
        # Load checkpoint if exists
        if config.resume:
            self.load_checkpoint()
    
    def _check_ground_truth_available(self):
        """Check if ground truth depth directory exists and has data"""
        gt_dir = Path(self.config.data_dir) / 'ground_truth_depth'
        if gt_dir.exists():
            # Check if there are any depth files
            depth_files = list(gt_dir.glob('*.npz'))
            return len(depth_files) > 0
        return False
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'photometric': 0, 'sparse': 0, 'smoothness': 0, 'regularization': 0, 'ground_truth': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            sparse_depth = batch['sparse_depth'].to(self.device)
            confidence = batch['confidence'].to(self.device)
            
            # Prepare input
            input_tensor = torch.cat([rgb, sparse_depth], dim=1)
            
            # Forward pass
            pred_depth = self.model(input_tensor)
            
            # For self-supervised loss, we need temporal frames
            # For now, use only sparse consistency and smoothness
            # In full implementation, you'd warp between frames
            rgb_warped = rgb  # Placeholder - should be warped from another frame
            
            # Get ground truth if available
            gt_depth = batch.get('gt_depth', None)
            if gt_depth is not None:
                gt_depth = gt_depth.to(self.device)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                pred_depth, sparse_depth, rgb, rgb_warped, gt_depth
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sparse': f"{loss_dict['sparse']:.4f}"
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, global_step)
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_losses = {'total': 0, 'photometric': 0, 'sparse': 0, 'smoothness': 0, 'regularization': 0, 'ground_truth': 0}
        
        # Initialize metrics accumulators
        val_metrics = {
            'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0,
            'a1': 0, 'a2': 0, 'a3': 0
        }
        
        # Initialize full GT metrics if evaluating this epoch
        evaluate_full_gt = (epoch % self.full_gt_eval_freq == 0) and self.has_ground_truth
        if evaluate_full_gt:
            full_gt_metrics = {
                'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0,
                'a1': 0, 'a2': 0, 'a3': 0
            }
            full_gt_count = 0
        
        # Initialize latency measurement
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                # Move to device
                rgb = batch['rgb'].to(self.device)
                sparse_depth = batch['sparse_depth'].to(self.device)
                confidence = batch['confidence'].to(self.device)
                
                # Prepare input
                input_tensor = torch.cat([rgb, sparse_depth], dim=1)
                
                # Forward pass for the batch
                pred_depth = self.model(input_tensor)
                
                # Measure single-image inference latency (only on first batch)
                if batch_idx == 0 and torch.cuda.is_available():
                    # Extract single image
                    single_input = input_tensor[:1]  # First image only
                    
                    # Warmup
                    for _ in range(10):
                        _ = self.model(single_input)
                    
                    # Time single image inference
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    # Measure multiple runs for stability
                    single_times = []
                    for _ in range(100):
                        torch.cuda.synchronize()
                        start_event.record()
                        _ = self.model(single_input)
                        end_event.record()
                        torch.cuda.synchronize()
                        single_times.append(start_event.elapsed_time(end_event))
                    
                    inference_times.extend(single_times)
                    
                    # Report single-image latency immediately
                    avg_single = np.mean(single_times)
                    std_single = np.std(single_times)
                    print(f"\n{'='*60}")
                    print(f"Single Image Inference Latency (batch_size=1):")
                    print(f"  Average: {avg_single:.2f} ms per image")
                    print(f"  Std Dev: {std_single:.2f} ms")
                    print(f"  Min: {np.min(single_times):.2f} ms")
                    print(f"  Max: {np.max(single_times):.2f} ms")
                    print(f"  FPS (single image): {1000.0/avg_single:.1f}")
                    print(f"{'='*60}\n")
                
                # Get ground truth if available
                gt_depth = batch.get('gt_depth', None)
                if gt_depth is not None:
                    gt_depth = gt_depth.to(self.device)
                
                # Compute loss
                rgb_warped = rgb  # Placeholder
                loss, loss_dict = self.criterion(
                    pred_depth, sparse_depth, rgb, rgb_warped, gt_depth
                )
                
                # Update losses
                for k, v in loss_dict.items():
                    val_losses[k] += v
                
                # Compute depth metrics on sparse points
                metrics = compute_depth_metrics(
                    pred_depth, sparse_depth, 
                    valid_mask=(sparse_depth > 0)
                )
                
                # Accumulate metrics
                for k, v in metrics.items():
                    val_metrics[k] += v
                
                # Full ground truth evaluation if enabled
                if evaluate_full_gt:
                    # Process each sample in batch for full GT evaluation
                    for i in range(len(batch['frame_id'])):
                        frame_id = batch['frame_id'][i].item()
                        
                        # Load ground truth depth
                        gt_depth = load_ground_truth_depth(frame_id, Path(self.config.data_dir))
                        if gt_depth is not None:
                            # Convert to tensor
                            gt_depth_tensor = torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0).to(self.device)
                            
                            # Compute metrics on ALL valid pixels
                            gt_valid_mask = gt_depth_tensor > 0
                            gt_metrics = compute_depth_metrics(
                                pred_depth[i:i+1], gt_depth_tensor, gt_valid_mask
                            )
                            
                            # Accumulate full GT metrics
                            for k, v in gt_metrics.items():
                                full_gt_metrics[k] += v
                            full_gt_count += 1
                
                # Save sample predictions
                if epoch % 5 == 0 and batch_idx == 0:
                    self.save_predictions(
                        rgb[0], sparse_depth[0], pred_depth[0], 
                        epoch, 'val'
                    )
        
        # Average losses and metrics
        for k in val_losses:
            val_losses[k] /= len(self.val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(self.val_loader)
        
        # Average full GT metrics if computed
        if evaluate_full_gt and full_gt_count > 0:
            for k in full_gt_metrics:
                full_gt_metrics[k] /= full_gt_count
            
            # Log full GT metrics to tensorboard
            for k, v in full_gt_metrics.items():
                self.writer.add_scalar(f'val/full_gt/{k}', v, epoch)
            
            # Store for printing
            val_losses['full_gt_metrics'] = full_gt_metrics
            val_losses['full_gt_count'] = full_gt_count
        
        # Log to tensorboard
        for k, v in val_losses.items():
            if k not in ['full_gt_metrics', 'full_gt_count']:
                self.writer.add_scalar(f'val/{k}', v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f'val/metrics/{k}', v, epoch)
        
        # Combine losses and metrics for return
        val_losses.update(val_metrics)
        
        # Calculate latency statistics
        if inference_times:
            val_losses['latency_mean'] = np.mean(inference_times)
            val_losses['latency_std'] = np.std(inference_times)
            val_losses['latency_min'] = np.min(inference_times)
            val_losses['latency_max'] = np.max(inference_times)
        
        return val_losses
    
    def save_predictions(self, rgb, sparse, pred, epoch, split):
        """Save prediction visualizations"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # RGB
        rgb_np = rgb.cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(rgb_np)
        axes[0].set_title('RGB Input')
        axes[0].axis('off')
        
        # Sparse depth
        sparse_np = sparse.cpu().squeeze().numpy()
        sparse_viz = sparse_np.copy()
        sparse_viz[sparse_viz == 0] = np.nan
        im1 = axes[1].imshow(sparse_viz, cmap='viridis')
        axes[1].set_title(f'Sparse Depth ({np.sum(sparse_np > 0)} points)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Predicted depth
        pred_np = pred.cpu().squeeze().numpy()
        im2 = axes[2].imshow(pred_np, cmap='viridis')
        axes[2].set_title('Predicted Dense Depth')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
        
        # Error map
        error = np.abs(pred_np - sparse_np)
        error[sparse_np == 0] = 0
        im3 = axes[3].imshow(error, cmap='hot')
        axes[3].set_title(f'Error (Mean: {error[sparse_np > 0].mean():.3f})')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{split}_epoch_{epoch}.png')
        plt.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint"""
        # Handle DataParallel wrapper
        model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_full_gt_metric': self.best_full_gt_metric,
            'config': self.config
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.output_dir / 'checkpoint_last.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
    
    def load_checkpoint(self):
        """Load training checkpoint"""
        checkpoint_path = self.output_dir / 'checkpoint_last.pth'
        if checkpoint_path.exists():
            # Load with weights_only=False for compatibility with saved config
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Handle DataParallel wrapper when loading
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint['best_loss']
            print(f"Resumed from epoch {self.start_epoch}")
            
            # Load best full GT metric if available
            if 'best_full_gt_metric' in checkpoint:
                self.best_full_gt_metric = checkpoint['best_full_gt_metric']
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.start_epoch, self.config.epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Train losses: {train_losses}")
            
            # Validate
            val_losses = self.validate(epoch)
            
            # Print validation losses
            print(f"Epoch {epoch} - Val losses: total={val_losses['total']:.4f}, "
                  f"sparse={val_losses['sparse']:.4f}, smooth={val_losses['smoothness']:.4f}")
            
            # Print validation metrics (sparse)
            print(f"Epoch {epoch} - Sparse metrics: abs_rel={val_losses['abs_rel']:.4f}, "
                  f"rmse={val_losses['rmse']:.4f}m, "
                  f"δ₁={val_losses['a1']:.4f} ({val_losses['a1']*100:.1f}%)")
            
            # Print full GT metrics if available
            if 'full_gt_metrics' in val_losses:
                full_gt = val_losses['full_gt_metrics']
                print(f"Epoch {epoch} - Full GT metrics: abs_rel={full_gt['abs_rel']:.4f}, "
                      f"rmse={full_gt['rmse']:.4f}m, "
                      f"δ₁={full_gt['a1']:.4f} ({full_gt['a1']*100:.1f}%) "
                      f"[{val_losses['full_gt_count']} frames]")
            
            # Latency was already printed during validation
            
            # Update learning rate
            self.scheduler.step(val_losses['total'])
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = val_losses['total']
            
            # Also track best full GT performance
            if 'full_gt_metrics' in val_losses:
                full_gt_rmse = val_losses['full_gt_metrics']['rmse']
                if full_gt_rmse < self.best_full_gt_metric:
                    self.best_full_gt_metric = full_gt_rmse
                    print(f"New best full GT RMSE: {full_gt_rmse:.4f}m")
                    # Save a special checkpoint for best full GT
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                        'full_gt_rmse': full_gt_rmse,
                        'config': self.config
                    }, self.output_dir / 'checkpoint_best_full_gt.pth')
            
            self.save_checkpoint(epoch, is_best)
            
            # Log learning rate
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', lr, epoch)
            print(f"Learning rate: {lr}")
            print("-" * 80)
        
        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train sparse-to-dense depth completion')
    
    # Data arguments
    parser.add_argument('--data_dir', required=True, help='Directory with processed SLAM data')
    parser.add_argument('--output_dir', default='experiments/depth_completion', 
                       help='Output directory')
    
    # Model arguments
    parser.add_argument('--model', choices=['unet', 'full'], default='unet',
                       help='Model architecture')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # Loss weights
    parser.add_argument('--sparse_weight', type=float, default=0.5)
    parser.add_argument('--smooth_weight', type=float, default=0.1)
    parser.add_argument('--regularization_weight', type=float, default=0.1)
    parser.add_argument('--gt_weight', type=float, default=1.0, help='Weight for ground truth supervision')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--full_gt_eval_freq', type=int, default=5,
                       help='Frequency of full ground truth evaluation (default: every 5 epochs)')
    
    # ADT specific arguments
    parser.add_argument('--target_size', type=int, nargs=2, default=None,
                       help='Target size (H W) for resizing. Default: None (keep original ADT size 480x640)')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = DepthCompletionTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()