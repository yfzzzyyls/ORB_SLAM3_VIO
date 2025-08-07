import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from spatial_dual_resolution_coordconv import SpatialDualResolutionGazeDepth
from spatial_dual_resolution_dataset import SpatialDualResolutionDataset, custom_collate_fn


class SILogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss."""
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target, mask):
        # Apply mask
        pred_masked = pred[mask > 0]
        target_masked = target[mask > 0]
        
        if pred_masked.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # Compute log difference
        log_diff = torch.log(pred_masked + 1e-8) - torch.log(target_masked + 1e-8)
        
        # Scale-invariant loss
        loss = torch.mean(log_diff ** 2) - self.alpha * (torch.mean(log_diff) ** 2)
        
        return loss


class BerHuLoss(nn.Module):
    """BerHu (Reverse Huber) Loss."""
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, pred, target, mask):
        # Apply mask
        pred_masked = pred[mask > 0]
        target_masked = target[mask > 0]
        
        if pred_masked.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # Compute absolute difference
        diff = torch.abs(pred_masked - target_masked)
        
        # Dynamic threshold
        c = self.threshold * torch.max(diff).detach()
        if c.item() == 0:
            return torch.mean(diff)  # all zeros -> 0 loss
        
        # BerHu loss
        loss = torch.where(diff <= c, diff, (diff**2 + c**2) / (2*c))
        
        return torch.mean(loss)


class EdgeAwareSmoothLoss(nn.Module):
    """Edge-aware smoothness loss."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, image):
        # Ensure same resolution - downsample image to match pred
        if pred.shape[-2:] != image.shape[-2:]:
            image = F.interpolate(image, size=pred.shape[-2:], mode='bilinear', align_corners=False)
            
        # Compute image gradients
        grad_img_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        grad_img_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        
        # Compute depth gradients
        grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        # Edge-aware weighting
        weight_x = torch.exp(-torch.mean(grad_img_x, dim=1, keepdim=True))
        weight_y = torch.exp(-torch.mean(grad_img_y, dim=1, keepdim=True))
        
        # Smoothness loss
        loss_x = torch.mean(weight_x * grad_pred_x)
        loss_y = torch.mean(weight_y * grad_pred_y)
        
        return loss_x + loss_y


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
        
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://',
                           world_size=world_size, rank=rank)
    
    return True, rank, world_size, gpu


def reduce_metric(metric, world_size):
    """Reduce metric across all processes."""
    if world_size == 1:
        return metric
    
    metric_tensor = torch.tensor(metric).cuda()
    dist.all_reduce(metric_tensor)
    return metric_tensor.item() / world_size


def add_weight_decay(model, wd, skip_modules=(nn.GroupNorm, nn.BatchNorm2d), skip_keywords=('bias',)):
    """Create param groups with selective weight decay."""
    decay, no_decay = [], []
    for m in model.modules():
        if isinstance(m, skip_modules):
            for p in m.parameters(recurse=False):
                if p.requires_grad: 
                    no_decay.append(p)
        else:
            for n, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if any(k in n for k in skip_keywords):
                    no_decay.append(p)
                else:
                    decay.append(p)
    return [{'params': decay, 'weight_decay': wd},
            {'params': no_decay, 'weight_decay': 0.0}]


@torch.no_grad()
def ema_update(student, teacher, decay):
    """Update EMA model weights."""
    s = student.module if isinstance(student, DDP) else student
    for ps, pt in zip(s.parameters(), teacher.parameters()):
        pt.data.mul_(decay).add_(ps.data, alpha=1.0 - decay)
    for bs, bt in zip(s.buffers(), teacher.buffers()):
        bt.copy_(bs)


def compute_metrics(pred, target, mask):
    """Compute evaluation metrics."""
    # Apply mask
    pred_masked = pred[mask > 0]
    target_masked = target[mask > 0]
    
    if pred_masked.numel() == 0:
        return {}
        
    # Absolute relative error
    abs_rel = torch.mean(torch.abs(pred_masked - target_masked) / target_masked)
    
    # Squared relative error
    sq_rel = torch.mean(((pred_masked - target_masked) ** 2) / target_masked)
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred_masked - target_masked) ** 2))
    
    # RMSE log
    rmse_log = torch.sqrt(torch.mean((torch.log(pred_masked) - torch.log(target_masked)) ** 2))
    
    # Threshold accuracies
    thresh = torch.maximum(pred_masked / target_masked, target_masked / pred_masked)
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    
    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'a1': a1.item(),
        'a2': a2.item(),
        'a3': a3.item()
    }


def train_epoch(model, train_loader, optimizer, scheduler, loss_fns, epoch, 
                writer, distributed, rank, world_size, model_ema=None, ema_decay=0.999):
    """Train for one epoch."""
    model.train()
    
    si_log_loss_fn, berhu_loss_fn, smooth_loss_fn = loss_fns
    
    total_loss = 0
    total_samples = 0
    metrics_sum = {}
    
    # Create progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(total=len(train_loader), desc=f'Epoch [{epoch}]', 
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue
            
        # Move to GPU
        context_rgb = batch['context_rgb'].cuda(non_blocking=True)
        patch_rgb = batch['patch_rgb'].cuda(non_blocking=True)
        depth_gt = batch['depth'].cuda(non_blocking=True)
        valid_mask = batch['valid_mask'].cuda(non_blocking=True)
        gaze_x = batch['gaze_x'].cuda(non_blocking=True)
        gaze_y = batch['gaze_y'].cuda(non_blocking=True)
        
        # Forward pass
        pred_depth, log_sigma = model(context_rgb, patch_rgb, gaze_x, gaze_y)
        
        # Compute losses
        si_log_loss = si_log_loss_fn(pred_depth.squeeze(1), depth_gt, valid_mask)
        berhu_loss = berhu_loss_fn(pred_depth.squeeze(1), depth_gt, valid_mask)
        smooth_loss = smooth_loss_fn(pred_depth, patch_rgb)
        
        # Heteroscedastic uncertainty loss with log_sigma
        log_sigma = log_sigma.squeeze(1).clamp(-8, 8)
        residual = pred_depth.squeeze(1) - depth_gt
        # Negative log likelihood: 0.5 * exp(-2*log_sigma) * residual^2 + log_sigma
        heteroscedastic_loss = 0.5 * torch.exp(-2 * log_sigma) * (residual**2) + log_sigma
        heteroscedastic_loss = heteroscedastic_loss[valid_mask > 0].mean()
        
        # Total loss
        loss = si_log_loss + 0.1 * berhu_loss + 0.01 * smooth_loss + 0.1 * heteroscedastic_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Update EMA model
        if model_ema is not None:
            ema_update(model, model_ema, ema_decay)
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(pred_depth.squeeze(1), depth_gt, valid_mask)
            
        # Accumulate
        total_loss += loss.item() * context_rgb.size(0)
        total_samples += context_rgb.size(0)
        
        for k, v in metrics.items():
            if k not in metrics_sum:
                metrics_sum[k] = 0
            metrics_sum[k] += v * context_rgb.size(0)
            
        # Update progress bar
        if rank == 0:
            # Calculate running average for display
            avg_loss = total_loss / total_samples
            avg_abs_rel = metrics_sum.get('abs_rel', 0) / total_samples
            avg_rmse = metrics_sum.get('rmse', 0) / total_samples
            avg_a1 = metrics_sum.get('a1', 0) / total_samples
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'AbsRel': f'{avg_abs_rel:.3f}',
                'RMSE': f'{avg_rmse:.3f}',
                'a1': f'{avg_a1:.3f}'
            })
            pbar.update(1)
    
    # Close progress bar
    if rank == 0:
        pbar.close()
                  
    # Reduce metrics across processes
    if distributed:
        total_loss = reduce_metric(total_loss, world_size)
        total_samples = reduce_metric(total_samples, world_size)
        for k in metrics_sum:
            metrics_sum[k] = reduce_metric(metrics_sum[k], world_size)
            
    # Average metrics
    avg_loss = total_loss / total_samples
    avg_metrics = {k: v / total_samples for k, v in metrics_sum.items()}
    
    # Update scheduler
    scheduler.step()
    
    # Log to tensorboard
    if rank == 0 and writer is not None:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/si_log_loss', si_log_loss.item(), epoch)
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        for k, v in avg_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
            
    return avg_loss, avg_metrics


def validate(model, val_loader, loss_fns, epoch, writer, distributed, rank, world_size):
    """Validate the model."""
    model.eval()
    
    si_log_loss_fn, berhu_loss_fn, smooth_loss_fn = loss_fns
    
    total_loss = 0
    total_samples = 0
    metrics_sum = {}
    
    # Create progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(total=len(val_loader), desc=f'Val [{epoch}]', 
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch is None:
                continue
                
            # Move to GPU
            context_rgb = batch['context_rgb'].cuda(non_blocking=True)
            patch_rgb = batch['patch_rgb'].cuda(non_blocking=True)
            depth_gt = batch['depth'].cuda(non_blocking=True)
            valid_mask = batch['valid_mask'].cuda(non_blocking=True)
            gaze_x = batch['gaze_x'].cuda(non_blocking=True)
            gaze_y = batch['gaze_y'].cuda(non_blocking=True)
            
            # Forward pass
            pred_depth, log_sigma = model(context_rgb, patch_rgb, gaze_x, gaze_y)
            
            # Compute losses
            si_log_loss = si_log_loss_fn(pred_depth.squeeze(1), depth_gt, valid_mask)
            berhu_loss = berhu_loss_fn(pred_depth.squeeze(1), depth_gt, valid_mask)
            smooth_loss = smooth_loss_fn(pred_depth, patch_rgb)
            
            # Heteroscedastic uncertainty loss with log_sigma
            log_sigma = log_sigma.squeeze(1).clamp(-8, 8)
            residual = pred_depth.squeeze(1) - depth_gt
            heteroscedastic_loss = 0.5 * torch.exp(-2 * log_sigma) * (residual**2) + log_sigma
            heteroscedastic_loss = heteroscedastic_loss[valid_mask > 0].mean()
            
            # Total loss
            loss = si_log_loss + 0.1 * berhu_loss + 0.01 * smooth_loss + 0.1 * heteroscedastic_loss
            
            # Compute metrics
            metrics = compute_metrics(pred_depth.squeeze(1), depth_gt, valid_mask)
            
            # Accumulate
            total_loss += loss.item() * context_rgb.size(0)
            total_samples += context_rgb.size(0)
            
            for k, v in metrics.items():
                if k not in metrics_sum:
                    metrics_sum[k] = 0
                metrics_sum[k] += v * context_rgb.size(0)
            
            # Update progress bar
            if rank == 0:
                avg_loss = total_loss / total_samples if total_samples > 0 else 0
                avg_abs_rel = metrics_sum.get('abs_rel', 0) / total_samples if total_samples > 0 else 0
                avg_a1 = metrics_sum.get('a1', 0) / total_samples if total_samples > 0 else 0
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'AbsRel': f'{avg_abs_rel:.3f}',
                    'a1': f'{avg_a1:.3f}'
                })
                pbar.update(1)
    
    # Close progress bar
    if rank == 0:
        pbar.close()
                
    # Reduce metrics across processes
    if distributed:
        total_loss = reduce_metric(total_loss, world_size)
        total_samples = reduce_metric(total_samples, world_size)
        for k in metrics_sum:
            metrics_sum[k] = reduce_metric(metrics_sum[k], world_size)
            
    # Average metrics
    avg_loss = total_loss / total_samples
    avg_metrics = {k: v / total_samples for k, v in metrics_sum.items()}
    
    # Log to tensorboard
    if rank == 0 and writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        for k, v in avg_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
            
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./processed_data',
                       help='Root directory of dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='./checkpoints/spatial_dual_coordconv',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str,
                       default='./logs/spatial_dual_coordconv',
                       help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    parser.add_argument('--max-train-sequences', type=int, default=30,
                       help='Maximum number of training sequences to use')
    parser.add_argument('--max-val-sequences', type=int, default=7,
                       help='Maximum number of validation sequences to use')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for sequence selection')
    
    args = parser.parse_args()
    
    # Setup distributed training
    distributed, rank, world_size, gpu = setup_distributed()
    
    # Create directories
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
    # Create model
    model = SpatialDualResolutionGazeDepth()
    model = model.cuda()
    
    if distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        
    # Create datasets
    train_dataset = SpatialDualResolutionDataset(
        data_root=args.data_root,
        split='train',
        augment=True,
        max_sequences=args.max_train_sequences,
        random_seed=args.random_seed
    )
    
    val_dataset = SpatialDualResolutionDataset(
        data_root=args.data_root,
        split='val',
        augment=False,
        max_sequences=args.max_val_sequences,
        random_seed=args.random_seed
    )
    
    # Create samplers
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Create optimizer with selective weight decay
    target_for_groups = model.module if isinstance(model, DDP) else model
    param_groups = add_weight_decay(target_for_groups, wd=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    
    # Create warm-up + cosine scheduler
    warmup_epochs = 5
    cosine_epochs = args.epochs - warmup_epochs
    
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    
    # Create EMA model
    ema_decay = 0.999  # Can be tuned (0.999-0.9995)
    if isinstance(model, DDP):
        model_ema = deepcopy(model.module).cuda().eval()
    else:
        model_ema = deepcopy(model).cuda().eval()
    for p in model_ema.parameters():
        p.requires_grad_(False)
    
    # Create loss functions
    si_log_loss = SILogLoss(alpha=0.85)
    berhu_loss = BerHuLoss(threshold=0.2)
    smooth_loss = EdgeAwareSmoothLoss()
    loss_fns = (si_log_loss, berhu_loss, smooth_loss)
    
    # Create tensorboard writer
    if rank == 0:
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
        
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")
            
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
            
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fns,
            epoch, writer, distributed, rank, world_size,
            model_ema=model_ema, ema_decay=ema_decay
        )
        
        # Validate regular model
        val_loss, val_metrics = validate(
            model, val_loader, loss_fns, epoch, writer,
            distributed, rank, world_size
        )
        
        # Validate EMA model
        ema_val_loss, ema_val_metrics = validate(
            model_ema, val_loader, loss_fns, epoch, writer,
            distributed, rank, world_size
        )
        
        # Print epoch summary
        if rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, AbsRel: {train_metrics['abs_rel']:.3f}, "
                  f"RMSE: {train_metrics['rmse']:.3f}, a1: {train_metrics['a1']:.3f}")
            print(f"  Val Loss: {val_loss:.4f}, AbsRel: {val_metrics['abs_rel']:.3f}, "
                  f"RMSE: {val_metrics['rmse']:.3f}, a1: {val_metrics['a1']:.3f}")
            print(f"  EMA Val Loss: {ema_val_loss:.4f}, AbsRel: {ema_val_metrics['abs_rel']:.3f}, "
                  f"RMSE: {ema_val_metrics['rmse']:.3f}, a1: {ema_val_metrics['a1']:.3f}")
                  
        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.module.state_dict() if distributed else model.state_dict(),
                'model_ema': model_ema.module.state_dict() if (distributed and hasattr(model_ema, 'module')) else model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'ema_val_loss': ema_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'ema_val_metrics': ema_val_metrics,
                'best_val_loss': best_val_loss,
                'args': args
            }
            
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Save best model (using EMA validation loss)
            if ema_val_loss < best_val_loss:
                best_val_loss = ema_val_loss
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint_best.pth'))
                print(f"  New best model saved (ema_val_loss: {ema_val_loss:.4f})")
                
            # Save latest
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth'))
            
    # Clean up
    if distributed:
        dist.destroy_process_group()
        
    if rank == 0 and writer is not None:
        writer.close()
        

if __name__ == '__main__':
    main()