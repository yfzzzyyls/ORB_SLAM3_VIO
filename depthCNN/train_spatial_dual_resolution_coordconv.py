import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import cv2
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Set OpenCV threads to prevent CPU oversubscription
cv2.setNumThreads(0)

from spatial_dual_resolution_coordconv import SpatialDualResolutionGazeDepth
from spatial_dual_resolution_dataset import SpatialDualResolutionDataset, custom_collate_fn

# WebDataset imports (optional, fallback to regular dataset if not available)
try:
    import webdataset as wds
    from PIL import Image
    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False
    print("WebDataset not available, using regular dataset loading")


def create_webdataset_shards(data_root, output_dir, split='train', shard_size_mb=1024, max_sequences=None):
    """Create WebDataset tar shards from ADT dataset with pre-computed edge maps."""
    if not HAS_WEBDATASET:
        raise ImportError("WebDataset not installed. Run: pip install webdataset")
    
    import json
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    split_dir = os.path.join(data_root, split)
    
    # Get all sequences
    all_sequences = sorted([d for d in os.listdir(split_dir) 
                          if os.path.isdir(os.path.join(split_dir, d))])
    
    if max_sequences and len(all_sequences) > max_sequences:
        sequences = all_sequences[:max_sequences]
    else:
        sequences = all_sequences
    
    print(f"Creating shards for {len(sequences)} sequences in {split} split")
    
    # Calculate samples per shard
    shard_size_bytes = shard_size_mb * 1024 * 1024
    estimated_sample_size = 3 * 1024 * 1024  # ~3MB per sample (RGB + depth + edge)
    samples_per_shard = max(1, shard_size_bytes // estimated_sample_size)
    
    shard_pattern = os.path.join(output_dir, f"{split}_%06d.tar")
    sink = wds.ShardWriter(shard_pattern, maxsize=shard_size_bytes)
    
    sample_count = 0
    for seq in tqdm(sequences, desc=f"Processing {split} sequences"):
        seq_dir = os.path.join(split_dir, seq)
        metadata_file = os.path.join(seq_dir, 'metadata.json')
        
        if not os.path.exists(metadata_file):
            continue
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for frame_info in metadata['frames']:
            if frame_info.get('depth') and frame_info.get('has_gaze', False):
                try:
                    # Load RGB
                    rgb_path = os.path.join(seq_dir, 'rgb', frame_info['rgb'])
                    with open(rgb_path, 'rb') as f:
                        rgb_data = f.read()
                    
                    # Load depth
                    depth_path = os.path.join(seq_dir, 'depth', frame_info['depth'])
                    depth_data = np.load(depth_path)
                    depth = (depth_data['depth'] / 1000.0).astype(np.float16)  # mm to meters, save as float16
                    
                    # Pre-compute edge map (CRITICAL for performance)
                    depth_clean = depth.copy()
                    depth_clean[depth <= 0] = 0
                    sobel_x = cv2.Sobel(depth_clean, cv2.CV_32F, 1, 0, ksize=3)
                    sobel_y = cv2.Sobel(depth_clean, cv2.CV_32F, 0, 1, ksize=3)
                    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                    edge_magnitude[depth <= 0] = 0
                    
                    # Save as .npy with proper shape encoding
                    depth_buffer = io.BytesIO()
                    np.save(depth_buffer, depth)
                    depth_bytes = depth_buffer.getvalue()
                    
                    edge_buffer = io.BytesIO()
                    np.save(edge_buffer, edge_magnitude.astype(np.float16))  # float16 to save space
                    edge_bytes = edge_buffer.getvalue()
                    
                    # Load gaze
                    gaze_path = os.path.join(seq_dir, 'gaze', frame_info['gaze'])
                    with open(gaze_path, 'r') as f:
                        gaze_json = f.read()
                    
                    # Create sample
                    sample = {
                        "__key__": f"{seq}_{frame_info.get('index', 0):06d}",
                        "rgb.jpg": rgb_data,
                        "depth.npy": depth_bytes,
                        "edge.npy": edge_bytes,
                        "gaze.json": gaze_json,
                        "meta.json": json.dumps({
                            "seq": seq,
                            "frame_id": frame_info.get('index', 0),
                            "height": 700,  # Correct shape
                            "width": 800
                        })
                    }
                    
                    sink.write(sample)
                    sample_count += 1
                    
                except Exception as e:
                    print(f"Error processing {seq} frame {frame_info.get('index', 0)}: {e}")
                    continue
    
    sink.close()
    print(f"Created {split} shards with {sample_count} samples")
    return sample_count


def make_webdataset_loader(shard_dir, split, batch_size, num_workers, distributed, world_size, rank):
    """Create WebDataset dataloader for efficient streaming."""
    if not HAS_WEBDATASET:
        return None
    
    shard_pattern = os.path.join(shard_dir, f"{split}_*.tar")
    
    # Check if shards exist
    import glob
    if not glob.glob(shard_pattern):
        print(f"No shards found at {shard_pattern}")
        return None
    
    # Create dataset pipeline
    dataset = wds.WebDataset(shard_pattern, resampled=True if split == 'train' else False)
    
    # WebDataset handles DDP distribution automatically with resampled=True
    # No need for explicit shard splitting in 0.2.x
    
    # Decode and preprocess
    def preprocess(sample):
        # Load RGB
        rgb = Image.open(io.BytesIO(sample['rgb.jpg'])).convert('RGB')
        rgb_tensor = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        
        # Load depth (already in meters)
        depth = np.load(io.BytesIO(sample['depth.npy']))
        
        # Load pre-computed edge map
        edge = np.load(io.BytesIO(sample['edge.npy']))
        
        # Load gaze
        gaze_data = json.loads(sample['gaze.json'])
        gaze_x = gaze_data['x_pixel']
        gaze_y = gaze_data['y_pixel']
        
        # Load metadata
        meta = json.loads(sample['meta.json'])
        
        # Multi-point sampling (16x: 1 real + 15 random)
        k_extra = 15 if split == 'train' else 0
        samples = []
        
        # Original gaze point
        samples.append({
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'is_real': True
        })
        
        # Random points with edge-biased sampling
        if k_extra > 0:
            for _ in range(k_extra):
                if np.random.random() < 0.3:  # 30% edge-biased
                    # Edge-biased sampling
                    valid_mask = (depth > 0).astype(np.float32)
                    edge_weights = edge * valid_mask + 0.01 * valid_mask
                    if edge_weights.sum() > 0:
                        edge_weights = edge_weights / edge_weights.sum()
                        flat_idx = np.random.choice(edge_weights.size, p=edge_weights.ravel())
                        y, x = np.unravel_index(flat_idx, edge_weights.shape)
                        # Scale to 1408x1408 (depth is 700x800)
                        x = x * 1408.0 / 800.0
                        y = y * 1408.0 / 700.0
                    else:
                        x = np.random.uniform(100, 1308)
                        y = np.random.uniform(100, 1308)
                else:
                    # Uniform sampling
                    x = np.random.uniform(100, 1308)
                    y = np.random.uniform(100, 1308)
                
                samples.append({
                    'gaze_x': x,
                    'gaze_y': y,
                    'is_real': False
                })
        
        return {
            'rgb': rgb_tensor,
            'depth': depth,
            'edge': edge,
            'samples': samples,
            'meta': meta
        }
    
    dataset = dataset.decode().map(preprocess)
    
    # Batching
    dataset = dataset.batched(batch_size, partial=False)
    
    # Create dataloader
    loader = wds.WebLoader(
        dataset,
        batch_size=None,  # Already batched
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Set epoch length for progress tracking
    if split == 'train':
        # Estimate based on dataset size with multi-point sampling
        loader.length = 220099 * 16 // batch_size // (world_size if distributed else 1)
    else:
        loader.length = 19364 // batch_size // (world_size if distributed else 1)
    
    return loader


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


class GradientConsistencyLoss(nn.Module):
    """Gradient consistency loss for sharper depth edges (Sobel-like)."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, image, valid_mask):
        """
        Args:
            pred: predicted depth [B, 1, H, W]
            target: ground truth depth [B, H, W]
            image: RGB image [B, 3, H', W']
            valid_mask: valid depth mask [B, H, W]
        """
        # Ensure same resolution
        if image.shape[-2:] != pred.shape[-2:]:
            image = F.interpolate(image, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        
        target = target.unsqueeze(1)  # [B, 1, H, W]
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
        
        # Sobel-like central difference for better edge detection
        if pred.shape[-1] > 2 and pred.shape[-2] > 2:
            # Depth gradients (central difference)
            grad_pred_x = pred[:, :, :, 2:] - pred[:, :, :, :-2]  # [B, 1, H, W-2]
            grad_pred_y = pred[:, :, 2:, :] - pred[:, :, :-2, :]  # [B, 1, H-2, W]
            
            grad_target_x = target[:, :, :, 2:] - target[:, :, :, :-2]
            grad_target_y = target[:, :, 2:, :] - target[:, :, :-2, :]
            
            # Valid masks for gradients
            valid_x = valid_mask[:, :, :, 1:-1] * valid_mask[:, :, :, 2:] * valid_mask[:, :, :, :-2]
            valid_y = valid_mask[:, :, 1:-1, :] * valid_mask[:, :, 2:, :] * valid_mask[:, :, :-2, :]
            
            # Image gradients for edge-aware weighting
            img_grad_x = (image[:, :, :, 2:] - image[:, :, :, :-2]).abs().mean(1, keepdim=True)
            img_grad_y = (image[:, :, 2:, :] - image[:, :, :-2, :]).abs().mean(1, keepdim=True)
            
            # Edge-aware weights (high weight where image has low gradient = smooth regions)
            weight_x = torch.exp(-10 * img_grad_x) * valid_x
            weight_y = torch.exp(-10 * img_grad_y) * valid_y
            
            # Gradient consistency loss
            loss_x = ((grad_pred_x - grad_target_x).abs() * weight_x).sum() / (weight_x.sum() + 1e-6)
            loss_y = ((grad_pred_y - grad_target_y).abs() * weight_y).sum() / (weight_y.sum() + 1e-6)
            
            return loss_x + loss_y
        else:
            return torch.tensor(0.0, device=pred.device)


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

def compute_gaze_metrics(pred_gaze, target_gaze, valid_mask):
    """Compute gaze-specific evaluation metrics."""
    # Filter valid gaze points
    valid = (valid_mask > 0) & (target_gaze > 0)
    
    if valid.sum() == 0:
        return {}
    
    pred_valid = pred_gaze[valid]
    target_valid = target_gaze[valid]
    
    # MAE
    mae = torch.mean(torch.abs(pred_valid - target_valid))
    
    # Relative error
    rel_err = torch.mean(torch.abs(pred_valid - target_valid) / target_valid)
    
    # Threshold accuracy (a1 for gaze)
    thresh = torch.maximum(pred_valid / target_valid, target_valid / pred_valid)
    gaze_a1 = (thresh < 1.25).float().mean()
    
    return {
        'gaze_mae': mae.item(),
        'gaze_rel': rel_err.item(),
        'gaze_a1': gaze_a1.item()
    }


def train_epoch(model, train_loader, optimizer, scheduler, loss_fns, epoch, 
                writer, distributed, rank, world_size, model_ema=None, ema_decay=0.999):
    """Train for one epoch."""
    model.train()
    
    si_log_loss_fn, berhu_loss_fn, smooth_loss_fn, grad_consistency_loss_fn = loss_fns
    
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
        
        # Forward pass - now returns 3 outputs (Fix #1)
        pred_depth, log_sigma, pred_gaze_depth = model(context_rgb, patch_rgb, gaze_x, gaze_y)
        
        # Get gaze depth GT from batch
        gaze_depth_gt = batch['gaze_depth_gt'].cuda(non_blocking=True)
        
        # Compute losses
        si_log_loss = si_log_loss_fn(pred_depth.squeeze(1), depth_gt, valid_mask)
        berhu_loss = berhu_loss_fn(pred_depth.squeeze(1), depth_gt, valid_mask)
        smooth_loss = smooth_loss_fn(pred_depth, patch_rgb)
        grad_consistency = grad_consistency_loss_fn(pred_depth, depth_gt, patch_rgb, valid_mask)
        
        # Heteroscedastic uncertainty loss with log_sigma
        log_sigma = log_sigma.squeeze(1).clamp(-8, 8)
        residual = pred_depth.squeeze(1) - depth_gt
        # Negative log likelihood: 0.5 * exp(-2*log_sigma) * residual^2 + log_sigma
        heteroscedastic_loss = 0.5 * torch.exp(-2 * log_sigma) * (residual**2) + log_sigma
        heteroscedastic_loss = heteroscedastic_loss[valid_mask > 0].mean()
        
        # NEW: Gaussian-weighted center loss (Fix #4)
        if hasattr(model, 'module'):  # DDP wrapper
            gaze_weights = model.module.gaze_weights.to(pred_depth.device)
        else:
            gaze_weights = model.gaze_weights.to(pred_depth.device)
        
        # Apply weights with valid mask
        w = gaze_weights.unsqueeze(0) * valid_mask  # [B, 22, 22]
        w = w / (w.sum(dim=(-1,-2), keepdim=True) + 1e-6)  # Renormalize
        center_loss = ((pred_depth.squeeze(1) - depth_gt)**2 * w).sum(dim=(-1,-2)).mean()
        
        # NEW: Scalar gaze depth loss (Fix #1)
        # Only compute loss for valid gaze points
        gaze_valid = (gaze_depth_gt > 0).float()
        
        # Check if we have real gaze points (weight them higher)
        if 'is_real_gaze' in batch:
            is_real_gaze = batch['is_real_gaze'].cuda(non_blocking=True).float()
            # Weight real gaze points 2x more than sampled points
            gaze_weight = torch.where(is_real_gaze > 0.5, 2.0, 1.0)
        else:
            gaze_weight = torch.ones_like(gaze_valid)
        
        if gaze_valid.sum() > 0:
            # Apply weighted loss
            gaze_diff = torch.abs(pred_gaze_depth.squeeze(-1) - gaze_depth_gt)
            weighted_loss = gaze_diff * gaze_valid * gaze_weight
            gaze_loss = weighted_loss.sum() / (gaze_valid * gaze_weight).sum()
        else:
            gaze_loss = torch.tensor(0.0, device=pred_depth.device)
        
        # Total loss with gaze as primary objective (point prediction focus)
        loss = 1.0 * si_log_loss + 0.05 * berhu_loss + 0.01 * smooth_loss + 0.05 * heteroscedastic_loss + \
               0.1 * center_loss + 2.0 * gaze_loss + 0.05 * grad_consistency
        
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
            gaze_metrics = compute_gaze_metrics(pred_gaze_depth.squeeze(-1), gaze_depth_gt, 
                                                 (gaze_depth_gt > 0).float())
            
        # Accumulate
        total_loss += loss.item() * context_rgb.size(0)
        total_samples += context_rgb.size(0)
        
        for k, v in metrics.items():
            if k not in metrics_sum:
                metrics_sum[k] = 0
            metrics_sum[k] += v * context_rgb.size(0)
            
        for k, v in gaze_metrics.items():
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
            avg_gaze_rel = metrics_sum.get('gaze_rel', 0) / total_samples
            avg_gaze_a1 = metrics_sum.get('gaze_a1', 0) / total_samples
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'AbsRel': f'{avg_abs_rel:.3f}',
                'a1': f'{avg_a1:.3f}',
                'GazeRel': f'{avg_gaze_rel:.3f}',
                'Gaze_a1': f'{avg_gaze_a1:.3f}'
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
            if 'gaze' in k:
                writer.add_scalar(f'train/gaze/{k}', v, epoch)
            else:
                writer.add_scalar(f'train/{k}', v, epoch)
            
    return avg_loss, avg_metrics


def validate(model, val_loader, loss_fns, epoch, writer, distributed, rank, world_size, use_tta=False):
    """Validate the model with optional test-time augmentation."""
    model.eval()
    
    si_log_loss_fn, berhu_loss_fn, smooth_loss_fn, grad_consistency_loss_fn = loss_fns
    
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
            
            # Forward pass with optional TTA
            if use_tta:
                # Original prediction
                pred_depth, log_sigma, pred_gaze_depth = model(context_rgb, patch_rgb, gaze_x, gaze_y)
                
                # Flipped prediction
                context_rgb_flip = torch.flip(context_rgb, dims=[-1])
                patch_rgb_flip = torch.flip(patch_rgb, dims=[-1])
                gaze_x_flip = -gaze_x  # Flip gaze x coordinate
                
                pred_depth_flip, log_sigma_flip, pred_gaze_depth_flip = model(
                    context_rgb_flip, patch_rgb_flip, gaze_x_flip, gaze_y
                )
                
                # Average predictions (flip back the flipped prediction)
                pred_depth = (pred_depth + torch.flip(pred_depth_flip, dims=[-1])) / 2
                log_sigma = (log_sigma + torch.flip(log_sigma_flip, dims=[-1])) / 2
                pred_gaze_depth = (pred_gaze_depth + pred_gaze_depth_flip) / 2
            else:
                # Standard forward pass
                pred_depth, log_sigma, pred_gaze_depth = model(context_rgb, patch_rgb, gaze_x, gaze_y)
            
            # Get gaze depth GT from batch
            gaze_depth_gt = batch['gaze_depth_gt'].cuda(non_blocking=True)
            
            # Compute losses
            si_log_loss = si_log_loss_fn(pred_depth.squeeze(1), depth_gt, valid_mask)
            berhu_loss = berhu_loss_fn(pred_depth.squeeze(1), depth_gt, valid_mask)
            smooth_loss = smooth_loss_fn(pred_depth, patch_rgb)
            grad_consistency = grad_consistency_loss_fn(pred_depth, depth_gt, patch_rgb, valid_mask)
            
            # Heteroscedastic uncertainty loss with log_sigma
            log_sigma = log_sigma.squeeze(1).clamp(-8, 8)
            residual = pred_depth.squeeze(1) - depth_gt
            heteroscedastic_loss = 0.5 * torch.exp(-2 * log_sigma) * (residual**2) + log_sigma
            heteroscedastic_loss = heteroscedastic_loss[valid_mask > 0].mean()
            
            # NEW: Gaussian-weighted center loss (Fix #4)
            if hasattr(model, 'module'):  # DDP wrapper
                gaze_weights = model.module.gaze_weights.to(pred_depth.device)
            else:
                gaze_weights = model.gaze_weights.to(pred_depth.device)
            
            # Apply weights with valid mask
            w = gaze_weights.unsqueeze(0) * valid_mask  # [B, 22, 22]
            w = w / (w.sum(dim=(-1,-2), keepdim=True) + 1e-6)  # Renormalize
            center_loss = ((pred_depth.squeeze(1) - depth_gt)**2 * w).sum(dim=(-1,-2)).mean()
            
            # NEW: Scalar gaze depth loss (Fix #1)
            # Only compute loss for valid gaze points
            gaze_valid = (gaze_depth_gt > 0).float()
            if gaze_valid.sum() > 0:
                # Squeeze pred_gaze_depth to match gaze_depth_gt shape
                gaze_loss = F.l1_loss(pred_gaze_depth.squeeze(-1)[gaze_valid > 0], 
                                     gaze_depth_gt[gaze_valid > 0], reduction='mean')
            else:
                gaze_loss = torch.tensor(0.0, device=pred_depth.device)
            
            # Total loss with gaze as primary objective (point prediction focus)
            loss = 1.0 * si_log_loss + 0.05 * berhu_loss + 0.01 * smooth_loss + 0.05 * heteroscedastic_loss + \
                   0.1 * center_loss + 2.0 * gaze_loss + 0.05 * grad_consistency
            
            # Compute metrics
            metrics = compute_metrics(pred_depth.squeeze(1), depth_gt, valid_mask)
            gaze_metrics = compute_gaze_metrics(pred_gaze_depth.squeeze(-1), gaze_depth_gt,
                                                 (gaze_depth_gt > 0).float())
            
            # Accumulate
            total_loss += loss.item() * context_rgb.size(0)
            total_samples += context_rgb.size(0)
            
            for k, v in metrics.items():
                if k not in metrics_sum:
                    metrics_sum[k] = 0
                metrics_sum[k] += v * context_rgb.size(0)
                
            for k, v in gaze_metrics.items():
                if k not in metrics_sum:
                    metrics_sum[k] = 0
                metrics_sum[k] += v * context_rgb.size(0)
            
            # Update progress bar
            if rank == 0:
                avg_loss = total_loss / total_samples if total_samples > 0 else 0
                avg_abs_rel = metrics_sum.get('abs_rel', 0) / total_samples if total_samples > 0 else 0
                avg_a1 = metrics_sum.get('a1', 0) / total_samples if total_samples > 0 else 0
                avg_gaze_rel = metrics_sum.get('gaze_rel', 0) / total_samples if total_samples > 0 else 0
                avg_gaze_a1 = metrics_sum.get('gaze_a1', 0) / total_samples if total_samples > 0 else 0
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'AbsRel': f'{avg_abs_rel:.3f}',
                    'a1': f'{avg_a1:.3f}',
                    'GazeRel': f'{avg_gaze_rel:.3f}',
                    'Gaze_a1': f'{avg_gaze_a1:.3f}'
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
            if 'gaze' in k:
                writer.add_scalar(f'val/gaze/{k}', v, epoch)
            else:
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
    parser.add_argument('--scheduler', type=str, default='cosine_restarts',
                       choices=['cosine_restarts', 'plateau', 'onecycle'],
                       help='Learning rate scheduler type')
    parser.add_argument('--use-swa', action='store_true',
                       help='Use Stochastic Weight Averaging for last 20 epochs')
    parser.add_argument('--swa-start', type=int, default=80,
                       help='Epoch to start SWA')
    parser.add_argument('--swa-lr', type=float, default=5e-5,
                       help='SWA learning rate')
    parser.add_argument('--use-webdataset', action='store_true',
                       help='Use WebDataset for faster loading')
    parser.add_argument('--shard-dir', type=str, 
                       default=os.path.expanduser('~/adt_webdataset_shards'),
                       help='Directory containing WebDataset shards')
    parser.add_argument('--create-shards', action='store_true',
                       help='Create WebDataset shards if they don\'t exist')
    
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
        model = DDP(model, device_ids=[gpu])
        
    # Check if we should use WebDataset
    use_webdataset = args.use_webdataset and HAS_WEBDATASET
    
    if use_webdataset:
        # Create shards if requested
        if args.create_shards and rank == 0:
            if not os.path.exists(args.shard_dir):
                print(f"Creating WebDataset shards at {args.shard_dir}...")
                os.makedirs(args.shard_dir, exist_ok=True)
                create_webdataset_shards(args.data_root, args.shard_dir, 'train', 
                                       max_sequences=args.max_train_sequences if args.max_train_sequences < 999 else None)
                create_webdataset_shards(args.data_root, args.shard_dir, 'val',
                                       max_sequences=args.max_val_sequences if args.max_val_sequences < 999 else None)
        
        # Wait for rank 0 to finish creating shards
        if distributed:
            dist.barrier()
        
        # Create WebDataset loaders
        train_loader = make_webdataset_loader(
            args.shard_dir, 'train', args.batch_size, args.num_workers,
            distributed, world_size, rank
        )
        
        val_loader = make_webdataset_loader(
            args.shard_dir, 'val', args.batch_size, args.num_workers,
            distributed, world_size, rank
        )
        
        if train_loader is None or val_loader is None:
            print("Failed to create WebDataset loaders, falling back to regular dataset")
            use_webdataset = False
    
    # Fallback to regular dataset if WebDataset not available or failed
    if not use_webdataset:
        # Create datasets
        # Note: k_extra=15 by default for 16x multi-point sampling (1 real + 15 random)
        train_dataset = SpatialDualResolutionDataset(
            data_root=args.data_root,
            split='train',
            augment=True,
            max_sequences=args.max_train_sequences,
            random_seed=args.random_seed
            # k_extra=15 is the default in the dataset class
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
    
    # Choose scheduler based on args (default to cosine warm restarts for plateau breaking)
    warmup_epochs = 5
    
    if getattr(args, 'scheduler', 'cosine_restarts') == 'plateau':
        # Option 1: ReduceLROnPlateau (monitors EMA val loss)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        # After warmup, we'll switch to ReduceLROnPlateau
        scheduler = warmup
        use_plateau_scheduler = False  # Will switch after warmup
    else:
        # Option 2: CosineAnnealingWarmRestarts (recommended for plateaus)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        # Restart every 10 epochs after warmup
        cosine_restarts = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine_restarts], milestones=[warmup_epochs]
        )
        use_plateau_scheduler = False
    
    # Create EMA model
    ema_decay = 0.999  # Can be tuned (0.999-0.9995)
    if isinstance(model, DDP):
        model_ema = deepcopy(model.module).cuda().eval()
    else:
        model_ema = deepcopy(model).cuda().eval()
    for p in model_ema.parameters():
        p.requires_grad_(False)
    
    # Setup SWA if requested
    swa_model = None
    swa_scheduler = None
    swa_n = 0
    if args.use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        if isinstance(model, DDP):
            swa_model = AveragedModel(model.module)
        else:
            swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr, anneal_epochs=5)
    
    # Create loss functions
    si_log_loss = SILogLoss(alpha=0.85)
    berhu_loss = BerHuLoss(threshold=0.2)
    smooth_loss = EdgeAwareSmoothLoss()
    grad_consistency_loss = GradientConsistencyLoss()
    loss_fns = (si_log_loss, berhu_loss, smooth_loss, grad_consistency_loss)
    
    # Create tensorboard writer
    if rank == 0:
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
        
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_gaze_a1 = 0.0  # Track best gaze_a1 (higher is better)
    best_abs_rel = float('inf')  # Track best abs_rel (lower is better)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if 'model_ema' in checkpoint:
            model_ema.load_state_dict(checkpoint['model_ema'])
            if rank == 0:
                print("EMA weights re-loaded")
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load best metrics if they exist
        best_gaze_a1 = checkpoint.get('best_gaze_a1', 0.0)
        best_abs_rel = checkpoint.get('best_abs_rel', float('inf'))
        
        # If metrics don't exist in checkpoint, try to extract from saved metrics
        if 'ema_val_metrics' in checkpoint:
            ema_metrics = checkpoint['ema_val_metrics']
            if best_gaze_a1 == 0.0 and 'gaze_a1' in ema_metrics:
                best_gaze_a1 = ema_metrics['gaze_a1']
            if best_abs_rel == float('inf') and 'abs_rel' in ema_metrics:
                best_abs_rel = ema_metrics['abs_rel']
        
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")
            print(f"  Best gaze_a1: {best_gaze_a1:.3f}")
            print(f"  Best abs_rel: {best_abs_rel:.3f}")
            
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        # Check if we should switch to SWA
        if args.use_swa and epoch >= args.swa_start:
            if swa_n == 0 and rank == 0:
                print(f"\n🔄 Starting SWA at epoch {epoch}")
            # Use SWA scheduler
            current_scheduler = swa_scheduler
        else:
            current_scheduler = scheduler
            
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, current_scheduler, loss_fns,
            epoch, writer, distributed, rank, world_size,
            model_ema=model_ema, ema_decay=ema_decay
        )
        
        # Update SWA model if in SWA phase
        if args.use_swa and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_n += 1
        
        # Use TTA for validation after epoch 50
        use_tta = (epoch >= 50)
        
        # Validate regular model
        val_loss, val_metrics = validate(
            model, val_loader, loss_fns, epoch, writer,
            distributed, rank, world_size, use_tta=use_tta
        )
        
        # Validate EMA model
        ema_val_loss, ema_val_metrics = validate(
            model_ema, val_loader, loss_fns, epoch, writer,
            distributed, rank, world_size, use_tta=use_tta
        )
        
        # Also validate SWA model if available
        if swa_n > 0:
            from torch.optim.swa_utils import update_bn
            # Update batch norm statistics
            update_bn(val_loader, swa_model)
            swa_val_loss, swa_val_metrics = validate(
                swa_model, val_loader, loss_fns, epoch, writer,
                distributed, rank, world_size, use_tta=use_tta
            )
            if rank == 0:
                print(f"  SWA Val Loss: {swa_val_loss:.4f}, AbsRel: {swa_val_metrics['abs_rel']:.3f}, "
                      f"RMSE: {swa_val_metrics['rmse']:.3f}, a1: {swa_val_metrics['a1']:.3f}")
        
        # Print epoch summary
        if rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Patch: Loss={train_loss:.4f}, AbsRel={train_metrics['abs_rel']:.3f}, "
                  f"a1={train_metrics['a1']:.3f}")
            print(f"        - Gaze:  MAE={train_metrics.get('gaze_mae', 0):.3f}m, "
                  f"RelErr={train_metrics.get('gaze_rel', 0):.3f}, "
                  f"a1={train_metrics.get('gaze_a1', 0):.3f}")
            print(f"  Val   - Patch: Loss={val_loss:.4f}, AbsRel={val_metrics['abs_rel']:.3f}, "
                  f"a1={val_metrics['a1']:.3f}")
            print(f"        - Gaze:  MAE={val_metrics.get('gaze_mae', 0):.3f}m, "
                  f"RelErr={val_metrics.get('gaze_rel', 0):.3f}, "
                  f"a1={val_metrics.get('gaze_a1', 0):.3f}")
            print(f"  EMA   - Patch: Loss={ema_val_loss:.4f}, AbsRel={ema_val_metrics['abs_rel']:.3f}, "
                  f"a1={ema_val_metrics['a1']:.3f}")
            print(f"        - Gaze:  MAE={ema_val_metrics.get('gaze_mae', 0):.3f}m, "
                  f"RelErr={ema_val_metrics.get('gaze_rel', 0):.3f}, "
                  f"a1={ema_val_metrics.get('gaze_a1', 0):.3f}")
                  
        # Save checkpoint
        if rank == 0:
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
                'best_gaze_a1': best_gaze_a1,
                'best_abs_rel': best_abs_rel,
                'args': args
            }
            
            # Get current EMA gaze_a1 and abs_rel
            current_gaze_a1 = ema_val_metrics.get('gaze_a1', 0.0)
            current_abs_rel = ema_val_metrics.get('abs_rel', float('inf'))
            
            # Save best model based on gaze_a1 (PRIMARY METRIC)
            if current_gaze_a1 > best_gaze_a1:
                best_gaze_a1 = current_gaze_a1
                checkpoint['best_gaze_a1'] = best_gaze_a1
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint_best.pth'))
                print(f"  🎯 New best model saved! (gaze_a1: {current_gaze_a1:.3f})")
                
            # Also save if abs_rel improved significantly (SECONDARY METRIC)
            if current_abs_rel < best_abs_rel:
                best_abs_rel = current_abs_rel
                checkpoint['best_abs_rel'] = best_abs_rel
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint_best_abs_rel.pth'))
                print(f"  📊 New best abs_rel saved! (abs_rel: {current_abs_rel:.3f})")
                
            # Keep track of old best_val_loss for compatibility
            if ema_val_loss < best_val_loss:
                best_val_loss = ema_val_loss
                checkpoint['best_val_loss'] = best_val_loss
                
            # Always save latest checkpoint every epoch
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth'))
            
    # Clean up
    if distributed:
        dist.destroy_process_group()
        
    if rank == 0 and writer is not None:
        writer.close()
        

if __name__ == '__main__':
    main()