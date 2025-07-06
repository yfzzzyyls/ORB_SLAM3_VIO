#!/usr/bin/env python3
"""
RT-MonoDepth-S model wrapper for training on ADT dataset.
Wraps the official RT-MonoDepth-S implementation from the cloned repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import sys
from pathlib import Path

# Add RT-MonoDepth to path
sys.path.insert(0, str(Path(__file__).parent / "RT-MonoDepth"))

from networks.RTMonoDepth.RTMonoDepth_s import DepthEncoder, DepthDecoder
from layers import disp_to_depth


class RTMonoDepthS(nn.Module):
    """
    Wrapper for official RT-MonoDepth-S model to work with ADT dataset.
    Combines DepthEncoder and DepthDecoder into a single model with
    interface expected by train.py and evaluate.py.
    """
    
    def __init__(self, max_depth=10.0, min_depth=0.1, scales=[0]):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.scales = scales
        
        # Create encoder and decoder from official RT-MonoDepth
        self.encoder = DepthEncoder()
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=scales)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass matching the interface expected by train.py
        
        Args:
            x: Input RGB image [B, 3, H, W]
            
        Returns:
            Dictionary containing:
                - 'depth': Predicted depth map [B, 1, H, W]
                - 'disp': Raw disparity outputs
                - 'features': Encoder features
        """
        # Encoder forward pass
        features = self.encoder(x)
        
        # Decoder forward pass
        outputs = self.decoder(features)
        
        # Get full resolution disparity
        disp = outputs[("disp", 0)]
        
        # Convert disparity to depth
        # RT-MonoDepth outputs normalized disparity, convert to metric depth
        _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        
        return {
            'depth': depth,
            'disp': outputs,
            'features': features
        }
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic loss for depth estimation.
    Adapted for RT-MonoDepth training on ADT dataset.
    """
    
    def __init__(self, lambda_weight=0.85):
        super().__init__()
        self.lambda_weight = lambda_weight
    
    def forward(self, pred_depth: torch.Tensor, 
                target_depth: torch.Tensor, 
                valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute SI-Log loss.
        
        Args:
            pred_depth: Predicted depth [B, 1, H, W]
            target_depth: Ground truth depth [B, 1, H, W]
            valid_mask: Valid depth mask [B, 1, H, W]
            
        Returns:
            Loss value
        """
        # Apply mask
        pred = pred_depth[valid_mask]
        target = target_depth[valid_mask]
        
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred_depth.device)
        
        # Compute log difference
        log_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
        
        # SI-Log loss
        loss = torch.sqrt(
            torch.mean(log_diff ** 2) - 
            self.lambda_weight * torch.mean(log_diff) ** 2
        )
        
        return loss


class RTMonoDepthLoss(nn.Module):
    """
    Combined loss function for RT-MonoDepth training on ADT dataset.
    Includes SI-Log loss and optional disparity smoothness.
    """
    
    def __init__(self, si_weight=0.85, smooth_weight=1e-3, scales=[0]):
        super().__init__()
        self.si_weight = si_weight
        self.smooth_weight = smooth_weight
        self.scales = scales
        self.silog = SILogLoss(lambda_weight=si_weight)
        
    def forward(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Model outputs including 'depth' and 'disp'
            targets: Dictionary with 'depth', 'valid_mask', and optionally 'rgb'
            
        Returns:
            Total loss value
        """
        pred_depth = predictions['depth']
        target_depth = targets['depth']
        valid_mask = targets['valid_mask']
        
        # SI-Log loss
        total_loss = self.silog(pred_depth, target_depth, valid_mask)
        
        # Optional: Add disparity smoothness loss
        if self.smooth_weight > 0 and 'disp' in predictions and 'rgb' in targets:
            smooth_loss = 0
            for scale in self.scales:
                disp = predictions['disp'][("disp", scale)]
                
                # Get color image at the same scale
                color = targets['rgb']
                if scale > 0:
                    h, w = disp.shape[2:]
                    color = F.interpolate(color, size=(h, w), mode='bilinear', align_corners=False)
                
                # Compute smoothness
                smooth_loss += self.compute_smooth_loss(disp, color) / (2 ** scale)
            
            total_loss += self.smooth_weight * smooth_loss
        
        return total_loss
    
    def compute_smooth_loss(self, disp, img):
        """Edge-aware smoothness loss."""
        # Normalize disparity
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        
        # Compute gradients
        grad_disp_x = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
        grad_disp_y = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])
        
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        
        # Edge-aware weighting
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        
        return grad_disp_x.mean() + grad_disp_y.mean()


class DepthMetrics:
    """Compute standard depth estimation metrics."""
    
    @staticmethod
    def compute_metrics(pred_depth: torch.Tensor,
                       target_depth: torch.Tensor,
                       valid_mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute depth metrics.
        
        Args:
            pred_depth: Predicted depth [B, 1, H, W]
            target_depth: Ground truth depth [B, 1, H, W]
            valid_mask: Valid depth mask [B, 1, H, W]
            
        Returns:
            Dictionary of metrics
        """
        # Apply mask and flatten
        pred = pred_depth[valid_mask].detach().cpu()
        target = target_depth[valid_mask].detach().cpu()
        
        if pred.numel() == 0:
            return {
                'abs_rel': 0.0,
                'sq_rel': 0.0,
                'rmse': 0.0,
                'rmse_log': 0.0,
                'a1': 0.0,
                'a2': 0.0,
                'a3': 0.0
            }
        
        # Absolute relative error
        abs_rel = torch.mean(torch.abs(pred - target) / target)
        
        # Squared relative error
        sq_rel = torch.mean(((pred - target) ** 2) / target)
        
        # RMSE
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        
        # RMSE log
        rmse_log = torch.sqrt(torch.mean(
            (torch.log(pred + 1e-8) - torch.log(target + 1e-8)) ** 2
        ))
        
        # Threshold accuracy
        thresh = torch.max(pred / target, target / pred)
        a1 = torch.mean((thresh < 1.25).float())
        a2 = torch.mean((thresh < 1.25 ** 2).float())
        a3 = torch.mean((thresh < 1.25 ** 3).float())
        
        return {
            'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(),
            'rmse': rmse.item(),
            'rmse_log': rmse_log.item(),
            'a1': a1.item() * 100,  # Convert to percentage
            'a2': a2.item() * 100,
            'a3': a3.item() * 100
        }


if __name__ == "__main__":
    # Test the wrapper
    print("Testing RT-MonoDepth-S wrapper...")
    
    # Create model
    model = RTMonoDepthS(max_depth=10.0)
    
    # Print model info
    num_params = model.get_num_params()
    print(f"\nModel parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 2
    height, width = 352, 352  # Test with smaller size
    x = torch.randn(batch_size, 3, height, width)
    
    # Forward pass
    outputs = model(x)
    depth = outputs['depth']
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min().item():.3f}, {depth.max().item():.3f}]")
    
    # Test loss
    target_depth = torch.rand_like(depth) * 10.0
    valid_mask = torch.ones_like(depth).bool()
    
    loss_fn = SILogLoss()
    loss = loss_fn(depth, target_depth, valid_mask)
    print(f"\nSI-Log loss: {loss.item():.4f}")
    
    # Test metrics
    metrics = DepthMetrics.compute_metrics(depth, target_depth, valid_mask)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")