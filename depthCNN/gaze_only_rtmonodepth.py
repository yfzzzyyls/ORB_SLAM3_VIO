#!/usr/bin/env python3
"""
Gaze-Only RT-MonoDepth: Predicts depth only at gaze location.
Uses RT-MonoDepth encoder with custom MLP decoder for single-point prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import RT-MonoDepth components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "RT-MonoDepth"))

from networks.RTMonoDepth.RTMonoDepth_s import DepthEncoder


class GazeFeatureExtractor(nn.Module):
    """Extracts features at gaze location from multiple encoder scales."""
    
    def __init__(self, num_ch_enc: List[int], output_dim: int = 64, use_layer_norm: bool = True):
        """
        Args:
            num_ch_enc: Number of channels from each encoder scale
            output_dim: Dimension to project each scale to (default: 64)
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        # For RT-MonoDepth-S: num_ch_enc = [24, 48, 96, 192]
        self.num_scales = len(num_ch_enc)
        self.output_dim = output_dim
        
        # Projection layers to common dimension
        self.projections = nn.ModuleList([
            nn.Linear(ch, output_dim) for ch in num_ch_enc
        ])
        
        # Layer normalization for each scale
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(output_dim) for _ in range(self.num_scales)
            ])
        else:
            self.layer_norms = nn.ModuleList([
                nn.Identity() for _ in range(self.num_scales)
            ])
    
    def extract_at_gaze_bilinear(self, feature_map: torch.Tensor, 
                                 gaze_x: torch.Tensor, gaze_y: torch.Tensor) -> torch.Tensor:
        """
        Extract features at gaze location using bilinear interpolation.
        
        Args:
            feature_map: [B, C, H, W] feature map
            gaze_x: [B] normalized gaze x coordinates in [0, W-1]
            gaze_y: [B] normalized gaze y coordinates in [0, H-1]
            
        Returns:
            [B, C] features at gaze location
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        # Normalize gaze coordinates to [-1, 1] for grid_sample
        gaze_norm_x = 2.0 * gaze_x / (W - 1) - 1.0
        gaze_norm_y = 2.0 * gaze_y / (H - 1) - 1.0
        
        # Create sampling grid for single point per batch
        grid = torch.stack([gaze_norm_x, gaze_norm_y], dim=-1)  # [B, 2]
        grid = grid.view(B, 1, 1, 2)  # [B, 1, 1, 2]
        
        # Sample with bilinear interpolation
        sampled = F.grid_sample(feature_map, grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)
        
        return sampled.squeeze(2).squeeze(2)  # [B, C]
    
    def forward(self, features: List[torch.Tensor], 
                gaze_x: torch.Tensor, gaze_y: torch.Tensor) -> torch.Tensor:
        """
        Extract and process features at gaze location from all scales.
        
        Args:
            features: List of feature maps from encoder, each [B, C, H, W]
            gaze_x: [B] gaze x coordinates in input resolution (e.g., 88)
            gaze_y: [B] gaze y coordinates in input resolution (e.g., 88)
            
        Returns:
            [B, num_scales * output_dim] concatenated features
        """
        processed_features = []
        
        # Process each scale
        for i, (feat, proj, ln) in enumerate(zip(features, self.projections, self.layer_norms)):
            B, C, H, W = feat.shape
            
            # Scale gaze coordinates to match feature map resolution
            # For 88x88 input: encoder downsamples by 2^(i+1)
            # Scale 0: 88/2 = 44, Scale 1: 88/4 = 22, Scale 2: 88/8 = 11, Scale 3: 88/16 = 5.5
            downsample_factor = 2 ** (i + 1)
            scaled_gaze_x = gaze_x / downsample_factor
            scaled_gaze_y = gaze_y / downsample_factor
            
            # Extract features at gaze location
            gaze_features = self.extract_at_gaze_bilinear(feat, scaled_gaze_x, scaled_gaze_y)
            
            # Project to common dimension
            projected = proj(gaze_features)
            
            # Apply layer norm and activation
            normalized = ln(projected)
            activated = F.relu(normalized)
            
            processed_features.append(activated)
        
        # Concatenate all scales
        return torch.cat(processed_features, dim=1)  # [B, num_scales * output_dim]


class ObjectAwareDepthMLP(nn.Module):
    """Two-stage MLP decoder for depth prediction from gaze features."""
    
    def __init__(self, input_dim: int = 256, hidden_dims: List[int] = [128, 64, 32, 16],
                 max_depth: float = 10.0, min_depth: float = 0.1):
        """
        Args:
            input_dim: Input feature dimension (4 scales * 64 = 256)
            hidden_dims: Hidden layer dimensions [stage1_hidden, stage1_out, stage2_hidden, stage2_out]
            max_depth: Maximum depth value
            min_depth: Minimum depth value
        """
        super().__init__()
        
        self.max_depth = max_depth
        self.min_depth = min_depth
        
        # Stage 1: Object understanding (256 → 128 → 64)
        self.object_understanding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(inplace=True),
        )
        
        # Stage 2: Depth prediction (64 → 32 → 16 → 1)
        self.depth_prediction = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.LayerNorm(hidden_dims[3]),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dims[3], 1)  # Single depth value
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization with special handling for output layer."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Special initialization for depth output
        # Initialize to predict median depth (~2m)
        with torch.no_grad():
            output_layer = self.depth_prediction[-1]
            # Small weights, bias to predict log(2.0/max_depth)
            nn.init.normal_(output_layer.weight, mean=0, std=0.01)
            median_depth = 2.0
            init_bias = np.log(median_depth / self.max_depth)
            nn.init.constant_(output_layer.bias, init_bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict depth from concatenated gaze features.
        
        Args:
            features: [B, 256] concatenated multi-scale features
            
        Returns:
            [B, 1] depth predictions
        """
        # Stage 1: Understand what object is at gaze
        object_features = self.object_understanding(features)
        
        # Stage 2: Predict depth based on object understanding
        depth_logits = self.depth_prediction(object_features)
        
        # Convert to depth using sigmoid and scaling
        depth = torch.sigmoid(depth_logits) * self.max_depth
        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
        
        return depth


class GazeOnlyRTMonoDepth(nn.Module):
    """
    Complete gaze-only depth prediction model.
    Uses RT-MonoDepth encoder with custom MLP decoder.
    """
    
    def __init__(self, max_depth: float = 10.0, min_depth: float = 0.1,
                 use_multi_scale_supervision: bool = True):
        """
        Args:
            max_depth: Maximum depth value in meters
            min_depth: Minimum depth value in meters
            use_multi_scale_supervision: Whether to use auxiliary losses
        """
        super().__init__()
        
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_multi_scale_supervision = use_multi_scale_supervision
        
        # RT-MonoDepth encoder
        self.encoder = DepthEncoder()
        
        # Get number of channels from each encoder scale
        # For RT-MonoDepth-S: [24, 48, 96, 192]
        num_ch_enc = self.encoder.num_ch_enc
        
        # Feature extractor at gaze location
        self.feature_extractor = GazeFeatureExtractor(
            num_ch_enc=num_ch_enc,
            output_dim=64,
            use_layer_norm=True
        )
        
        # Main depth decoder
        self.depth_decoder = ObjectAwareDepthMLP(
            input_dim=len(num_ch_enc) * 64,  # 4 * 64 = 256
            hidden_dims=[128, 64, 32, 16],
            max_depth=max_depth,
            min_depth=min_depth
        )
        
        # Auxiliary predictors for multi-scale supervision
        if use_multi_scale_supervision:
            self.aux_predictors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(64, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 1)
                ) for _ in range(2)  # Auxiliary from scale 2 and 3
            ])
    
    def forward(self, rgb: torch.Tensor, gaze_x: torch.Tensor, gaze_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for gaze-only depth prediction.
        
        Args:
            rgb: [B, 3, H, W] input RGB image (e.g., 88x88)
            gaze_x: [B] gaze x coordinates in pixels
            gaze_y: [B] gaze y coordinates in pixels
            
        Returns:
            Dictionary containing:
                - 'depth': [B, 1] main depth prediction
                - 'aux_depths': List of [B, 1] auxiliary predictions (if enabled)
        """
        # Encode image to multi-scale features
        features = self.encoder(rgb)
        
        # Extract features at gaze location
        gaze_features = self.feature_extractor(features, gaze_x, gaze_y)
        
        # Main depth prediction
        depth = self.depth_decoder(gaze_features)
        
        outputs = {'depth': depth}
        
        # Auxiliary predictions if enabled
        if self.use_multi_scale_supervision and self.training:
            aux_depths = []
            
            # Extract features from individual scales for auxiliary predictions
            for i, aux_pred in enumerate(self.aux_predictors):
                scale_idx = i + 2  # Use scale 2 and 3
                
                # Extract features at single scale
                B, C, H, W = features[scale_idx].shape
                # Scale gaze coordinates properly
                downsample_factor = 2 ** (scale_idx + 1)
                scaled_gaze_x = gaze_x / downsample_factor
                scaled_gaze_y = gaze_y / downsample_factor
                
                single_scale_features = self.feature_extractor.extract_at_gaze_bilinear(
                    features[scale_idx], scaled_gaze_x, scaled_gaze_y
                )
                
                # Project and predict
                projected = self.feature_extractor.projections[scale_idx](single_scale_features)
                aux_depth = torch.sigmoid(aux_pred(projected)) * self.max_depth
                aux_depth = torch.clamp(aux_depth, min=self.min_depth, max=self.max_depth)
                
                aux_depths.append(aux_depth)
            
            outputs['aux_depths'] = aux_depths
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GazeDepthLoss(nn.Module):
    """
    Gaze-specific loss function for depth prediction.
    Combines scale-invariant log loss with gradient and relative error terms.
    """
    
    def __init__(self, alpha: float = 0.85, grad_weight: float = 0.1, 
                 rel_weight: float = 0.1):
        """
        Args:
            alpha: Weight for scale-invariant term
            grad_weight: Weight for gradient loss
            rel_weight: Weight for relative error loss
        """
        super().__init__()
        self.alpha = alpha
        self.grad_weight = grad_weight
        self.rel_weight = rel_weight
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, 
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gaze-specific depth loss.
        
        Args:
            pred: [B, 1] predicted depth
            gt: [B, 1] ground truth depth
            valid_mask: [B, 1] optional mask for valid depths
            
        Returns:
            Scalar loss value
        """
        # Ensure positive values
        pred = torch.clamp(pred, min=1e-8)
        gt = torch.clamp(gt, min=1e-8)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            pred = pred[valid_mask]
            gt = gt[valid_mask]
            
            if pred.numel() == 0:
                return torch.tensor(0.0, device=pred.device)
        
        # 1. Scale-invariant log loss
        log_diff = torch.log(pred) - torch.log(gt)
        si_loss = torch.sqrt(
            torch.mean(log_diff ** 2) - 
            self.alpha * torch.mean(log_diff) ** 2
        )
        
        # 2. Gradient loss (penalizes large errors)
        grad_loss = torch.mean(torch.abs(log_diff))
        
        # 3. Relative error loss
        rel_loss = torch.mean(torch.abs(pred - gt) / gt)
        
        # Combine losses
        total_loss = si_loss + self.grad_weight * grad_loss + self.rel_weight * rel_loss
        
        return total_loss


if __name__ == "__main__":
    # Test the model
    print("Testing GazeOnlyRTMonoDepth...")
    
    # Create model
    model = GazeOnlyRTMonoDepth(max_depth=10.0, min_depth=0.1)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 88, 88)
    gaze_x = torch.randint(0, 88, (batch_size,)).float()
    gaze_y = torch.randint(0, 88, (batch_size,)).float()
    
    # Forward pass
    outputs = model(rgb, gaze_x, gaze_y)
    print(f"Output depth shape: {outputs['depth'].shape}")
    print(f"Depth range: [{outputs['depth'].min():.3f}, {outputs['depth'].max():.3f}]")
    
    # Test loss
    loss_fn = GazeDepthLoss()
    gt_depth = torch.rand(batch_size, 1) * 5 + 0.5  # Random depth 0.5-5.5m
    loss = loss_fn(outputs['depth'], gt_depth)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nModel test passed!")