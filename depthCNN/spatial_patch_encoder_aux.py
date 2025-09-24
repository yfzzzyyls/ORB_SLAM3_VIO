#!/usr/bin/env python3
"""
Spatial patch prediction architecture with auxiliary losses.
Enhanced version that adds intermediate supervision from encoder layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math

from spatial_patch_encoder import SpatialPatchExtractor, SpatialCNNDecoder


class AuxiliarySpatialDecoder(nn.Module):
    """Lightweight decoder for auxiliary supervision at intermediate scales."""
    
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        output_size: int = 16
    ):
        """
        Args:
            input_channels: Number of input feature channels
            input_size: Size of input spatial region
            output_size: Size of output depth patch
        """
        super().__init__()
        
        # Simple decoder with fewer layers
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(size=output_size, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode spatial features to depth patch."""
        return self.decoder(x)


class SpatialPatchDepthPredictorWithAux(nn.Module):
    """
    Spatial patch depth prediction model with auxiliary losses.
    Adds intermediate supervision from multiple encoder levels.
    """
    
    def __init__(
        self,
        image_size: int = 88,
        num_encoder_levels: int = 3,
        base_channels: int = 32,
        spatial_region_size: int = 5,
        patch_size: int = 16,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        use_auxiliary_losses: bool = True,
        auxiliary_weights: Optional[List[float]] = None
    ):
        """
        Args:
            image_size: Input image size
            num_encoder_levels: Number of encoder levels (3-4 recommended)
            base_channels: Base channels in encoder
            spatial_region_size: Size of spatial feature region to extract
            patch_size: Size of output depth patch
            max_depth: Maximum depth value
            min_depth: Minimum depth value
            use_auxiliary_losses: Whether to use auxiliary losses
            auxiliary_weights: Weights for auxiliary losses (default: [0.3, 0.2])
        """
        super().__init__()
        
        self.image_size = image_size
        self.spatial_region_size = spatial_region_size
        self.patch_size = patch_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_auxiliary_losses = use_auxiliary_losses
        self.auxiliary_weights = auxiliary_weights or [0.3, 0.2]  # For level -2 and -3
        
        # Import the encoder
        from flexible_gaze_encoder import FlexibleGazeEncoder
        
        # Encoder - processes full image
        self.encoder = FlexibleGazeEncoder(
            num_levels=num_encoder_levels,
            base_channels=base_channels,
            image_size=image_size
        )
        
        # Spatial region extractors for each level
        self.spatial_extractors = nn.ModuleList([
            SpatialPatchExtractor(region_size=spatial_region_size)
            for _ in range(num_encoder_levels)
        ])
        
        # Main decoder for deepest features
        deepest_channels = self.encoder.num_ch_enc[-1]
        self.main_decoder = SpatialCNNDecoder(
            input_channels=deepest_channels,
            input_size=spatial_region_size,
            output_size=patch_size,
            hidden_channels=128
        )
        
        # Auxiliary decoders for intermediate supervision
        if use_auxiliary_losses and num_encoder_levels > 1:
            self.aux_decoders = nn.ModuleList()
            
            # Add auxiliary decoders for second-to-last and third-to-last levels
            for level_idx in range(-2, max(-num_encoder_levels-1, -4), -1):
                if abs(level_idx) <= num_encoder_levels:
                    aux_channels = self.encoder.num_ch_enc[level_idx]
                    self.aux_decoders.append(
                        AuxiliarySpatialDecoder(
                            input_channels=aux_channels,
                            input_size=spatial_region_size,
                            output_size=patch_size
                        )
                    )
        else:
            self.aux_decoders = None
        
        # Multi-scale fusion for main decoder
        self.use_multiscale = num_encoder_levels > 2
        if self.use_multiscale:
            # Combine features from last two levels
            fusion_channels = deepest_channels + self.encoder.num_ch_enc[-2]
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(fusion_channels, deepest_channels, 1),
                nn.BatchNorm2d(deepest_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(
        self,
        rgb: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with auxiliary outputs.
        
        Args:
            rgb: [B, 3, H, W] input image
            gaze_x: [B] gaze x coordinates in image space
            gaze_y: [B] gaze y coordinates in image space
            
        Returns:
            Dictionary with:
                - 'depth': Main depth prediction [B, patch_size, patch_size]
                - 'aux_depths': List of auxiliary predictions (if training)
        """
        # Encode full image
        features = self.encoder(rgb)
        
        # Extract spatial regions at multiple scales
        spatial_regions = []
        for level_idx, (feat, extractor) in enumerate(zip(features, self.spatial_extractors)):
            scale_factor = 2 ** (level_idx + 1)
            gaze_x_scaled = gaze_x / scale_factor
            gaze_y_scaled = gaze_y / scale_factor
            
            region = extractor(feat, gaze_x_scaled, gaze_y_scaled)
            spatial_regions.append(region)
        
        # Main prediction using deepest features
        deepest_region = spatial_regions[-1]
        
        # Multi-scale fusion if enabled
        if self.use_multiscale and len(spatial_regions) > 1:
            # Get second deepest region and resize to match
            second_region = spatial_regions[-2]
            second_region = F.interpolate(
                second_region,
                size=(self.spatial_region_size, self.spatial_region_size),
                mode='bilinear',
                align_corners=True
            )
            
            # Fuse features
            fused_region = torch.cat([deepest_region, second_region], dim=1)
            deepest_region = self.feature_fusion(fused_region)
        
        # Main depth prediction
        main_depth = self.main_decoder(deepest_region)
        main_depth = main_depth.squeeze(1) * self.max_depth
        main_depth = torch.clamp(main_depth, min=self.min_depth, max=self.max_depth)
        
        outputs = {'depth': main_depth}
        
        # Auxiliary predictions (only during training)
        if self.training and self.use_auxiliary_losses and self.aux_decoders is not None:
            aux_depths = []
            
            # Process intermediate levels
            aux_level_indices = [-2, -3] if len(features) >= 3 else [-2]
            for aux_idx, level_idx in enumerate(aux_level_indices):
                if abs(level_idx) <= len(spatial_regions) and aux_idx < len(self.aux_decoders):
                    aux_region = spatial_regions[level_idx]
                    
                    # Resize to consistent size if needed
                    if aux_region.shape[-1] != self.spatial_region_size:
                        aux_region = F.interpolate(
                            aux_region,
                            size=(self.spatial_region_size, self.spatial_region_size),
                            mode='bilinear',
                            align_corners=True
                        )
                    
                    # Auxiliary prediction
                    aux_depth = self.aux_decoders[aux_idx](aux_region)
                    aux_depth = aux_depth.squeeze(1) * self.max_depth
                    aux_depth = torch.clamp(aux_depth, min=self.min_depth, max=self.max_depth)
                    aux_depths.append(aux_depth)
            
            if aux_depths:
                outputs['aux_depths'] = aux_depths
                # Don't include aux_weights in outputs during training with DataParallel
                # The loss function can access them directly from the model
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialPatchLossWithAux(nn.Module):
    """Loss function that combines main and auxiliary losses."""
    
    def __init__(self, alpha=0.85, smooth_weight=0.1, auxiliary_weights=None):
        super().__init__()
        self.alpha = alpha
        self.smooth_weight = smooth_weight
        self.auxiliary_weights = auxiliary_weights or [0.3, 0.2]
    
    def compute_si_log_loss(self, pred, target, valid_mask):
        """Compute scale-invariant logarithmic loss."""
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]
        
        if valid_pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # SI-log loss
        d = torch.log(valid_pred + 1e-6) - torch.log(valid_target + 1e-6)
        loss = torch.sqrt(torch.mean(d ** 2) - self.alpha * torch.mean(d) ** 2)
        
        return loss
    
    def forward(self, outputs, gt_patch, valid_mask):
        """
        Compute total loss including auxiliary losses.
        
        Args:
            outputs: Model outputs dictionary with 'depth' and optionally 'aux_depths'
            gt_patch: Ground truth depth patch
            valid_mask: Valid pixel mask
        """
        # Main loss
        main_depth = outputs['depth']
        main_loss = self.compute_si_log_loss(main_depth, gt_patch, valid_mask)
        
        # Add smoothness
        if self.smooth_weight > 0:
            dx = main_depth[:, :, 1:] - main_depth[:, :, :-1]
            dy = main_depth[:, 1:, :] - main_depth[:, :-1, :]
            smooth_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
            main_loss = main_loss + self.smooth_weight * smooth_loss
        
        total_loss = main_loss
        loss_dict = {'main_loss': main_loss.item()}
        
        # Auxiliary losses
        if 'aux_depths' in outputs:
            aux_depths = outputs['aux_depths']
            
            for i, aux_depth in enumerate(aux_depths):
                if i < len(self.auxiliary_weights):
                    weight = self.auxiliary_weights[i]
                    aux_loss = self.compute_si_log_loss(aux_depth, gt_patch, valid_mask)
                    total_loss = total_loss + weight * aux_loss
                    loss_dict[f'aux_loss_{i}'] = aux_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the model with auxiliary losses
    print("Testing Spatial Patch Predictor with Auxiliary Losses...")
    
    # Create model
    model = SpatialPatchDepthPredictorWithAux(
        image_size=88,
        num_encoder_levels=3,
        base_channels=32,
        spatial_region_size=5,
        patch_size=16,
        use_auxiliary_losses=True
    )
    
    print(f"Total parameters: {model.get_num_params():,}")
    
    # Count parameters by component
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    main_decoder_params = sum(p.numel() for p in model.main_decoder.parameters())
    aux_params = 0
    if model.aux_decoders:
        aux_params = sum(p.numel() for p in model.aux_decoders.parameters())
    
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Main decoder: {main_decoder_params:,}")
    print(f"  Auxiliary decoders: {aux_params:,}")
    
    # Test forward pass
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 88, 88)
    gaze_x = torch.randint(20, 68, (batch_size,)).float()
    gaze_y = torch.randint(20, 68, (batch_size,)).float()
    
    # Training mode - should get auxiliary outputs
    model.train()
    outputs = model(rgb, gaze_x, gaze_y)
    
    print("\nTraining mode outputs:")
    for key, value in outputs.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
            for i, v in enumerate(value):
                if isinstance(v, torch.Tensor):
                    print(f"    [{i}]: shape {v.shape}")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Eval mode - should only get main output
    model.eval()
    outputs = model(rgb, gaze_x, gaze_y)
    
    print("\nEval mode outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")