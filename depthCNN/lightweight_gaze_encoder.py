#!/usr/bin/env python3
"""
Lightweight encoder specifically designed for 88x88 gaze-only depth prediction.
Much more efficient than using the full RT-MonoDepth encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class LightweightGazeEncoder(nn.Module):
    """
    Efficient encoder for 88x88 input with 2-3 encoding levels.
    Designed specifically for gaze-based depth prediction.
    """
    
    def __init__(self, num_levels: int = 3, base_channels: int = 32):
        """
        Args:
            num_levels: Number of encoding levels (2 or 3 recommended)
            base_channels: Base number of channels (doubled at each level)
        """
        super().__init__()
        
        assert num_levels in [2, 3], "Only 2 or 3 levels supported for 88x88 input"
        self.num_levels = num_levels
        self.base_channels = base_channels
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Level 1: 88 -> 44
        self.level1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Level 2: 44 -> 22
        self.level2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # Level 3: 22 -> 11 (optional)
        if num_levels >= 3:
            self.level3 = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
        
        # Output channel counts for each level
        if num_levels == 2:
            self.num_ch_enc = [base_channels, base_channels * 2]
        else:  # num_levels == 3
            self.num_ch_enc = [base_channels, base_channels * 2, base_channels * 4]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with Kaiming for ReLU activation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: [B, 3, 88, 88] input RGB image
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # Initial processing
        x = self.input_conv(x)  # [B, C, 88, 88]
        
        # Level 1
        x1 = self.level1(x)  # [B, C, 44, 44]
        features.append(x1)
        
        # Level 2
        x2 = self.level2(x1)  # [B, 2C, 22, 22]
        features.append(x2)
        
        # Level 3 (if enabled)
        if self.num_levels >= 3:
            x3 = self.level3(x2)  # [B, 4C, 11, 11]
            features.append(x3)
        
        return features


class LightweightGazeOnlyDepth(nn.Module):
    """
    Complete lightweight model for gaze-only depth prediction on 88x88 input.
    """
    
    def __init__(
        self,
        num_encoder_levels: int = 3,
        base_channels: int = 32,
        gaze_feature_dim: int = 64,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        use_multi_scale_supervision: bool = True
    ):
        """
        Args:
            num_encoder_levels: Number of encoding levels (2 or 3)
            base_channels: Base channels for encoder
            gaze_feature_dim: Dimension for each scale's gaze features
            max_depth: Maximum depth value
            min_depth: Minimum depth value
            use_multi_scale_supervision: Whether to use auxiliary losses
        """
        super().__init__()
        
        self.num_encoder_levels = num_encoder_levels
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_multi_scale_supervision = use_multi_scale_supervision
        
        # Lightweight encoder
        self.encoder = LightweightGazeEncoder(
            num_levels=num_encoder_levels,
            base_channels=base_channels
        )
        
        # Gaze feature extractors for each scale
        self.gaze_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, gaze_feature_dim),
                nn.LayerNorm(gaze_feature_dim),
                nn.ReLU(inplace=True)
            ) for ch in self.encoder.num_ch_enc
        ])
        
        # Main depth predictor
        total_feature_dim = num_encoder_levels * gaze_feature_dim
        self.depth_predictor = nn.Sequential(
            nn.Linear(total_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            
            nn.Linear(32, 1)
        )
        
        # Auxiliary predictors (simpler)
        if use_multi_scale_supervision:
            self.aux_predictors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(gaze_feature_dim, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 1)
                ) for _ in range(num_encoder_levels - 1)  # All scales except first
            ])
        
        # Initialize output layer for reasonable depth predictions
        self._init_depth_output()
    
    def _init_depth_output(self):
        """Initialize depth output to predict reasonable initial values."""
        with torch.no_grad():
            # Main predictor
            output_layer = self.depth_predictor[-1]
            nn.init.normal_(output_layer.weight, mean=0, std=0.01)
            # Initialize to predict ~2m depth
            init_bias = torch.log(torch.tensor(2.0 / self.max_depth))
            nn.init.constant_(output_layer.bias, init_bias.item())
            
            # Auxiliary predictors
            if hasattr(self, 'aux_predictors'):
                for aux_pred in self.aux_predictors:
                    output_layer = aux_pred[-1]
                    nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                    nn.init.constant_(output_layer.bias, init_bias.item())
    
    def extract_gaze_features(
        self,
        feature_map: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_y: torch.Tensor
    ) -> torch.Tensor:
        """Extract features at gaze location using bilinear interpolation."""
        B, C, H, W = feature_map.shape
        
        # Normalize gaze coordinates to [-1, 1] for grid_sample
        gaze_norm_x = 2.0 * gaze_x / (W - 1) - 1.0
        gaze_norm_y = 2.0 * gaze_y / (H - 1) - 1.0
        
        # Create sampling grid
        grid = torch.stack([gaze_norm_x, gaze_norm_y], dim=-1)  # [B, 2]
        grid = grid.view(B, 1, 1, 2)  # [B, 1, 1, 2]
        
        # Sample features
        sampled = F.grid_sample(
            feature_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return sampled.squeeze(2).squeeze(2)  # [B, C]
    
    def forward(
        self,
        rgb: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for lightweight gaze-only depth prediction.
        
        Args:
            rgb: [B, 3, 88, 88] input RGB image
            gaze_x: [B] gaze x coordinates
            gaze_y: [B] gaze y coordinates
            
        Returns:
            Dictionary with 'depth' and optionally 'aux_depths'
        """
        # Encode image
        features = self.encoder(rgb)
        
        # Extract and project gaze features from each scale
        gaze_features = []
        for i, (feat, proj) in enumerate(zip(features, self.gaze_projections)):
            # Scale gaze coordinates for this level
            scale_factor = 2 ** (i + 1)  # 2, 4, 8 for levels 1, 2, 3
            scaled_gaze_x = gaze_x / scale_factor
            scaled_gaze_y = gaze_y / scale_factor
            
            # Extract features at gaze
            gaze_feat = self.extract_gaze_features(feat, scaled_gaze_x, scaled_gaze_y)
            
            # Project to common dimension
            projected = proj(gaze_feat)
            gaze_features.append(projected)
        
        # Concatenate all gaze features
        combined_features = torch.cat(gaze_features, dim=1)
        
        # Predict depth
        depth_logit = self.depth_predictor(combined_features)
        depth = torch.sigmoid(depth_logit) * self.max_depth
        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
        
        outputs = {'depth': depth}
        
        # Auxiliary predictions
        if self.use_multi_scale_supervision and self.training:
            aux_depths = []
            # Use features from scales 2 and 3 (skip scale 1)
            for i in range(1, len(gaze_features)):
                aux_idx = i - 1  # Auxiliary predictor index
                if aux_idx < len(self.aux_predictors):
                    aux_logit = self.aux_predictors[aux_idx](gaze_features[i])
                    aux_depth = torch.sigmoid(aux_logit) * self.max_depth
                    aux_depth = torch.clamp(aux_depth, min=self.min_depth, max=self.max_depth)
                    aux_depths.append(aux_depth)
            
            outputs['aux_depths'] = aux_depths
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the lightweight models
    print("Testing Lightweight Gaze Encoder...")
    
    # Test 2-level encoder
    print("\n1. Testing 2-level encoder (ultra-lightweight):")
    model_2level = LightweightGazeOnlyDepth(
        num_encoder_levels=2,
        base_channels=32,
        gaze_feature_dim=64
    )
    print(f"   Parameters: {model_2level.get_num_params():,}")
    
    # Test 3-level encoder
    print("\n2. Testing 3-level encoder (balanced):")
    model_3level = LightweightGazeOnlyDepth(
        num_encoder_levels=3,
        base_channels=32,
        gaze_feature_dim=64
    )
    print(f"   Parameters: {model_3level.get_num_params():,}")
    
    # Test forward pass
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 88, 88)
    gaze_x = torch.randint(0, 88, (batch_size,)).float()
    gaze_y = torch.randint(0, 88, (batch_size,)).float()
    
    print("\n3. Testing forward pass (3-level):")
    outputs = model_3level(rgb, gaze_x, gaze_y)
    print(f"   Output depth shape: {outputs['depth'].shape}")
    print(f"   Depth range: [{outputs['depth'].min():.3f}, {outputs['depth'].max():.3f}]")
    if 'aux_depths' in outputs:
        print(f"   Number of auxiliary outputs: {len(outputs['aux_depths'])}")
    
    # Compare sizes
    print("\n4. Model comparison:")
    print(f"   2-level: {model_2level.get_num_params():,} parameters")
    print(f"   3-level: {model_3level.get_num_params():,} parameters")
    print(f"   Original RT-MonoDepth: ~1,234,161 parameters")
    
    # Breakdown for 3-level model
    encoder_params = sum(p.numel() for p in model_3level.encoder.parameters())
    predictor_params = sum(p.numel() for p in model_3level.depth_predictor.parameters())
    print(f"\n   3-level breakdown:")
    print(f"   - Encoder: {encoder_params:,}")
    print(f"   - Predictor: {predictor_params:,}")