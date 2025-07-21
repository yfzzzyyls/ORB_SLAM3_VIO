#!/usr/bin/env python3
"""
Flexible encoder for gaze-only depth prediction supporting various image sizes.
Based on the proven lightweight_gaze_encoder.py but with flexible resolution support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class FlexibleGazeEncoder(nn.Module):
    """
    Efficient encoder supporting flexible input sizes for gaze-based depth prediction.
    """
    
    def __init__(self, num_levels: int = 3, base_channels: int = 32, image_size: int = 88):
        """
        Args:
            num_levels: Number of encoding levels (2 or 3 recommended)
            base_channels: Base number of channels (doubled at each level)
            image_size: Input image size (square images assumed)
        """
        super().__init__()
        
        assert num_levels in [2, 3, 4, 5], "Only 2-5 levels supported"
        self.num_levels = num_levels
        self.base_channels = base_channels
        self.image_size = image_size
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Level 1: size -> size/2
        self.level1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Level 2: size/2 -> size/4
        self.level2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # Level 3: size/4 -> size/8 (optional)
        if num_levels >= 3:
            self.level3 = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
        
        # Level 4: size/8 -> size/16 (optional)
        if num_levels >= 4:
            self.level4 = nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            )
        
        # Level 5: size/16 -> size/32 (optional)
        if num_levels >= 5:
            self.level5 = nn.Sequential(
                nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 16),
                nn.ReLU(inplace=True)
            )
        
        # Output channel counts for each level
        self.num_ch_enc = []
        channel_multipliers = [1, 2, 4, 8, 16]  # For levels 1-5
        for i in range(num_levels):
            self.num_ch_enc.append(base_channels * channel_multipliers[i])
        
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
            x: [B, 3, H, W] input RGB image (H=W=image_size)
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # Initial processing
        x = self.input_conv(x)  # [B, C, H, W]
        
        # Level 1
        x1 = self.level1(x)  # [B, C, H/2, W/2]
        features.append(x1)
        
        # Level 2
        x2 = self.level2(x1)  # [B, 2C, H/4, W/4]
        features.append(x2)
        
        # Level 3 (if enabled)
        if self.num_levels >= 3:
            x3 = self.level3(x2)  # [B, 4C, H/8, W/8]
            features.append(x3)
        
        # Level 4 (if enabled)
        if self.num_levels >= 4:
            x4 = self.level4(x3)  # [B, 8C, H/16, W/16]
            features.append(x4)
        
        # Level 5 (if enabled)
        if self.num_levels >= 5:
            x5 = self.level5(x4)  # [B, 16C, H/32, W/32]
            features.append(x5)
        
        return features


class FlexibleGazeOnlyDepth(nn.Module):
    """
    Complete flexible model for gaze-only depth prediction supporting various image sizes.
    """
    
    def __init__(
        self,
        num_encoder_levels: int = 3,
        base_channels: int = 32,
        gaze_feature_dim: int = 64,
        image_size: int = 88,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        use_multi_scale_supervision: bool = True
    ):
        """
        Args:
            num_encoder_levels: Number of encoding levels (2 or 3)
            base_channels: Base channels for encoder
            gaze_feature_dim: Dimension for each scale's gaze features
            image_size: Input image size (square images assumed)
            max_depth: Maximum depth value
            min_depth: Minimum depth value
            use_multi_scale_supervision: Whether to use auxiliary losses
        """
        super().__init__()
        
        self.num_encoder_levels = num_encoder_levels
        self.image_size = image_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_multi_scale_supervision = use_multi_scale_supervision
        
        # Flexible encoder
        self.encoder = FlexibleGazeEncoder(
            num_levels=num_encoder_levels,
            base_channels=base_channels,
            image_size=image_size
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
        Forward pass for flexible gaze-only depth prediction.
        
        Args:
            rgb: [B, 3, H, W] input RGB image (H=W=image_size)
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
    # Test the flexible models
    print("Testing Flexible Gaze Encoder...")
    
    # Test different image sizes
    image_sizes = [88, 128, 176, 256]
    
    for img_size in image_sizes:
        print(f"\nTesting with image size {img_size}x{img_size}:")
        
        model = FlexibleGazeOnlyDepth(
            num_encoder_levels=3,
            base_channels=32,
            gaze_feature_dim=64,
            image_size=img_size
        )
        
        print(f"  Parameters: {model.get_num_params():,}")
        
        # Test forward pass
        batch_size = 4
        rgb = torch.randn(batch_size, 3, img_size, img_size)
        gaze_x = torch.randint(0, img_size, (batch_size,)).float()
        gaze_y = torch.randint(0, img_size, (batch_size,)).float()
        
        outputs = model(rgb, gaze_x, gaze_y)
        print(f"  Output depth shape: {outputs['depth'].shape}")
        print(f"  Depth range: [{outputs['depth'].min():.3f}, {outputs['depth'].max():.3f}]")
        
        # Calculate output feature map sizes
        feat_sizes = []
        size = img_size
        for i in range(3):
            size = size // 2
            feat_sizes.append(size)
        print(f"  Feature map sizes: {feat_sizes}")
    
    # Compare model sizes
    print("\nModel size comparison:")
    for levels in [2, 3]:
        model = FlexibleGazeOnlyDepth(
            num_encoder_levels=levels,
            base_channels=32,
            image_size=88
        )
        print(f"  {levels}-level model: {model.get_num_params():,} parameters")