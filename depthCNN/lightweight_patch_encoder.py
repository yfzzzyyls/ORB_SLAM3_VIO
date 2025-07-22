#!/usr/bin/env python3
"""
Lightweight encoder for high-resolution patches centered at gaze location.
Designed to extract fine details with balanced parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from flexible_gaze_encoder import FlexibleGazeEncoder


class LightweightPatchEncoder(nn.Module):
    """
    Efficient encoder for 96×96 high-res patches.
    Uses wider channels (48 vs 32) to capture high-res information.
    Outputs 192-dim feature vector for fusion with context.
    """
    
    def __init__(
        self,
        num_levels: int = 3,
        base_channels: int = 48,
        output_dim: int = 192,
        image_size: int = 96
    ):
        """
        Args:
            num_levels: Number of encoding levels (3 recommended for 96×96)
            base_channels: Base channels (48 for high-res information)
            output_dim: Output feature dimension (192)
            image_size: Expected patch size (96)
        """
        super().__init__()
        
        self.num_levels = num_levels
        self.base_channels = base_channels
        self.output_dim = output_dim
        self.image_size = image_size
        
        # Use the flexible encoder with more channels for high-res
        self.encoder = FlexibleGazeEncoder(
            num_levels=num_levels,
            base_channels=base_channels,
            image_size=image_size
        )
        
        # Calculate total channels from all levels
        # For 3 levels with base=48: [48, 96, 192] = 336 total
        total_channels = sum(self.encoder.num_ch_enc)
        
        # Feature projector: aggregate multi-scale features to fixed dimension
        self.feature_projector = nn.Sequential(
            nn.Linear(total_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the feature projector weights."""
        for m in self.feature_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def extract_center_features(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract features from the center of each feature map.
        Since the patch is already centered at gaze, we extract from center.
        
        Args:
            feature_maps: List of feature maps at different scales
            
        Returns:
            Concatenated features from all scales
        """
        center_features = []
        
        for feat_map in feature_maps:
            B, C, H, W = feat_map.shape
            
            # Extract center location (gaze is already centered in patch)
            center_y = H // 2
            center_x = W // 2
            
            # Extract features using a small window around center
            # This is more robust than single pixel
            window_size = 3
            half_window = window_size // 2
            
            y_start = max(0, center_y - half_window)
            y_end = min(H, center_y + half_window + 1)
            x_start = max(0, center_x - half_window)
            x_end = min(W, center_x + half_window + 1)
            
            # Extract window and pool
            window_features = feat_map[:, :, y_start:y_end, x_start:x_end]
            pooled_features = F.adaptive_avg_pool2d(window_features, (1, 1))
            center_features.append(pooled_features.squeeze(-1).squeeze(-1))
        
        # Concatenate all scale features
        return torch.cat(center_features, dim=1)
    
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through patch encoder.
        
        Args:
            patch: [B, 3, 96, 96] high-res RGB patch centered at gaze
            
        Returns:
            [B, output_dim] feature vector
        """
        # Encode patch at multiple scales
        feature_maps = self.encoder(patch)
        
        # Extract features from center (where gaze is)
        center_features = self.extract_center_features(feature_maps)
        
        # Project to output dimension
        patch_features = self.feature_projector(center_features)
        
        return patch_features
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FeatureFusionModule(nn.Module):
    """
    Fuses context features (from 88×88) with patch features (from 96×96).
    Uses attention mechanism to weight the contributions.
    """
    
    def __init__(
        self,
        context_dim: int = 256,
        patch_dim: int = 192,
        output_dim: int = 384,
        use_attention: bool = True
    ):
        """
        Args:
            context_dim: Dimension of context features
            patch_dim: Dimension of patch features
            output_dim: Output dimension after fusion
            use_attention: Whether to use attention-based fusion
        """
        super().__init__()
        
        self.context_dim = context_dim
        self.patch_dim = patch_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        if use_attention:
            # Attention mechanism to weight context vs patch
            self.context_attention = nn.Sequential(
                nn.Linear(context_dim + patch_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
            self.patch_attention = nn.Sequential(
                nn.Linear(context_dim + patch_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        # Fusion projection
        self.fusion_projector = nn.Sequential(
            nn.Linear(context_dim + patch_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        context_features: torch.Tensor,
        patch_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse context and patch features.
        
        Args:
            context_features: [B, context_dim] from 88×88 encoder
            patch_features: [B, patch_dim] from 96×96 encoder
            
        Returns:
            [B, output_dim] fused features
        """
        # Concatenate for attention computation
        combined = torch.cat([context_features, patch_features], dim=1)
        
        if self.use_attention:
            # Compute attention weights
            context_weight = self.context_attention(combined)
            patch_weight = self.patch_attention(combined)
            
            # Normalize weights
            total_weight = context_weight + patch_weight + 1e-8
            context_weight = context_weight / total_weight
            patch_weight = patch_weight / total_weight
            
            # Weighted features
            weighted_context = context_features * context_weight
            weighted_patch = patch_features * patch_weight
            
            # Concatenate weighted features
            fused = torch.cat([weighted_context, weighted_patch], dim=1)
        else:
            # Simple concatenation
            fused = combined
        
        # Project to output dimension
        output = self.fusion_projector(fused)
        
        return output


if __name__ == "__main__":
    # Test the patch encoder
    print("Testing Lightweight Patch Encoder...")
    
    # Create encoder
    encoder = LightweightPatchEncoder(
        num_levels=3,
        base_channels=48,
        output_dim=192,
        image_size=96
    )
    
    print(f"Total parameters: {encoder.get_num_params():,}")
    
    # Test forward pass
    batch_size = 4
    patch = torch.randn(batch_size, 3, 96, 96)
    
    features = encoder(patch)
    print(f"Input shape: {patch.shape}")
    print(f"Output shape: {features.shape}")
    
    # Test fusion module
    print("\nTesting Feature Fusion Module...")
    
    fusion = FeatureFusionModule(
        context_dim=256,
        patch_dim=192,
        output_dim=384,
        use_attention=True
    )
    
    context_feat = torch.randn(batch_size, 256)
    patch_feat = torch.randn(batch_size, 192)
    
    fused = fusion(context_feat, patch_feat)
    print(f"Context features: {context_feat.shape}")
    print(f"Patch features: {patch_feat.shape}")
    print(f"Fused features: {fused.shape}")
    
    # Count fusion parameters
    fusion_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"Fusion module parameters: {fusion_params:,}")