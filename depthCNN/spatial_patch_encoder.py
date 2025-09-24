#!/usr/bin/env python3
"""
Spatial patch prediction architecture that processes regions with CNNs.
Key difference from flexible_gaze_encoder: extracts spatial feature blocks instead of individual points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class SpatialPatchExtractor(nn.Module):
    """Extracts a spatial feature region around gaze location."""
    
    def __init__(self, region_size: int = 5):
        """
        Args:
            region_size: Size of the spatial region to extract (e.g., 5x5)
        """
        super().__init__()
        self.region_size = region_size
        
    def forward(
        self, 
        feature_map: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract a spatial region centered at gaze location.
        
        Args:
            feature_map: [B, C, H, W] feature map
            gaze_x: [B] normalized gaze x in feature space
            gaze_y: [B] normalized gaze y in feature space
            
        Returns:
            [B, C, region_size, region_size] spatial feature region
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        # Convert gaze to integer indices
        gaze_x_int = torch.round(gaze_x).long()
        gaze_y_int = torch.round(gaze_y).long()
        
        # Calculate region bounds
        half_size = self.region_size // 2
        
        # Extract regions for each sample in batch
        regions = []
        for b in range(B):
            # Get bounds for this sample
            x_center = gaze_x_int[b].item()
            y_center = gaze_y_int[b].item()
            
            # Calculate valid bounds
            x_start = max(0, x_center - half_size)
            x_end = min(W, x_center + half_size + 1)
            y_start = max(0, y_center - half_size)
            y_end = min(H, y_center + half_size + 1)
            
            # Extract region
            region = feature_map[b:b+1, :, y_start:y_end, x_start:x_end]
            
            # Pad if necessary to maintain consistent size
            pad_left = max(0, half_size - x_center)
            pad_right = max(0, x_center + half_size + 1 - W)
            pad_top = max(0, half_size - y_center)
            pad_bottom = max(0, y_center + half_size + 1 - H)
            
            if any([pad_left, pad_right, pad_top, pad_bottom]):
                region = F.pad(region, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
            
            regions.append(region)
        
        # Stack all regions
        return torch.cat(regions, dim=0)


class SpatialCNNDecoder(nn.Module):
    """CNN decoder that upsamples spatial feature region to depth patch."""
    
    def __init__(
        self,
        input_channels: int,
        input_size: int = 5,
        output_size: int = 16,
        hidden_channels: int = 128
    ):
        """
        Args:
            input_channels: Number of input feature channels
            input_size: Size of input spatial region (e.g., 5x5)
            output_size: Size of output depth patch (e.g., 16x16)
            hidden_channels: Number of channels in hidden layers
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Calculate upsampling factor
        scale_factor = output_size / input_size
        num_upsamples = int(math.log2(scale_factor))
        
        layers = []
        in_ch = input_channels
        out_ch = hidden_channels
        
        # Initial processing
        layers.extend([
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ])
        
        # Upsampling layers
        for i in range(num_upsamples):
            layers.extend([
                nn.ConvTranspose2d(out_ch, out_ch//2, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch//2, out_ch//2, 3, padding=1),
                nn.BatchNorm2d(out_ch//2),
                nn.ReLU(inplace=True)
            ])
            out_ch = out_ch // 2
        
        # Final layers
        layers.extend([
            nn.Conv2d(out_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode spatial features to depth patch.
        
        Args:
            x: [B, C, H, W] spatial feature region
            
        Returns:
            [B, 1, output_size, output_size] depth patch
        """
        depth = self.decoder(x)
        
        # Ensure output is exactly the right size
        if depth.shape[-1] != self.output_size:
            depth = F.interpolate(
                depth, 
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=True
            )
        
        return depth


class SpatialPatchDepthPredictor(nn.Module):
    """
    Complete spatial patch depth prediction model.
    Processes spatial regions with CNNs instead of individual points.
    """
    
    def __init__(
        self,
        image_size: int = 88,
        num_encoder_levels: int = 3,
        base_channels: int = 32,
        spatial_region_size: int = 5,
        patch_size: int = 16,
        max_depth: float = 10.0,
        min_depth: float = 0.1
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
        """
        super().__init__()
        
        self.image_size = image_size
        self.spatial_region_size = spatial_region_size
        self.patch_size = patch_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        
        # Import the encoder from flexible_gaze_encoder
        from flexible_gaze_encoder import FlexibleGazeEncoder
        
        # Encoder - same as before, processes full image
        self.encoder = FlexibleGazeEncoder(
            num_levels=num_encoder_levels,
            base_channels=base_channels,
            image_size=image_size
        )
        
        # Spatial region extractor
        self.spatial_extractor = SpatialPatchExtractor(region_size=spatial_region_size)
        
        # Get output channels from deepest encoder level
        deepest_channels = self.encoder.num_ch_enc[-1]
        
        # CNN decoder for spatial upsampling
        self.spatial_decoder = SpatialCNNDecoder(
            input_channels=deepest_channels,
            input_size=spatial_region_size,
            output_size=patch_size,
            hidden_channels=128
        )
        
        # Optional: Multi-scale fusion (use features from multiple encoder levels)
        self.use_multiscale = num_encoder_levels > 2
        if self.use_multiscale:
            # Combine features from last two levels
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(deepest_channels + self.encoder.num_ch_enc[-2], deepest_channels, 1),
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
        Forward pass for spatial patch prediction.
        
        Args:
            rgb: [B, 3, H, W] input image
            gaze_x: [B] gaze x coordinates in image space
            gaze_y: [B] gaze y coordinates in image space
            
        Returns:
            Dictionary with 'depth' containing [B, patch_size, patch_size] depth patch
        """
        # Encode full image
        features = self.encoder(rgb)
        
        # Use deepest feature level
        deepest_features = features[-1]
        B, C, H, W = deepest_features.shape
        
        # Convert gaze coordinates to feature space
        # For level 3 with 88x88 input: 88->44->22->11
        scale_factor = 2 ** len(features)  # 8 for 3 levels
        gaze_x_feat = gaze_x / scale_factor
        gaze_y_feat = gaze_y / scale_factor
        
        # Extract spatial region around gaze
        spatial_region = self.spatial_extractor(deepest_features, gaze_x_feat, gaze_y_feat)
        
        # Optional: Multi-scale fusion
        if self.use_multiscale and len(features) > 1:
            # Get second deepest features
            second_features = features[-2]
            scale_factor_2 = 2 ** (len(features) - 1)
            gaze_x_feat_2 = gaze_x / scale_factor_2
            gaze_y_feat_2 = gaze_y / scale_factor_2
            
            # Extract and upsample to match deepest resolution
            second_region = self.spatial_extractor(second_features, gaze_x_feat_2, gaze_y_feat_2)
            second_region = F.interpolate(
                second_region,
                size=(self.spatial_region_size, self.spatial_region_size),
                mode='bilinear',
                align_corners=True
            )
            
            # Fuse features
            combined = torch.cat([spatial_region, second_region], dim=1)
            spatial_region = self.feature_fusion(combined)
        
        # Decode to depth patch
        depth_patch = self.spatial_decoder(spatial_region)
        
        # Remove channel dimension and scale to depth range
        depth_patch = depth_patch.squeeze(1) * self.max_depth
        depth_patch = torch.clamp(depth_patch, min=self.min_depth, max=self.max_depth)
        
        outputs = {
            'depth': depth_patch
        }
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HybridSpatialPatchPredictor(nn.Module):
    """
    Hybrid approach combining point-based and spatial methods.
    Can help transition from existing point-based models.
    """
    
    def __init__(
        self,
        image_size: int = 88,
        patch_size: int = 16,
        base_channels: int = 32,
        fusion_mode: str = 'add'  # 'add', 'concat', or 'weighted'
    ):
        """
        Args:
            image_size: Input image size
            patch_size: Output patch size
            base_channels: Base channels for encoders
            fusion_mode: How to combine point and spatial predictions
        """
        super().__init__()
        
        from flexible_gaze_encoder import FlexibleGazeOnlyDepth
        
        self.patch_size = patch_size
        self.fusion_mode = fusion_mode
        
        # Point-based predictor (existing approach)
        self.point_predictor = FlexibleGazeOnlyDepth(
            num_encoder_levels=3,
            base_channels=base_channels,
            image_size=image_size,
            predict_patch=True,
            patch_size=patch_size
        )
        
        # Spatial predictor (new approach)
        self.spatial_predictor = SpatialPatchDepthPredictor(
            image_size=image_size,
            num_encoder_levels=3,
            base_channels=base_channels,
            patch_size=patch_size
        )
        
        # Fusion layer
        if fusion_mode == 'concat':
            self.fusion = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            )
        elif fusion_mode == 'weighted':
            # Learn weights for each approach
            self.weight_net = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 2, 1),
                nn.Softmax(dim=1)
            )
    
    def forward(
        self,
        rgb: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining both approaches.
        
        Args:
            rgb: [B, 3, H, W] input image
            gaze_x: [B] gaze x coordinates
            gaze_y: [B] gaze y coordinates
            
        Returns:
            Dictionary with combined depth prediction
        """
        # Get predictions from both models
        point_output = self.point_predictor(rgb, gaze_x, gaze_y)
        spatial_output = self.spatial_predictor(rgb, gaze_x, gaze_y)
        
        point_depth = point_output['depth']
        spatial_depth = spatial_output['depth']
        
        # Combine predictions based on fusion mode
        if self.fusion_mode == 'add':
            combined_depth = (point_depth + spatial_depth) / 2.0
        
        elif self.fusion_mode == 'concat':
            # Stack and process with CNN
            combined = torch.stack([point_depth, spatial_depth], dim=1)
            combined_depth = self.fusion(combined).squeeze(1) * 10.0
        
        elif self.fusion_mode == 'weighted':
            # Learn adaptive weights
            combined = torch.stack([point_depth, spatial_depth], dim=1)
            weights = self.weight_net(combined / 10.0)  # Normalize for stability
            combined_depth = (weights[:, 0:1] * point_depth.unsqueeze(1) + 
                            weights[:, 1:2] * spatial_depth.unsqueeze(1)).squeeze(1)
        
        outputs = {
            'depth': combined_depth,
            'point_depth': point_depth,
            'spatial_depth': spatial_depth
        }
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the spatial patch predictor
    print("Testing Spatial Patch Depth Predictor...")
    
    # Create model
    model = SpatialPatchDepthPredictor(
        image_size=88,
        num_encoder_levels=3,
        base_channels=32,
        spatial_region_size=5,
        patch_size=16
    )
    
    print(f"Total parameters: {model.get_num_params():,}")
    
    # Test forward pass
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 88, 88)
    gaze_x = torch.randint(20, 68, (batch_size,)).float()
    gaze_y = torch.randint(20, 68, (batch_size,)).float()
    
    outputs = model(rgb, gaze_x, gaze_y)
    
    print("\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Test hybrid model
    print("\n\nTesting Hybrid Model...")
    hybrid = HybridSpatialPatchPredictor(
        image_size=88,
        patch_size=16,
        fusion_mode='weighted'
    )
    
    print(f"Total parameters: {hybrid.get_num_params():,}")
    
    outputs = hybrid(rgb, gaze_x, gaze_y)
    print("\nHybrid outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")