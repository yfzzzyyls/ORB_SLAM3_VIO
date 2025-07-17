#!/usr/bin/env python3
"""
Specialized MonoDepth architecture for gaze-only depth prediction.
Modified from RT-MonoDepth to predict depth at a single gaze location.

Key modifications:
1. Encoder remains similar but optimized for 88x88 input
2. Decoder completely replaced - no spatial reconstruction needed
3. Multi-scale feature extraction only at gaze location
4. Two-stage reasoning: object understanding → depth prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


class GazeDepthEncoder(nn.Module):
    """
    Modified encoder for gaze-only depth prediction.
    Similar to RT-MonoDepth but optimized for 88x88 input.
    """
    
    def __init__(self):
        super().__init__()
        
        # Channel configurations for each stage
        # Reduced from original [64, 64, 128, 192] since we have smaller input
        self.num_ch_enc = [32, 64, 128, 256]
        
        # Initial convolution (88x88 → 44x44)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        # Build encoder stages
        self.convs = nn.ModuleList()
        
        # Stage 1: 44x44 → 22x22
        self.convs.append(nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        ))
        
        # Stage 2: 22x22 → 11x11
        self.convs.append(nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        ))
        
        # Stage 3: 11x11 → 5x5 (roughly, depends on exact input)
        self.convs.append(nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        ))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: Input image [B, 3, 88, 88]
            
        Returns:
            List of features at different scales
        """
        features = []
        
        # Normalize input (ImageNet normalization)
        x = (x - 0.45) / 0.225
        
        # Initial convolution
        x = self.conv1(x)  # 88 → 44
        x = self.relu(x)
        features.append(x)  # 44x44x32
        
        # Progressive encoding
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        
        # features[0]: 44x44x32 (local details)
        # features[1]: 22x22x64 (object parts)
        # features[2]: 11x11x128 (object level)
        # features[3]: 5x5x256 (scene context)
        
        return features


class GazeFeatureExtractor(nn.Module):
    """
    Extracts and processes features at gaze location from multi-scale encoder outputs.
    """
    
    def __init__(self, num_ch_enc):
        super().__init__()
        
        # Feature processors for each scale
        self.scale_processors = nn.ModuleList([
            nn.Linear(num_ch_enc[0], 32),  # 44x44 scale
            nn.Linear(num_ch_enc[1], 32),  # 22x22 scale
            nn.Linear(num_ch_enc[2], 32),  # 11x11 scale
            nn.Linear(num_ch_enc[3], 32),  # 5x5 scale
        ])
        
        # Gaze position encoder
        self.gaze_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
    def forward(self, features: List[torch.Tensor], 
                gaze_x: float, gaze_y: float, 
                input_size: int = 1408) -> torch.Tensor:
        """
        Extract features at gaze location from each scale.
        
        Args:
            features: List of feature maps from encoder
            gaze_x, gaze_y: Gaze coordinates in original image space
            input_size: Original input size before pooling
            
        Returns:
            Combined features at gaze location
        """
        batch_size = features[0].shape[0]
        device = features[0].device
        
        # Extract features at gaze from each scale
        gaze_features = []
        
        # Scale 0: 44x44
        scale_size = 44
        gx = int(gaze_x * scale_size / input_size)
        gy = int(gaze_y * scale_size / input_size)
        gx = min(max(gx, 0), scale_size - 1)
        gy = min(max(gy, 0), scale_size - 1)
        feat = features[0][:, :, gy, gx]  # [B, C]
        gaze_features.append(self.scale_processors[0](feat))
        
        # Scale 1: 22x22
        scale_size = 22
        gx = int(gaze_x * scale_size / input_size)
        gy = int(gaze_y * scale_size / input_size)
        gx = min(max(gx, 0), scale_size - 1)
        gy = min(max(gy, 0), scale_size - 1)
        feat = features[1][:, :, gy, gx]
        gaze_features.append(self.scale_processors[1](feat))
        
        # Scale 2: 11x11
        scale_size = 11
        gx = int(gaze_x * scale_size / input_size)
        gy = int(gaze_y * scale_size / input_size)
        gx = min(max(gx, 0), scale_size - 1)
        gy = min(max(gy, 0), scale_size - 1)
        feat = features[2][:, :, gy, gx]
        gaze_features.append(self.scale_processors[2](feat))
        
        # Scale 3: 5x5 (or global if using adaptive pooling)
        if features[3].shape[2] > 1:
            scale_size = features[3].shape[2]
            gx = int(gaze_x * scale_size / input_size)
            gy = int(gaze_y * scale_size / input_size)
            gx = min(max(gx, 0), scale_size - 1)
            gy = min(max(gy, 0), scale_size - 1)
            feat = features[3][:, :, gy, gx]
        else:
            # Global features (1x1)
            feat = features[3].squeeze(-1).squeeze(-1)
        gaze_features.append(self.scale_processors[3](feat))
        
        # Encode normalized gaze position
        gaze_norm = torch.tensor([gaze_x / input_size, gaze_y / input_size], 
                                 device=device, dtype=torch.float32)
        gaze_norm = gaze_norm.unsqueeze(0).expand(batch_size, -1)
        gaze_encoded = self.gaze_encoder(gaze_norm)
        
        # Concatenate all features
        # 4 scales * 32 features + 16 gaze features = 144 features
        combined = torch.cat(gaze_features + [gaze_encoded], dim=1)
        
        return combined


class GazeDepthDecoder(nn.Module):
    """
    Specialized decoder for single-point depth prediction.
    Implements two-stage reasoning: object understanding → depth prediction.
    """
    
    def __init__(self, input_dim: int = 176):  # 144 from multi-scale + 32 from global
        super().__init__()
        
        # Stage 1: Object/Scene Understanding
        # This stage learns to understand what object the gaze is on
        # and its general properties (size, type, distance range)
        self.object_understanding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Stage 2: Depth Refinement
        # Uses object understanding to predict precise depth
        self.depth_prediction = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to single depth value.
        
        Args:
            features: Combined multi-scale features at gaze location
            
        Returns:
            Predicted depth value [B, 1]
        """
        # Stage 1: Understand the object/scene
        object_features = self.object_understanding(features)
        
        # Stage 2: Predict depth based on understanding
        depth = self.depth_prediction(object_features)
        
        # Apply sigmoid and scale to depth range
        # Using same convention as RT-MonoDepth
        depth = torch.sigmoid(depth) * 10.0  # 0-10m range
        
        return depth


class GazeMonoDepth(nn.Module):
    """
    Complete gaze-only depth prediction model.
    Combines encoder, feature extractor, and decoder.
    """
    
    def __init__(self, min_depth: float = 0.1, max_depth: float = 10.0):
        super().__init__()
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Create components
        self.encoder = GazeDepthEncoder()
        self.feature_extractor = GazeFeatureExtractor(self.encoder.num_ch_enc)
        self.decoder = GazeDepthDecoder()
        
        # Optional: Global context branch (processes entire feature map)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU()
        )
        
    def forward(self, image: torch.Tensor, 
                gaze_x: torch.Tensor, 
                gaze_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for gaze-only depth prediction.
        
        Args:
            image: Input image [B, 3, 88, 88]
            gaze_x: Gaze x-coordinate in original image space [B]
            gaze_y: Gaze y-coordinate in original image space [B]
            
        Returns:
            Dictionary containing:
                - 'depth': Predicted depth at gaze location [B, 1]
                - 'features': Encoder features (for visualization/debugging)
        """
        batch_size = image.shape[0]
        
        # Encode image
        features = self.encoder(image)
        
        # Extract features at gaze location
        gaze_features_list = []
        for b in range(batch_size):
            gaze_feat = self.feature_extractor(
                [f[b:b+1] for f in features],  # Single batch element
                gaze_x[b].item(),
                gaze_y[b].item()
            )
            gaze_features_list.append(gaze_feat)
        
        gaze_features = torch.cat(gaze_features_list, dim=0)
        
        # Add global context
        global_feat = self.global_context(features[-1])
        combined_features = torch.cat([gaze_features, global_feat], dim=1)
        
        # Decode to depth
        depth = self.decoder(combined_features)
        
        return {
            'depth': depth,
            'features': features,
            'gaze_features': gaze_features
        }
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Simplified version without global context
class SimpleGazeMonoDepth(nn.Module):
    """
    Simplified version that directly processes features to depth.
    """
    
    def __init__(self):
        super().__init__()
        
        # Progressive encoder (88 → 44 → 22 → 11 → 1)
        self.encoder = nn.Sequential(
            # 88 → 44
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            # 44 → 22
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            # 22 → 11
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            # 11 → 1 (global)
            nn.Conv2d(128, 256, 11),
            nn.ReLU()
        )
        
        # Direct depth prediction
        self.depth_head = nn.Sequential(
            nn.Linear(256 + 2, 128),  # features + gaze coords
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, image: torch.Tensor, 
                gaze_x: torch.Tensor, 
                gaze_y: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        # Extract global features
        features = self.encoder(image)
        features = features.squeeze(-1).squeeze(-1)  # [B, 256]
        
        # Add normalized gaze coordinates
        gaze_norm = torch.stack([gaze_x / 1408, gaze_y / 1408], dim=1)
        combined = torch.cat([features, gaze_norm], dim=1)
        
        # Predict depth
        depth = self.depth_head(combined)
        depth = torch.sigmoid(depth) * 10.0  # 0-10m range
        
        return depth


if __name__ == "__main__":
    # Test the model
    print("Testing GazeMonoDepth...")
    
    # Create model
    model = GazeMonoDepth()
    
    # Test input
    batch_size = 2
    image = torch.randn(batch_size, 3, 88, 88)
    gaze_x = torch.tensor([704.0, 1000.0])
    gaze_y = torch.tensor([704.0, 400.0])
    
    # Forward pass
    output = model(image, gaze_x, gaze_y)
    depth = output['depth']
    
    print(f"\nInput shape: {image.shape}")
    print(f"Gaze coordinates: x={gaze_x.tolist()}, y={gaze_y.tolist()}")
    print(f"Output depth shape: {depth.shape}")
    print(f"Output depth values: {depth.squeeze().tolist()}")
    
    # Count parameters
    total_params = model.get_num_params()
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Reduction from original RT-MonoDepth: {1.23e6 / total_params:.1f}x smaller")
    
    # Test simple version
    print("\n\nTesting SimpleGazeMonoDepth...")
    simple_model = SimpleGazeMonoDepth()
    simple_depth = simple_model(image, gaze_x, gaze_y)
    print(f"Simple model output: {simple_depth.squeeze().tolist()}")
    print(f"Simple model parameters: {sum(p.numel() for p in simple_model.parameters()):,}")