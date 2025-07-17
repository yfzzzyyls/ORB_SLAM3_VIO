#!/usr/bin/env python3
"""
Fixed version of RT-MonoDepth decoder that works with DataParallel.
"""

from collections import OrderedDict
import torch
import torch.nn as nn

# Import necessary components from the original
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "RT-MonoDepth"))

from networks.RTMonoDepth.RTMonoDepth_s import DepthEncoder, ConvBlock, Decoder


class DepthDecoderFixed(nn.Module):
    """Fixed version of DepthDecoder that properly registers all modules."""
    
    def __init__(self, num_ch_enc, scales=range(3), use_skips=False):
        super().__init__()
        
        self.use_skips = use_skips
        self.scales = scales
        self.groups = 1
        
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 96, 192]
        
        # Create ModuleDict instead of OrderedDict for proper registration
        self.upconvs = nn.ModuleDict()
        self.dispconvs = nn.ModuleDict()
        
        # Single layer decoder - only one processing block
        # Input from encoder bottleneck
        num_ch_in = self.num_ch_enc[-1]  # 192 for RT-MonoDepth-S
        num_ch_out = 32  # Reduced channels for single layer
        
        # Single processing block instead of 4 stages
        self.upconvs[f"upconv_0_0"] = ConvBlock(num_ch_in, num_ch_out)
        
        # Output disparity
        for s in self.scales:
            self.dispconvs[f"dispconv_{s}"] = Decoder(num_ch_out)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features):
        outputs = {}
        
        # Take bottleneck features
        x = input_features[-1]  # Lowest resolution features
        
        # Single processing block
        x = self.upconvs[f"upconv_0_0"](x)
        
        # Upsample directly to full resolution
        # For 88x88 input: encoder produces [44, 22, 11, 5] so we need 16x upsampling
        # to get back to 88x88 (5 * 16 ≈ 88)
        target_h = input_features[0].shape[2] * 2  # First encoder is half resolution
        target_w = input_features[0].shape[3] * 2
        
        x = nn.functional.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
        
        # Output disparity at scale 0 (full resolution)
        if 0 in self.scales:
            depth = self.sigmoid(self.dispconvs[f"dispconv_0"](x))
            outputs[("disp", 0)] = depth
        
        return outputs