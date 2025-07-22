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
        
        # decoder
        for i in range(3, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs[f"upconv_{i}_0"] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i == 1:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs[f"upconv_{i}_1"] = ConvBlock(num_ch_in, num_ch_out)
        
        for s in self.scales:
            self.dispconvs[f"dispconv_{s}"] = Decoder(self.num_ch_dec[s])
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features):
        outputs = {}
        
        x = input_features[-1]  # 1/16
        for i in range(3, -1, -1):
            x = self.upconvs[f"upconv_{i}_0"](x)
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            
            if self.use_skips and i > 1:
                x += input_features[i - 1]
            elif self.use_skips and i == 1:
                x = torch.cat([x, input_features[i - 1]], 1)
            
            x = self.upconvs[f"upconv_{i}_1"](x)
            
            if i in self.scales:
                depth = self.sigmoid(self.dispconvs[f"dispconv_{i}"](x))
                outputs[("disp", i)] = depth
        
        return outputs