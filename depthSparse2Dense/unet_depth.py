import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DepthCompletionUNet(nn.Module):
    """
    UNet for sparse-to-dense depth completion
    Input: RGB (3) + Sparse Depth (1) = 4 channels
    Output: Dense Depth (1 channel)
    """
    def __init__(self, n_channels=4, n_classes=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Output activation for depth (ensure positive values)
        self.output_activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 4, H, W] (RGB + sparse depth)
        Returns:
            Dense depth map [B, 1, H, W]
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        # Ensure positive depth values
        return self.output_activation(x)


class DepthCompletionNet(nn.Module):
    """
    Complete depth completion network with RGB encoder branch
    """
    def __init__(self, use_skip_connections=True):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        
        # Separate encoders for RGB and sparse depth
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Fusion and main UNet
        self.fusion_conv = nn.Conv2d(96, 64, kernel_size=1)
        self.unet = DepthCompletionUNet(n_channels=64, n_classes=1)
    
    def forward(self, rgb, sparse_depth):
        """
        Args:
            rgb: RGB image [B, 3, H, W]
            sparse_depth: Sparse depth map [B, 1, H, W]
        Returns:
            Dense depth map [B, 1, H, W]
        """
        # Encode inputs separately
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(sparse_depth)
        
        # Concatenate and fuse
        fused = torch.cat([rgb_features, depth_features], dim=1)
        fused = self.fusion_conv(fused)
        
        # Pass through UNet
        dense_depth = self.unet(fused)
        
        # Optional: Add residual connection with sparse depth
        if self.use_skip_connections:
            # Where we have sparse depth, use it directly
            mask = (sparse_depth > 0).float()
            dense_depth = dense_depth * (1 - mask) + sparse_depth * mask
        
        return dense_depth