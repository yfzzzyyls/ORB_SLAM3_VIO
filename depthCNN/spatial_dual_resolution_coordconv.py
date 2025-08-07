import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    """Antialiased downsampling with fixed Gaussian blur."""
    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride = stride
        filt = torch.tensor([1., 4., 6., 4., 1.])
        kernel = (filt[:, None] * filt[None, :]).float()
        kernel /= kernel.sum()
        self.register_buffer('kernel', kernel[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=2, groups=x.size(1))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for conditioning features with gaze."""
    def __init__(self, gaze_dim, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(gaze_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        # Initialize to identity: gamma=1, beta=0
        with torch.no_grad():
            self.fc[-1].weight.zero_()
            self.fc[-1].bias.zero_()
            self.fc[-1].bias.data[:feature_dim] = 1.0  # gamma=1 for first half
            # beta=0 for second half (already zero)
        
    def forward(self, features, gaze):
        # gaze: [B, 2]
        # features: [B, C, H, W]
        B, C, H, W = features.shape
        film_params = self.fc(gaze)  # [B, 2*C]
        gamma = film_params[:, :C].view(B, C, 1, 1)
        beta = film_params[:, C:].view(B, C, 1, 1)
        return features * gamma + beta


class SpatialDualResolutionGazeDepth(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Context encoder - processes downsampled full image with coordinate channels
        # Input: 7 channels (RGB + gaze_heatmap + x_coord + y_coord + r_to_gaze)
        # Scaled up width for more capacity
        self.context_conv0_dw = nn.Conv2d(7, 7, kernel_size=11, stride=1, padding=5, groups=7)
        self.context_conv0_pw = nn.Conv2d(7, 96, kernel_size=1)
        self.context_gn0 = nn.GroupNorm(16, 96)  # GroupNorm for coordinate channels
        self.context_blur0 = BlurPool(channels=96, stride=2)  # 88 -> 44
        
        # Multi-scale gaze injection
        self.gaze_inject_44 = FiLMLayer(gaze_dim=2, feature_dim=96)
        
        # Removed C1 and C2 since only C0 is used for ROI alignment
        
        # Patch encoder - processes high-res crop at gaze
        # Input: 5 channels (RGB + delta_x + delta_y)
        # Scaled up width for more capacity
        self.patch_conv0_dw = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, groups=5)  # 88 -> 88
        self.patch_conv0_pw = nn.Conv2d(5, 96, kernel_size=1)
        self.patch_gn0 = nn.GroupNorm(16, 96)
        self.patch_blur0 = BlurPool(channels=96, stride=2)  # 88 -> 44 (anti-aliased)
        
        self.patch_conv1_dw = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96)  # 44 -> 44
        self.patch_conv1_pw = nn.Conv2d(96, 192, kernel_size=1)
        self.patch_gn1 = nn.GroupNorm(32, 192)
        self.patch_blur1 = BlurPool(channels=192, stride=2)  # 44 -> 22 (anti-aliased)
        
        self.patch_conv2_dw = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192)  # 22 -> 22
        self.patch_conv2_pw = nn.Conv2d(192, 288, kernel_size=1)
        self.patch_gn2 = nn.GroupNorm(48, 288)
        
        self.patch_conv3_dw = nn.Conv2d(288, 288, kernel_size=3, stride=1, padding=1, groups=288)  # 22 -> 22
        self.patch_conv3_pw = nn.Conv2d(288, 288, kernel_size=1)
        self.patch_gn3 = nn.GroupNorm(48, 288)
        
        # Project context features from RoIAlign (96ch) to match patch (288ch)
        self.ctx_proj = nn.Conv2d(96, 288, kernel_size=1)
        
        # Pointwise fusion only (no spatial convs) as specified
        self.fuse_1x1 = nn.Sequential(
            nn.Conv2d(288 + 288, 192, kernel_size=1),
            nn.GroupNorm(32, 192),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(p=0.10)  # After pointwise fusion
        
        # Lightweight decoder at 22x22 only (no 44x44 detour)
        self.decode_dw = nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192)
        self.decode_pw = nn.Conv2d(192, 96, kernel_size=1)
        self.decode_gn = nn.GroupNorm(16, 96)
        self.head_dropout = nn.Dropout2d(p=0.10)  # Just before final convs
        
        # Final channel reduction
        self.decode_final = nn.Conv2d(96, 48, kernel_size=1)
        
        # Final prediction
        self.pred_conv = nn.Conv2d(48, 1, kernel_size=1)
        
        # Uncertainty prediction
        self.uncertainty_conv = nn.Conv2d(48, 1, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        # Initialize final layer for depth prediction
        nn.init.xavier_uniform_(self.pred_conv.weight)
        nn.init.constant_(self.pred_conv.bias, 2.0)  # median depth ~2m
        
        # Initialize uncertainty to predict small values initially
        nn.init.xavier_uniform_(self.uncertainty_conv.weight, gain=0.01)
        nn.init.constant_(self.uncertainty_conv.bias, -3.0)  # log(sigma) = -3 -> sigma ~0.05
        
    def create_coordinate_channels(self, height, width, device):
        """Create coordinate channels for spatial awareness."""
        # Create meshgrid
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        return x_grid, y_grid
        
    def create_gaze_heatmap(self, gaze_x, gaze_y, height, width, device, sigma_px=2.5):
        """Create Gaussian heatmap at gaze location."""
        # convert pixel sigma to normalized units in [-1, 1]
        sigma = (2.0 * sigma_px) / (max(height, width) - 1)
        # gaze_x, gaze_y are in normalized coordinates [-1, 1]
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Expand for batch
        y_grid = y_grid.unsqueeze(0).expand(gaze_x.shape[0], -1, -1)
        x_grid = x_grid.unsqueeze(0).expand(gaze_x.shape[0], -1, -1)
        
        # Compute Gaussian
        gaze_x = gaze_x.view(-1, 1, 1)
        gaze_y = gaze_y.view(-1, 1, 1)
        
        dist_sq = (x_grid - gaze_x)**2 + (y_grid - gaze_y)**2
        heatmap = torch.exp(-dist_sq / (2 * sigma**2))
        
        return heatmap.unsqueeze(1)  # Add channel dimension
        
    def roi_align_to_patch(self, ctx_44, gaze_x, gaze_y, img_size=1408, patch_size=88):
        """
        Align context features to the same gaze-centered window as the RGB patch,
        using reflection padding to match dataset extraction. Returns [B, C, 22, 22].
        """
        B, C, H, W = ctx_44.shape  # H=W=44
        device = ctx_44.device

        # 1) Map normalized gaze [-1,1] -> original pixels
        cx = ((gaze_x + 1.0) * 0.5) * (img_size - 1)
        cy = ((gaze_y + 1.0) * 0.5) * (img_size - 1)

        # 2) ROI bounds in original pixels (fractional is fine)
        half = patch_size / 2.0
        x1 = cx - half
        y1 = cy - half
        x2 = cx + half
        y2 = cy + half

        # 3) Convert ROI to feature-map coordinates (44x44 corresponds to img_size)
        scale = H / float(img_size)  # 44 / 1408 = 1/32
        x1f = x1 * scale
        y1f = y1 * scale
        x2f = x2 * scale
        y2f = y2 * scale

        # 4) Build sampling grid for 22x22 crop
        ys = torch.linspace(0, 1, steps=22, device=device).view(1, 22, 1).expand(B, -1, 22)
        xs = torch.linspace(0, 1, steps=22, device=device).view(1, 1, 22).expand(B, 22, -1)
        x = x1f.view(B, 1, 1) * (1 - xs) + x2f.view(B, 1, 1) * xs
        y = y1f.view(B, 1, 1) * (1 - ys) + y2f.view(B, 1, 1) * ys

        # Normalize to [-1,1] for grid_sample with align_corners=True
        xn = 2.0 * (x / (W - 1)) - 1.0
        yn = 2.0 * (y / (H - 1)) - 1.0
        grid = torch.stack([xn, yn], dim=-1)  # [B, 22, 22, 2]

        # 5) Sample with reflection padding to match dataset patch behavior
        return F.grid_sample(ctx_44, grid, mode='bilinear', align_corners=True, padding_mode='reflection')
        
    def forward(self, context_rgb, patch_rgb, gaze_x, gaze_y):
        B = context_rgb.shape[0]
        device = context_rgb.device
        
        # Create coordinate channels for context
        x_coord, y_coord = self.create_coordinate_channels(88, 88, device)
        x_coord = x_coord.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        y_coord = y_coord.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        # Create gaze heatmap
        gaze_heatmap = self.create_gaze_heatmap(gaze_x, gaze_y, 88, 88, device)
        
        # Distance to gaze channel (design requirement)
        gx = gaze_x.view(B, 1, 1, 1).expand(B, 1, 88, 88)  # already in [-1,1]
        gy = gaze_y.view(B, 1, 1, 1).expand(B, 1, 88, 88)
        r_to_gaze = torch.sqrt((x_coord - gx)**2 + (y_coord - gy)**2) / (2.0 ** 0.5)  # Use Python float instead of numpy
        
        # Concatenate context input
        context_input = torch.cat([context_rgb, gaze_heatmap, x_coord, y_coord, r_to_gaze], dim=1)
        
        # Context encoder with multi-scale gaze injection
        c0_dw = self.context_conv0_dw(context_input)
        c0 = F.relu(self.context_gn0(self.context_conv0_pw(c0_dw)))
        c0_down = self.context_blur0(c0)
        c0_down = self.gaze_inject_44(c0_down, torch.stack([gaze_x, gaze_y], dim=1))
        
        # Build relative coordinate channels for patch (delta_x, delta_y)
        H = W = 88
        dy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)
        dx = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)
        delta_x = dx.expand(B, 1, H, W)
        delta_y = dy.expand(B, 1, H, W)
        patch_input = torch.cat([patch_rgb, delta_x, delta_y], dim=1)  # 5 channels
        
        # Patch encoder with anti-aliased downsampling
        p0_dw = self.patch_conv0_dw(patch_input)
        p0 = F.relu(self.patch_gn0(self.patch_conv0_pw(p0_dw)))  # 88x88
        p0_down = self.patch_blur0(p0)  # 88 -> 44
        p1_dw = self.patch_conv1_dw(p0_down)
        p1 = F.relu(self.patch_gn1(self.patch_conv1_pw(p1_dw)))    # 44x44
        p1_down = self.patch_blur1(p1)  # 44 -> 22
        p2_dw = self.patch_conv2_dw(p1_down)
        p2 = F.relu(self.patch_gn2(self.patch_conv2_pw(p2_dw)))    # 22x22
        p3_dw = self.patch_conv3_dw(p2)
        p3 = F.relu(self.patch_gn3(self.patch_conv3_pw(p3_dw)))    # 22x22
        
        # No skip connections in 22-only design
        
        # RoIAlign from 44x44 context features to extract patch-aligned context
        ctx_roi = self.roi_align_to_patch(c0_down, gaze_x, gaze_y)  # [B, 96, 22, 22]
        ctx_roi = self.ctx_proj(ctx_roi)  # Project to 288 channels [B, 288, 22, 22]
        
        # Pointwise fusion only (as specified)
        fused = torch.cat([ctx_roi, p3], dim=1)  # [B, 384, 22, 22]
        fused = self.fuse_1x1(fused)  # [B, 128, 22, 22]
        fused = self.dropout(fused)  # Apply dropout after fusion
        
        # Lightweight decoder at 22x22 only
        d_dw = self.decode_dw(fused)
        d = F.relu(self.decode_gn(self.decode_pw(d_dw)))  # [B, 64, 22, 22]
        d = self.head_dropout(d)  # Apply dropout before final layers
        d_final = self.decode_final(d)  # [B, 32, 22, 22]
        
        # Final predictions at 22x22
        depth = self.pred_conv(d_final)
        depth = F.softplus(depth) + 0.1  # Ensure positive depth
        
        log_sigma = self.uncertainty_conv(d_final)  # Output log(sigma) directly
        
        return depth, log_sigma


if __name__ == '__main__':
    # Test the model
    model = SpatialDualResolutionGazeDepth()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    B = 2
    context_rgb = torch.randn(B, 3, 88, 88)
    patch_rgb = torch.randn(B, 3, 88, 88)
    gaze_x = torch.rand(B) * 2 - 1  # [-1, 1]
    gaze_y = torch.rand(B) * 2 - 1  # [-1, 1]
    
    depth, uncertainty = model(context_rgb, patch_rgb, gaze_x, gaze_y)
    print(f"Depth shape: {depth.shape}")  # Should be [2, 1, 22, 22]
    print(f"Uncertainty shape: {uncertainty.shape}")  # Should be [2, 1, 22, 22]
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")