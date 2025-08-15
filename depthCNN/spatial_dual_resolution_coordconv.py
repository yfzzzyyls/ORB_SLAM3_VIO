import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        
    def forward(self, x):
        # Global average pooling
        w = F.adaptive_avg_pool2d(x, 1)
        # Squeeze and excitation
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Reweight channels
        return x * w


class GazeBiasedCrossAttention(nn.Module):
    """Cross-attention fusion with gaze bias and grouped-query attention (GQA).
    
    Uses 4 Q heads but only 2 KV heads to save parameters while maintaining expressiveness.
    """
    def __init__(self, c_in_q=288, c_in_kv=288, d_model=160, h_q=4, h_kv=2, c_out=192):
        super().__init__()
        self.hq, self.hkv = h_q, h_kv
        self.d_head = d_model // h_q
        
        # Projections (GQA: smaller KV, no bias as they're followed by norm)
        self.q_proj = nn.Conv2d(c_in_q, d_model, 1, bias=False)  # 288->160
        self.k_proj = nn.Conv2d(c_in_kv, self.hkv * self.d_head, 1, bias=False)  # 288->80
        self.v_proj = nn.Conv2d(c_in_kv, self.hkv * self.d_head, 1, bias=False)  # 288->80
        self.o_proj = nn.Conv2d(d_model, c_out, 1, bias=False)  # 160->192
        
        # Learnable per-head temperature
        self.log_tau = nn.Parameter(torch.zeros(h_q) - 0.5)  # Initialize slightly negative for sharper attention
        
        # Relative position bias (per-head)
        size = 22
        table_size = 2 * size - 1  # 43
        self.rpb = nn.Parameter(torch.zeros(h_q, table_size, table_size))
        
        # Gaze bias strength (learnable, constrained positive via softplus)
        self._log_alpha = nn.Parameter(torch.zeros(1))  # start ~1.0 after softplus
        
        # Normalization
        self.norm_q = nn.GroupNorm(32, c_in_q)
        self.norm_kv = nn.GroupNorm(32, c_in_kv)
        
        # Precompute RPB indices once (for performance)
        coords = torch.stack(torch.meshgrid(
            torch.arange(size), torch.arange(size), indexing='ij'), dim=-1)  # [22,22,2]
        rel = coords.view(-1, 1, 2) - coords.view(1, -1, 2)  # [484, 484, 2]
        self.register_buffer("rpb_ix", (rel[..., 0] + (size - 1)).long(), persistent=False)
        self.register_buffer("rpb_iy", (rel[..., 1] + (size - 1)).long(), persistent=False)
        
        # Precompute key coordinates for gaze bias (avoid rebuilding every forward)
        H = W = 22
        gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        self.register_buffer("key_y", gy.reshape(-1).float(), persistent=False)  # [484]
        self.register_buffer("key_x", gx.reshape(-1).float(), persistent=False)  # [484]
        
        # Dropout for attention and residuals
        self.attn_drop = nn.Dropout(p=0.03)
        self.resid_drop = nn.Dropout2d(p=0.03)
        
        # Initialize
        nn.init.trunc_normal_(self.rpb, std=0.02)
        
    def forward(self, q_feat, kv_feat, gaze_xy_norm=None):
        """
        Args:
            q_feat: [B, C_q, H, W] query features (patch)
            kv_feat: [B, C_kv, H, W] key/value features (context)
            gaze_xy_norm: [B, 2] normalized gaze coordinates in [-1, 1]
        """
        B, _, H, W = q_feat.shape
        
        # Project Q, K, V
        q = self.q_proj(self.norm_q(q_feat))  # [B, 160, 22, 22]
        k = self.k_proj(self.norm_kv(kv_feat))  # [B, 80, 22, 22]
        v = self.v_proj(self.norm_kv(kv_feat))  # [B, 80, 22, 22]
        
        # Reshape to heads
        def split_heads(x, h):
            # x: [B, h*d, H, W] -> [B, h, HW, d]
            hd = x.shape[1]
            d_head = hd // h
            x = x.view(B, h, d_head, H, W).permute(0, 1, 3, 4, 2).reshape(B, h, H*W, d_head)
            return x
        
        qh = split_heads(q, self.hq)  # [B, 4, 484, 40]
        kh = split_heads(k, self.hkv)  # [B, 2, 484, 40]
        vh = split_heads(v, self.hkv)  # [B, 2, 484, 40]
        
        # Broadcast KV heads to match Q heads (GQA)
        kh = kh.repeat_interleave(self.hq // self.hkv, dim=1)  # [B, 4, 484, 40]
        vh = vh.repeat_interleave(self.hq // self.hkv, dim=1)  # [B, 4, 484, 40]
        
        # Compute attention scores with cosine similarity for stability
        # L2 normalize Q and K for cosine attention (explicit eps to avoid NaNs)
        qh = F.normalize(qh, dim=-1, eps=1e-6)
        kh = F.normalize(kh, dim=-1, eps=1e-6)
        logits = torch.einsum('bhid,bhjd->bhij', qh, kh)  # [B, 4, 484, 484]
        
        # Add relative position bias (using precomputed indices)
        rpb = self.rpb[:, self.rpb_ix, self.rpb_iy]  # [4, 484, 484]
        logits = logits + rpb.unsqueeze(0)  # Broadcast over batch
        
        # Add gaze radial bias (always compute to ensure gradient flow)
        # Apply positive alpha via softplus
        alpha = F.softplus(self._log_alpha)
        
        if gaze_xy_norm is not None:
            # Convert gaze from [-1, 1] to [0, 21] pixel coordinates
            gx = (gaze_xy_norm[:, 0] + 1.0) * (H - 1) / 2
            gy = (gaze_xy_norm[:, 1] + 1.0) * (W - 1) / 2
            
            # Use precomputed key coordinates (moved to device if needed)
            ky = self.key_y  # [484]
            kx = self.key_x  # [484]
            
            # Distance from each key position to gaze
            dist = torch.sqrt((kx[None, :] - gx[:, None])**2 + (ky[None, :] - gy[:, None])**2)
            
            gaze_bias = -alpha * dist  # Closer = less negative = higher attention
            logits = logits + gaze_bias[:, None, None, :]  # Add to keys dimension
        else:
            # For self-attention, add zero bias but ensure alpha is still computed (for gradient)
            logits = logits + 0.0 * alpha  # Ensures gradient flow through alpha
        
        # Apply temperature per head (clamped for stability)
        tau = torch.exp(self.log_tau).clamp(0.25, 4.0).view(1, self.hq, 1, 1)
        attn = (logits / tau).softmax(dim=-1)
        attn = self.attn_drop(attn)  # Small dropout on attention weights
        
        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, vh)  # [B, 4, 484, 40]
        
        # Reshape back to spatial
        out = out.reshape(B, self.hq, H, W, self.d_head).permute(0, 1, 4, 2, 3)
        out = out.reshape(B, -1, H, W)  # [B, 160, 22, 22]
        out = self.o_proj(out)  # [B, 192, 22, 22]
        
        return self.resid_drop(out)  # Apply residual dropout


class ASPPLite(nn.Module):
    """Lightweight ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context."""
    def __init__(self, c=96, rates=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=r, dilation=r, groups=c, bias=False),  # Depthwise
                nn.Conv2d(c, c, 1, bias=True),  # Pointwise
                nn.SiLU(inplace=True)  # Changed from ReLU to SiLU for better regression
            ) for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * c, c, 1, bias=False),
            nn.GroupNorm(16, c),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        xs = [branch(x) for branch in self.branches]
        x = torch.cat(xs, dim=1)  # [B, 3*96, 22, 22]
        return self.project(x)  # [B, 96, 22, 22]


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


class GaussianDownsample(nn.Module):
    """Gaussian downsampling for depth targets - replaces box average (Fix #5)."""
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Create Gaussian kernel
        sigma = scale_factor / 2.0
        kernel_size = 2 * scale_factor + 1
        kernel = self._gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)
        
    def _gaussian_kernel(self, size, sigma):
        """Create 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= (size - 1) / 2.0
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x):
        """Apply Gaussian downsampling."""
        B, C, H, W = x.shape
        assert H % self.scale_factor == 0 and W % self.scale_factor == 0
        
        # Apply Gaussian filter
        padding = self.kernel.shape[-1] // 2
        x_filtered = F.conv2d(x, self.kernel.repeat(C, 1, 1, 1), 
                              padding=padding, groups=C)
        
        # Subsample
        return x_filtered[:, :, ::self.scale_factor, ::self.scale_factor]


def create_gaussian_weight_map(size=22, sigma=3.0):
    """Create Gaussian weight map centered at (size//2, size//2) for loss weighting (Fix #4)."""
    center = size / 2.0 - 0.5  # Center of 22x22 grid
    y, x = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                          torch.arange(size, dtype=torch.float32), 
                          indexing='ij')
    
    # Distance from center
    dist_sq = (x - center)**2 + (y - center)**2
    weights = torch.exp(-dist_sq / (2 * sigma**2))
    
    # Normalize so sum = 1
    weights = weights / weights.sum()
    return weights


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
        self.context_conv0_dw = nn.Conv2d(7, 7, kernel_size=11, stride=1, padding=5, groups=7)
        self.context_conv0_pw = nn.Conv2d(7, 96, kernel_size=1)
        self.context_gn0 = nn.GroupNorm(16, 96)
        self.context_blur0 = BlurPool(channels=96, stride=2)  # 88 -> 44
        
        # Multi-scale gaze injection
        self.gaze_inject_44 = FiLMLayer(gaze_dim=2, feature_dim=96)
        
        # Patch encoder - WIDENED channels: 96->128, 192->224, 288->288
        # Input: 5 channels (RGB + delta_x + delta_y)
        self.patch_conv0_dw = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, groups=5)
        self.patch_conv0_pw = nn.Conv2d(5, 128, kernel_size=1)  # WIDENED: 96->128
        self.patch_gn0 = nn.GroupNorm(16, 128)
        self.patch_se0 = SEBlock(128, reduction=8)  # NEW: SE block
        self.patch_blur0 = BlurPool(channels=128, stride=2)  # 88 -> 44
        
        self.patch_conv1_dw = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.patch_conv1_pw = nn.Conv2d(128, 224, kernel_size=1)  # WIDENED: 192->224
        self.patch_gn1 = nn.GroupNorm(28, 224)  # Adjusted for 224 channels
        self.patch_se1 = SEBlock(224, reduction=8)  # NEW: SE block
        self.patch_blur1 = BlurPool(channels=224, stride=2)  # 44 -> 22
        
        self.patch_conv2_dw = nn.Conv2d(224, 224, kernel_size=3, stride=1, padding=1, groups=224)
        self.patch_conv2_pw = nn.Conv2d(224, 288, kernel_size=1)  # Keep 288 for fusion
        self.patch_gn2 = nn.GroupNorm(48, 288)
        
        self.patch_conv3_dw = nn.Conv2d(288, 288, kernel_size=3, stride=1, padding=1, groups=288)
        self.patch_conv3_pw = nn.Conv2d(288, 288, kernel_size=1)
        self.patch_gn3 = nn.GroupNorm(48, 288)
        self.patch_se3 = SEBlock(288, reduction=12)  # NEW: SE block with higher reduction
        
        # Project context features from RoIAlign (96ch) to match patch (288ch)
        self.ctx_proj = nn.Conv2d(96, 288, kernel_size=1, bias=False)  # No bias, followed by norm in attention
        
        # NEW: Cross-attention fusion (replaces concat + 1x1)
        self.cross_attn = GazeBiasedCrossAttention(
            c_in_q=288, c_in_kv=288, d_model=160, h_q=4, h_kv=2, c_out=192
        )
        
        # NEW: Self-attention for patch refinement (no gaze bias needed)
        self.self_attn = GazeBiasedCrossAttention(
            c_in_q=192, c_in_kv=192, d_model=160, h_q=4, h_kv=2, c_out=192
        )
        
        # NEW: Tiny FFN after attention (adds ~30k params for better mixing)
        self.post_attn_ffn = nn.Sequential(
            nn.Conv2d(192, 256, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(256, 192, 1, bias=False)
        )
        
        # NEW: Skip connection from pre-attention
        self.skip_proj = nn.Conv2d(288, 192, kernel_size=1, bias=False)  # No bias for consistency
        
        # Reduce fused features to decoder width
        self.fuse_reduce = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, bias=False),  # No bias before norm
            nn.GroupNorm(16, 96),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(p=0.10)  # After fusion
        
        # Decoder at 22x22
        self.decode_dw = nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96)
        self.decode_pw = nn.Conv2d(96, 96, kernel_size=1)
        self.decode_gn = nn.GroupNorm(16, 96)
        
        # NEW: ASPP-lite for multi-scale context
        self.aspp = ASPPLite(c=96, rates=(1, 2, 3))
        
        self.head_dropout = nn.Dropout2d(p=0.10)
        
        # Final channel reduction
        self.decode_final = nn.Conv2d(96, 48, kernel_size=1)
        
        # Final prediction
        self.pred_conv = nn.Conv2d(48, 1, kernel_size=1)
        
        # Uncertainty prediction
        self.uncertainty_conv = nn.Conv2d(48, 1, kernel_size=1)
        
        # NEW: Precise K×K gaze sampling head (multi-scale point query)
        self.gaze_k = 5  # K×K neighborhood sampling
        
        # Lightweight channel projections (keep params < 1M)
        self.gaze_proj_p0 = nn.Conv2d(128, 16, 1)  # 88×88 features → 16 channels
        self.gaze_proj_p1 = nn.Conv2d(224, 16, 1)  # 44×44 features → 16 channels
        self.gaze_proj_d = nn.Conv2d(48, 16, 1)    # 22×22 features → 16 channels
        
        # Small MLP for gaze depth prediction
        # Input: 3 scales × 16 channels × 5×5 + 2 gaze coords = 1202
        self.gaze_mlp = nn.Sequential(
            nn.Linear(3 * 16 * 5 * 5 + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # NEW: Register Gaussian weight map for loss (Fix #4)
        self.register_buffer('gaze_weights', create_gaussian_weight_map(22, sigma=3.0))
        
        # NEW: Gaussian downsampler for targets (Fix #5)
        self.gaussian_downsample = GaussianDownsample(scale_factor=4)
        
        # Initialize weights
        self._init_weights()
        
        # Zero-init last conv of FFN for identity start
        nn.init.zeros_(self.post_attn_ffn[-1].weight)
        
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
        
    def sample_kxk_at_gaze(self, feat, K=5):
        """
        Sample K×K patch centered at gaze location.
        Since our patch is already gaze-centered, sample around the center.
        Uses grid_sample with align_corners=False for precise sub-pixel alignment.
        """
        B, C, H, W = feat.shape
        device = feat.device
        
        # Create K×K grid centered at origin (gaze is at center of our gaze-centered patches)
        # Scale the grid based on feature map size
        ks = torch.linspace(-(K-1)/2, (K-1)/2, K, device=device) * 2.0 / H
        grid_y, grid_x = torch.meshgrid(ks, ks, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [K, K, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, K, K, 2]
        
        # Sample with bilinear interpolation
        sampled = F.grid_sample(
            feat, grid,
            mode='bilinear',
            align_corners=False,  # Critical for correct sub-pixel alignment
            padding_mode='reflection'
        )
        
        return sampled  # [B, C, K, K]
    
    def create_gaze_heatmap(self, gaze_x, gaze_y, height, width, device, sigma_px=2.5):
        """Create Gaussian heatmap at gaze location."""
        # convert pixel sigma to normalized units in [-1, 1] with align_corners=False
        sigma = (2.0 * sigma_px) / max(height, width)
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

        # 1) Map normalized gaze [-1,1] -> original pixels with align_corners=False
        cx = ((gaze_x + 1.0) * 0.5) * img_size - 0.5
        cy = ((gaze_y + 1.0) * 0.5) * img_size - 0.5

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

        # 4) Build sampling grid for 22x22 crop at BIN CENTERS (align_corners=False)
        # Sample at centers, not endpoints, to match dataset
        i = torch.arange(22, device=device, dtype=torch.float32)
        x = x1f[:, None, None] + (i + 0.5)[None, None, :] * (x2f - x1f)[:, None, None] / 22
        y = y1f[:, None, None] + (i + 0.5)[None, :, None] * (y2f - y1f)[:, None, None] / 22
        
        # Expand to [B, 22, 22]
        x = x.expand(-1, 22, -1)
        y = y.expand(-1, -1, 22)

        # Normalize to [-1,1] for grid_sample with align_corners=False
        # x and y are bin centers in feature space, normalize with +0.5
        xn = 2.0 * (x + 0.5) / W - 1.0
        yn = 2.0 * (y + 0.5) / H - 1.0
        grid = torch.stack([xn, yn], dim=-1)  # [B, 22, 22, 2]

        # 5) Sample with reflection padding to match dataset patch behavior
        return F.grid_sample(ctx_44, grid, mode='bilinear', align_corners=False, padding_mode='reflection')
        
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
        
        # Patch encoder with SE blocks and anti-aliased downsampling
        p0_dw = self.patch_conv0_dw(patch_input)
        p0 = F.relu(self.patch_gn0(self.patch_conv0_pw(p0_dw)))  # 88x88, 128ch
        p0 = self.patch_se0(p0)  # Apply SE block
        p0_down = self.patch_blur0(p0)  # 88 -> 44
        
        p1_dw = self.patch_conv1_dw(p0_down)
        p1 = F.relu(self.patch_gn1(self.patch_conv1_pw(p1_dw)))  # 44x44, 224ch
        p1 = self.patch_se1(p1)  # Apply SE block
        p1_down = self.patch_blur1(p1)  # 44 -> 22
        
        p2_dw = self.patch_conv2_dw(p1_down)
        p2 = F.relu(self.patch_gn2(self.patch_conv2_pw(p2_dw)))  # 22x22, 288ch
        
        p3_dw = self.patch_conv3_dw(p2)
        p3 = F.relu(self.patch_gn3(self.patch_conv3_pw(p3_dw)))  # 22x22, 288ch
        p3 = self.patch_se3(p3)  # Apply SE block
        
        # RoIAlign from 44x44 context features to extract patch-aligned context
        ctx_roi = self.roi_align_to_patch(c0_down, gaze_x, gaze_y)  # [B, 96, 22, 22]
        ctx_roi = self.ctx_proj(ctx_roi)  # Project to 288 channels [B, 288, 22, 22]
        
        # NEW: Cross-attention fusion (patch queries context)
        gaze_coords = torch.stack([gaze_x, gaze_y], dim=1)  # [B, 2]
        fused_ca = self.cross_attn(p3, ctx_roi, gaze_coords)  # [B, 192, 22, 22]
        fused_ca = self.dropout(fused_ca)
        
        # NEW: Self-attention refinement (no gaze bias for self-attention)
        fused_sa = self.self_attn(fused_ca, fused_ca, None)  # [B, 192, 22, 22]
        
        # NEW: Apply FFN after attention (transformer-style MLP)
        # Note: residual dropout is already applied inside self_attn via resid_drop
        fused_sa = fused_sa + self.post_attn_ffn(fused_sa)
        
        # NEW: Add skip connection from pre-attention
        skip = self.skip_proj(p3)  # [B, 288, 22, 22] -> [B, 192, 22, 22]
        fused_sa = fused_sa + skip
        
        # Reduce to decoder width
        fused = self.fuse_reduce(fused_sa)  # [B, 96, 22, 22]
        
        # Decoder at 22x22
        d_dw = self.decode_dw(fused)
        d = F.relu(self.decode_gn(self.decode_pw(d_dw)))  # [B, 96, 22, 22]
        
        # NEW: Apply ASPP for multi-scale context
        d = self.aspp(d)  # [B, 96, 22, 22]
        
        d = self.head_dropout(d)
        d_final = self.decode_final(d)  # [B, 48, 22, 22]
        
        # Final predictions at 22x22
        depth = self.pred_conv(d_final)
        depth = F.softplus(depth) + 0.1  # Ensure positive depth
        
        log_sigma = self.uncertainty_conv(d_final)  # Output log(sigma) directly
        
        # NEW: Precise multi-scale gaze depth prediction
        # Project features to lightweight channels
        p0_proj = self.gaze_proj_p0(p0)  # [B, 16, 88, 88]
        p1_proj = self.gaze_proj_p1(p1)  # [B, 16, 44, 44]
        d_proj = self.gaze_proj_d(d_final)  # [B, 16, 22, 22]
        
        # Sample K×K neighborhoods at gaze location from each scale
        p0_sample = self.sample_kxk_at_gaze(p0_proj, self.gaze_k)  # [B, 16, 5, 5]
        p1_sample = self.sample_kxk_at_gaze(p1_proj, self.gaze_k)  # [B, 16, 5, 5]
        d_sample = self.sample_kxk_at_gaze(d_proj, self.gaze_k)    # [B, 16, 5, 5]
        
        # Flatten and concatenate with gaze coordinates
        gaze_features = torch.cat([
            p0_sample.flatten(1),  # 16*5*5 = 400
            p1_sample.flatten(1),  # 16*5*5 = 400
            d_sample.flatten(1),   # 16*5*5 = 400
            gaze_x.view(B, 1),     # 1
            gaze_y.view(B, 1)      # 1
        ], dim=1)  # Total: 1202
        
        # MLP to predict exact gaze depth
        gaze_depth = self.gaze_mlp(gaze_features)  # [B, 1]
        gaze_depth = F.softplus(gaze_depth) + 0.1  # [B, 1] with minimum depth
        
        return depth, log_sigma, gaze_depth


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
    
    depth, uncertainty, gaze_depth = model(context_rgb, patch_rgb, gaze_x, gaze_y)
    print(f"Depth shape: {depth.shape}")  # Should be [2, 1, 22, 22]
    print(f"Uncertainty shape: {uncertainty.shape}")  # Should be [2, 1, 22, 22]
    print(f"Gaze depth shape: {gaze_depth.shape}")  # Should be [2, 1]
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Gaze depth: {gaze_depth}")
    
    # Test Gaussian downsampling
    test_depth = torch.randn(2, 1, 88, 88).abs()
    downsampler = GaussianDownsample(scale_factor=4)
    downsampled = downsampler(test_depth)
    print(f"Downsampled shape: {downsampled.shape}")  # Should be [2, 1, 22, 22]
    print(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")