import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfSupervisedDepthLoss(nn.Module):
    """Self-supervised loss for sparse-to-dense depth completion"""
    
    def __init__(self, alpha=0.85, sparse_weight=0.5, smooth_weight=0.1):
        super().__init__()
        self.alpha = alpha  # SSIM vs L1 weight
        self.sparse_weight = sparse_weight
        self.smooth_weight = smooth_weight
        
    def photometric_loss(self, target_img, warped_img):
        """Combination of L1 and SSIM loss"""
        l1_loss = torch.abs(target_img - warped_img).mean()
        ssim_loss = self.compute_ssim(target_img, warped_img).mean()
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss
    
    def compute_ssim(self, x, y):
        """Simplified SSIM computation"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, padding=1)
        mu_y = F.avg_pool2d(y, 3, 1, padding=1)
        
        sigma_x = F.avg_pool2d(x**2, 3, 1, padding=1) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1, padding=1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1, padding=1) - mu_x*mu_y
        
        ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        
        return (1 - ssim) / 2
    
    def sparse_depth_loss(self, pred_depth, sparse_depth):
        """Ensure predicted depth matches input sparse points"""
        mask = (sparse_depth > 0)
        if mask.sum() == 0:
            return torch.tensor(0.0).to(pred_depth.device)
        
        return F.l1_loss(pred_depth[mask], sparse_depth[mask])
    
    def smoothness_loss(self, depth, image):
        """Edge-aware smoothness loss"""
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        image_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        image_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        
        # Reduce depth gradients where image gradients are small
        weight_x = torch.exp(-torch.mean(image_dx, dim=1, keepdim=True))
        weight_y = torch.exp(-torch.mean(image_dy, dim=1, keepdim=True))
        
        smoothness_x = (depth_dx * weight_x).mean()
        smoothness_y = (depth_dy * weight_y).mean()
        
        return smoothness_x + smoothness_y
    
    def forward(self, pred_depth, sparse_depth, rgb_current, rgb_warped):
        """
        Args:
            pred_depth: Predicted dense depth [B, 1, H, W]
            sparse_depth: Input sparse depth from SLAM [B, 1, H, W]
            rgb_current: Current frame RGB [B, 3, H, W]
            rgb_warped: Warped frame using predicted depth [B, 3, H, W]
        """
        # Photometric loss (main self-supervision)
        photo_loss = self.photometric_loss(rgb_current, rgb_warped)
        
        # Sparse depth consistency
        sparse_loss = self.sparse_depth_loss(pred_depth, sparse_depth)
        
        # Edge-aware smoothness
        smooth_loss = self.smoothness_loss(pred_depth, rgb_current)
        
        total_loss = photo_loss + \
                     self.sparse_weight * sparse_loss + \
                     self.smooth_weight * smooth_loss
        
        return total_loss, {
            'photometric': photo_loss.item(),
            'sparse': sparse_loss.item(),
            'smoothness': smooth_loss.item(),
            'total': total_loss.item()
        }


def warp_frame(depth, ref_img, K, K_inv, T_curr_to_ref):
    """
    Warp reference image to current view using depth and relative pose
    
    Args:
        depth: Dense depth map [B, 1, H, W]
        ref_img: Reference image [B, 3, H, W]
        K: Camera intrinsics [B, 3, 3]
        K_inv: Inverse intrinsics [B, 3, 3]
        T_curr_to_ref: Transform from current to reference [B, 4, 4]
    """
    B, _, H, W = depth.shape
    
    # Create pixel grid
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=0).float()
    pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1).to(depth.device)
    
    # Backproject to 3D using depth
    cam_coords = torch.matmul(K_inv.unsqueeze(1).unsqueeze(1), 
                              pixel_coords.reshape(B, 3, -1))
    cam_coords = cam_coords.reshape(B, 3, H, W)
    cam_coords = cam_coords * depth
    
    # Transform to reference frame
    cam_coords_hom = torch.cat([cam_coords, 
                                torch.ones(B, 1, H, W).to(depth.device)], dim=1)
    cam_coords_hom = cam_coords_hom.reshape(B, 4, -1)
    ref_coords = torch.matmul(T_curr_to_ref, cam_coords_hom)
    ref_coords = ref_coords[:, :3, :].reshape(B, 3, H, W)
    
    # Project to reference image
    ref_coords = torch.matmul(K.unsqueeze(1).unsqueeze(1),
                             ref_coords.reshape(B, 3, -1))
    ref_coords = ref_coords.reshape(B, 3, H, W)
    
    # Normalize pixel coordinates
    ref_x = ref_coords[:, 0, :, :] / (ref_coords[:, 2, :, :] + 1e-7)
    ref_y = ref_coords[:, 1, :, :] / (ref_coords[:, 2, :, :] + 1e-7)
    
    # Normalize to [-1, 1] for grid_sample
    ref_x = 2.0 * ref_x / (W - 1) - 1.0
    ref_y = 2.0 * ref_y / (H - 1) - 1.0
    
    grid = torch.stack([ref_x, ref_y], dim=-1)
    
    # Sample reference image
    warped = F.grid_sample(ref_img, grid, mode='bilinear', 
                          padding_mode='zeros', align_corners=False)
    
    # Create valid mask (pixels visible in both views)
    valid_mask = (ref_x > -1) & (ref_x < 1) & (ref_y > -1) & (ref_y < 1)
    valid_mask = valid_mask.unsqueeze(1).float()
    
    return warped, valid_mask