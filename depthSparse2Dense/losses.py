import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfSupervisedDepthLoss(nn.Module):
    """Self-supervised loss for sparse-to-dense depth completion"""
    
    def __init__(self, alpha=0.85, sparse_weight=0.5, smooth_weight=0.1, regularization_weight=0.1, gt_weight=0.0):
        super().__init__()
        self.alpha = alpha  # SSIM vs L1 weight
        self.sparse_weight = sparse_weight
        self.smooth_weight = smooth_weight
        self.regularization_weight = regularization_weight
        self.gt_weight = gt_weight  # Weight for ground truth supervision
        
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
        """Enhanced edge-aware smoothness loss with second-order terms"""
        # First order gradients
        depth_dx = depth[:, :, :, :-1] - depth[:, :, :, 1:]
        depth_dy = depth[:, :, :-1, :] - depth[:, :, 1:, :]
        
        # Second order gradients for better propagation
        depth_dxx = depth_dx[:, :, :, :-1] - depth_dx[:, :, :, 1:]
        depth_dyy = depth_dy[:, :, :-1, :] - depth_dy[:, :, 1:, :]
        
        # Image gradients for edge-aware weighting
        image_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
        image_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)
        
        # Stronger edge-aware weighting
        weight_x = torch.exp(-10 * image_dx)
        weight_y = torch.exp(-10 * image_dy)
        
        # First order smoothness
        smoothness_x = (torch.abs(depth_dx[:, :, :, :-1]) * weight_x[:, :, :, :-1]).mean()
        smoothness_y = (torch.abs(depth_dy[:, :, :-1, :]) * weight_y[:, :, :-1, :]).mean()
        
        # Second order smoothness (penalizes curvature)
        smooth_2nd_x = (torch.abs(depth_dxx) * weight_x[:, :, :, :-1]).mean()
        smooth_2nd_y = (torch.abs(depth_dyy) * weight_y[:, :, :-1, :]).mean()
        
        return smoothness_x + smoothness_y + 0.5 * (smooth_2nd_x + smooth_2nd_y)
    
    def depth_regularization_loss(self, pred_depth, sparse_depth):
        """Regularize predicted depth to match sparse depth statistics"""
        mask = (sparse_depth > 0)
        if mask.sum() < 10:  # Need enough points for statistics
            return torch.tensor(0.0).to(pred_depth.device)
        
        # Get statistics from sparse points
        sparse_values = sparse_depth[mask]
        sparse_mean = sparse_values.mean()
        sparse_std = sparse_values.std()
        
        # For predicted depth, sample around sparse points
        # Create dilated mask to sample predictions near sparse points
        from torch.nn.functional import max_pool2d
        mask_float = mask.float()
        dilated_mask = max_pool2d(mask_float, kernel_size=21, stride=1, padding=10)
        dilated_mask = dilated_mask > 0
        
        # Get predictions in neighborhood of sparse points
        pred_near_sparse = pred_depth[dilated_mask]
        if pred_near_sparse.numel() == 0:
            return torch.tensor(0.0).to(pred_depth.device)
            
        pred_mean = pred_near_sparse.mean()
        pred_std = pred_near_sparse.std()
        
        # Softer penalty with clamping
        mean_diff = torch.clamp(torch.abs(pred_mean - sparse_mean) / (sparse_mean + 0.1), max=2.0)
        std_diff = torch.clamp(torch.abs(pred_std - sparse_std) / (sparse_std + 0.1), max=2.0)
        
        return 0.1 * (mean_diff + 0.5 * std_diff)
    
    def ground_truth_loss(self, pred_depth, gt_depth):
        """Supervised loss using ground truth depth"""
        # Only compute loss where GT is valid
        valid_mask = gt_depth > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0).to(pred_depth.device)
        
        # L1 loss on all valid pixels
        return F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
    
    def forward(self, pred_depth, sparse_depth, rgb_current, rgb_warped, gt_depth=None):
        """
        Args:
            pred_depth: Predicted dense depth [B, 1, H, W]
            sparse_depth: Input sparse depth from SLAM [B, 1, H, W]
            rgb_current: Current frame RGB [B, 3, H, W]
            rgb_warped: Warped frame using predicted depth [B, 3, H, W]
            gt_depth: Optional ground truth depth [B, 1, H, W]
        """
        loss_dict = {}
        
        # Ground truth loss (if available and weight > 0)
        if gt_depth is not None and self.gt_weight > 0:
            gt_loss = self.ground_truth_loss(pred_depth, gt_depth)
            loss_dict['ground_truth'] = gt_loss.item()
        else:
            gt_loss = torch.tensor(0.0).to(pred_depth.device)
            loss_dict['ground_truth'] = 0.0
        
        # Photometric loss (main self-supervision)
        photo_loss = self.photometric_loss(rgb_current, rgb_warped)
        loss_dict['photometric'] = photo_loss.item()
        
        # Sparse depth consistency
        sparse_loss = self.sparse_depth_loss(pred_depth, sparse_depth)
        loss_dict['sparse'] = sparse_loss.item()
        
        # Enhanced edge-aware smoothness
        smooth_loss = self.smoothness_loss(pred_depth, rgb_current)
        loss_dict['smoothness'] = smooth_loss.item()
        
        # Depth statistics regularization
        reg_loss = self.depth_regularization_loss(pred_depth, sparse_depth)
        loss_dict['regularization'] = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0
        
        # Combined loss
        total_loss = self.gt_weight * gt_loss + \
                     photo_loss + \
                     self.sparse_weight * sparse_loss + \
                     self.smooth_weight * smooth_loss + \
                     self.regularization_weight * reg_loss
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


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