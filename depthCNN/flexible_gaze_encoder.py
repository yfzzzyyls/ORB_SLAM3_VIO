#!/usr/bin/env python3
"""
Flexible encoder for gaze-only depth prediction supporting various image sizes.
Based on the proven lightweight_gaze_encoder.py but with flexible resolution support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class FlexibleGazeEncoder(nn.Module):
    """
    Efficient encoder supporting flexible input sizes for gaze-based depth prediction.
    """
    
    def __init__(self, num_levels: int = 3, base_channels: int = 32, image_size: int = 88):
        """
        Args:
            num_levels: Number of encoding levels (2 or 3 recommended)
            base_channels: Base number of channels (doubled at each level)
            image_size: Input image size (square images assumed)
        """
        super().__init__()
        
        assert num_levels in [2, 3, 4, 5], "Only 2-5 levels supported"
        self.num_levels = num_levels
        self.base_channels = base_channels
        self.image_size = image_size
        
        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Level 1: size -> size/2
        self.level1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Level 2: size/2 -> size/4
        self.level2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # Level 3: size/4 -> size/8 (optional)
        if num_levels >= 3:
            self.level3 = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
        
        # Level 4: size/8 -> size/16 (optional)
        if num_levels >= 4:
            self.level4 = nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            )
        
        # Level 5: size/16 -> size/32 (optional)
        if num_levels >= 5:
            self.level5 = nn.Sequential(
                nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels * 16),
                nn.ReLU(inplace=True)
            )
        
        # Output channel counts for each level
        self.num_ch_enc = []
        channel_multipliers = [1, 2, 4, 8, 16]  # For levels 1-5
        for i in range(num_levels):
            self.num_ch_enc.append(base_channels * channel_multipliers[i])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with Kaiming for ReLU activation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: [B, 3, H, W] input RGB image (H=W=image_size)
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # Initial processing
        x = self.input_conv(x)  # [B, C, H, W]
        
        # Level 1
        x1 = self.level1(x)  # [B, C, H/2, W/2]
        features.append(x1)
        
        # Level 2
        x2 = self.level2(x1)  # [B, 2C, H/4, W/4]
        features.append(x2)
        
        # Level 3 (if enabled)
        if self.num_levels >= 3:
            x3 = self.level3(x2)  # [B, 4C, H/8, W/8]
            features.append(x3)
        
        # Level 4 (if enabled)
        if self.num_levels >= 4:
            x4 = self.level4(x3)  # [B, 8C, H/16, W/16]
            features.append(x4)
        
        # Level 5 (if enabled)
        if self.num_levels >= 5:
            x5 = self.level5(x4)  # [B, 16C, H/32, W/32]
            features.append(x5)
        
        return features


class FlexibleGazeOnlyDepth(nn.Module):
    """
    Complete flexible model for gaze-only depth prediction supporting various image sizes.
    Now with multi-task learning capabilities.
    """
    
    def __init__(
        self,
        num_encoder_levels: int = 3,
        base_channels: int = 32,
        gaze_feature_dim: int = 64,
        image_size: int = 88,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        use_multi_scale_supervision: bool = True,
        use_multi_task: bool = False
    ):
        """
        Args:
            num_encoder_levels: Number of encoding levels (2 or 3)
            base_channels: Base channels for encoder
            gaze_feature_dim: Dimension for each scale's gaze features
            image_size: Input image size (square images assumed)
            max_depth: Maximum depth value
            min_depth: Minimum depth value
            use_multi_scale_supervision: Whether to use auxiliary losses
            use_multi_task: Whether to use multi-task learning with patch statistics
        """
        super().__init__()
        
        self.num_encoder_levels = num_encoder_levels
        self.image_size = image_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_multi_scale_supervision = use_multi_scale_supervision
        self.use_multi_task = use_multi_task
        
        # Flexible encoder
        self.encoder = FlexibleGazeEncoder(
            num_levels=num_encoder_levels,
            base_channels=base_channels,
            image_size=image_size
        )
        
        # Gaze feature extractors for each scale
        self.gaze_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, gaze_feature_dim),
                nn.LayerNorm(gaze_feature_dim),
                nn.ReLU(inplace=True)
            ) for ch in self.encoder.num_ch_enc
        ])
        
        # Main depth predictor
        total_feature_dim = num_encoder_levels * gaze_feature_dim
        self.depth_predictor = nn.Sequential(
            nn.Linear(total_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            
            nn.Linear(32, 1)
        )
        
        # Multi-task prediction heads
        if use_multi_task:
            # Shared feature processing for auxiliary tasks
            self.aux_feature_processor = nn.Sequential(
                nn.Linear(total_feature_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
            
            # Mean depth predictor
            self.mean_predictor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1)
            )
            
            # Std depth predictor
            self.std_predictor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1)
            )
            
            # Gradient magnitude predictor
            self.gradient_predictor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1)
            )
            
            # Edge score predictor (binary classification)
            self.edge_predictor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1)
            )
            
            # Depth bin classifier (5 classes: 0-2m, 2-4m, 4-6m, 6-8m, 8m+)
            self.depth_bin_predictor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 5)
            )
        
        # Auxiliary predictors for multi-scale supervision (simpler)
        if use_multi_scale_supervision:
            self.aux_predictors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(gaze_feature_dim, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 1)
                ) for _ in range(num_encoder_levels - 1)  # All scales except first
            ])
        
        # Initialize output layer for reasonable depth predictions
        self._init_depth_output()
    
    def _init_depth_output(self):
        """Initialize depth output to predict reasonable initial values."""
        with torch.no_grad():
            # Main predictor
            output_layer = self.depth_predictor[-1]
            nn.init.normal_(output_layer.weight, mean=0, std=0.01)
            # Initialize to predict ~2m depth
            init_bias = torch.log(torch.tensor(2.0 / self.max_depth))
            nn.init.constant_(output_layer.bias, init_bias.item())
            
            # Multi-task predictors
            if self.use_multi_task:
                # Mean predictor - initialize to predict around 2m
                output_layer = self.mean_predictor[-1]
                nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                nn.init.constant_(output_layer.bias, init_bias.item())
                
                # Std predictor - initialize to predict small std
                output_layer = self.std_predictor[-1]
                nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                nn.init.constant_(output_layer.bias, -2.0)  # sigmoid(-2) ≈ 0.12
                
                # Gradient predictor - initialize to predict small gradients
                output_layer = self.gradient_predictor[-1]
                nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                nn.init.constant_(output_layer.bias, -2.0)
                
                # Edge predictor - initialize to predict non-edge
                output_layer = self.edge_predictor[-1]
                nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                nn.init.constant_(output_layer.bias, -2.0)  # sigmoid(-2) ≈ 0.12
                
                # Depth bin predictor - no special initialization needed
                output_layer = self.depth_bin_predictor[-1]
                nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                nn.init.constant_(output_layer.bias, 0.0)
            
            # Auxiliary predictors for multi-scale supervision
            if hasattr(self, 'aux_predictors'):
                for aux_pred in self.aux_predictors:
                    output_layer = aux_pred[-1]
                    nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                    nn.init.constant_(output_layer.bias, init_bias.item())
    
    def extract_gaze_features(
        self,
        feature_map: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_y: torch.Tensor
    ) -> torch.Tensor:
        """Extract features at gaze location using bilinear interpolation."""
        B, C, H, W = feature_map.shape
        
        # Normalize gaze coordinates to [-1, 1] for grid_sample
        gaze_norm_x = 2.0 * gaze_x / (W - 1) - 1.0
        gaze_norm_y = 2.0 * gaze_y / (H - 1) - 1.0
        
        # Create sampling grid
        grid = torch.stack([gaze_norm_x, gaze_norm_y], dim=-1)  # [B, 2]
        grid = grid.view(B, 1, 1, 2)  # [B, 1, 1, 2]
        
        # Sample features
        sampled = F.grid_sample(
            feature_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return sampled.squeeze(2).squeeze(2)  # [B, C]
    
    def forward(
        self,
        rgb: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for flexible gaze-only depth prediction.
        
        Args:
            rgb: [B, 3, H, W] input RGB image (H=W=image_size)
            gaze_x: [B] gaze x coordinates
            gaze_y: [B] gaze y coordinates
            
        Returns:
            Dictionary with 'depth' and optionally 'aux_depths' and multi-task predictions
        """
        # Encode image
        features = self.encoder(rgb)
        
        # Extract and project gaze features from each scale
        gaze_features = []
        for i, (feat, proj) in enumerate(zip(features, self.gaze_projections)):
            # Scale gaze coordinates for this level
            scale_factor = 2 ** (i + 1)  # 2, 4, 8 for levels 1, 2, 3
            scaled_gaze_x = gaze_x / scale_factor
            scaled_gaze_y = gaze_y / scale_factor
            
            # Extract features at gaze
            gaze_feat = self.extract_gaze_features(feat, scaled_gaze_x, scaled_gaze_y)
            
            # Project to common dimension
            projected = proj(gaze_feat)
            gaze_features.append(projected)
        
        # Concatenate all gaze features
        combined_features = torch.cat(gaze_features, dim=1)
        
        # Predict depth
        depth_logit = self.depth_predictor(combined_features)
        depth = torch.sigmoid(depth_logit) * self.max_depth
        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
        
        outputs = {'depth': depth}
        
        # Multi-task predictions
        if self.use_multi_task and self.training:
            # Process features for auxiliary tasks
            aux_features = self.aux_feature_processor(combined_features)
            
            # Mean depth prediction
            mean_logit = self.mean_predictor(aux_features)
            pred_mean = torch.sigmoid(mean_logit) * self.max_depth
            outputs['pred_mean'] = torch.clamp(pred_mean, min=self.min_depth, max=self.max_depth)
            
            # Std depth prediction (ensure positive)
            std_logit = self.std_predictor(aux_features)
            pred_std = torch.sigmoid(std_logit) * 2.0  # Max std of 2m
            outputs['pred_std'] = pred_std
            
            # Gradient magnitude prediction
            grad_logit = self.gradient_predictor(aux_features)
            pred_gradient = torch.sigmoid(grad_logit) * 1.0  # Max gradient of 1.0
            outputs['pred_gradient'] = pred_gradient
            
            # Edge score prediction
            edge_logit = self.edge_predictor(aux_features)
            pred_edge = torch.sigmoid(edge_logit)  # Probability of being an edge
            outputs['pred_edge'] = pred_edge
            
            # Depth bin classification
            bin_logits = self.depth_bin_predictor(aux_features)
            outputs['pred_depth_bin'] = bin_logits  # Raw logits for cross-entropy loss
        
        # Auxiliary predictions for multi-scale supervision
        if self.use_multi_scale_supervision and self.training:
            aux_depths = []
            # Use features from scales 2 and 3 (skip scale 1)
            for i in range(1, len(gaze_features)):
                aux_idx = i - 1  # Auxiliary predictor index
                if aux_idx < len(self.aux_predictors):
                    aux_logit = self.aux_predictors[aux_idx](gaze_features[i])
                    aux_depth = torch.sigmoid(aux_logit) * self.max_depth
                    aux_depth = torch.clamp(aux_depth, min=self.min_depth, max=self.max_depth)
                    aux_depths.append(aux_depth)
            
            outputs['aux_depths'] = aux_depths
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTaskGazeLoss(nn.Module):
    """Multi-task loss for gaze-based depth prediction with auxiliary patch statistics."""
    
    def __init__(
        self,
        alpha: float = 0.85,
        task_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            alpha: Weight for scale-invariant term in SI-log loss
            task_weights: Weights for different tasks
        """
        super().__init__()
        self.alpha = alpha
        
        # Default task weights
        self.task_weights = task_weights or {
            'depth': 1.0,      # Main task
            'mean': 0.1,       # Auxiliary tasks
            'std': 0.1,
            'gradient': 0.05,
            'edge': 0.05,
            'depth_bin': 0.1,
            'aux_depths': 0.1  # Multi-scale supervision
        }
        
        # Individual loss functions
        self.si_loss = nn.MSELoss(reduction='none')  # Will implement SI-log manually
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def compute_si_log_loss(self, pred, target):
        """Compute scale-invariant logarithmic loss."""
        # Avoid log(0)
        pred = torch.clamp(pred, min=1e-6)
        target = torch.clamp(target, min=1e-6)
        
        # Log difference
        log_diff = torch.log(pred) - torch.log(target)
        
        # Scale-invariant loss
        loss = torch.mean(log_diff ** 2) - self.alpha * (torch.mean(log_diff) ** 2)
        
        return loss
    
    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model predictions
            batch: Batch data including ground truth
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual losses for logging
        """
        losses = {}
        
        # Main depth loss
        pred_depth = outputs['depth']
        gt_depth = batch['gt_depth_at_gaze'].to(pred_depth.device)
        losses['depth'] = self.compute_si_log_loss(pred_depth, gt_depth)
        
        # Multi-task losses (only if model predicts them)
        if 'pred_mean' in outputs and 'gt_mean' in batch:
            # Mean depth loss - ensure tensors are on same device
            gt_mean = batch['gt_mean'].to(outputs['pred_mean'].device)
            losses['mean'] = self.l1_loss(outputs['pred_mean'], gt_mean.unsqueeze(1))
            
        if 'pred_std' in outputs and 'gt_std' in batch:
            # Std depth loss - ensure tensors are on same device
            gt_std = batch['gt_std'].to(outputs['pred_std'].device)
            losses['std'] = self.l1_loss(outputs['pred_std'], gt_std.unsqueeze(1))
            
        if 'pred_gradient' in outputs and 'gt_grad_magnitude' in batch:
            # Gradient magnitude loss - ensure tensors are on same device
            gt_grad = batch['gt_grad_magnitude'].to(outputs['pred_gradient'].device)
            losses['gradient'] = self.l1_loss(outputs['pred_gradient'], gt_grad.unsqueeze(1))
            
        if 'pred_edge' in outputs and 'gt_edge_score' in batch:
            # Edge detection loss (treat as binary classification)
            # Threshold edge score to create binary labels
            edge_threshold = 0.1
            gt_edge_score = batch['gt_edge_score'].to(outputs['pred_edge'].device)
            gt_edges = (gt_edge_score > edge_threshold).float().unsqueeze(1)
            losses['edge'] = self.bce_loss(outputs['pred_edge'], gt_edges)
            
        if 'pred_depth_bin' in outputs and 'gt_depth_bin' in batch:
            # Depth bin classification loss - ensure tensors are on same device
            gt_depth_bin = batch['gt_depth_bin'].to(outputs['pred_depth_bin'].device)
            losses['depth_bin'] = self.ce_loss(outputs['pred_depth_bin'], gt_depth_bin)
        
        # Multi-scale supervision losses
        if 'aux_depths' in outputs:
            aux_loss = 0
            for i, aux_depth in enumerate(outputs['aux_depths']):
                aux_loss += self.compute_si_log_loss(aux_depth, gt_depth)
            losses['aux_depths'] = aux_loss / len(outputs['aux_depths'])
        
        # Weighted sum
        total_loss = sum(losses[k] * self.task_weights.get(k, 0) for k in losses)
        
        return total_loss, losses


class DualResolutionGazeDepth(nn.Module):
    """
    Combines low-res context (88×88) with high-res patch (96×96) for accurate gaze depth.
    This achieves high accuracy while maintaining efficiency.
    """
    
    def __init__(
        self,
        # Context encoder parameters
        context_size: int = 88,
        context_levels: int = 3,
        context_channels: int = 32,
        
        # Patch encoder parameters  
        patch_size: int = 96,
        patch_levels: int = 3,
        patch_channels: int = 48,
        
        # Output parameters
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        
        # Feature dimensions
        context_feature_dim: int = 64,
        patch_feature_dim: int = 192,
        
        # Options
        use_attention_fusion: bool = True,
        use_multi_scale_supervision: bool = True
    ):
        """Initialize dual-resolution model."""
        super().__init__()
        
        # Import patch encoder
        from lightweight_patch_encoder import LightweightPatchEncoder, FeatureFusionModule
        
        self.context_size = context_size
        self.patch_size = patch_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_multi_scale_supervision = use_multi_scale_supervision
        
        # Context encoder - processes full 88×88 image
        self.context_encoder = FlexibleGazeOnlyDepth(
            num_encoder_levels=context_levels,
            base_channels=context_channels,
            gaze_feature_dim=context_feature_dim,
            image_size=context_size,
            max_depth=max_depth,
            min_depth=min_depth,
            use_multi_scale_supervision=False,  # We'll handle this separately
            use_multi_task=False  # Simplify for now
        )
        
        # Patch encoder - processes 96×96 high-res patch
        self.patch_encoder = LightweightPatchEncoder(
            num_levels=patch_levels,
            base_channels=patch_channels,
            output_dim=patch_feature_dim,
            image_size=patch_size
        )
        
        # Calculate total feature dimension
        context_total_dim = context_levels * context_feature_dim  # 3 * 64 = 192
        
        # Feature fusion module
        self.fusion = FeatureFusionModule(
            context_dim=context_total_dim,
            patch_dim=patch_feature_dim,
            output_dim=384,
            use_attention=use_attention_fusion
        )
        
        # Final depth predictor
        self.depth_predictor = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
        # Auxiliary predictors for multi-scale supervision
        if use_multi_scale_supervision:
            # Predict from context features alone (for auxiliary loss)
            self.context_aux_predictor = nn.Sequential(
                nn.Linear(context_total_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )
            
            # Predict from patch features alone (for auxiliary loss)
            self.patch_aux_predictor = nn.Sequential(
                nn.Linear(patch_feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )
        
        # Initialize depth output layers
        self._init_depth_outputs()
        
    def _init_depth_outputs(self):
        """Initialize depth prediction layers for reasonable outputs."""
        with torch.no_grad():
            # Initialize main predictor
            output_layer = self.depth_predictor[-1]
            nn.init.normal_(output_layer.weight, mean=0, std=0.01)
            init_bias = torch.log(torch.tensor(2.0 / self.max_depth))
            nn.init.constant_(output_layer.bias, init_bias.item())
            
            # Initialize auxiliary predictors
            if hasattr(self, 'context_aux_predictor'):
                output_layer = self.context_aux_predictor[-1]
                nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                nn.init.constant_(output_layer.bias, init_bias.item())
                
            if hasattr(self, 'patch_aux_predictor'):
                output_layer = self.patch_aux_predictor[-1]
                nn.init.normal_(output_layer.weight, mean=0, std=0.01)
                nn.init.constant_(output_layer.bias, init_bias.item())
    
    def forward(
        self,
        context_rgb: torch.Tensor,
        patch_rgb: torch.Tensor,
        gaze_x: torch.Tensor,
        gaze_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining context and patch information.
        
        Args:
            context_rgb: [B, 3, 88, 88] low-res full image
            patch_rgb: [B, 3, 96, 96] high-res patch at gaze
            gaze_x: [B] gaze x coordinate in context image
            gaze_y: [B] gaze y coordinate in context image
            
        Returns:
            Dictionary with 'depth' and optionally 'aux_depths'
        """
        # Get context features using gaze
        context_outputs = self.context_encoder(context_rgb, gaze_x, gaze_y)
        context_features = context_outputs['depth'].squeeze(1)  # Remove depth dimension
        
        # Actually, we need to get the features before final prediction
        # Let's access the encoder directly
        context_encoder_features = self.context_encoder.encoder(context_rgb)
        
        # Extract gaze features from each scale
        context_gaze_features = []
        for i, (feat, proj) in enumerate(zip(context_encoder_features, self.context_encoder.gaze_projections)):
            scale_factor = 2 ** (i + 1)
            scaled_gaze_x = gaze_x / scale_factor
            scaled_gaze_y = gaze_y / scale_factor
            
            gaze_feat = self.context_encoder.extract_gaze_features(feat, scaled_gaze_x, scaled_gaze_y)
            projected = proj(gaze_feat)
            context_gaze_features.append(projected)
        
        # Concatenate context features
        context_combined = torch.cat(context_gaze_features, dim=1)
        
        # Get patch features (centered at gaze, so extract from center)
        patch_features = self.patch_encoder(patch_rgb)
        
        # Fuse context and patch features
        fused_features = self.fusion(context_combined, patch_features)
        
        # Predict depth from fused features
        depth_logit = self.depth_predictor(fused_features)
        depth = torch.sigmoid(depth_logit) * self.max_depth
        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
        
        outputs = {'depth': depth}
        
        # Auxiliary predictions for multi-scale supervision
        if self.use_multi_scale_supervision and self.training:
            # Context-only prediction
            context_depth_logit = self.context_aux_predictor(context_combined)
            context_depth = torch.sigmoid(context_depth_logit) * self.max_depth
            context_depth = torch.clamp(context_depth, min=self.min_depth, max=self.max_depth)
            
            # Patch-only prediction
            patch_depth_logit = self.patch_aux_predictor(patch_features)
            patch_depth = torch.sigmoid(patch_depth_logit) * self.max_depth
            patch_depth = torch.clamp(patch_depth, min=self.min_depth, max=self.max_depth)
            
            outputs['aux_depths'] = [context_depth, patch_depth]
            outputs['aux_names'] = ['context_only', 'patch_only']
        
        return outputs
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the flexible models
    print("Testing Flexible Gaze Encoder...")
    
    # Test multi-task model
    print("\nTesting multi-task model:")
    model = FlexibleGazeOnlyDepth(
        num_encoder_levels=3,
        base_channels=32,
        gaze_feature_dim=64,
        image_size=352,
        use_multi_task=True
    )
    
    print(f"  Total parameters: {model.get_num_params():,}")
    
    # Count parameters by component
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    depth_params = sum(p.numel() for p in model.depth_predictor.parameters())
    multitask_params = 0
    if model.use_multi_task:
        multitask_params += sum(p.numel() for p in model.aux_feature_processor.parameters())
        multitask_params += sum(p.numel() for p in model.mean_predictor.parameters())
        multitask_params += sum(p.numel() for p in model.std_predictor.parameters())
        multitask_params += sum(p.numel() for p in model.gradient_predictor.parameters())
        multitask_params += sum(p.numel() for p in model.edge_predictor.parameters())
        multitask_params += sum(p.numel() for p in model.depth_bin_predictor.parameters())
    
    print(f"  Encoder: {encoder_params:,} params")
    print(f"  Depth predictor: {depth_params:,} params")
    print(f"  Multi-task heads: {multitask_params:,} params")
    print(f"  Overhead for multi-task: {multitask_params / model.get_num_params() * 100:.1f}%")
    
    # Test forward pass
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 352, 352)
    gaze_x = torch.randint(0, 352, (batch_size,)).float()
    gaze_y = torch.randint(0, 352, (batch_size,)).float()
    
    model.train()  # Enable training mode to get all outputs
    outputs = model(rgb, gaze_x, gaze_y)
    
    print("\n  Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, list):
            print(f"    {key}: {len(value)} items, shape {value[0].shape}")
        else:
            print(f"    {key}: shape {value.shape}")
    
    # Test different image sizes
    print("\nTesting different image sizes:")
    for img_size in [88, 176, 352]:
        model = FlexibleGazeOnlyDepth(
            num_encoder_levels=3,
            base_channels=32,
            image_size=img_size,
            use_multi_task=False
        )
        print(f"  {img_size}x{img_size}: {model.get_num_params():,} parameters")