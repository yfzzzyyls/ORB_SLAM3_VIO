#!/usr/bin/env python3
"""
Depth evaluation metrics for sparse-to-dense training
"""

import torch
import numpy as np


def compute_depth_metrics(pred, gt, valid_mask=None):
    """
    Compute standard depth evaluation metrics
    
    Args:
        pred: Predicted depth map (B, 1, H, W) or (B, H, W)
        gt: Ground truth sparse depth (B, 1, H, W) or (B, H, W)
        valid_mask: Binary mask for valid pixels (optional)
    
    Returns:
        Dictionary of metrics
    """
    # Ensure 4D tensors
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    
    # Create valid mask if not provided (gt > 0)
    if valid_mask is None:
        valid_mask = gt > 0
    
    # Flatten and apply mask
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    if len(pred_valid) == 0:
        return {
            'abs_rel': 0.0,
            'sq_rel': 0.0,
            'rmse': 0.0,
            'rmse_log': 0.0,
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0
        }
    
    # Avoid division by zero
    gt_valid = torch.clamp(gt_valid, min=1e-3)
    pred_valid = torch.clamp(pred_valid, min=1e-3)
    
    # Compute errors
    thresh = torch.max((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    
    rmse = torch.sqrt(torch.mean((gt_valid - pred_valid) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt_valid) - torch.log(pred_valid)) ** 2))
    
    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'a1': a1.item(),
        'a2': a2.item(),
        'a3': a3.item()
    }


def compute_depth_metrics_numpy(pred, gt, valid_mask=None):
    """
    Numpy version for post-processing evaluation
    """
    if valid_mask is None:
        valid_mask = gt > 0
    
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    if len(pred_valid) == 0:
        return {
            'abs_rel': 0.0,
            'sq_rel': 0.0,
            'rmse': 0.0,
            'rmse_log': 0.0,
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0
        }
    
    # Avoid log of zero
    pred_valid = np.maximum(pred_valid, 1e-3)
    gt_valid = np.maximum(gt_valid, 1e-3)
    
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }