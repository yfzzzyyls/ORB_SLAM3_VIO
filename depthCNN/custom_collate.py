"""
Custom collate function for handling the flexible dataset output.
"""

import torch
from typing import List, Dict, Any, Optional


def flexible_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that handles:
    - None values in high_res_patch
    - List of dicts for gaze information
    - Variable tensor types
    """
    # Filter out None samples and samples without valid high_res_patch (for dual-resolution mode)
    batch = [item for item in batch if item is not None]
    
    # Additional filtering for dual-resolution mode - require high_res_patch
    if len(batch) > 0 and 'high_res_patch' in batch[0]:
        batch = [item for item in batch if item.get('high_res_patch') is not None]
    
    if len(batch) == 0:
        return None
    
    # Initialize output dict
    collated = {}
    
    # Get first sample to check keys
    sample = batch[0]
    
    for key in sample.keys():
        if key == 'high_res_patch':
            # We've already filtered out None values above
            values = [item[key] for item in batch]
            collated[key] = torch.stack(values)
                
        elif key == 'gaze':
            # Keep gaze as list of dicts (or None)
            collated[key] = [item[key] for item in batch]
            
            # Also extract gaze coordinates as tensors for convenience
            gaze_list = [item[key] for item in batch]
            valid_gaze = [g for g in gaze_list if g is not None]
            if valid_gaze:
                collated['gaze_coords'] = torch.stack([
                    torch.tensor([g['x'], g['y']], dtype=torch.float32) 
                    for g in valid_gaze
                ])
            
        elif key == 'patch_coords':
            # Keep as list of tuples
            collated[key] = [item.get(key) for item in batch]
            
        elif key in ['sequence', 'frame_idx']:
            # Keep as lists
            collated[key] = [item[key] for item in batch]
            
        elif key == 'depth_patch_stats':
            # Keep as list of dicts
            collated[key] = [item.get(key) for item in batch]
            
        elif key == 'gt_depth_patch':
            # Handle gt_depth_patch which might not exist for all samples
            patches = []
            for item in batch:
                if key in item:
                    patches.append(item[key])
            if patches:
                collated[key] = torch.stack(patches)
            else:
                # Skip this key if no samples have it
                continue
                
        elif key == 'gt_depth_patch_mask':
            # Handle gt_depth_patch_mask similarly
            masks = []
            for item in batch:
                if key in item:
                    masks.append(item[key])
            if masks:
                collated[key] = torch.stack(masks)
            else:
                continue
                
        elif isinstance(sample[key], torch.Tensor):
            # Stack tensors
            collated[key] = torch.stack([item[key] for item in batch])
            
        elif isinstance(sample[key], (int, float)):
            # Convert numbers to tensor
            collated[key] = torch.tensor([item[key] for item in batch])
            
        else:
            # Keep other types as lists
            collated[key] = [item[key] for item in batch]
    
    return collated


def validation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for validation that filters out samples without gaze data.
    Only keeps samples with real gaze and corresponding depth patches.
    """
    # Filter out None samples and samples without gaze/depth patches
    valid_samples = []
    for item in batch:
        if item is not None and item.get('gaze') is not None:
            # Check if it has real gaze (not synthetic)
            if not item['gaze'].get('is_synthetic', False):
                # Check if it has depth patch
                if 'gt_depth_patch' in item and item['gt_depth_patch'] is not None:
                    valid_samples.append(item)
    
    if len(valid_samples) == 0:
        return None
    
    # Use flexible_collate_fn for the valid samples
    return flexible_collate_fn(valid_samples)