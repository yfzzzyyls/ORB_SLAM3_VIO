#!/usr/bin/env python3
"""
Simple wrapper around FlexibleResolutionDataset that generates 16 patch centers
per image during training for dense supervision.
"""

import torch
from torch.utils.data import Dataset
from flexible_dataset import FlexibleResolutionDataset

class DensePatchDataset(Dataset):
    """
    Wrapper that generates multiple patch centers per image during training.
    For 88x88 images with 22x22 patches, we generate 16 centers.
    """
    
    def __init__(self, base_dataset: FlexibleResolutionDataset, patch_size: int = 22):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.is_training = base_dataset.split == 'train'
        
        # Calculate patch centers for 88x88 image
        # For 22x22 patches, we have 4x4 grid
        self.image_size = base_dataset.target_size
        self.patches_per_dim = self.image_size // patch_size
        
        # Generate all patch centers
        self.patch_centers = []
        half_patch = patch_size // 2
        for i in range(self.patches_per_dim):
            for j in range(self.patches_per_dim):
                center_x = half_patch + j * patch_size
                center_y = half_patch + i * patch_size
                self.patch_centers.append((center_x, center_y))
        
        # For training, we have 16x more samples
        self.multiplier = len(self.patch_centers) if self.is_training else 1
        
        print(f"DensePatchDataset initialized:")
        print(f"  Base dataset size: {len(self.base_dataset)}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Patches per image: {len(self.patch_centers)}")
        print(f"  Total samples: {len(self)}")
        print(f"  Mode: {'Training (dense)' if self.is_training else 'Validation/Test (gaze only)'}")
    
    def __len__(self):
        return len(self.base_dataset) * self.multiplier
    
    def __getitem__(self, idx):
        # For training, map idx to (image_idx, patch_idx)
        if self.is_training:
            image_idx = idx // len(self.patch_centers)
            patch_idx = idx % len(self.patch_centers)
            
            # Get base sample
            sample = self.base_dataset[image_idx]
            
            # Replace gaze with patch center
            center_x, center_y = self.patch_centers[patch_idx]
            
            # Create synthetic gaze info for this patch center
            sample['gaze'] = {
                'x': float(center_x),
                'y': float(center_y),
                'x_original': float(center_x * self.base_dataset.scale_factor),
                'y_original': float(center_y * self.base_dataset.scale_factor),
                'is_real_gaze': False,
                'patch_idx': patch_idx
            }
            
            # Extract GT depth at this new center
            depth_numpy = sample['depth'].squeeze(0).numpy()
            sample['gt_depth_at_gaze'] = depth_numpy[int(center_y), int(center_x)]
            
            # Extract depth patch around new center
            if 'gt_depth_patch' in sample:
                # Re-extract patch for new center
                depth_patch, valid_mask_patch = self.base_dataset._extract_depth_patch(
                    depth_numpy,
                    center_x,
                    center_y,
                    self.patch_size
                )
                sample['gt_depth_patch'] = torch.from_numpy(depth_patch).float()
                sample['gt_depth_patch_mask'] = torch.from_numpy(valid_mask_patch).bool()
            
        else:
            # For validation/test, just return original sample with real gaze
            sample = self.base_dataset[idx]
            if sample['gaze'] is not None:
                sample['gaze']['is_real_gaze'] = True
        
        return sample


def create_dense_patch_dataloader(data_root, split, batch_size, patch_size=22, 
                                 image_size=88, num_workers=4, shuffle=None):
    """Create dataloader with dense patch sampling for training."""
    
    # Create base dataset
    base_dataset = FlexibleResolutionDataset(
        data_root=data_root,
        split=split,
        target_size=image_size,
        return_depth_patch=True,
        depth_patch_size=patch_size
    )
    
    # Wrap with dense patch dataset
    dataset = DensePatchDataset(base_dataset, patch_size=patch_size)
    
    # Adjust batch size for training
    # Since we have 16x more samples, we might want to reduce batch size
    if split == 'train':
        # Reduce batch size to manage memory
        effective_batch_size = max(1, batch_size // 8)  # Reduce by 8x instead of 16x
        print(f"  Adjusted batch size for training: {batch_size} -> {effective_batch_size}")
    else:
        effective_batch_size = batch_size
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle if shuffle is not None else (split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader, dataset


if __name__ == '__main__':
    # Test the dataset
    print("Testing DensePatchDataset...")
    
    # Create base dataset
    base_dataset = FlexibleResolutionDataset(
        data_root='/mnt/ssd_ext/incSeg-data/processed_adt',
        split='train',
        target_size=88,
        return_depth_patch=True,
        depth_patch_size=22
    )
    
    # Wrap with dense patch
    dataset = DensePatchDataset(base_dataset, patch_size=22)
    
    # Test first image's patches
    print("\nTesting first image's 16 patches:")
    for i in range(16):
        sample = dataset[i]
        gaze = sample['gaze']
        print(f"  Patch {i}: center at ({gaze['x']:.0f}, {gaze['y']:.0f})")