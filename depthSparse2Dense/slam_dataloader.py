#!/usr/bin/env python3
"""
PyTorch Dataset for loading SLAM sparse depth data
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import json
import random


class SLAMDepthDataset(Dataset):
    """Dataset for sparse-to-dense depth training from SLAM outputs"""
    
    def __init__(self, data_dir, split='train', transform=None, 
                 load_poses=True, augment=True):
        """
        Args:
            data_dir: Directory with processed SLAM data (from process_slam_to_sparse_depth.py)
            split: 'train' or 'val' 
            transform: Optional image transforms
            load_poses: Whether to load camera poses
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.load_poses = load_poses
        self.augment = augment
        
        # Load metadata
        with open(self.data_dir / 'metadata' / 'frames.json', 'r') as f:
            self.frames = json.load(f)
        
        # Split data (80/20 train/val)
        n_frames = len(self.frames)
        indices = list(range(n_frames))
        random.seed(42)
        random.shuffle(indices)
        
        split_idx = int(0.8 * n_frames)
        if split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
        # Load camera intrinsics
        self.K = np.load(self.data_dir / 'metadata' / 'intrinsics.npy')
        
        print(f"Loaded {len(self.indices)} frames for {split} split")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        frame_info = self.frames[self.indices[idx]]
        frame_id = frame_info['frame_id']
        frame_str = f"{frame_id:06d}"
        
        # Load RGB image
        rgb_path = self.data_dir / 'rgb' / f'{frame_str}.png'
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            # Create blank image if RGB not available
            h, w = 480, 640  # Default size
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Load sparse depth
        sparse_depth = np.load(self.data_dir / 'sparse_depth' / f'{frame_str}.npy')
        confidence = np.load(self.data_dir / 'sparse_depth' / f'{frame_str}_conf.npy')
        
        # Load pose if requested
        if self.load_poses:
            pose = np.load(self.data_dir / 'poses' / f'{frame_str}.npy')
        else:
            pose = np.eye(4)
        
        # Apply augmentation
        if self.augment and self.split == 'train':
            rgb, sparse_depth, confidence = self._augment(rgb, sparse_depth, confidence)
        
        # Convert to tensors
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).float()
        confidence = torch.from_numpy(confidence).unsqueeze(0).float()
        pose = torch.from_numpy(pose).float()
        K = torch.from_numpy(self.K).float()
        
        # Apply additional transforms if provided
        if self.transform:
            rgb = self.transform(rgb)
        
        return {
            'rgb': rgb,
            'sparse_depth': sparse_depth,
            'confidence': confidence,
            'pose': pose,
            'intrinsics': K,
            'frame_id': frame_id,
            'timestamp': frame_info['timestamp']
        }
    
    def _augment(self, rgb, sparse_depth, confidence):
        """Apply data augmentation"""
        h, w = rgb.shape[:2]
        
        # Random horizontal flip
        if random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            sparse_depth = np.fliplr(sparse_depth)
            confidence = np.fliplr(confidence)
        
        # Random crop
        if random.random() > 0.5:
            crop_h = int(h * 0.9)
            crop_w = int(w * 0.9)
            y = random.randint(0, h - crop_h)
            x = random.randint(0, w - crop_w)
            
            rgb = rgb[y:y+crop_h, x:x+crop_w]
            sparse_depth = sparse_depth[y:y+crop_h, x:x+crop_w]
            confidence = confidence[y:y+crop_h, x:x+crop_w]
            
            # Resize back
            rgb = cv2.resize(rgb, (w, h))
            sparse_depth = cv2.resize(sparse_depth, (w, h), interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Color jittering for RGB only
        if random.random() > 0.5:
            # Brightness
            factor = random.uniform(0.8, 1.2)
            rgb = np.clip(rgb * factor, 0, 255).astype(np.uint8)
            
        return rgb, sparse_depth, confidence


class SequentialSLAMDataset(Dataset):
    """Dataset that provides sequential frames for temporal consistency"""
    
    def __init__(self, data_dir, sequence_length=3, stride=1, **kwargs):
        """
        Args:
            data_dir: Directory with processed SLAM data
            sequence_length: Number of frames in sequence
            stride: Frame stride
        """
        self.base_dataset = SLAMDepthDataset(data_dir, **kwargs)
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Filter valid sequences
        self.valid_indices = []
        for i in range(len(self.base_dataset) - (sequence_length - 1) * stride):
            # Check if frames are continuous
            frame_ids = []
            valid = True
            for j in range(sequence_length):
                idx = i + j * stride
                if idx < len(self.base_dataset):
                    frame_ids.append(self.base_dataset.frames[self.base_dataset.indices[idx]]['frame_id'])
                else:
                    valid = False
                    break
            
            if valid and all(frame_ids[j+1] - frame_ids[j] == stride for j in range(len(frame_ids)-1)):
                self.valid_indices.append(i)
        
        print(f"Found {len(self.valid_indices)} valid sequences")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        sequence = []
        
        for i in range(self.sequence_length):
            frame_idx = start_idx + i * self.stride
            frame_data = self.base_dataset[frame_idx]
            sequence.append(frame_data)
        
        return sequence


def create_dataloaders(data_dir, batch_size=8, num_workers=4, 
                      sequential=False, sequence_length=3):
    """Create train and validation dataloaders"""
    
    if sequential:
        train_dataset = SequentialSLAMDataset(
            data_dir, 
            split='train',
            sequence_length=sequence_length,
            augment=True
        )
        val_dataset = SequentialSLAMDataset(
            data_dir,
            split='val', 
            sequence_length=sequence_length,
            augment=False
        )
    else:
        train_dataset = SLAMDepthDataset(data_dir, split='train', augment=True)
        val_dataset = SLAMDepthDataset(data_dir, split='val', augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def test_dataloader():
    """Test the dataloader"""
    import matplotlib.pyplot as plt
    
    # Assume test data exists
    data_dir = "/home/external/ORB_SLAM3_AEA/processed_depth_data"
    
    if not os.path.exists(data_dir):
        print(f"Test data not found at {data_dir}")
        return
    
    # Create dataset
    dataset = SLAMDepthDataset(data_dir, split='train', augment=False)
    print(f"Dataset size: {len(dataset)}")
    
    # Load one sample
    sample = dataset[0]
    
    print("Sample keys:", sample.keys())
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"Sparse depth shape: {sample['sparse_depth'].shape}")
    print(f"Confidence shape: {sample['confidence'].shape}")
    print(f"Pose shape: {sample['pose'].shape}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB
    rgb = sample['rgb'].permute(1, 2, 0).numpy()
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Sparse depth
    sparse = sample['sparse_depth'].squeeze().numpy()
    sparse_viz = sparse.copy()
    sparse_viz[sparse_viz == 0] = np.nan
    im = axes[1].imshow(sparse_viz, cmap='viridis')
    axes[1].set_title(f'Sparse Depth ({np.sum(sparse > 0)} points)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Confidence
    conf = sample['confidence'].squeeze().numpy()
    im2 = axes[2].imshow(conf, cmap='hot')
    axes[2].set_title('Confidence')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('dataloader_test.png')
    plt.show()


if __name__ == '__main__':
    test_dataloader()