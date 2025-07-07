#!/usr/bin/env python3
"""
PyTorch Dataset for loading pre-processed RGB-Depth pairs from extracted ADT data.
This is faster than loading directly from VRS files as data is already extracted.
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import List, Optional, Dict


class ProcessedADTDataset(Dataset):
    """Dataset for loading pre-processed RGB-Depth pairs."""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None
    ):
        """
        Args:
            data_root: Root directory containing train/val/test folders
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.split_dir = self.data_root / split
        
        # Check if data exists
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        # Load all frame info
        self.frame_index = []
        self._build_frame_index()
        
    def _build_frame_index(self):
        """Build index of all frames."""
        # Get all sequence directories
        seq_dirs = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])
        
        print(f"\nLoading {self.split} dataset from {self.split_dir}")
        print(f"Found {len(seq_dirs)} sequences")
        
        for seq_dir in seq_dirs:
            # Load metadata
            metadata_path = seq_dir / 'metadata.json'
            if not metadata_path.exists():
                print(f"Warning: No metadata found for {seq_dir.name}")
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Add all frames
            for frame_info in metadata['frames']:
                self.frame_index.append({
                    'sequence': seq_dir.name,
                    'seq_dir': seq_dir,
                    'rgb_file': frame_info['rgb'],
                    'depth_file': frame_info['depth'],
                    'timestamp_ns': frame_info['timestamp_ns'],
                    'index': frame_info['index']
                })
            
            print(f"  {seq_dir.name}: {metadata['num_frames']} frames")
        
        print(f"Total {self.split} frames: {len(self.frame_index)}")
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, index):
        frame_info = self.frame_index[index]
        
        # Load RGB image
        rgb_path = frame_info['seq_dir'] / 'rgb' / frame_info['rgb_file']
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load depth
        depth_path = frame_info['seq_dir'] / 'depth' / frame_info['depth_file']
        depth_data = np.load(depth_path)
        depth = depth_data['depth']
        
        # Convert depth from millimeters to meters
        depth = depth.astype(np.float32) / 1000.0
        
        # Convert to tensors
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        # Create valid mask (depth > 0)
        valid_mask = depth > 0
        
        # Apply transforms if any
        if self.transform:
            rgb, depth, valid_mask = self.transform(rgb, depth, valid_mask)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'valid_mask': valid_mask,
            'sequence': frame_info['sequence'],
            'frame_idx': frame_info['index']
        }


def create_data_loaders(data_root: str, batch_size: int = 4, 
                       num_workers: int = 4, crop_size: int = 1024):
    """Create train, val, and test data loaders."""
    from torch.utils.data import DataLoader
    from vrs_dataset import Compose, RandomCrop, RandomHorizontalFlip
    
    # Training transforms
    train_transforms = Compose([
        RandomCrop((crop_size, crop_size)),
        RandomHorizontalFlip(p=0.5)
    ])
    
    # Create datasets
    train_dataset = ProcessedADTDataset(
        data_root=data_root,
        split='train',
        transform=train_transforms
    )
    
    val_dataset = ProcessedADTDataset(
        data_root=data_root,
        split='val',
        transform=None  # No augmentation for validation
    )
    
    test_dataset = ProcessedADTDataset(
        data_root=data_root,
        split='test',
        transform=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Root directory of processed data')
    args = parser.parse_args()
    
    # Test loading
    print("Testing ProcessedADTDataset...")
    
    dataset = ProcessedADTDataset(
        data_root=args.data_root,
        split='train'
    )
    
    if len(dataset) > 0:
        # Load one sample
        sample = dataset[0]
        print(f"\nSample loaded successfully!")
        print(f"  RGB shape: {sample['rgb'].shape}")
        print(f"  Depth shape: {sample['depth'].shape}")
        print(f"  Valid mask shape: {sample['valid_mask'].shape}")
        print(f"  Sequence: {sample['sequence']}")
        print(f"  Frame index: {sample['frame_idx']}")
        print(f"  Valid pixels: {sample['valid_mask'].sum().item() / sample['valid_mask'].numel() * 100:.1f}%")
        print(f"  Depth range: [{sample['depth'][sample['valid_mask']].min():.3f}, "
              f"{sample['depth'][sample['valid_mask']].max():.3f}] meters")
    else:
        print("No data found! Run extract_dataset.py first.")