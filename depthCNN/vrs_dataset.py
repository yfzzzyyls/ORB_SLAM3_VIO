#!/usr/bin/env python3
"""
PyTorch Dataset for loading RGB-Depth pairs from ADT VRS files.
Uses 7 sequences for training, 1 for validation, 2 for evaluation.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import List, Tuple, Optional
import cv2

# Fix projectaria_tools import
sys.path.append('/home/external/.local/lib/python3.9/site-packages')
from projectaria_tools.core import data_provider


class ADTVRSDataset(Dataset):
    """Dataset for loading RGB-Depth pairs from ADT VRS files."""
    
    def __init__(
        self,
        adt_root: str,
        split: str = 'train',
        transform=None,
        cache_dir: Optional[str] = None,
        subsample_factor: int = 1
    ):
        """
        Args:
            adt_root: Root directory containing ADT sequences
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            cache_dir: Directory to cache extracted frames
            subsample_factor: Sample every N frames (default 1 = all frames)
        """
        self.adt_root = Path(adt_root)
        self.split = split
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.subsample_factor = subsample_factor
        
        # Define splits (10 sequences total)
        all_sequences = sorted([
            d for d in os.listdir(adt_root)
            if d.startswith("Apartment_release_clean_seq") and 
            os.path.isdir(os.path.join(adt_root, d))
        ])[:10]  # Use first 10 sequences
        
        # Split: 7 train, 1 val, 2 test
        self.sequences = {
            'train': all_sequences[:7],
            'val': all_sequences[7:8],
            'test': all_sequences[8:10]
        }[split]
        
        print(f"\n{split.upper()} sequences ({len(self.sequences)}):")
        for seq in self.sequences:
            print(f"  - {seq}")
        
        # Build frame index
        self.frame_index = []
        self._build_frame_index()
        
    def _build_frame_index(self):
        """Build index of all frames across sequences."""
        for seq_name in self.sequences:
            seq_dir = self.adt_root / seq_name
            
            # Find VRS files
            rgb_vrs = None
            depth_vrs = None
            
            for file in os.listdir(seq_dir):
                if file.endswith('_main_recording.vrs'):
                    rgb_vrs = seq_dir / file
                elif file == 'depth_images.vrs':
                    depth_vrs = seq_dir / file
            
            if not rgb_vrs or not depth_vrs:
                print(f"Warning: Missing VRS files in {seq_name}")
                continue
            
            # Get number of frames
            try:
                rgb_provider = data_provider.create_vrs_data_provider(str(rgb_vrs))
                rgb_stream_id = rgb_provider.get_stream_id_from_label("camera-rgb")
                num_frames = rgb_provider.get_num_data(rgb_stream_id)
                
                # Subsample frames
                frame_indices = range(0, num_frames, self.subsample_factor)
                
                for frame_idx in frame_indices:
                    self.frame_index.append({
                        'sequence': seq_name,
                        'rgb_vrs': str(rgb_vrs),
                        'depth_vrs': str(depth_vrs),
                        'frame_idx': frame_idx
                    })
                
                print(f"  {seq_name}: {len(frame_indices)} frames")
                
            except Exception as e:
                print(f"Error processing {seq_name}: {e}")
        
        print(f"\nTotal {self.split} frames: {len(self.frame_index)}")
    
    def _load_frame(self, index):
        """Load RGB and depth for a single frame."""
        frame_info = self.frame_index[index]
        
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{frame_info['sequence']}_{frame_info['frame_idx']:06d}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                return data['rgb'], data['depth']
        
        # Load from VRS
        # RGB
        rgb_provider = data_provider.create_vrs_data_provider(frame_info['rgb_vrs'])
        rgb_stream_id = rgb_provider.get_stream_id_from_label("camera-rgb")
        rgb_data = rgb_provider.get_image_data_by_index(rgb_stream_id, frame_info['frame_idx'])
        rgb_image = rgb_data[0].to_numpy_array()
        
        # Depth (stream 345-1 for RGB camera)
        depth_provider = data_provider.create_vrs_data_provider(frame_info['depth_vrs'])
        depth_stream_id = depth_provider.get_all_streams()[0]  # First stream is RGB depth
        depth_data = depth_provider.get_image_data_by_index(depth_stream_id, frame_info['frame_idx'])
        depth_image = depth_data[0].to_numpy_array()
        
        # Convert depth from millimeters to meters
        depth_meters = depth_image.astype(np.float32) / 1000.0
        
        # Cache if enabled
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                rgb=rgb_image,
                depth=depth_meters
            )
        
        return rgb_image, depth_meters
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, index):
        rgb, depth = self._load_frame(index)
        
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
            'sequence': self.frame_index[index]['sequence'],
            'frame_idx': self.frame_index[index]['frame_idx']
        }


class RandomCrop:
    """Random crop transform for training."""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, rgb, depth, valid_mask):
        h, w = rgb.shape[-2:]
        th, tw = self.size
        
        if h < th or w < tw:
            return rgb, depth, valid_mask
        
        x1 = torch.randint(0, w - tw + 1, (1,)).item()
        y1 = torch.randint(0, h - th + 1, (1,)).item()
        
        rgb = rgb[:, y1:y1+th, x1:x1+tw]
        depth = depth[:, y1:y1+th, x1:x1+tw]
        valid_mask = valid_mask[:, y1:y1+th, x1:x1+tw]
        
        return rgb, depth, valid_mask


class RandomHorizontalFlip:
    """Random horizontal flip transform."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, rgb, depth, valid_mask):
        if torch.rand(1) < self.p:
            rgb = torch.flip(rgb, [-1])
            depth = torch.flip(depth, [-1])
            valid_mask = torch.flip(valid_mask, [-1])
        return rgb, depth, valid_mask


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, rgb, depth, valid_mask):
        for t in self.transforms:
            rgb, depth, valid_mask = t(rgb, depth, valid_mask)
        return rgb, depth, valid_mask


if __name__ == "__main__":
    # Test dataset
    dataset = ADTVRSDataset(
        adt_root="/mnt/ssd_ext/incSeg-data/adt",
        split='train',
        subsample_factor=100  # Sample every 100 frames for testing
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load one sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  RGB: {sample['rgb'].shape}")
    print(f"  Depth: {sample['depth'].shape}")
    print(f"  Valid mask: {sample['valid_mask'].shape}")
    print(f"  Valid pixels: {sample['valid_mask'].sum().item() / sample['valid_mask'].numel() * 100:.1f}%")