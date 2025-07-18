#!/usr/bin/env python3
"""
Low-resolution dataset for efficient depth prediction training.
Applies average pooling to downsample images and properly scales gaze coordinates.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import Dict, Optional, Tuple

from processed_dataset import ProcessedADTDataset


class LowResADTDataset(ProcessedADTDataset):
    """
    Low-resolution dataset that downsamples RGB and depth images.
    Also properly scales gaze coordinates to match the downsampled resolution.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        scale_factor: int = 16,
        transform=None
    ):
        """
        Args:
            data_root: Root directory containing train/val/test folders
            split: 'train', 'val', or 'test'
            scale_factor: Downsampling factor (16 means 1/16 of original size)
            transform: Optional transforms to apply
        """
        super().__init__(data_root, split, transform)
        self.scale_factor = scale_factor
        self.lowres_size = 1408 // scale_factor  # 88 for scale_factor=16
        
        print(f"Low-resolution dataset initialized:")
        print(f"  Original size: 1408×1408")
        print(f"  Scale factor: {scale_factor}")
        print(f"  Low-res size: {self.lowres_size}×{self.lowres_size}")
        
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
        
        # Keep a reference to numpy depth for gaze extraction
        depth_numpy = depth.copy()
        
        # Convert to tensors
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        # Apply average pooling to downsample
        # Add batch dimension for pooling
        rgb_lowres = F.avg_pool2d(
            rgb.unsqueeze(0), 
            kernel_size=self.scale_factor,
            stride=self.scale_factor
        ).squeeze(0)
        
        depth_lowres = F.avg_pool2d(
            depth.unsqueeze(0),
            kernel_size=self.scale_factor,
            stride=self.scale_factor
        ).squeeze(0)

        # need to 


        # Create valid mask at low resolution
        valid_mask_lowres = depth_lowres > 0
        
        # Load and scale gaze information if available
        gaze_info = None
        gt_depth_at_gaze = None
        if 'gaze' in frame_info and frame_info.get('has_gaze', False):
            try:
                gaze_path = frame_info['seq_dir'] / 'gaze' / frame_info['gaze']
                if gaze_path.exists():
                    with open(gaze_path, 'r') as f:
                        gaze_data = json.load(f)
                    
                    # Extract exact GT depth at original gaze location BEFORE downsampling
                    gaze_x_orig = int(round(gaze_data['x_pixel']))
                    gaze_y_orig = int(round(gaze_data['y_pixel']))
                    
                    # Ensure coordinates are within bounds
                    gaze_x_orig = max(0, min(gaze_x_orig, 1407))
                    gaze_y_orig = max(0, min(gaze_y_orig, 1407))
                    
                    # Extract exact depth value at gaze position from numpy array
                    gt_depth_at_gaze = depth_numpy[gaze_y_orig, gaze_x_orig]
                    
                    # Scale gaze coordinates for 88x88
                    gaze_info = {
                        'x': gaze_data['x_pixel'] / self.scale_factor,
                        'y': gaze_data['y_pixel'] / self.scale_factor,
                        'x_original': gaze_data['x_pixel'],
                        'y_original': gaze_data['y_pixel'],
                        'pitch': gaze_data['pitch_rad'],
                        'yaw': gaze_data['yaw_rad'],
                        'time_diff_ms': gaze_data['time_diff_ms']
                    }
                    
                    # Ensure scaled coordinates are within bounds
                    gaze_info['x'] = max(0, min(gaze_info['x'], self.lowres_size - 1))
                    gaze_info['y'] = max(0, min(gaze_info['y'], self.lowres_size - 1))
            except Exception as e:
                # If there's any error loading gaze, just skip it
                gaze_info = None
                gt_depth_at_gaze = None
        
        # Apply transforms if any (on low-res data)
        if self.transform:
            rgb_lowres, depth_lowres, valid_mask_lowres = self.transform(
                rgb_lowres, depth_lowres, valid_mask_lowres
            )
        
        return {
            'rgb': rgb_lowres,
            'depth': depth_lowres,
            'valid_mask': valid_mask_lowres,
            'sequence': frame_info['sequence'],
            'frame_idx': frame_info['index'],
            'gaze': gaze_info,
            'gt_depth_at_gaze': gt_depth_at_gaze,  # Exact depth at gaze position
            'scale_factor': self.scale_factor,
            'original_size': 1408,
            'lowres_size': self.lowres_size
        }
    
    def _build_frame_index(self):
        """Build index of all frames, including gaze file information."""
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
                frame_index_entry = {
                    'sequence': seq_dir.name,
                    'seq_dir': seq_dir,
                    'rgb_file': frame_info['rgb'],
                    'depth_file': frame_info['depth'],
                    'timestamp_ns': frame_info['rgb_timestamp_ns'],
                    'index': frame_info['index']
                }
                
                # Add gaze information if available
                if 'gaze' in frame_info:
                    frame_index_entry['gaze'] = frame_info['gaze']
                    frame_index_entry['has_gaze'] = frame_info.get('has_gaze', True)
                else:
                    frame_index_entry['has_gaze'] = False
                
                self.frame_index.append(frame_index_entry)
            
            print(f"  {seq_dir.name}: {metadata['num_frames']} frames")
        
        print(f"Total {self.split} frames: {len(self.frame_index)}")


def visualize_lowres_sample(sample: Dict, save_path: Optional[str] = None):
    """Visualize a low-resolution sample with RGB, depth, and gaze location."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # RGB image
    rgb = sample['rgb'].permute(1, 2, 0).numpy()
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB ({sample['lowres_size']}×{sample['lowres_size']})")
    
    # Depth image
    depth = sample['depth'].squeeze().numpy()
    valid_mask = sample['valid_mask'].squeeze().numpy()
    depth_vis = depth.copy()
    depth_vis[~valid_mask] = np.nan
    
    im = axes[1].imshow(depth_vis, cmap='viridis')
    axes[1].set_title(f"Depth ({sample['lowres_size']}×{sample['lowres_size']})")
    plt.colorbar(im, ax=axes[1], label='Depth (m)')
    
    # Add gaze point if available
    if sample.get('gaze') is not None:
        gaze_x = sample['gaze']['x']
        gaze_y = sample['gaze']['y']
        
        for ax in axes:
            ax.scatter(gaze_x, gaze_y, c='red', s=100, marker='x', linewidths=3)
            ax.scatter(gaze_x, gaze_y, c='white', s=80, marker='x', linewidths=2)
    
    plt.suptitle(f"Sequence: {sample['sequence']}, Frame: {sample['frame_idx']}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test low-resolution dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Root directory of processed data')
    parser.add_argument('--scale-factor', type=int, default=16,
                        help='Downsampling factor')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample')
    args = parser.parse_args()
    
    # Test loading
    print("Testing LowResADTDataset...")
    
    dataset = LowResADTDataset(
        data_root=args.data_root,
        split='train',
        scale_factor=args.scale_factor
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
        print(f"  Scale factor: {sample['scale_factor']}")
        print(f"  Low-res size: {sample['lowres_size']}×{sample['lowres_size']}")
        print(f"  Valid pixels: {sample['valid_mask'].sum().item() / sample['valid_mask'].numel() * 100:.1f}%")
        
        if sample['depth'][sample['valid_mask']].numel() > 0:
            print(f"  Depth range: [{sample['depth'][sample['valid_mask']].min():.3f}, "
                  f"{sample['depth'][sample['valid_mask']].max():.3f}] meters")
        
        if sample.get('gaze') is not None:
            print(f"  Gaze coordinates: ({sample['gaze']['x']:.1f}, {sample['gaze']['y']:.1f})")
            print(f"  Original gaze: ({sample['gaze']['x_original']}, {sample['gaze']['y_original']})")
        else:
            print("  No gaze information available")
        
        if args.visualize:
            print("\nVisualizing sample...")
            visualize_lowres_sample(sample, save_path='lowres_sample.png')
            print("Saved visualization to lowres_sample.png")
    else:
        print("No data found! Run extract_dataset.py first.")