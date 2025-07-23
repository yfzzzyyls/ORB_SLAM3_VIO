#!/usr/bin/env python3
"""
Flexible resolution dataset for efficient depth prediction training.
Supports arbitrary image sizes by resizing images to the target resolution.
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


class FlexibleResolutionDataset(ProcessedADTDataset):
    """
    Flexible resolution dataset that resizes RGB and depth images to target size.
    Also properly scales gaze coordinates to match the target resolution.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        target_size: int = 88,
        transform=None,
        use_high_res_patch: bool = False,
        patch_size: int = 96,
        return_depth_patch: bool = False,
        depth_patch_size: int = 16
    ):
        """
        Args:
            data_root: Root directory containing train/val/test folders
            split: 'train', 'val', or 'test'
            target_size: Target image size (square images)
            transform: Optional transforms to apply
            use_high_res_patch: Whether to extract high-res patch at gaze
            patch_size: Size of the high-res patch (default 96)
            return_depth_patch: Whether to return depth patch around gaze
            depth_patch_size: Size of the depth patch to extract (default 16)
        """
        super().__init__(data_root, split, transform)
        self.target_size = target_size
        self.original_size = 1408
        self.scale_factor = self.original_size / target_size
        self.use_high_res_patch = use_high_res_patch
        self.patch_size = patch_size
        self.return_depth_patch = return_depth_patch
        self.depth_patch_size = depth_patch_size
        
        print(f"Flexible resolution dataset initialized:")
        print(f"  Original size: {self.original_size}×{self.original_size}")
        print(f"  Target size: {target_size}×{target_size}")
        print(f"  Scale factor: {self.scale_factor:.2f}")
        if use_high_res_patch:
            print(f"  High-res patch: {patch_size}×{patch_size}")
            print(f"  Patch coverage: {patch_size/self.original_size*100:.1f}% of original")
        
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
        
        # Resize to target size
        # Add batch dimension for interpolation
        rgb_resized = F.interpolate(
            rgb.unsqueeze(0), 
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        depth_resized = F.interpolate(
            depth.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Create valid mask at target resolution
        valid_mask_resized = depth_resized > 0
        
        # Load and scale gaze information if available
        gaze_info = None
        gt_depth_at_gaze = None
        depth_patch_stats = None
        
        if 'gaze' in frame_info and frame_info.get('has_gaze', False):
            try:
                gaze_path = frame_info['seq_dir'] / 'gaze' / frame_info['gaze']
                if gaze_path.exists():
                    with open(gaze_path, 'r') as f:
                        gaze_data = json.load(f)
                    
                    # Extract exact GT depth at original gaze location BEFORE resizing
                    gaze_x_orig = int(round(gaze_data['x_pixel']))
                    gaze_y_orig = int(round(gaze_data['y_pixel']))
                    
                    # Ensure coordinates are within bounds
                    gaze_x_orig = max(0, min(gaze_x_orig, self.original_size - 1))
                    gaze_y_orig = max(0, min(gaze_y_orig, self.original_size - 1))
                    
                    # Extract exact depth value at gaze position from numpy array
                    gt_depth_at_gaze = depth_numpy[gaze_y_orig, gaze_x_orig]
                    
                    # Scale gaze coordinates for target size
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
                    gaze_info['x'] = max(0, min(gaze_info['x'], self.target_size - 1))
                    gaze_info['y'] = max(0, min(gaze_info['y'], self.target_size - 1))
                    
                    # Extract depth patch if requested
                    if self.return_depth_patch:
                        # Extract patch from resized depth for consistency
                        depth_patch, valid_mask_patch = self._extract_depth_patch(
                            depth_resized.squeeze(0).numpy(),
                            gaze_info['x'],
                            gaze_info['y'],
                            self.depth_patch_size
                        )
                    
                    # Extract depth patch and compute statistics from RESIZED depth
                    # This ensures consistency with what the model sees
                    depth_patch_stats = self._extract_depth_patch_statistics(
                        depth_resized.squeeze(0).numpy(),  # Remove channel dimension
                        gaze_info['x'],
                        gaze_info['y']
                    )
                    
            except Exception as e:
                # If there's any error loading gaze, just skip it
                gaze_info = None
                gt_depth_at_gaze = None
                depth_patch_stats = None
        
        # Apply transforms if any (on resized data)
        if self.transform:
            rgb_resized, depth_resized, valid_mask_resized = self.transform(
                rgb_resized, depth_resized, valid_mask_resized
            )
        
        # Extract high-res patch if requested
        high_res_patch = None
        patch_coords = None
        if self.use_high_res_patch and gaze_info is not None:
            # Load original resolution RGB for patch extraction
            rgb_original = cv2.imread(str(rgb_path))
            rgb_original = cv2.cvtColor(rgb_original, cv2.COLOR_BGR2RGB)
            
            # Extract patch centered at original gaze coordinates
            high_res_patch, patch_coords = self.extract_gaze_patch(
                rgb_original,
                gaze_info['x_original'],
                gaze_info['y_original']
            )
            
            # Convert patch to tensor and normalize
            high_res_patch = torch.from_numpy(high_res_patch).permute(2, 0, 1).float() / 255.0
        
        sample_dict = {
            'rgb': rgb_resized,
            'depth': depth_resized,
            'valid_mask': valid_mask_resized,
            'sequence': frame_info['sequence'],
            'frame_idx': frame_info['index'],
            'gaze': gaze_info,
            'gt_depth_at_gaze': gt_depth_at_gaze,  # Exact depth at gaze position
            'depth_patch_stats': depth_patch_stats,  # Statistics from patch around gaze
            'scale_factor': self.scale_factor,
            'original_size': self.original_size,
            'target_size': self.target_size
        }
        
        # Add depth patch if requested and gaze is available
        if self.return_depth_patch and gaze_info is not None:
            sample_dict['gt_depth_patch'] = torch.from_numpy(depth_patch).float()
            sample_dict['gt_depth_patch_mask'] = torch.from_numpy(valid_mask_patch).bool()
        
        # Add high-res patch data if available
        if self.use_high_res_patch:
            sample_dict['high_res_patch'] = high_res_patch
            sample_dict['patch_coords'] = patch_coords
            
        return sample_dict
    
    def _extract_depth_patch_statistics(self, depth_map, gaze_x, gaze_y):
        """Extract statistics from a patch around the gaze point.
        
        Args:
            depth_map: 2D numpy array of depth values (already resized)
            gaze_x: x coordinate in resized image
            gaze_y: y coordinate in resized image
            
        Returns:
            Dictionary of statistics or None if patch extraction fails
        """
        # Patch size (16x16 for consistency across resolutions)
        patch_size = 16
        half_patch = patch_size // 2
        
        # Get patch boundaries
        x_center = int(round(gaze_x))
        y_center = int(round(gaze_y))
        
        x_start = max(0, x_center - half_patch)
        x_end = min(depth_map.shape[1], x_center + half_patch)
        y_start = max(0, y_center - half_patch)
        y_end = min(depth_map.shape[0], y_center + half_patch)
        
        # Extract patch
        depth_patch = depth_map[y_start:y_end, x_start:x_end]
        
        # Check if patch is valid
        if depth_patch.size == 0:
            return None
            
        # Filter out invalid depths (0 or negative)
        valid_depths = depth_patch[depth_patch > 0]
        
        if len(valid_depths) < 10:  # Need at least 10 valid pixels
            return None
        
        # Compute statistics
        stats = {
            # Basic statistics
            'mean': float(np.mean(valid_depths)),
            'std': float(np.std(valid_depths) + 1e-6),  # Add epsilon for stability
            'median': float(np.median(valid_depths)),
            'min': float(np.min(valid_depths)),
            'max': float(np.max(valid_depths)),
            'range': float(np.max(valid_depths) - np.min(valid_depths)),
            
            # Gradient information (computed on full patch including zeros)
            'grad_x_mean': float(np.mean(np.abs(np.gradient(depth_patch, axis=1)))),
            'grad_y_mean': float(np.mean(np.abs(np.gradient(depth_patch, axis=0)))),
            'grad_magnitude': float(np.sqrt(
                np.mean(np.gradient(depth_patch, axis=1)**2) + 
                np.mean(np.gradient(depth_patch, axis=0)**2)
            )),
            
            # Relative metrics
            'coeff_var': float(np.std(valid_depths) / (np.mean(valid_depths) + 1e-6)),  # Normalized variance
            
            # Edge detection (high gradient relative to mean depth)
            'edge_score': float(np.max([
                np.mean(np.abs(np.gradient(depth_patch, axis=1))),
                np.mean(np.abs(np.gradient(depth_patch, axis=0)))
            ]) / (np.mean(valid_depths) + 1e-6)),
            
            # Depth bin (for classification task)
            'depth_bin': self._get_depth_bin(float(np.mean(valid_depths))),
            
            # Valid pixel ratio (how much of patch is valid)
            'valid_ratio': float(len(valid_depths) / depth_patch.size)
        }
        
        return stats
    
    def _extract_depth_patch(self, depth_map, gaze_x, gaze_y, patch_size):
        """
        Extract a patch_size x patch_size depth patch centered at gaze location.
        
        Args:
            depth_map: 2D numpy array of depth values
            gaze_x: x coordinate in the depth map
            gaze_y: y coordinate in the depth map
            patch_size: Size of the patch to extract
            
        Returns:
            depth_patch: numpy array of shape (patch_size, patch_size)
            valid_mask: boolean array indicating valid depth values
        """
        H, W = depth_map.shape
        half_size = patch_size // 2
        
        # Calculate patch boundaries
        x_center = int(round(gaze_x))
        y_center = int(round(gaze_y))
        
        # Initialize with zeros
        depth_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        
        # Calculate source and destination regions
        # Source (from depth_map)
        src_x_start = max(0, x_center - half_size)
        src_x_end = min(W, x_center + half_size)
        src_y_start = max(0, y_center - half_size)
        src_y_end = min(H, y_center + half_size)
        
        # Destination (in patch)
        dst_x_start = max(0, half_size - x_center)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_start = max(0, half_size - y_center)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # Copy the data
        depth_patch[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            depth_map[src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Create valid mask
        valid_mask = depth_patch > 0
        
        return depth_patch, valid_mask
    
    def _get_depth_bin(self, depth):
        """Convert depth to categorical bin."""
        bins = [0, 2, 4, 6, 8, float('inf')]
        for i, threshold in enumerate(bins[1:]):
            if depth < threshold:
                return i
        return len(bins) - 2  # Last bin
    
    def extract_gaze_patch(self, image: np.ndarray, gaze_x: float, gaze_y: float) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract a high-resolution patch centered at gaze location.
        
        Args:
            image: Original resolution image (1408×1408)
            gaze_x: X coordinate in original image
            gaze_y: Y coordinate in original image
            
        Returns:
            patch: Extracted patch of size patch_size×patch_size
            patch_coords: (x_start, y_start) coordinates of patch in original image
        """
        # Convert gaze coordinates to integers
        gaze_x_int = int(round(gaze_x))
        gaze_y_int = int(round(gaze_y))
        
        # Calculate patch boundaries
        half_patch = self.patch_size // 2
        
        # Initial boundaries
        x_start = gaze_x_int - half_patch
        y_start = gaze_y_int - half_patch
        x_end = x_start + self.patch_size
        y_end = y_start + self.patch_size
        
        # Handle boundaries - shift patch to stay within image
        if x_start < 0:
            x_end -= x_start
            x_start = 0
        elif x_end > self.original_size:
            x_start -= (x_end - self.original_size)
            x_end = self.original_size
            
        if y_start < 0:
            y_end -= y_start
            y_start = 0
        elif y_end > self.original_size:
            y_start -= (y_end - self.original_size)
            y_end = self.original_size
        
        # Extract patch
        patch = image[y_start:y_end, x_start:x_end]
        
        # Ensure patch is exactly patch_size×patch_size (pad if necessary)
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            # This should rarely happen, only at extreme boundaries
            padded_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=patch.dtype)
            h, w = patch.shape[:2]
            padded_patch[:h, :w] = patch
            patch = padded_patch
        
        return patch, (x_start, y_start)
    
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


def visualize_flexible_sample(sample: Dict, save_path: Optional[str] = None):
    """Visualize a flexible resolution sample with RGB, depth, and gaze location."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # RGB image
    rgb = sample['rgb'].permute(1, 2, 0).numpy()
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB ({sample['target_size']}×{sample['target_size']})")
    
    # Depth image
    depth = sample['depth'].squeeze().numpy()
    valid_mask = sample['valid_mask'].squeeze().numpy()
    depth_vis = depth.copy()
    depth_vis[~valid_mask] = np.nan
    
    im = axes[1].imshow(depth_vis, cmap='viridis')
    axes[1].set_title(f"Depth ({sample['target_size']}×{sample['target_size']})")
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
    # Test flexible resolution dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./processed_data',
                        help='Root directory of processed data')
    parser.add_argument('--target-size', type=int, default=88,
                        help='Target image size')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample')
    args = parser.parse_args()
    
    # Test loading with different sizes
    print("Testing FlexibleResolutionDataset...")
    
    for size in [88, 128, 176]:
        print(f"\nTesting with target size {size}×{size}:")
        
        dataset = FlexibleResolutionDataset(
            data_root=args.data_root,
            split='train',
            target_size=size
        )
        
        if len(dataset) > 0:
            # Load one sample
            sample = dataset[0]
            print(f"  Sample loaded successfully!")
            print(f"  RGB shape: {sample['rgb'].shape}")
            print(f"  Depth shape: {sample['depth'].shape}")
            print(f"  Valid mask shape: {sample['valid_mask'].shape}")
            print(f"  Scale factor: {sample['scale_factor']:.2f}")
            print(f"  Valid pixels: {sample['valid_mask'].sum().item() / sample['valid_mask'].numel() * 100:.1f}%")
            
            if sample['depth'][sample['valid_mask']].numel() > 0:
                print(f"  Depth range: [{sample['depth'][sample['valid_mask']].min():.3f}, "
                      f"{sample['depth'][sample['valid_mask']].max():.3f}] meters")
            
            if sample.get('gaze') is not None:
                print(f"  Gaze coordinates: ({sample['gaze']['x']:.1f}, {sample['gaze']['y']:.1f})")
                print(f"  Original gaze: ({sample['gaze']['x_original']}, {sample['gaze']['y_original']})")
            
            if args.visualize and size == args.target_size:
                print(f"\nVisualizing sample at {size}×{size}...")
                visualize_flexible_sample(sample, save_path=f'flexible_sample_{size}.png')
                print(f"Saved visualization to flexible_sample_{size}.png")
        else:
            print("No data found! Run extract_dataset.py first.")
            break