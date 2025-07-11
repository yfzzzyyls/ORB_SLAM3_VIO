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
                 load_poses=True, augment=True, target_size=None, load_gt=False):
        """
        Args:
            data_dir: Directory with processed SLAM data (from process_slam_to_sparse_depth_adt.py)
            split: 'train' or 'val' 
            transform: Optional image transforms
            load_poses: Whether to load camera poses
            augment: Whether to apply data augmentation
            target_size: (H, W) tuple for resizing, None to keep original
            load_gt: Whether to load ground truth depth
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.load_poses = load_poses
        self.augment = augment
        self.target_size = target_size
        self.load_gt = load_gt
        
        # Check if ground truth directory exists
        self.gt_dir = self.data_dir / 'ground_truth_depth'
        self.has_gt = self.gt_dir.exists() and load_gt
        
        # Load metadata
        with open(self.data_dir / 'metadata' / 'frames.json', 'r') as f:
            self.frames = json.load(f)
        
        # Load camera params if available (ADT specific)
        camera_params_path = self.data_dir / 'metadata' / 'camera_params.json'
        if camera_params_path.exists():
            with open(camera_params_path, 'r') as f:
                self.camera_params = json.load(f)
        else:
            # Default ADT SLAM camera params
            self.camera_params = {
                'width': 640,
                'height': 480,
                'fx': 242.7,
                'fy': 242.7,
                'cx': 318.08,
                'cy': 235.65
            }
        
        # Check if this is a merged dataset with predefined splits
        split_file = self.data_dir / 'metadata' / 'split.json'
        if split_file.exists():
            # Use predefined split from merge_sequences.py
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            
            if split == 'train':
                self.indices = split_info['train']
            else:
                self.indices = split_info['val']
            
            print(f"Using predefined split from merged dataset")
        else:
            # Default split for single sequence (80/20 train/val)
            n_frames = len(self.frames)
            indices = list(range(n_frames))
            random.seed(42)
            random.shuffle(indices)
            
            split_idx = int(0.8 * n_frames)
            if split == 'train':
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]
            
        # Load sequence info if available (for merged datasets)
        sequences_file = self.data_dir / 'metadata' / 'sequences.json'
        if sequences_file.exists():
            with open(sequences_file, 'r') as f:
                self.sequences = json.load(f)
            print(f"Loaded multi-sequence dataset with {len(self.sequences)} sequences")
        else:
            self.sequences = None
            
        # Load camera intrinsics
        self.K = np.load(self.data_dir / 'metadata' / 'intrinsics.npy')
        
        print(f"Loaded {len(self.indices)} frames for {split} split")
        print(f"Camera resolution: {self.camera_params['height']}x{self.camera_params['width']}")
        
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
            
            # The RGB images are already in correct orientation (640x480)
            # But sparse depth is 480x640, so we need to transpose the sparse depth
            # or resize RGB to match. Let's resize RGB to match sparse depth dimensions
        else:
            # Create blank image if RGB not available
            h = self.camera_params['height']
            w = self.camera_params['width']
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Load sparse depth
        sparse_depth = np.load(self.data_dir / 'sparse_depth' / f'{frame_str}.npy')
        confidence = np.load(self.data_dir / 'sparse_depth' / f'{frame_str}_conf.npy')
        
        # Ensure consistent shapes
        expected_h = self.camera_params['height']
        expected_w = self.camera_params['width']
        if sparse_depth.shape != (expected_h, expected_w):
            print(f"Warning: Frame {frame_id} sparse depth has shape {sparse_depth.shape}, expected ({expected_h}, {expected_w})")
        if confidence.shape != (expected_h, expected_w):
            print(f"Warning: Frame {frame_id} confidence has shape {confidence.shape}, expected ({expected_h}, {expected_w})")
        
        # Load pose if requested
        if self.load_poses:
            pose = np.load(self.data_dir / 'poses' / f'{frame_str}.npy')
        else:
            pose = np.eye(4)
        
        # Apply augmentation
        if self.augment and self.split == 'train':
            rgb, sparse_depth, confidence = self._augment(rgb, sparse_depth, confidence)
        
        # Resize if target size is specified
        if self.target_size is not None and self.target_size != (rgb.shape[0], rgb.shape[1]):
            rgb = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))
            sparse_depth = cv2.resize(sparse_depth, (self.target_size[1], self.target_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, (self.target_size[1], self.target_size[0]),
                                  interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensors (use .copy() to ensure contiguous memory)
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
        sparse_depth = torch.from_numpy(sparse_depth.copy()).unsqueeze(0).float()
        confidence = torch.from_numpy(confidence.copy()).unsqueeze(0).float()
        pose = torch.from_numpy(pose.copy()).float()
        K = torch.from_numpy(self.K.copy()).float()
        
        # Apply additional transforms if provided
        if self.transform:
            rgb = self.transform(rgb)
        
        # Build return dictionary
        # Load ground truth if available
        gt_depth = None
        if self.has_gt:
            gt_path = self.gt_dir / f'{frame_str}.npz'
            if gt_path.exists():
                gt_data = np.load(gt_path)
                gt_depth = gt_data['depth'].astype(np.float32)
                # Ensure GT matches expected dimensions (sparse depth is 640x480)
                expected_shape = (self.camera_params['height'], self.camera_params['width'])
                if gt_depth.shape != expected_shape:
                    # GT is likely 640x480 but we need height x width order
                    if gt_depth.shape == (expected_shape[1], expected_shape[0]):
                        gt_depth = gt_depth.T  # Transpose if dimensions are swapped
                    else:
                        # Resize if different size
                        gt_depth = cv2.resize(gt_depth, (expected_shape[1], expected_shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
            else:
                # Create zero GT if file not found
                gt_depth = np.zeros((self.camera_params['height'], self.camera_params['width']), dtype=np.float32)
        
        ret_dict = {
            'rgb': rgb,
            'sparse_depth': sparse_depth,
            'confidence': confidence,
            'pose': pose,
            'intrinsics': K,
            'frame_id': frame_id,
            'timestamp': frame_info['timestamp'],
            'num_sparse_points': frame_info.get('num_features', -1),
            'sparsity': frame_info.get('sparsity', 0.0)
        }
        
        if gt_depth is not None:
            # Convert to tensor
            gt_depth = torch.from_numpy(gt_depth.copy()).unsqueeze(0).float()
            ret_dict['gt_depth'] = gt_depth
        
        # Add sequence information if available
        if 'sequence' in frame_info:
            ret_dict['sequence'] = frame_info['sequence']
            ret_dict['sequence_idx'] = frame_info.get('sequence_idx', -1)
            ret_dict['original_frame_id'] = frame_info.get('original_frame_id', frame_id)
        
        return ret_dict
    
    def _augment(self, rgb, sparse_depth, confidence):
        """Apply data augmentation"""
        h, w = rgb.shape[:2]
        
        # Random horizontal flip
        if random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            sparse_depth = np.fliplr(sparse_depth).copy()  # Make contiguous copy
            confidence = np.fliplr(confidence).copy()  # Make contiguous copy
        
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
                      sequential=False, sequence_length=3, target_size=None, load_gt=False):
    """Create train and validation dataloaders
    
    Args:
        data_dir: Directory with processed SLAM data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        sequential: Whether to use sequential dataset
        sequence_length: Length of sequences if sequential
        target_size: (H, W) tuple for resizing, None to keep original ADT size
        load_gt: Whether to load ground truth depth
    """
    
    if sequential:
        train_dataset = SequentialSLAMDataset(
            data_dir, 
            split='train',
            sequence_length=sequence_length,
            augment=True,
            target_size=target_size
        )
        val_dataset = SequentialSLAMDataset(
            data_dir,
            split='val', 
            sequence_length=sequence_length,
            augment=False,
            target_size=target_size
        )
    else:
        train_dataset = SLAMDepthDataset(data_dir, split='train', augment=True, target_size=target_size, load_gt=load_gt)
        val_dataset = SLAMDepthDataset(data_dir, split='val', augment=False, target_size=target_size, load_gt=load_gt)
    
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