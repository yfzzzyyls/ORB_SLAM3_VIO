import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
import random


class SpatialDualResolutionDataset(Dataset):
    def __init__(self, data_root, split='train', context_size=88, patch_size=88, 
                 output_size=22, augment=True, max_sequences=None, random_seed=42):
        """
        Dataset for spatial dual-resolution training.
        
        Args:
            data_root: Root directory containing train/val/test splits
            split: 'train', 'val', or 'test'
            context_size: Size of downsampled context (88x88)
            patch_size: Size of high-res patch (88x88)
            output_size: Size of output depth map (22x22)
            augment: Whether to apply augmentations
            max_sequences: Maximum number of sequences to use (None = use all)
            random_seed: Random seed for reproducible sequence selection
        """
        self.data_root = data_root
        self.split = split
        self.context_size = context_size
        self.patch_size = patch_size
        self.output_size = output_size
        self.augment = augment and (split == 'train')
        
        # Original image size
        self.original_size = 1408
        
        # Build file list
        self.samples = []
        split_dir = os.path.join(data_root, split)
        
        # Get all sequences in split
        all_sequences = sorted([d for d in os.listdir(split_dir) 
                          if os.path.isdir(os.path.join(split_dir, d))])
        
        # Optionally limit number of sequences
        if max_sequences is not None and len(all_sequences) > max_sequences:
            # Set random seed for reproducibility
            rng = random.Random(random_seed)
            sequences = rng.sample(all_sequences, max_sequences)
            sequences.sort()  # Sort for consistent ordering
            print(f"[{split}] Randomly selected {max_sequences} sequences from {len(all_sequences)} available:")
            for seq in sequences:
                print(f"  - {seq}")
        else:
            sequences = all_sequences
            print(f"[{split}] Using all {len(sequences)} sequences")
        
        for seq in sequences:
            seq_dir = os.path.join(split_dir, seq)
            metadata_file = os.path.join(seq_dir, 'metadata.json')
            
            if not os.path.exists(metadata_file):
                continue
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Add each frame
            for frame_info in metadata['frames']:
                # Check if this frame has both depth and gaze data
                if frame_info.get('depth') and frame_info.get('has_gaze', False):
                    self.samples.append({
                        'seq': seq,
                        'frame_id': frame_info.get('index', frame_info.get('frame_number', 0)),
                        'rgb_path': os.path.join(seq_dir, 'rgb', frame_info['rgb']),
                        'depth_path': os.path.join(seq_dir, 'depth', frame_info['depth']),
                        'gaze_path': os.path.join(seq_dir, 'gaze', frame_info['gaze'])
                    })
        
        print(f"Found {len(self.samples)} samples for {split} split")
        
    def __len__(self):
        return len(self.samples)
        
    def _load_rgb(self, path):
        """Load RGB image."""
        return Image.open(path).convert('RGB')
        
    def _load_depth(self, path):
        """Load depth map from npz file."""
        data = np.load(path)
        depth = data['depth'].astype(np.float32) / 1000.0  # mm to meters
        return depth
        
    def _load_gaze(self, path):
        """Load gaze data from json file."""
        with open(path, 'r') as f:
            gaze_data = json.load(f)
        return gaze_data
        
    def _extract_patch(self, image, center_x, center_y, patch_size):
        """Extract a gaze-centered patch using reflect padding at borders.
        Supports torch tensors [C,H,W] or [H,W] and numpy arrays [H,W]."""
        half = patch_size // 2

        # Helper to slice after padding
        def _slice(arr, cx, cy):
            x1, y1 = cx - half, cy - half
            x2, y2 = x1 + patch_size, y1 + patch_size
            return (arr[:, y1:y2, x1:x2] if isinstance(arr, torch.Tensor) and arr.dim() == 3
                    else arr[y1:y2, x1:x2])

        if isinstance(image, torch.Tensor):
            H, W = (image.shape[-2], image.shape[-1]) if image.dim() == 3 else image.shape
            pad_l = max(0, half - center_x)
            pad_r = max(0, center_x + half - W)
            pad_t = max(0, half - center_y)
            pad_b = max(0, center_y + half - H)
            if pad_l or pad_r or pad_t or pad_b:
                if image.dim() == 3:
                    image = F.pad(image, (pad_l, pad_r, pad_t, pad_b), mode='reflect')
                else:
                    image = F.pad(image.unsqueeze(0), (pad_l, pad_r, pad_t, pad_b), mode='reflect').squeeze(0)
                center_x += pad_l
                center_y += pad_t
            return _slice(image, center_x, center_y)

        # numpy depth maps
        H, W = image.shape[-2:]
        pad_l = max(0, half - center_x)
        pad_r = max(0, center_x + half - W)
        pad_t = max(0, half - center_y)
        pad_b = max(0, center_y + half - H)
        if pad_l or pad_r or pad_t or pad_b:
            image = np.pad(image, ((pad_t, pad_b), (pad_l, pad_r)), mode='reflect')
            center_x += pad_l
            center_y += pad_t
        return _slice(image, center_x, center_y)
            
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load data
        rgb = self._load_rgb(sample['rgb_path'])
        depth = self._load_depth(sample['depth_path'])
        gaze_data = self._load_gaze(sample['gaze_path'])
        
        # Get gaze coordinates (already in pixels)
        gaze_x = gaze_data['x_pixel']
        gaze_y = gaze_data['y_pixel']
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                depth = np.fliplr(depth).copy()  # .copy() ensures contiguous array
                gaze_x = self.original_size - 1 - gaze_x
                
            # Random color jitter
            rgb = TF.adjust_brightness(rgb, 0.8 + random.random() * 0.4)
            rgb = TF.adjust_contrast(rgb, 0.8 + random.random() * 0.4)
            
            # Small random shift to gaze position (±10 pixels)
            gaze_x += random.randint(-10, 10)
            gaze_y += random.randint(-10, 10)
            
        # Clamp gaze to valid range
        gaze_x = np.clip(gaze_x, 0, self.original_size - 1)
        gaze_y = np.clip(gaze_y, 0, self.original_size - 1)
        
        # Convert RGB to tensor
        rgb_tensor = TF.to_tensor(rgb)
        
        # Create context: downsample full image
        context_rgb = TF.resize(rgb_tensor, [self.context_size, self.context_size], 
                               interpolation=InterpolationMode.BILINEAR)
        
        # Create patch: extract high-res crop at gaze
        patch_rgb = self._extract_patch(rgb_tensor, int(gaze_x), int(gaze_y), self.patch_size)
        
        # Extract GT depth patch (88x88) then downsample to output size
        depth_patch_88 = self._extract_patch(depth, int(gaze_x), int(gaze_y), self.patch_size)
        # Ensure contiguous array before converting to tensor
        if not depth_patch_88.flags['C_CONTIGUOUS']:
            depth_patch_88 = depth_patch_88.copy()
        depth_patch = torch.from_numpy(depth_patch_88).unsqueeze(0)
        
        # Create mask before downsampling
        mask_88 = (depth_patch > 0).float()
        
        # Downsample mask and depth separately to avoid mixing invalid pixels
        mask_22 = F.avg_pool2d(mask_88, kernel_size=4, stride=4)  # 88 -> 22
        depth_sum = F.avg_pool2d(depth_patch * mask_88, kernel_size=4, stride=4)
        depth_output = depth_sum / (mask_22 + 1e-6)  # Avoid division by zero
        
        # Create valid mask (consider valid if > 50% of pixels in 4x4 region were valid)
        valid_mask = (mask_22 > 0.5).float()
        
        # Normalize gaze coordinates to [-1, 1]
        gaze_x_norm = (gaze_x / (self.original_size - 1)) * 2 - 1
        gaze_y_norm = (gaze_y / (self.original_size - 1)) * 2 - 1
        
        return {
            'context_rgb': context_rgb,
            'patch_rgb': patch_rgb,
            'depth': depth_output.squeeze(0),  # Remove channel dim
            'valid_mask': valid_mask.squeeze(0),
            'gaze_x': torch.tensor(gaze_x_norm, dtype=torch.float32),
            'gaze_y': torch.tensor(gaze_y_norm, dtype=torch.float32),
            'seq': sample['seq'],
            'frame_id': sample['frame_id']
        }


def custom_collate_fn(batch):
    """Custom collate function to handle samples with missing data."""
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    if len(batch) == 0:
        return None
        
    # Stack tensors
    context_rgb = torch.stack([s['context_rgb'] for s in batch])
    patch_rgb = torch.stack([s['patch_rgb'] for s in batch])
    depth = torch.stack([s['depth'] for s in batch])
    valid_mask = torch.stack([s['valid_mask'] for s in batch])
    gaze_x = torch.stack([s['gaze_x'] for s in batch])
    gaze_y = torch.stack([s['gaze_y'] for s in batch])
    
    # Keep metadata as lists
    seqs = [s['seq'] for s in batch]
    frame_ids = [s['frame_id'] for s in batch]
    
    return {
        'context_rgb': context_rgb,
        'patch_rgb': patch_rgb,
        'depth': depth,
        'valid_mask': valid_mask,
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
        'seqs': seqs,
        'frame_ids': frame_ids
    }


if __name__ == '__main__':
    # Test the dataset
    import torch.nn.functional as F
    
    dataset = SpatialDualResolutionDataset(
        data_root='./processed_data',
        split='train',
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Sequence: {sample['seq']}")
        print(f"  Frame ID: {sample['frame_id']}")
        print(f"  Context RGB shape: {sample['context_rgb'].shape}")
        print(f"  Patch RGB shape: {sample['patch_rgb'].shape}")
        print(f"  Depth shape: {sample['depth'].shape}")
        print(f"  Valid mask shape: {sample['valid_mask'].shape}")
        print(f"  Gaze: ({sample['gaze_x']:.3f}, {sample['gaze_y']:.3f})")
        print(f"  Valid pixels: {sample['valid_mask'].sum().item()}/{sample['valid_mask'].numel()}")
        
    # Test dataloader with collate function
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, 
                       collate_fn=custom_collate_fn, num_workers=2)
    
    batch = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  Context RGB shape: {batch['context_rgb'].shape}")
    print(f"  Patch RGB shape: {batch['patch_rgb'].shape}")
    print(f"  Depth shape: {batch['depth'].shape}")
    print(f"  Gaze shape: ({batch['gaze_x'].shape}, {batch['gaze_y'].shape})")