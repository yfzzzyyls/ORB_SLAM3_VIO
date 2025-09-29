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
import cv2
from spatial_dual_resolution_coordconv import GaussianDownsample


class SpatialDualResolutionDataset(Dataset):
    def __init__(self, data_root, split='train', context_size=88, patch_size=88, 
                 output_size=22, augment=True, max_sequences=None, random_seed=42,
                 k_extra=4, edge_bias_prob=0.3):
        """
        Dataset for spatial dual-resolution training with multi-point sampling.
        
        Args:
            data_root: Root directory containing train/val/test splits
            split: 'train', 'val', or 'test'
            context_size: Size of downsampled context (88x88)
            patch_size: Size of high-res patch (88x88)
            output_size: Size of output depth map (22x22)
            augment: Whether to apply augmentations
            max_sequences: Maximum number of sequences to use (None = use all)
            random_seed: Random seed for reproducible sequence selection
            k_extra: Number of extra points to sample per image (training only)
            edge_bias_prob: Probability of edge-biased sampling (vs uniform)
        """
        self.data_root = data_root
        self.split = split
        self.context_size = context_size
        self.patch_size = patch_size
        self.output_size = output_size
        self.augment = augment and (split == 'train')
        self.k_extra = k_extra if split == 'train' else 0  # Only add extra points during training
        self.edge_bias_prob = edge_bias_prob
        
        # Original image size
        self.original_size = 1408
        
        # Create Gaussian downsampler once in __init__ (Fix #3 - avoid per-item construction)
        self.gaussian_downsample = GaussianDownsample(scale_factor=4)
        
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
        if self.k_extra > 0:
            print(f"Will sample {self.k_extra} extra points per image (edge bias: {edge_bias_prob*100:.0f}%)")
        
    def __len__(self):
        # During training, we effectively have more samples
        base_len = len(self.samples)
        if self.k_extra > 0:
            return base_len * (1 + self.k_extra)
        return base_len
        
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
    
    def _compute_edge_map(self, depth):
        """Compute edge magnitude using Sobel filter."""
        # Convert invalid depths to 0 for edge detection
        depth_clean = depth.copy()
        depth_clean[depth <= 0] = 0
        
        # Compute Sobel gradients
        sobel_x = cv2.Sobel(depth_clean, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_clean, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute magnitude
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Zero out edges at invalid regions
        edge_magnitude[depth <= 0] = 0
        
        return edge_magnitude
    
    def _sample_valid_point(self, depth, edge_map=None, edge_biased=False):
        """Sample a valid point from the depth map.
        
        Args:
            depth: Full resolution depth map [H, W]
            edge_map: Edge magnitude map for edge-biased sampling
            edge_biased: Whether to use edge-biased sampling
        
        Returns:
            (x, y) coordinates in pixel space, or None if no valid point found
        """
        # Create validity mask (with some margin from borders)
        margin = self.patch_size // 2 + 10
        valid_mask = np.zeros_like(depth, dtype=bool)
        valid_mask[margin:-margin, margin:-margin] = depth[margin:-margin, margin:-margin] > 0
        
        if not valid_mask.any():
            return None
            
        if edge_biased and edge_map is not None:
            # Edge-biased sampling: sample proportional to edge magnitude
            # Apply validity mask
            edge_weights = edge_map * valid_mask.astype(np.float32)
            
            # Add small epsilon to ensure some probability everywhere valid
            edge_weights = edge_weights + 0.01 * valid_mask.astype(np.float32)
            
            # Normalize to probability distribution
            edge_weights = edge_weights / edge_weights.sum()
            
            # Sample from distribution
            flat_idx = np.random.choice(
                edge_weights.size, 
                p=edge_weights.ravel()
            )
            y, x = np.unravel_index(flat_idx, edge_weights.shape)
        else:
            # Uniform sampling from valid pixels
            valid_coords = np.where(valid_mask)
            idx = np.random.randint(len(valid_coords[0]))
            y, x = valid_coords[0][idx], valid_coords[1][idx]
        
        # Add small random offset for sub-pixel coordinates
        x = x + np.random.uniform(-0.5, 0.5)
        y = y + np.random.uniform(-0.5, 0.5)
        
        # Clamp to valid range
        x = np.clip(x, margin, self.original_size - margin - 1)
        y = np.clip(y, margin, self.original_size - margin - 1)
        
        return x, y
        
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
    
    def _extract_patch_float(self, image, gaze_x_float, gaze_y_float, patch_size):
        """Extract patch using float coordinates with grid_sample for sub-pixel accuracy (Fix #2)."""
        if isinstance(image, torch.Tensor):
            # For RGB tensors
            B = 1 if image.dim() == 3 else image.shape[0]
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dim
            
            H, W = image.shape[-2:]
            device = image.device
            
            # Create sampling grid for patch_size x patch_size
            half = patch_size / 2.0
            # Grid in pixel coordinates
            y_coords = torch.linspace(-half + 0.5, half - 0.5, patch_size, device=device)
            x_coords = torch.linspace(-half + 0.5, half - 0.5, patch_size, device=device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Add gaze offset and normalize to [-1, 1] with align_corners=False convention
            # FIX: Use (coord + 0.5) / size normalization for align_corners=False
            grid_x = 2.0 * (grid_x + gaze_x_float + 0.5) / self.original_size - 1.0
            grid_y = 2.0 * (grid_y + gaze_y_float + 0.5) / self.original_size - 1.0
            
            # Stack for grid_sample [1, H, W, 2]
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            
            # Sample with reflection padding, align_corners=False for consistency
            patch = F.grid_sample(image, grid, mode='bilinear', 
                                padding_mode='reflection', align_corners=False)
            
            return patch.squeeze(0) if B == 1 else patch
        else:
            # For numpy depth - convert to tensor, process, convert back
            depth_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            patch_tensor = self._extract_patch_float(depth_tensor, gaze_x_float, gaze_y_float, patch_size)
            return patch_tensor.squeeze(0).squeeze(0).numpy()
            
    def __getitem__(self, idx):
        # Determine which base sample and which point (gaze or extra)
        base_idx = idx % len(self.samples)
        point_idx = idx // len(self.samples)  # 0 = gaze, 1-k_extra = extra points
        
        sample = self.samples[base_idx]
        
        # Load data
        rgb = self._load_rgb(sample['rgb_path'])
        depth = self._load_depth(sample['depth_path'])
        gaze_data = self._load_gaze(sample['gaze_path'])
        
        # Get original gaze coordinates (already in pixels)
        orig_gaze_x = gaze_data['x_pixel']
        orig_gaze_y = gaze_data['y_pixel']
        
        # Data augmentation (apply same augmentation to all points from this image)
        if self.augment:
            # Store augmentation parameters
            do_flip = random.random() > 0.5
            brightness_factor = 0.8 + random.random() * 0.4
            contrast_factor = 0.8 + random.random() * 0.4
            gaze_shift_x = random.randint(-10, 10)
            gaze_shift_y = random.randint(-10, 10)
            
            # Apply augmentations
            if do_flip:
                rgb = TF.hflip(rgb)
                depth = np.fliplr(depth).copy()
                orig_gaze_x = self.original_size - 1 - orig_gaze_x
                
            rgb = TF.adjust_brightness(rgb, brightness_factor)
            rgb = TF.adjust_contrast(rgb, contrast_factor)
        else:
            gaze_shift_x = gaze_shift_y = 0
        
        # Determine which point to use for this sample
        if point_idx == 0:
            # Use original gaze point (with potential augmentation shift)
            gaze_x = orig_gaze_x + gaze_shift_x
            gaze_y = orig_gaze_y + gaze_shift_y
            is_real_gaze = True
        else:
            # Sample an extra point
            # Compute edge map for potential edge-biased sampling
            edge_map = self._compute_edge_map(depth) if self.edge_bias_prob > 0 else None
            
            # Decide sampling strategy
            use_edge_bias = random.random() < self.edge_bias_prob
            
            # Sample a valid point
            point = self._sample_valid_point(depth, edge_map, use_edge_bias)
            
            if point is None:
                # Fallback to a random point near center if no valid point found
                gaze_x = self.original_size // 2 + random.randint(-100, 100)
                gaze_y = self.original_size // 2 + random.randint(-100, 100)
            else:
                gaze_x, gaze_y = point
            
            is_real_gaze = False
        
        # Clamp to valid range
        gaze_x = np.clip(gaze_x, 0, self.original_size - 1)
        gaze_y = np.clip(gaze_y, 0, self.original_size - 1)
        
        # Convert RGB to tensor (normalized to [0,1])
        rgb_tensor = TF.to_tensor(rgb)
        
        # Extract GT depth patch (88x88) using FLOAT coordinates (Fix #2)
        depth_patch_88 = self._extract_patch_float(depth, gaze_x, gaze_y, self.patch_size)
        # Ensure contiguous array before converting to tensor
        if not depth_patch_88.flags['C_CONTIGUOUS']:
            depth_patch_88 = depth_patch_88.copy()
        depth_patch = torch.from_numpy(depth_patch_88).unsqueeze(0)
        
        # Create mask before downsampling
        mask_88 = (depth_patch > 0).float()
        
        # Use Gaussian downsampling instead of box average (Fix #5)
        # Now using self.gaussian_downsample created in __init__
        
        # Downsample depth with mask weighting
        # FIX: Ensure 4D tensor for GaussianDownsample [B, C, H, W]
        depth_weighted = depth_patch * mask_88
        # depth_patch is already [1, 88, 88], need to add batch dim for downsampler
        if depth_weighted.dim() == 3:
            depth_weighted = depth_weighted.unsqueeze(0)  # [1, 1, 88, 88]
            mask_88_4d = mask_88.unsqueeze(0)  # [1, 1, 88, 88]
        else:
            mask_88_4d = mask_88
            
        depth_22_weighted = self.gaussian_downsample(depth_weighted)  # [1, 1, 22, 22]
        mask_22 = self.gaussian_downsample(mask_88_4d)  # [1, 1, 22, 22]
        
        # Remove batch dim if added
        if depth_22_weighted.shape[0] == 1:
            depth_22_weighted = depth_22_weighted.squeeze(0)  # [1, 22, 22]
            mask_22 = mask_22.squeeze(0)  # [1, 22, 22]
            
        depth_output = depth_22_weighted / (mask_22 + 1e-6)
        
        # Create valid mask (consider valid if > 25% of Gaussian support was valid)
        valid_mask = (mask_22 > 0.25).float()
        
        # NEW: Get scalar gaze depth at exact float coordinates (Fix #1)
        # Bilinear sample the original full-resolution depth
        depth_full_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
        # Create a single-point grid at gaze location with align_corners=False convention
        gaze_x_norm_temp = 2.0 * (gaze_x + 0.5) / self.original_size - 1.0
        gaze_y_norm_temp = 2.0 * (gaze_y + 0.5) / self.original_size - 1.0
        gaze_grid = torch.tensor([[[[gaze_x_norm_temp, gaze_y_norm_temp]]]], dtype=torch.float32)
        
        # Sample depth at exact gaze point (use reflection padding for consistency)
        gaze_depth_sample = F.grid_sample(depth_full_tensor, gaze_grid, 
                                         mode='bilinear', padding_mode='reflection', 
                                         align_corners=False)
        gaze_depth_gt = gaze_depth_sample.squeeze().item()
        
        # FIX: Check validity using bilinear sampled mask, not nearest pixel
        valid_map = torch.from_numpy((depth > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        gaze_valid_sample = F.grid_sample(valid_map, gaze_grid,
                                         mode='bilinear', padding_mode='reflection',
                                         align_corners=False)
        gaze_is_valid = gaze_valid_sample.squeeze().item() > 0.5
        if not gaze_is_valid:
            gaze_depth_gt = 0.0
        
        # Normalize gaze coordinates to [-1, 1] with align_corners=False convention
        gaze_x_norm = 2.0 * (gaze_x + 0.5) / self.original_size - 1.0
        gaze_y_norm = 2.0 * (gaze_y + 0.5) / self.original_size - 1.0
        
        return {
            'rgb_full': rgb_tensor,
            'depth': depth_output.squeeze(0),  # Remove channel dim
            'valid_mask': valid_mask.squeeze(0),
            'gaze_x': torch.tensor(gaze_x_norm, dtype=torch.float32),
            'gaze_y': torch.tensor(gaze_y_norm, dtype=torch.float32),
            'gaze_depth_gt': torch.tensor(gaze_depth_gt, dtype=torch.float32),  # NEW: scalar gaze GT
            'is_real_gaze': torch.tensor(is_real_gaze, dtype=torch.bool),  # Track if real gaze
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
    rgb_full = torch.stack([s['rgb_full'] for s in batch])
    depth = torch.stack([s['depth'] for s in batch])
    valid_mask = torch.stack([s['valid_mask'] for s in batch])
    gaze_x = torch.stack([s['gaze_x'] for s in batch])
    gaze_y = torch.stack([s['gaze_y'] for s in batch])
    gaze_depth_gt = torch.stack([s['gaze_depth_gt'] for s in batch])  # NEW
    is_real_gaze = torch.stack([s['is_real_gaze'] for s in batch])  # NEW
    
    # Keep metadata as lists
    seqs = [s['seq'] for s in batch]
    frame_ids = [s['frame_id'] for s in batch]
    
    return {
        'rgb_full': rgb_full,
        'depth': depth,
        'valid_mask': valid_mask,
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
        'gaze_depth_gt': gaze_depth_gt,  # NEW
        'is_real_gaze': is_real_gaze,  # NEW: track real vs sampled
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
        print(f"  RGB full shape: {sample['rgb_full'].shape}")
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
    print(f"  RGB full shape: {batch['rgb_full'].shape}")
    print(f"  Depth shape: {batch['depth'].shape}")
    print(f"  Gaze shape: ({batch['gaze_x'].shape}, {batch['gaze_y'].shape})")
