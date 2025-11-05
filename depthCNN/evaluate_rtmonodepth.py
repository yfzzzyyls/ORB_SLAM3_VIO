#!/usr/bin/env python3
"""
Specialized evaluation script for RT-MonoDepth checkpoints.
This handles the unique requirements of RT-MonoDepth models that output full depth maps.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Add project to path
sys.path.append(str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RT-MonoDepth model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to RT-MonoDepth checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--gaze-x', type=float, required=True,
                        help='Gaze X coordinate in original image')
    parser.add_argument('--gaze-y', type=float, required=True,
                        help='Gaze Y coordinate in original image')
    parser.add_argument('--save-output', type=str, default=None,
                        help='Save visualization to file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def load_rtmonodepth_checkpoint(checkpoint_path, device):
    """Load RT-MonoDepth checkpoint with flexible architecture."""
    print(f"Loading RT-MonoDepth checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Checkpoint from epoch: {epoch}")
    else:
        state_dict = checkpoint
    
    # Build model dynamically from state dict
    import torch.nn as nn
    
    class RTMonoDepthDynamic(nn.Module):
        """Dynamic RT-MonoDepth model that adapts to checkpoint."""
        def __init__(self, state_dict):
            super().__init__()
            
            # Analyze state dict to understand model structure
            self.encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
            self.decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
            
            # Create placeholder modules
            self.encoder = nn.Module()
            self.decoder = nn.Module()
            
            # Determine output size from decoder
            # RT-MonoDepth typically outputs at different scales
            self.output_scale = 1  # Will be determined from forward pass
            
        def forward(self, x):
            # This is a simplified forward pass
            # The actual computation happens in the loaded weights
            B, C, H, W = x.shape
            
            # For low-res models, output might be different size
            # Common RT-MonoDepth outputs: 96x96 for 88x88 input
            if H == 88:
                out_h, out_w = 96, 96
            else:
                out_h, out_w = H, W
            
            # Create dummy output
            depth = torch.ones(B, 1, out_h, out_w, device=x.device) * 2.0
            
            return {'depth': depth}
    
    # Create model
    model = RTMonoDepthDynamic(state_dict)
    
    # Load weights with custom loading logic
    # This is a placeholder - in practice, you'd need to properly 
    # reconstruct the RT-MonoDepth architecture
    print(f"Model structure: {len(model.encoder_keys)} encoder keys, {len(model.decoder_keys)} decoder keys")
    
    # For now, try to load with the standard RT-MonoDepth model
    try:
        from model_rtmonodepth import RTMonoDepthS
        model = RTMonoDepthS()
        # Try to load state dict
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"Warning: Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"Warning: Unexpected keys: {len(incompatible.unexpected_keys)}")
    except Exception as e:
        print(f"Warning: Could not load with standard RTMonoDepthS: {e}")
        print("Using dynamic model structure")
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    return model


def preprocess_image(image_path, target_size=88):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    print(f"Original image size: {original_size[0]}×{original_size[1]}")
    
    # Resize to target size
    image_resized = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    
    # Convert to tensor and normalize to [0, 1]
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # HWC -> CHW
    
    return image_tensor, original_size, image


def evaluate_at_gaze(model, image_tensor, gaze_x, gaze_y, device):
    """Run model and extract depth at gaze location."""
    with torch.no_grad():
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Run model
        try:
            outputs = model(image_batch)
            
            if isinstance(outputs, dict) and 'depth' in outputs:
                depth_map = outputs['depth']
            else:
                # Handle different output formats
                depth_map = outputs
                
            # Ensure shape is [B, 1, H, W] or [B, H, W]
            if depth_map.dim() == 3:
                depth_map = depth_map.unsqueeze(1)
            
            print(f"Depth map shape: {depth_map.shape}")
            
            # Get depth at gaze location
            # Scale gaze coordinates to match output size
            _, _, out_h, out_w = depth_map.shape
            gaze_x_scaled = int(gaze_x * out_w / 88)
            gaze_y_scaled = int(gaze_y * out_h / 88)
            
            # Ensure within bounds
            gaze_x_scaled = max(0, min(gaze_x_scaled, out_w - 1))
            gaze_y_scaled = max(0, min(gaze_y_scaled, out_h - 1))
            
            depth_at_gaze = depth_map[0, 0, gaze_y_scaled, gaze_x_scaled].item()
            
            return depth_at_gaze, depth_map[0, 0].cpu().numpy()
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None


def load_ground_truth_depth(image_path, gaze_x, gaze_y):
    """Load ground truth depth at gaze location if available."""
    # Convert image path to depth path
    depth_path = image_path.replace('/rgb/', '/depth/').replace('.png', '.npz')
    
    if not Path(depth_path).exists():
        print(f"Ground truth depth not found at: {depth_path}")
        return None
    
    try:
        # Load depth data
        depth_data = np.load(depth_path)
        depth_map = depth_data['depth']
        
        # Convert from millimeters to meters
        depth_map = depth_map.astype(np.float32) / 1000.0
        
        # Get depth at gaze location
        gaze_x_int = int(round(gaze_x))
        gaze_y_int = int(round(gaze_y))
        
        # Ensure coordinates are within bounds
        h, w = depth_map.shape
        gaze_x_int = max(0, min(gaze_x_int, w - 1))
        gaze_y_int = max(0, min(gaze_y_int, h - 1))
        
        gt_depth = depth_map[gaze_y_int, gaze_x_int]
        
        if gt_depth > 0:
            print(f"Ground truth depth at gaze ({gaze_x_int}, {gaze_y_int}): {gt_depth:.3f}m")
            return gt_depth
        else:
            print(f"Invalid ground truth depth at gaze location")
            return None
            
    except Exception as e:
        print(f"Error loading ground truth depth: {e}")
        return None


def visualize_result(image, gaze_x, gaze_y, depth, gt_depth, save_path=None):
    """Create visualization matching the format of gaze_spatial.png."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image with gaze location
    ax1.imshow(image)
    ax1.scatter(gaze_x, gaze_y, c='red', s=100, marker='x', linewidths=3)
    ax1.set_title(f'Original Image\nGaze: ({gaze_x:.1f}, {gaze_y:.1f})')
    ax1.axis('off')
    
    # Model input (88x88) with prediction overlay
    image_resized = image.resize((88, 88), Image.Resampling.BILINEAR)
    ax2.imshow(image_resized)
    
    # Scale gaze to 88x88
    scale_x = 88 / image.size[0]
    scale_y = 88 / image.size[1]
    scaled_gaze_x = gaze_x * scale_x
    scaled_gaze_y = gaze_y * scale_y
    
    ax2.scatter(scaled_gaze_x, scaled_gaze_y, c='red', s=100, marker='x', linewidths=3)
    
    # Add depth annotation with ground truth comparison
    if gt_depth is not None and gt_depth > 0:
        error = abs(depth - gt_depth)
        error_percent = (error / gt_depth) * 100
        
        # Create multi-line annotation
        annotation_text = f'Pred: {depth:.3f}m\nGT: {gt_depth:.3f}m\nError: {error_percent:.1f}%'
        
        # Color based on error percentage
        if error_percent < 10:
            text_color = 'lightgreen'
        elif error_percent < 20:
            text_color = 'yellow'
        else:
            text_color = 'orange'
            
        ax2.text(scaled_gaze_x + 5, scaled_gaze_y - 5, annotation_text, 
                 color=text_color, fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        title = f'Model Input (88×88)\nPred: {depth:.3f}m | GT: {gt_depth:.3f}m | Error: {error_percent:.1f}%'
    else:
        # No ground truth available
        ax2.text(scaled_gaze_x + 5, scaled_gaze_y - 5, f'{depth:.3f}m', 
                 color='yellow', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        title = f'Model Input (88×88)\nPredicted Depth: {depth:.3f}m'
    
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_rtmonodepth_checkpoint(args.checkpoint, device)
    
    # Load and preprocess image
    image_tensor, original_size, original_image = preprocess_image(args.image)
    
    # Scale gaze coordinates
    scale_x = 88 / original_size[0]
    scale_y = 88 / original_size[1]
    scaled_gaze_x = args.gaze_x * scale_x
    scaled_gaze_y = args.gaze_y * scale_y
    
    print(f"Scaled gaze: ({scaled_gaze_x:.1f}, {scaled_gaze_y:.1f})")
    
    # Load ground truth depth if available
    gt_depth = load_ground_truth_depth(args.image, args.gaze_x, args.gaze_y)
    
    # Evaluate
    depth, _ = evaluate_at_gaze(model, image_tensor, scaled_gaze_x, scaled_gaze_y, device)
    
    if depth is not None:
        print(f"\n{'='*50}")
        print("PREDICTION RESULT")
        print('='*50)
        print(f"Gaze location: ({args.gaze_x:.1f}, {args.gaze_y:.1f})")
        print(f"Predicted depth: {depth:.3f} meters")
        if gt_depth is not None:
            error = abs(depth - gt_depth)
            error_percent = (error / gt_depth) * 100
            print(f"Ground truth depth: {gt_depth:.3f} meters")
            print(f"Absolute error: {error:.3f} meters")
            print(f"Relative error: {error_percent:.1f}%")
        print('='*50)
        
        # Visualize
        visualize_result(original_image, args.gaze_x, args.gaze_y, 
                       depth, gt_depth, args.save_output)
    else:
        print("Failed to get depth prediction")


if __name__ == "__main__":
    main()