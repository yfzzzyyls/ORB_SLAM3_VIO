#!/usr/bin/env python3
"""
Demo evaluation script for gaze-based depth prediction.
Shows how to use a trained model to predict depth at a gaze location in a single image.

Usage:
python evaluate_demo.py --checkpoint path/to/checkpoint.pth --image path/to/image.png --gaze-x 44 --gaze-y 44
"""

import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Add project to path
sys.path.append(str(Path(__file__).parent))

from flexible_gaze_encoder import FlexibleGazeOnlyDepth, DualResolutionGazeDepth
from lightweight_dual_resolution import LightweightDualResolution
from spatial_patch_encoder import SpatialPatchDepthPredictor
from spatial_patch_encoder_aux import SpatialPatchDepthPredictorWithAux
from model_rtmonodepth import RTMonoDepthS


def parse_args():
    parser = argparse.ArgumentParser(description='Demo: Predict depth at gaze location in a single image')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image (PNG or JPG)')
    parser.add_argument('--gaze-x', type=float, required=True,
                        help='Gaze X coordinate in image pixels')
    parser.add_argument('--gaze-y', type=float, required=True,
                        help='Gaze Y coordinate in image pixels')
    
    # Optional model configuration
    parser.add_argument('--model-type', type=str, default='auto',
                        choices=['auto', 'single', 'multitask', 'dual', 'lightweight_dual', 'spatial', 'spatial_aux'],
                        help='Model type (auto-detect from checkpoint)')
    parser.add_argument('--image-size', type=int, default=88,
                        help='Model input size (default: 88)')
    parser.add_argument('--encoder-levels', type=int, default=3,
                        help='Number of encoder levels')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='Base channels for encoder')
    
    # Dual-resolution specific
    parser.add_argument('--patch-size', type=int, default=96,
                        help='High-res patch size for dual-resolution models')
    parser.add_argument('--context-region-size', type=int, default=6,
                        help='Context region size for lightweight dual model')
    
    # Output options
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization of input and gaze location')
    parser.add_argument('--save-output', type=str, default=None,
                        help='Save visualization to file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def detect_model_type(checkpoint_path):
    """Auto-detect model type from checkpoint path and state dict."""
    path_lower = checkpoint_path.lower()
    
    # First, try to load checkpoint to check state dict and args
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Also check for saved args which might have model info
            args = checkpoint.get('args', {})
        else:
            state_dict = checkpoint
            args = {}
        
        # Check for model type in saved args first
        if 'model_type' in args:
            return args['model_type']
        
        # Check state dict keys for model architecture hints
        state_keys = list(state_dict.keys())
        
        # Check for RT-MonoDepth signature
        if any('encoder.model' in key for key in state_keys):
            return 'rtmonodepth'
        
        # Check for auxiliary decoders (spatial models with aux)
        if any('aux_decoders' in key for key in state_keys):
            return 'spatial_aux'
        
        # Check for main_decoder (also indicates spatial with aux)
        if any('main_decoder' in key for key in state_keys):
            return 'spatial_aux'
        
        # Check for dual resolution signatures
        if any('context_encoder' in key and 'patch_encoder' in key for key in state_keys):
            return 'dual'
        
        # Check for spatial patch predictor
        if any('spatial_extractor' in key or 'spatial_decoder' in key for key in state_keys):
            return 'spatial'
            
    except Exception as e:
        print(f"Warning: Could not load checkpoint for type detection: {e}")
    
    # Fallback to path-based detection
    if 'lowres' in path_lower or 'rtmonodepth' in path_lower:
        return 'rtmonodepth'
    elif 'spatial' in path_lower and 'aux' in path_lower:
        return 'spatial_aux'
    elif 'spatial' in path_lower:
        return 'spatial'
    elif 'lightweight' in path_lower and 'dual' in path_lower:
        return 'lightweight_dual'
    elif 'dual' in path_lower:
        return 'dual'
    elif 'multitask' in path_lower or 'multi_task' in path_lower:
        return 'multitask'
    else:
        return 'single'


def load_model(args, device):
    """Load model based on checkpoint and configuration."""
    # Auto-detect model type if needed
    if args.model_type == 'auto':
        model_type = detect_model_type(args.checkpoint)
        print(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type
    
    # Try to load checkpoint first to get saved args
    checkpoint_data = None
    saved_args = {}
    try:
        checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if isinstance(checkpoint_data, dict):
            saved_args = checkpoint_data.get('args', {})
    except:
        pass
    
    # Create appropriate model
    if model_type == 'spatial_aux':
        print(f"Loading spatial patch model with auxiliary losses")
        # Use saved args if available
        patch_size = saved_args.get('depth_patch_size', 16)
        spatial_region_size = saved_args.get('spatial_region_size', 5)
        
        model = SpatialPatchDepthPredictorWithAux(
            image_size=args.image_size,
            num_encoder_levels=saved_args.get('encoder_levels', args.encoder_levels),
            base_channels=saved_args.get('base_channels', args.base_channels),
            spatial_region_size=spatial_region_size,
            patch_size=patch_size,
            use_auxiliary_losses=True
        )
        is_dual = False
        is_spatial = True
        print(f"  Patch size: {patch_size}, Region size: {spatial_region_size}")
        
    elif model_type == 'spatial':
        print(f"Loading spatial patch model")
        # Use saved args if available
        patch_size = saved_args.get('depth_patch_size', 16)
        spatial_region_size = saved_args.get('spatial_region_size', 5)
        
        model = SpatialPatchDepthPredictor(
            image_size=args.image_size,
            num_encoder_levels=saved_args.get('encoder_levels', args.encoder_levels),
            base_channels=saved_args.get('base_channels', args.base_channels),
            spatial_region_size=spatial_region_size,
            patch_size=patch_size
        )
        is_dual = False
        is_spatial = True
        print(f"  Patch size: {patch_size}, Region size: {spatial_region_size}")
        
    elif model_type == 'lightweight_dual':
        print(f"Loading lightweight dual-resolution model")
        model = LightweightDualResolution(
            base_channels=saved_args.get('base_channels', args.base_channels),
            encoder_levels=saved_args.get('encoder_levels', args.encoder_levels),
            patch_size=saved_args.get('patch_size', args.patch_size),
            context_region_size=saved_args.get('context_region_size', args.context_region_size)
        )
        is_dual = True
        is_spatial = False
        
    elif model_type == 'dual':
        print(f"Loading dual-resolution model")
        # Try to infer configuration from saved args or checkpoint name
        context_channels = saved_args.get('context_channels', args.base_channels)
        patch_channels = saved_args.get('patch_channels', args.base_channels)
        
        # Override from checkpoint name if present
        if 'ch32' in args.checkpoint:
            patch_channels = 32
            print(f"Detected patch_channels=32 from checkpoint path")
        elif 'ch48' in args.checkpoint:
            patch_channels = 48
            print(f"Detected patch_channels=48 from checkpoint path")
        
        # Check for output size in saved args or filename
        output_size = 11  # default
        if saved_args.get('output_size'):
            output_size = saved_args['output_size']
        elif '3x3' in args.checkpoint:
            output_size = 3
            print(f"Detected 3x3 output from checkpoint path")
        
        model = DualResolutionGazeDepth(
            context_size=args.image_size,
            context_levels=saved_args.get('context_levels', args.encoder_levels),
            context_channels=context_channels,
            patch_size=saved_args.get('patch_size', args.patch_size),
            patch_levels=saved_args.get('patch_levels', args.encoder_levels),
            patch_channels=patch_channels,
            output_size=output_size,
            max_depth=10.0,
            min_depth=0.1
        )
        is_dual = True
        is_spatial = False
        print(f"  Output size: {output_size}x{output_size}")
        
    elif model_type == 'rtmonodepth':
        print(f"RT-MonoDepth detected - using specialized evaluation")
        
        # For RT-MonoDepth, we need special handling due to architecture variations
        # Defer to the specialized evaluate_rtmonodepth.py script
        import subprocess
        import sys
        
        # Build command for RT-MonoDepth evaluation
        cmd = [
            sys.executable, 'evaluate_rtmonodepth.py',
            '--checkpoint', args.checkpoint,
            '--image', args.image,
            '--gaze-x', str(args.gaze_x),
            '--gaze-y', str(args.gaze_y)
        ]
        
        if args.save_output:
            cmd.extend(['--save-output', args.save_output])
        
        if args.device:
            cmd.extend(['--device', args.device])
        
        print("Launching RT-MonoDepth evaluation...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Exit after RT-MonoDepth evaluation
        sys.exit(result.returncode)
        
        # This code won't be reached but kept for reference
        model = RTMonoDepthS()
        is_dual = False
        is_spatial = False
        
    else:
        print(f"Loading single-resolution model ({'multi-task' if model_type == 'multitask' else 'standard'})")
        use_multi_task = (model_type == 'multitask')
        
        # Check for predict_patch in saved args
        # Handle both dict and Namespace types
        if isinstance(saved_args, dict):
            predict_patch = saved_args.get('predict_patch', False)
            patch_size = saved_args.get('depth_patch_size', 16) if predict_patch else None
        else:
            # It's a Namespace object from older checkpoints
            predict_patch = getattr(saved_args, 'predict_patch', False)
            patch_size = getattr(saved_args, 'depth_patch_size', 16) if predict_patch else None
        
        # Get parameters from saved args
        if isinstance(saved_args, dict):
            encoder_levels = saved_args.get('encoder_levels', args.encoder_levels)
            base_channels = saved_args.get('base_channels', args.base_channels)
        else:
            encoder_levels = getattr(saved_args, 'encoder_levels', args.encoder_levels)
            base_channels = getattr(saved_args, 'base_channels', args.base_channels)
            
        model = FlexibleGazeOnlyDepth(
            num_encoder_levels=encoder_levels,
            base_channels=base_channels,
            gaze_feature_dim=64,
            image_size=args.image_size,
            max_depth=10.0,
            min_depth=0.1,
            use_multi_scale_supervision=True,
            use_multi_task=use_multi_task,
            predict_patch=predict_patch,
            patch_size=patch_size
        )
        is_dual = False
        is_spatial = predict_patch  # If predicting patch, treat as spatial
        if predict_patch:
            print(f"  Patch prediction mode: {patch_size}x{patch_size}")
    
    # Load checkpoint
    print(f"Loading weights from: {args.checkpoint}")
    if checkpoint_data is None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    else:
        checkpoint = checkpoint_data
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Try to load state dict with strict=False to handle minor mismatches
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        except RuntimeError as e:
            print(f"Warning: Loading with strict=False due to: {e}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded checkpoint (partial match) from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    return model, is_dual, is_spatial


def preprocess_image(image_path, target_size=88):
    """Load and preprocess image for model input."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    print(f"Original image size: {original_size[0]}×{original_size[1]}")
    
    # Resize to target size
    image_resized = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    
    # Convert to tensor and normalize to [0, 1]
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # HWC -> CHW
    
    return image_tensor, original_size, image


def extract_patch_around_gaze(image_tensor, gaze_x, gaze_y, patch_size=96, context_size=88):
    """Extract high-resolution patch around gaze location."""
    # For demo purposes, we'll extract a patch from the resized image
    # In a real application, you might want to extract from the original high-res image
    
    # Simple center crop for now (as gaze is typically near center)
    # In practice, you'd extract around the actual gaze location
    C, H, W = image_tensor.shape
    
    # Create a slightly upscaled version to extract patch from
    upscaled = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0),
        size=(patch_size, patch_size),
        mode='bilinear',
        align_corners=True
    ).squeeze(0)
    
    return upscaled


def scale_gaze_coordinates(gaze_x, gaze_y, original_size, target_size):
    """Scale gaze coordinates from original image to target size."""
    scale_x = target_size / original_size[0]
    scale_y = target_size / original_size[1]
    
    scaled_gaze_x = gaze_x * scale_x
    scaled_gaze_y = gaze_y * scale_y
    
    # Ensure within bounds
    scaled_gaze_x = max(0, min(scaled_gaze_x, target_size - 1))
    scaled_gaze_y = max(0, min(scaled_gaze_y, target_size - 1))
    
    return scaled_gaze_x, scaled_gaze_y


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


def predict_depth(model, image_tensor, patch_tensor, gaze_x, gaze_y, is_dual, is_spatial, device):
    """Run model inference to predict depth at gaze location."""
    with torch.no_grad():
        # Prepare inputs
        image_batch = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        gaze_x_tensor = torch.tensor([gaze_x], dtype=torch.float32).to(device)
        gaze_y_tensor = torch.tensor([gaze_y], dtype=torch.float32).to(device)
        
        # Check if it's RT-MonoDepth model (outputs full depth map)
        if isinstance(model, RTMonoDepthS):
            # RT-MonoDepth outputs full depth map
            outputs = model(image_batch)
            depth_map = outputs['depth']  # Shape: [1, 1, H, W]
            
            # Extract depth at gaze location
            # Convert gaze coordinates to integers
            gaze_x_int = int(round(gaze_x))
            gaze_y_int = int(round(gaze_y))
            
            # Get depth map dimensions
            h, w = depth_map.shape[2], depth_map.shape[3]
            
            # Ensure coordinates are within bounds
            gaze_x_int = max(0, min(gaze_x_int, w - 1))
            gaze_y_int = max(0, min(gaze_y_int, h - 1))
            
            depth = depth_map[0, 0, gaze_y_int, gaze_x_int].item()
        elif is_dual:
            # Dual-resolution model
            patch_batch = patch_tensor.unsqueeze(0).to(device)
            outputs = model(image_batch, patch_batch, gaze_x_tensor, gaze_y_tensor)
            depth = outputs['depth'].item()
        elif is_spatial:
            # Spatial patch models
            outputs = model(image_batch, gaze_x_tensor, gaze_y_tensor)
            # For spatial models, output is a 16x16 patch
            # Extract the center pixel as the gaze depth
            depth_patch = outputs['depth']  # Shape: [1, 16, 16]
            center_y = depth_patch.shape[1] // 2  # 8
            center_x = depth_patch.shape[2] // 2  # 8
            depth = depth_patch[0, center_y, center_x].item()
        else:
            # Standard gaze-only models
            outputs = model(image_batch, gaze_x_tensor, gaze_y_tensor)
            depth = outputs['depth'].item()
        
    return depth


def visualize_prediction(image, gaze_x, gaze_y, depth, scaled_gaze_x, scaled_gaze_y, 
                        target_size, gt_depth=None, save_path=None):
    """Create visualization showing input image, gaze location, and predicted depth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image with gaze location
    ax1.imshow(image)
    ax1.scatter(gaze_x, gaze_y, c='red', s=100, marker='x', linewidths=3)
    ax1.set_title(f'Original Image\nGaze: ({gaze_x:.1f}, {gaze_y:.1f})')
    ax1.axis('off')
    
    # Resized image with scaled gaze and depth prediction
    image_resized = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    ax2.imshow(image_resized)
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
        
        title = f'Model Input ({target_size}×{target_size})\nPred: {depth:.3f}m | GT: {gt_depth:.3f}m | Error: {error_percent:.1f}%'
    else:
        # No ground truth available
        ax2.text(scaled_gaze_x + 5, scaled_gaze_y - 5, f'{depth:.3f}m', 
                 color='yellow', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        title = f'Model Input ({target_size}×{target_size})\nPredicted Depth: {depth:.3f}m'
    
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    if not save_path:  # Only show if not saving
        plt.show()
    
    plt.close()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Load model
    model, is_dual, is_spatial = load_model(args, device)
    
    # Load and preprocess image
    print(f"\nProcessing image: {args.image}")
    image_tensor, original_size, original_image = preprocess_image(args.image, args.image_size)
    
    # Scale gaze coordinates
    scaled_gaze_x, scaled_gaze_y = scale_gaze_coordinates(
        args.gaze_x, args.gaze_y, original_size, args.image_size
    )
    print(f"Original gaze: ({args.gaze_x}, {args.gaze_y})")
    print(f"Scaled gaze: ({scaled_gaze_x:.1f}, {scaled_gaze_y:.1f})")
    
    # Extract patch for dual-resolution models
    patch_tensor = None
    if is_dual:
        patch_tensor = extract_patch_around_gaze(
            image_tensor, scaled_gaze_x, scaled_gaze_y, 
            args.patch_size, args.image_size
        )
    
    # Load ground truth depth if available
    gt_depth = load_ground_truth_depth(args.image, args.gaze_x, args.gaze_y)
    
    # Predict depth
    print("\nRunning inference...")
    depth = predict_depth(model, image_tensor, patch_tensor, 
                         scaled_gaze_x, scaled_gaze_y, is_dual, is_spatial, device)
    
    # Display result
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Gaze location: ({args.gaze_x:.1f}, {args.gaze_y:.1f})")
    print(f"Predicted depth: {depth:.3f} meters")
    if gt_depth is not None:
        error = abs(depth - gt_depth)
        error_percent = (error / gt_depth) * 100
        print(f"Ground truth depth: {gt_depth:.3f} meters")
        print(f"Absolute error: {error:.3f} meters")
        print(f"Relative error: {error_percent:.1f}%")
    print("="*50)
    
    # Visualize if requested
    if args.visualize or args.save_output:
        print("\nCreating visualization...")
        visualize_prediction(
            original_image, args.gaze_x, args.gaze_y, depth,
            scaled_gaze_x, scaled_gaze_y, args.image_size,
            gt_depth=gt_depth, save_path=args.save_output
        )
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()