#!/usr/bin/env python3
"""
Test script for depth prediction functions.
Only contains the essential code to test create_depth_predictor and use_depth_predictor.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.nn import functional as F

# Add depth CNN path to system path
depth_cnn_path = '/home/external/ORB_SLAM3_VIO/depthCNN'
if depth_cnn_path not in sys.path:
    sys.path.append(depth_cnn_path)

def create_depth_predictor():
    """
    Load depth prediction model from checkpoint.
    Uses default checkpoint path from the command line example.
    
    Returns:
        tuple: (model, device) - The loaded model and the device it's on
    """
    from spatial_patch_encoder_aux import SpatialPatchDepthPredictorWithAux
    
    # Set checkpoint path (default from command line example)
    checkpoint_path = '/home/external/ORB_SLAM3_VIO/depthCNN/checkpoints/spatial_gaze_replication_16x16/checkpoint_best.pth'
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    # Load checkpoint to get saved args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get saved args from checkpoint
    saved_args = checkpoint.get('args', {})
    if hasattr(saved_args, 'depth_patch_size'):
        patch_size = getattr(saved_args, 'depth_patch_size', 16)
        spatial_region_size = getattr(saved_args, 'spatial_region_size', 5)
        encoder_levels = getattr(saved_args, 'encoder_levels', 3)
        base_channels = getattr(saved_args, 'base_channels', 32)
    else:
        # Default values if args not found
        patch_size = 16
        spatial_region_size = 5
        encoder_levels = 3
        base_channels = 32
    
    # Create model with correct architecture (spatial_aux)
    model = SpatialPatchDepthPredictorWithAux(
        image_size=88,  # Default input size
        num_encoder_levels=encoder_levels,
        base_channels=base_channels,
        spatial_region_size=spatial_region_size,
        patch_size=patch_size,
        use_auxiliary_losses=True
    )
    
    # Load model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded spatial_aux depth predictor from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device

def use_depth_predictor(model, device, image_path=None, gaze_x=None, gaze_y=None):
    """
    Predict depth at gaze point using the loaded model.
    Uses default values from command line example if not provided.
    
    Args:
        model: Loaded depth prediction model
        device: Torch device (cuda or cpu)
        image_path: Path to input image (default: ADT test image)
        gaze_x: X coordinate of gaze pixel (default: 1050)
        gaze_y: Y coordinate of gaze pixel (default: 750)
    
    Returns:
        float: Predicted depth at the gaze point in meters
    """
    # Use defaults from command line example if not provided
    if image_path is None:
        image_path = '/mnt/ssd_ext/incSeg-data/processed_adt/test/Apartment_release_clean_seq148_M1292/rgb/frame_000450.png'
    if gaze_x is None:
        gaze_x = 1050
    if gaze_y is None:
        gaze_y = 750
    
    # Load image
    if isinstance(image_path, str):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    elif isinstance(image_path, np.ndarray):
        # Handle numpy array input
        image = image_path
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = torch.from_numpy(image).float()
    elif hasattr(image_path, 'convert'):  # PIL Image
        image_np = np.array(image_path.convert('RGB')).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    else:
        image_tensor = image_path
    
    # Model expects 88x88 input image
    # Resize full image to 88x88
    input_image = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(88, 88),
        mode='bilinear',
        align_corners=True
    )
    
    # Scale gaze coordinates from 1408x1408 to 88x88
    scaled_x = gaze_x * 88 / 1408
    scaled_y = gaze_y * 88 / 1408
    
    # Prepare inputs
    input_image = input_image.to(device)
    gaze_x_tensor = torch.tensor([scaled_x], dtype=torch.float32).to(device)
    gaze_y_tensor = torch.tensor([scaled_y], dtype=torch.float32).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_image, gaze_x_tensor, gaze_y_tensor)
        depth_output = outputs['depth']  # Shape: [1, 16, 16]
        
        # Extract center pixel as the depth at gaze point
        center_y = depth_output.shape[1] // 2  # 8
        center_x = depth_output.shape[2] // 2  # 8
        depth = depth_output[0, center_y, center_x].item()
    
    return depth

if __name__ == '__main__':
    print("=== Testing Depth Predictor ===")
    print()
    
    # Load the depth prediction model
    print("1. Loading model...")
    try:
        model, device = create_depth_predictor()
        print(f"   ✓ Model loaded successfully on {device}")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        sys.exit(1)
    
    # Use the model with default values from command line example
    print("\n2. Running inference...")
    print(f"   - Image: /mnt/ssd_ext/incSeg-data/processed_adt/test/Apartment_release_clean_seq148_M1292/rgb/frame_000450.png")
    print(f"   - Gaze point: (1050, 750)")
    
    try:
        predicted_depth = use_depth_predictor(model, device)
        print(f"   ✓ Inference successful")
    except Exception as e:
        print(f"   ✗ Error during inference: {e}")
        sys.exit(1)
    
    # Print the result
    print("\n3. Result:")
    print(f"   Predicted depth at gaze point: {predicted_depth:.3f} meters")
    print("\n=== Test completed successfully! ===")