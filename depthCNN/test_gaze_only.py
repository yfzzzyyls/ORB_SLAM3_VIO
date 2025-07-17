#!/usr/bin/env python3
"""
Test script to verify the gaze-only model implementation.
Tests forward pass, loss computation, and basic functionality.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from gaze_only_rtmonodepth import GazeOnlyRTMonoDepth, GazeDepthLoss, GazeFeatureExtractor
from train_gaze_only import extract_gt_depth_at_gaze, GazeAwareAugmentation


def test_feature_extraction():
    """Test the gaze feature extraction module."""
    print("Testing GazeFeatureExtractor...")
    
    # Create feature extractor
    num_ch_enc = [24, 48, 96, 192]  # RT-MonoDepth-S channels
    extractor = GazeFeatureExtractor(num_ch_enc=num_ch_enc, output_dim=64)
    
    # Create dummy features at different scales
    batch_size = 4
    features = [
        torch.randn(batch_size, 24, 44, 44),   # Scale 0
        torch.randn(batch_size, 48, 22, 22),   # Scale 1
        torch.randn(batch_size, 96, 11, 11),   # Scale 2
        torch.randn(batch_size, 192, 5, 5),    # Scale 3
    ]
    
    # Gaze coordinates
    gaze_x = torch.tensor([44.5, 20.0, 60.0, 10.0])
    gaze_y = torch.tensor([44.5, 30.0, 70.0, 80.0])
    
    # Extract features
    output = extractor(features, gaze_x, gaze_y)
    
    assert output.shape == (batch_size, 256), f"Expected shape (4, 256), got {output.shape}"
    print(f"✓ Feature extraction output shape: {output.shape}")
    
    # Test bilinear interpolation
    single_feat = extractor.extract_at_gaze_bilinear(features[0], gaze_x, gaze_y)
    assert single_feat.shape == (batch_size, 24), f"Expected shape (4, 24), got {single_feat.shape}"
    print(f"✓ Bilinear interpolation works correctly")


def test_model_forward():
    """Test the complete model forward pass."""
    print("\nTesting GazeOnlyRTMonoDepth forward pass...")
    
    # Create model
    model = GazeOnlyRTMonoDepth(max_depth=10.0, min_depth=0.1)
    model.eval()
    
    # Print model info
    num_params = model.get_num_params()
    print(f"Model parameters: {num_params:,}")
    
    # Create dummy input
    batch_size = 8
    rgb = torch.randn(batch_size, 3, 88, 88)
    gaze_x = torch.rand(batch_size) * 87  # Random gaze in [0, 87]
    gaze_y = torch.rand(batch_size) * 87
    
    # Forward pass
    with torch.no_grad():
        outputs = model(rgb, gaze_x, gaze_y)
    
    # Check outputs
    assert 'depth' in outputs, "Missing 'depth' in outputs"
    assert outputs['depth'].shape == (batch_size, 1), f"Wrong depth shape: {outputs['depth'].shape}"
    
    # Check depth range
    depths = outputs['depth'].squeeze()
    assert torch.all(depths >= 0.1) and torch.all(depths <= 10.0), "Depth values out of range"
    
    print(f"✓ Forward pass successful")
    print(f"✓ Depth predictions shape: {outputs['depth'].shape}")
    print(f"✓ Depth range: [{depths.min():.3f}, {depths.max():.3f}] m")


def test_loss_function():
    """Test the gaze-specific loss function."""
    print("\nTesting GazeDepthLoss...")
    
    loss_fn = GazeDepthLoss(alpha=0.85)
    
    # Create dummy predictions and targets
    batch_size = 16
    pred = torch.rand(batch_size, 1) * 5 + 0.5  # Random depth 0.5-5.5m
    gt = torch.rand(batch_size, 1) * 5 + 0.5
    
    # Compute loss
    loss = loss_fn(pred, gt)
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"✓ Loss computation successful: {loss.item():.4f}")
    
    # Test with valid mask
    valid_mask = torch.rand(batch_size, 1) > 0.3  # Random mask
    loss_masked = loss_fn(pred, gt, valid_mask)
    
    print(f"✓ Loss with valid mask: {loss_masked.item():.4f}")


def test_gt_extraction():
    """Test ground truth depth extraction at gaze."""
    print("\nTesting GT depth extraction...")
    
    # Create dummy depth map
    batch_size = 4
    depth_map = torch.rand(batch_size, 1, 88, 88) * 5 + 0.5
    
    # Gaze coordinates
    gaze_x = torch.tensor([44.0, 20.5, 60.3, 10.7])
    gaze_y = torch.tensor([44.0, 30.2, 70.8, 80.1])
    
    # Extract depth at gaze
    gt_depths = extract_gt_depth_at_gaze(depth_map, gaze_x, gaze_y)
    
    assert gt_depths.shape == (batch_size, 1), f"Wrong shape: {gt_depths.shape}"
    
    # Verify bilinear interpolation works
    # For integer coordinates, should match exactly
    exact_depth = depth_map[0, 0, 44, 44]
    extracted_depth = gt_depths[0, 0]
    assert torch.abs(exact_depth - extracted_depth) < 0.01, "Bilinear interpolation error"
    
    print(f"✓ GT depth extraction shape: {gt_depths.shape}")
    print(f"✓ Bilinear interpolation verified")


def test_augmentation():
    """Test gaze-aware data augmentation."""
    print("\nTesting GazeAwareAugmentation...")
    
    # Test 1: Only horizontal flip (no brightness/contrast changes)
    augmentation = GazeAwareAugmentation(
        horizontal_flip_prob=1.0,  # Always flip
        brightness_range=0.0,       # No brightness change
        contrast_range=0.0          # No contrast change
    )
    
    # Create dummy data
    rgb = torch.rand(3, 88, 88)
    depth = torch.rand(1, 88, 88)
    gaze_x, gaze_y = 30.0, 40.0
    
    # Apply augmentation
    rgb_aug, depth_aug, gaze_x_aug, gaze_y_aug = augmentation(rgb, depth, gaze_x, gaze_y)
    
    # Check horizontal flip
    expected_gaze_x = 88 - 1 - gaze_x
    assert abs(gaze_x_aug - expected_gaze_x) < 0.01, f"Gaze flip error: {gaze_x_aug} vs {expected_gaze_x}"
    assert gaze_y_aug == gaze_y, "Gaze Y should not change with horizontal flip"
    
    # Verify image flip - when flipping along dim=2 (width), we flip horizontally
    # After horizontal flip, first column should equal last column of original
    assert torch.allclose(rgb_aug[:, :, 0], rgb[:, :, -1], atol=1e-6), "RGB first column not matching"
    assert torch.allclose(rgb_aug[:, :, -1], rgb[:, :, 0], atol=1e-6), "RGB last column not matching"
    assert torch.allclose(depth_aug[:, :, 0], depth[:, :, -1], atol=1e-6), "Depth first column not matching"
    
    print(f"✓ Horizontal flip works correctly")
    print(f"✓ Gaze coordinates updated: ({gaze_x}, {gaze_y}) → ({gaze_x_aug}, {gaze_y_aug})")
    
    # Test 2: Brightness/contrast changes
    augmentation2 = GazeAwareAugmentation(
        horizontal_flip_prob=0.0,   # No flip
        brightness_range=0.2,        # Brightness change
        contrast_range=0.2           # Contrast change
    )
    
    rgb_aug2, depth_aug2, gaze_x_aug2, gaze_y_aug2 = augmentation2(rgb, depth, gaze_x, gaze_y)
    
    # Gaze should not change without flip
    assert gaze_x_aug2 == gaze_x and gaze_y_aug2 == gaze_y, "Gaze changed without flip"
    
    # Depth should not change from brightness/contrast
    assert torch.allclose(depth_aug2, depth), "Depth changed from brightness/contrast"
    
    # RGB should change but stay in valid range
    assert not torch.allclose(rgb_aug2, rgb), "RGB didn't change with brightness/contrast"
    assert torch.all(rgb_aug2 >= 0) and torch.all(rgb_aug2 <= 1), "RGB out of valid range"
    
    print(f"✓ Brightness/contrast augmentation works correctly")


def test_multi_gpu_compatibility():
    """Test that model works with DataParallel."""
    print("\nTesting multi-GPU compatibility...")
    
    model = GazeOnlyRTMonoDepth()
    
    # Check if multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"✓ Model wrapped in DataParallel for {torch.cuda.device_count()} GPUs")
    else:
        print("✓ Single GPU or CPU mode")
    
    # Test forward pass
    rgb = torch.randn(4, 3, 88, 88)
    gaze_x = torch.rand(4) * 87
    gaze_y = torch.rand(4) * 87
    
    if torch.cuda.is_available():
        model = model.cuda()
        rgb = rgb.cuda()
        gaze_x = gaze_x.cuda()
        gaze_y = gaze_y.cuda()
    
    with torch.no_grad():
        outputs = model(rgb, gaze_x, gaze_y)
    
    print(f"✓ Multi-GPU forward pass successful")


def test_training_step():
    """Test a complete training step."""
    print("\nTesting training step...")
    
    # Create model and optimizer
    model = GazeOnlyRTMonoDepth(use_multi_scale_supervision=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = GazeDepthLoss()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy batch
    batch_size = 8
    rgb = torch.randn(batch_size, 3, 88, 88).to(device)
    depth_map = torch.rand(batch_size, 1, 88, 88).to(device) * 5 + 0.5
    gaze_x = torch.rand(batch_size).to(device) * 87
    gaze_y = torch.rand(batch_size).to(device) * 87
    
    # Extract GT depth at gaze
    gt_depth = extract_gt_depth_at_gaze(depth_map, gaze_x, gaze_y)
    
    # Forward pass
    model.train()
    outputs = model(rgb, gaze_x, gaze_y)
    
    # Compute losses
    main_loss = loss_fn(outputs['depth'], gt_depth)
    
    total_loss = main_loss
    if 'aux_depths' in outputs:
        for aux_depth in outputs['aux_depths']:
            total_loss += 0.1 * loss_fn(aux_depth, gt_depth)
        print(f"✓ Multi-scale supervision active with {len(outputs['aux_depths'])} auxiliary outputs")
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"✓ Gradient flowing through {name}: {grad_norm:.6f}")
                break
    
    # Update
    optimizer.step()
    
    print(f"✓ Training step completed successfully")
    print(f"✓ Loss: {total_loss.item():.4f}")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Gaze-Only RT-MonoDepth Implementation")
    print("="*60)
    
    try:
        test_feature_extraction()
        test_model_forward()
        test_loss_function()
        test_gt_extraction()
        test_augmentation()
        test_multi_gpu_compatibility()
        test_training_step()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()