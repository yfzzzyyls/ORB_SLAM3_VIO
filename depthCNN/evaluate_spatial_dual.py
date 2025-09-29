#!/usr/bin/env python3
"""
Focused evaluation script for SpatialDualResolutionGazeDepth model.
Matches training preprocessing exactly.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import the model
from spatial_dual_resolution_coordconv import SpatialDualResolutionGazeDepth


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SpatialDualResolutionGazeDepth model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to RGB image')
    
    # Gaze coordinates (in pixel space, 0-1407)
    parser.add_argument('--gaze-x', type=float, required=True,
                        help='Gaze X coordinate in pixels (0-1407)')
    parser.add_argument('--gaze-y', type=float, required=True,
                        help='Gaze Y coordinate in pixels (0-1407)')
    
    # Optional arguments
    parser.add_argument('--depth', type=str, default=None,
                        help='Path to depth npz file for comparison')
    parser.add_argument('--use-ema', action='store_true',
                        help='Use EMA model weights if available')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization')
    parser.add_argument('--save-output', type=str, default=None,
                        help='Save visualization to file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--patch-coverage', type=int, default=88,
                        help='Size of region to extract before downsampling to 88x88 (default: 88, try: 352)')
    
    return parser.parse_args()


def load_model(checkpoint_path, device='cuda', use_ema=False):
    """Load the SpatialDualResolutionGazeDepth model."""
    
    # Create model with exact same config as training
    model = SpatialDualResolutionGazeDepth()
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load appropriate weights
    if use_ema and 'model_ema' in checkpoint:
        model.load_state_dict(checkpoint['model_ema'])
        print(f"Loaded EMA model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'ema_val_loss' in checkpoint:
            print(f"  EMA val loss: {checkpoint['ema_val_loss']:.4f}")
        if 'ema_val_metrics' in checkpoint:
            metrics = checkpoint['ema_val_metrics']
            print(f"  EMA metrics: abs_rel={metrics.get('abs_rel', 'N/A'):.3f}, "
                  f"a1={metrics.get('a1', 'N/A'):.3f}, "
                  f"gaze_a1={metrics.get('gaze_a1', 'N/A'):.3f}")
    else:
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded regular model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    return model

def load_image(image_path):
    """Load RGB image and convert to tensor."""
    image = Image.open(image_path).convert('RGB')
    rgb_tensor = TF.to_tensor(image)
    return rgb_tensor, image


def load_depth(depth_path, gaze_x, gaze_y):
    """Load ground truth depth if available."""
    if depth_path is None:
        return None
    
    # Load depth from npz file
    data = np.load(depth_path)
    depth = data['depth'].astype(np.float32) / 1000.0  # mm to meters
    
    # Sample depth at gaze location using bilinear interpolation
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    
    # Normalize gaze coordinates with align_corners=False convention
    gaze_x_norm = 2.0 * (gaze_x + 0.5) / 1408 - 1.0
    gaze_y_norm = 2.0 * (gaze_y + 0.5) / 1408 - 1.0
    gaze_grid = torch.tensor([[[[gaze_x_norm, gaze_y_norm]]]], dtype=torch.float32)
    
    # Sample depth at exact gaze point
    gaze_depth_sample = F.grid_sample(depth_tensor, gaze_grid, 
                                     mode='bilinear', padding_mode='reflection', 
                                     align_corners=False)
    gt_depth = gaze_depth_sample.squeeze().item()
    
    return gt_depth, depth


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device, args.use_ema)
    
    # Load full-resolution image
    print(f"\nProcessing image: {args.image}")
    print(f"Gaze location (pixels): ({args.gaze_x}, {args.gaze_y})")

    rgb_tensor, original_image = load_image(args.image)
    original_size = original_image.size  # (width, height)
    print(f"Original image size: {original_size[0]}×{original_size[1]}")
    coverage_pct = args.patch_coverage / 1408 * 100
    print(f"Patch coverage: {args.patch_coverage}×{args.patch_coverage} pixels ({coverage_pct:.1f}% of image)")
    
    # Load ground truth depth if available
    gt_depth = None
    if args.depth:
        gt_depth, depth_map = load_depth(args.depth, args.gaze_x, args.gaze_y)
        print(f"Ground truth depth at gaze: {gt_depth:.3f}m")
    elif args.image.replace('/rgb/', '/depth/').replace('.png', '.npz'):
        # Try to find matching depth file
        depth_path = args.image.replace('/rgb/', '/depth/').replace('.png', '.npz')
        try:
            gt_depth, depth_map = load_depth(depth_path, args.gaze_x, args.gaze_y)
            print(f"Ground truth depth at gaze: {gt_depth:.3f}m")
        except:
            print("No ground truth depth available")
    
    # Normalize gaze coordinates to [-1, 1] with align_corners=False
    gaze_x_norm = 2.0 * (args.gaze_x + 0.5) / 1408 - 1.0
    gaze_y_norm = 2.0 * (args.gaze_y + 0.5) / 1408 - 1.0
    
    print(f"Normalized gaze: ({gaze_x_norm:.3f}, {gaze_y_norm:.3f})")
    
    # Run inference and measure latency
    print("\nRunning inference...")
    import time
    
    with torch.no_grad():
        gaze_x_tensor = torch.tensor([gaze_x_norm], dtype=torch.float32, device=device)
        gaze_y_tensor = torch.tensor([gaze_y_norm], dtype=torch.float32, device=device)

        full_rgb_batch = rgb_tensor.unsqueeze(0).to(device)
        context_batch, patch_batch = model.prepare_inputs(
            full_rgb_batch,
            gaze_x_tensor,
            gaze_y_tensor,
            patch_size=args.patch_coverage
        )

        # Store CPU copies for visualization before any in-place ops
        context_rgb = context_batch[0].detach().cpu()
        patch_rgb = patch_batch[0].detach().cpu()
        
        # Warm up (first run is slower due to CUDA kernel compilation)
        _ = model(context_batch, patch_batch, gaze_x_tensor, gaze_y_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure pure inference latency on 88×88 branches
        num_runs = 10 if device.type == 'cpu' else 100
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_runs):
            outputs = model(context_batch, patch_batch, gaze_x_tensor, gaze_y_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        end_time = time.perf_counter()
        avg_latency = (end_time - start_time) / num_runs * 1000  # Convert to ms

        # Measure end-to-end latency (GPU preprocessing + inference, assuming frame already resides on device)
        pipeline_runs = 5 if device.type == 'cpu' else 25
        preprocess_times = []
        end_to_end_times = []

        for _ in range(pipeline_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            pipeline_start = time.perf_counter()

            context_e2e, patch_e2e = model.prepare_inputs(
                full_rgb_batch,
                gaze_x_tensor,
                gaze_y_tensor,
                patch_size=args.patch_coverage
            )

            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_start = time.perf_counter()

            _ = model(context_e2e, patch_e2e, gaze_x_tensor, gaze_y_tensor)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            pipeline_end = time.perf_counter()

            preprocess_times.append((inference_start - pipeline_start) * 1000)
            end_to_end_times.append((pipeline_end - pipeline_start) * 1000)

        # Get final outputs for visualization
        outputs = model(context_batch, patch_batch, gaze_x_tensor, gaze_y_tensor)
        
        # Extract predictions
        depth_22x22 = outputs[0]  # [1, 1, 22, 22] dense depth map
        log_sigma = outputs[1]    # [1, 1, 22, 22] uncertainty
        gaze_depth = outputs[2]    # [1, 1] scalar gaze depth
        
        pred_depth = gaze_depth.squeeze().item()
        
        print(f"Predicted gaze depth: {pred_depth:.3f}m")
        
        # Also show center pixel from 22x22 for comparison
        center_depth = depth_22x22[0, 0, 11, 11].item()
        print(f"Center pixel of 22x22: {center_depth:.3f}m (for reference)")
        
        # Print latency measurements
        print("\n" + "="*50)
        print("LATENCY MEASUREMENT")
        print("="*50)
        print(f"Device: {device}")
        print(f"Average inference time: {avg_latency:.2f} ms")
        print(f"FPS capability: {1000/avg_latency:.1f} fps")
        if end_to_end_times:
            mean_preproc = sum(preprocess_times) / len(preprocess_times)
            mean_end_to_end = sum(end_to_end_times) / len(end_to_end_times)
            print(f"Average GPU preprocessing time: {mean_preproc:.2f} ms")
            print(f"Average end-to-end time (on-GPU): {mean_end_to_end:.2f} ms")
            print(f"End-to-end FPS capability: {1000/mean_end_to_end:.1f} fps")
        print("="*50)
    
    # Compare with ground truth
    if gt_depth is not None:
        abs_error = abs(pred_depth - gt_depth)
        rel_error = abs_error / (gt_depth + 1e-6) * 100
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Gaze location: ({args.gaze_x}, {args.gaze_y})")
        print(f"Predicted depth: {pred_depth:.3f} meters")
        print(f"Ground truth depth: {gt_depth:.3f} meters")
        print(f"Absolute error: {abs_error:.3f} meters")
        print(f"Relative error: {rel_error:.1f}%")
        print(f"Within 25% threshold: {'YES' if rel_error < 25 else 'NO'}")
        print("="*50)
    
    # Visualization
    if args.visualize or args.save_output:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image with gaze
        axes[0, 0].imshow(original_image)
        axes[0, 0].plot(args.gaze_x, args.gaze_y, 'r+', markersize=20, markeredgewidth=3)
        circle = patches.Circle((args.gaze_x, args.gaze_y), 44, fill=False, edgecolor='red', linewidth=2)
        axes[0, 0].add_patch(circle)
        axes[0, 0].set_title(f'Original Image with Gaze\n({args.gaze_x:.0f}, {args.gaze_y:.0f})')
        axes[0, 0].axis('off')
        
        # Context (88x88)
        axes[0, 1].imshow(context_rgb.permute(1, 2, 0).cpu())
        axes[0, 1].set_title('Context (88×88)')
        axes[0, 1].axis('off')
        
        # Patch (88x88)
        axes[0, 2].imshow(patch_rgb.permute(1, 2, 0).cpu())
        axes[0, 2].set_title('Patch at Gaze (88×88)')
        axes[0, 2].axis('off')
        
        # Predicted depth map (22x22)
        im1 = axes[1, 0].imshow(depth_22x22.squeeze().cpu(), cmap='viridis')
        axes[1, 0].set_title('Predicted Depth (22×22)')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])
        
        # Uncertainty map
        im2 = axes[1, 1].imshow(torch.exp(log_sigma).squeeze().cpu(), cmap='hot')
        axes[1, 1].set_title('Uncertainty (σ)')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Results text
        axes[1, 2].axis('off')
        result_text = f"Gaze Depth Prediction:\n\n"
        result_text += f"Predicted: {pred_depth:.3f}m\n"
        if gt_depth is not None:
            result_text += f"Ground Truth: {gt_depth:.3f}m\n"
            result_text += f"Absolute Error: {abs_error:.3f}m\n"
            result_text += f"Relative Error: {rel_error:.1f}%\n"
            result_text += f"Within 25%: {'✓' if rel_error < 25 else '✗'}"
        axes[1, 2].text(0.1, 0.5, result_text, fontsize=14, verticalalignment='center')
        
        plt.tight_layout()
        
        if args.save_output:
            plt.savefig(args.save_output, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {args.save_output}")
        
        if args.visualize:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':
    main()
