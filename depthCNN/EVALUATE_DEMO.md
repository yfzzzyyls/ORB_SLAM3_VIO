# Evaluate Demo - Model Checkpoint Evaluation Guide

This guide provides commands to evaluate different model architectures and checkpoints using the `evaluate_demo.py` script.

## Overview

The `evaluate_demo.py` script is a unified evaluation tool that supports multiple model architectures:
- RT-MonoDepth models (low-resolution full depth maps)
- Spatial patch models (16×16 or 22×22 patches around gaze)
- Dual-resolution models (context + high-res patch)
- Single-point gaze models (depth at gaze location only)
- Multi-task models (with auxiliary outputs)

The script automatically detects the model type from the checkpoint and loads the appropriate architecture.

## Basic Usage

```bash
python evaluate_demo.py \
    --checkpoint path/to/checkpoint.pth \
    --image path/to/image.png \
    --gaze-x <x-coordinate> \
    --gaze-y <y-coordinate> \
    --save-output output.png
```

### Arguments
- `--checkpoint`: Path to trained model checkpoint
- `--image`: Input image (PNG or JPG)
- `--gaze-x`: Gaze X coordinate in original image pixels
- `--gaze-y`: Gaze Y coordinate in original image pixels
- `--save-output`: (Optional) Save visualization to file
- `--device`: (Optional) Device to use (cuda or cpu, default: cuda)
- `--model-type`: (Optional) Force specific model type (auto-detect by default)

## Supported Model Types

### 1. RT-MonoDepth Models
Low-resolution models that output full depth maps (typically 88×88 or 96×96).

```bash
# Example: Evaluate lowres_16x checkpoint
python evaluate_demo.py \
    --checkpoint ./checkpoints/lowres_16x/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output rtmonodepth_result.png
```

### 2. Spatial Patch Models
Models that predict a spatial patch (e.g., 16×16 or 22×22) around the gaze location.

#### Standard Spatial Models
```bash
# Example: Evaluate spatial_flexible checkpoint
python evaluate_demo.py \
    --checkpoint ./checkpoints/spatial_flexible/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output spatial_flexible_result.png
```

#### Spatial Models with Auxiliary Losses
```bash
# Example: Evaluate spatial model trained with auxiliary losses
python evaluate_demo.py \
    --checkpoint ./checkpoints/spatial_gaze_replication_16x16/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output spatial_aux_result.png
```

#### Dense Patch Training Models
```bash
# Example: Evaluate spatial model trained with dense patches
python evaluate_demo.py \
    --checkpoint ./checkpoints/spatial_dense_optimal/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output spatial_dense_result.png
```

### 3. Dual-Resolution Models
Models that process both low-resolution context (88×88) and high-resolution patch (44×44 or 96×96).

```bash
# Example: Evaluate dual-resolution model
python evaluate_demo.py \
    --checkpoint ./checkpoints/dual_resolution_3x3/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output dual_resolution_result.png

# Example: Lightweight dual-resolution model
python evaluate_demo.py \
    --checkpoint ./checkpoints/lightweight_dual/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output lightweight_dual_result.png
```

### 4. Single-Point Gaze Models
Models that predict depth at a single gaze location.

```bash
# Example: Standard gaze-only model
python evaluate_demo.py \
    --checkpoint ./checkpoints/gaze_352_4level/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output gaze_only_result.png

# Example: Multi-task gaze model
python evaluate_demo.py \
    --checkpoint ./checkpoints/gaze_multitask/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output multitask_result.png
```

### 5. Flexible Gaze Models with Patch Prediction
Models that can predict either single points or patches based on configuration.

```bash
# Example: Flexible gaze model in patch mode
python evaluate_demo.py \
    --checkpoint ./checkpoints/flexible_gaze_patch_16/checkpoint_best.pth \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --save-output flexible_patch_result.png
```

## Batch Evaluation

To compare multiple models on the same image, use the `compare_all_models.py` script:

```bash
# Compare all models in checkpoints directory
python compare_all_models.py \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --output-dir ./model_comparison

# Compare specific models only
python compare_all_models.py \
    --image ./test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png \
    --gaze-x 704 \
    --gaze-y 704 \
    --models spatial_flexible spatial_dense_optimal lowres_16x \
    --output-dir ./model_comparison
```

## Output Format

The evaluation produces a visualization showing:
1. **Left panel**: Original image (1408×1408) with gaze location marked
2. **Right panel**: Model input (88×88) with:
   - Predicted depth value
   - Ground truth depth (if available)
   - Error percentage (if ground truth available)
   - Color-coded error indicator:
     - Green: < 10% error
     - Yellow: 10-20% error
     - Orange: > 20% error

## Using Test Data

### Finding Test Images
```bash
# List available test images
find test_data_minimal -name "*.png" | head -10

# Common test image path
test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png
```

### Gaze Coordinates
The gaze coordinates should be in the original image space (1408×1408 for ADT dataset). Common test points:
- Center: `--gaze-x 704 --gaze-y 704`
- Upper left quadrant: `--gaze-x 352 --gaze-y 352`
- Lower right quadrant: `--gaze-x 1056 --gaze-y 1056`

## Troubleshooting

### Model Loading Errors
If you encounter model loading errors:
1. The script uses `strict=False` to handle minor architecture mismatches
2. RT-MonoDepth models are handled by a specialized loader
3. Check the console output for detected model type

### Missing Ground Truth
If ground truth is not found:
- Ensure depth files are in the expected location: `../depth/` relative to RGB images
- Depth files should be `.npz` format with the same base name as the image

### GPU Memory Issues
For large models or batch evaluation:
```bash
# Use CPU instead
python evaluate_demo.py --checkpoint model.pth --image img.png --gaze-x 704 --gaze-y 704 --device cpu

# Or limit GPU memory growth
export CUDA_VISIBLE_DEVICES=0
```

## Example Commands for Common Checkpoints

```bash
# RT-MonoDepth (low-res full depth map)
python evaluate_demo.py --checkpoint ./checkpoints/lowres_16x/checkpoint_best.pth --image test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png --gaze-x 704 --gaze-y 704 --save-output lowres_demo.png

# Spatial patch with auxiliary losses
python evaluate_demo.py --checkpoint ./checkpoints/spatial_gaze_replication_16x16/checkpoint_best.pth --image test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png --gaze-x 704 --gaze-y 704 --save-output spatial_aux_demo.png

# Dual-resolution 3x3 output
python evaluate_demo.py --checkpoint ./checkpoints/dual_resolution_3x3/checkpoint_best.pth --image test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png --gaze-x 704 --gaze-y 704 --save-output dual_3x3_demo.png

# Flexible gaze with patch prediction
python evaluate_demo.py --checkpoint ./checkpoints/flexible_gaze_patch_16/checkpoint_best.pth --image test_data_minimal/train/Apartment_release_clean_seq131_M1292/rgb/frame_000000.png --gaze-x 704 --gaze-y 704 --save-output flexible_patch_demo.png
```

## Model Architecture Details

The script automatically detects and loads the correct architecture based on checkpoint contents:

- **RT-MonoDepth**: Encoder-decoder outputting full depth maps
- **Spatial Models**: CNN decoder upsampling spatial features to patches
- **Dual-Resolution**: Separate encoders for context and patch with fusion
- **Single-Point**: MLP decoder for single depth value
- **Flexible**: Configurable for either point or patch prediction

For more details on specific architectures, see the corresponding training scripts and model definitions.