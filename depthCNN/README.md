# Depth Prediction CNN with ADT Dataset

This project implements RT-MonoDepth-S for metric depth prediction using the Aria Digital Twin (ADT) dataset.

## Dataset Information

ADT provides:
- RGB images: 1408×1408 at 30Hz from camera-rgb (214-1)
- Depth maps: 1408×1408 synthetic ground truth depth
- Gaze data: Eye tracking pitch/yaw angles from eyegaze.csv files
- Depth format: 16-bit uint millimeters (divide by 1000 for meters)
- Depth range: 0-7.6 meters (typical indoor scenes)
- Timestamp offset: Depth recording starts ~10-16 seconds after RGB

## Extracted Dataset

The dataset has been extracted with timestamp-based matching:
- **Train**: 7 sequences (~20,154 RGB-depth pairs)
- **Val**: 1 sequence (~2,881 pairs)
- **Test**: 2 sequences (~5,731 pairs)
- **Total**: ~28,766 matched RGB-depth pairs at full 30Hz
- **Quality**: 1-to-1 RGB-depth matching with 1ms tolerance
- **Coverage**: ~86% valid depth pixels per frame
- **Gaze data**: Pitch/yaw angles converted to pixel coordinates (x, y)

## Quick Start

### 1. Setup Environment
```bash
# Use existing orbslam conda environment
source ~/miniconda3/bin/activate
conda activate orbslam

# Navigate to depthCNN directory
cd /home/external/ORB_SLAM3_VIO/depthCNN
```

### 2. Extract Data (if not already done)
```bash
# Uses timestamp-based matching to handle RGB-depth time offset
python extract_dataset.py  # All defaults configured for full 30Hz extraction
```

### 3. Train Model - All Approaches

#### Approach 1: Full Resolution Dense Depth (1408×1408)
```bash
# Single GPU - Original RT-MonoDepth on full resolution
python train.py \
    --data-root ./processed_data \
    --epochs 20 \
    --batch-size 4 \
    --lr 1e-4 \
    --crop-size 1408

# Multi-GPU (automatically uses all available GPUs)
python train.py \
    --data-root ./processed_data \
    --epochs 20 \
    --batch-size 16 \
    --lr 2e-4 \
    --crop-size 1408
```
**Results**: Dense depth map at 1408×1408, can evaluate at any pixel including gaze

#### Approach 2: Low Resolution Dense Depth (88×88)
```bash
# Average pooling to 88×88, predict full depth map
python train_lowres.py \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --epochs 20 \
    --batch-size 32 \
    --lr 2e-4

# Evaluate at gaze location from 88×88 depth map
python evaluate_lowres.py \
    --checkpoint ./checkpoints/lowres_16x/checkpoint_best.pth \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --save-results
```
**Results**: 88×88 dense depth map, 3.58ms latency, 16.8% error at gaze

#### Approach 3: Gaze-Only with RT-MonoDepth Encoder (Original)
```bash
# Use RT-MonoDepth encoder + MLP decoder for single point
python train_gaze_only_original_init.py \
    --data-root ./processed_data \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --use-patch-encoder \
    --patch-size 96 \
    --patch-output-dim 128
```
**Results**: 1.41M params, MAE ~0.53m at gaze location only

#### Approach 4: Lightweight Gaze-Only (RECOMMENDED)
```bash
# 3-Level lightweight encoder (Best performance)
python train_lightweight_gaze.py \
    --data-root ./processed_data \
    --encoder-levels 3 \
    --base-channels 32 \
    --batch-size 128 \
    --lr 4e-4 \
    --epochs 30

# 2-Level ultra-lightweight variant
python train_lightweight_gaze.py \
    --data-root ./processed_data \
    --encoder-levels 2 \
    --base-channels 32 \
    --batch-size 256 \
    --lr 5e-4 \
    --epochs 30
```
**Results**: 354K params (3-level) or 114K params (2-level), MAE 0.41m, 71% fewer parameters

#### Comparison Summary:
| Approach | Parameters | Output | MAE at Gaze | Speed | Use Case |
|----------|------------|---------|-------------|--------|----------|
| Full Res Dense | 1.23M | 1408×1408 map | ~0.5m | 20-50ms | Need full depth map |
| Low Res Dense | 1.23M | 88×88 map | ~0.5m | 3.58ms | Fast dense depth |
| Gaze-Only Original | 1.41M | Single point | ~0.53m | 2-3ms | Baseline gaze depth |
| **Lightweight Gaze** | **354K** | **Single point** | **0.41m** | **2-3ms** | **Best for gaze** |

### 4. Evaluate

#### Full Resolution Evaluation
```bash
# Evaluate on test dataset (default)
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data-root ./processed_data

# With batch size adjustment for larger GPUs
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data-root ./processed_data \
    --batch-size 16
```

#### Low Resolution Evaluation
```bash
# Evaluate low-res model on test dataset
python evaluate_lowres.py \
    --checkpoint ./checkpoints/lowres_16x/checkpoint_best.pth \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --save-results

# Evaluate with larger batch size (faster)
python evaluate_lowres.py \
    --checkpoint ./checkpoints/lowres_16x/checkpoint_best.pth \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --batch-size 64 \
    --save-results

# Test downsampling pipeline
python test_lowres_pipeline.py --data-root ./processed_data --visualize
```

The evaluation will report:
- **Standard metrics**: abs_rel, sq_rel, RMSE, a1-a3 accuracy
- **Gaze-specific metrics**: MAE, RMSE, and relative error at gaze location
- **Latency statistics**: Mean, median, min/max, percentiles, and throughput (FPS)
  - Low-res (88×88): Expected ~2-5ms per frame on GPU
  - Full-res (1408×1408): Expected ~20-50ms per frame on GPU

### 5. Export to TensorRT (optional)
```bash
python export_tensorrt.py \
    --checkpoint checkpoints/best_model.pth \
    --output models/rtmonodepth_fp16.trt
```

## Model Details

- **Architecture**: RT-MonoDepth-S (1.23M parameters)
- **Input**: Full 1408×1408 resolution (no cropping)
- **Loss**: Scale-Invariant Log loss (SI-Log) with α=0.85
- **Output**: Depth predictions scaled to [0.1, 10.0] meters
- **Training**: ~20,000 frames at 30Hz (15x more than 2Hz subsampling)

## Data Pipeline

1. Load RGB (PNG), depth (NPZ), and gaze (JSON) from processed_data/
2. Convert depth: `depth_m = depth_uint16.float() / 1000.0`
3. Create valid mask: `valid = depth > 0`
4. Load gaze data: pitch/yaw angles and (x, y) pixel coordinates
5. Apply data augmentation (random horizontal flip)
6. Normalize RGB to [0, 1]
7. No cropping - use full 1408×1408 resolution

## Key Implementation Details

- **Timestamp matching**: Handles ~300-500 frame offset between RGB and depth
- **Frame filtering**: Only extracts frames with valid RGB-depth matches
- **Memory optimization**: Depth stored as compressed uint16 NPZ files
- **Fast loading**: Pre-extracted PNG/NPZ faster than VRS reading
- **Clean pairs**: Sequential frame numbering with perfect 1-to-1 correspondence

## Training Tips

- Start with lr=1e-4, reduce to 1e-5 if plateauing
- Monitor validation metrics every epoch
- Best model saved based on lowest validation loss
- Expect significant improvement over 2Hz subsampled training
- Full resolution preserves spatial context for better predictions

## Monitoring Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training log in real-time
tail -f logs/lowres_16x/training_*.log

# Check training progress
cat logs/lowres_16x/training_log.json | jq '.[].val_metrics.abs_rel'
```

## Common Commands

```bash
# Test the downsampling pipeline
python test_lowres_pipeline.py --data-root ./processed_data --visualize

# Visualize a single sample from the dataset
python lowres_dataset.py --data-root ./processed_data --scale-factor 16 --visualize

# Resume training from checkpoint
python train_lowres.py \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --resume ./checkpoints/lowres_16x/checkpoint_latest.pth \
    --lr 1e-5
```

## Gaze-Only Depth Prediction (New Architecture)

### Overview
A specialized architecture for predicting depth only at the gaze location, based on the insight that we only need to understand what object is being gazed at, not reconstruct the entire spatial map.

### Architecture Design
1. **Encoder**: Keep the same RT-MonoDepth encoder for multi-scale feature extraction
2. **Decoder**: Replace spatial decoder with an MLP that outputs a single depth value
3. **Key Innovation**: Extract features at gaze location from multiple encoder scales

### Implementation Details

#### Multi-Scale Feature Extraction
- Extract features at gaze location from encoder scales: [44×44, 22×22, 11×11, 5×5]
- Use bilinear interpolation for sub-pixel accuracy
- Project features to common dimension (64) before concatenation
- Total feature vector: 256 dimensions (4 scales × 64)

#### Two-Stage MLP Decoder
1. **Object Understanding Stage** (256→128→64)
   - Learns what object is at the gaze location
   - Uses ReLU activations and layer normalization
   
2. **Depth Prediction Stage** (64→32→16→1)
   - Predicts depth based on object understanding
   - Final sigmoid activation for [0.1, 10.0]m range

#### Training Configuration
- **Data Augmentation**: Gaze-aware augmentation that adjusts gaze coordinates
- **Loss Function**: Scale-invariant log loss with gaze-specific weighting
- **Multi-Scale Supervision**: Auxiliary losses at intermediate scales
- **Layer Normalization**: Applied after each linear layer for stability
- **Initialization**: Xavier initialization with median depth bias

#### Performance Optimizations
- **Batch Processing**: Extract features for all gaze points in batch
- **Mixed Precision**: FP16 training for efficiency
- **Gradient Checkpointing**: For memory-efficient training
- **Learning Rate Scaling**: Scale with batch size (linear or square root)

### Advantages
- **Efficiency**: ~0.5M parameters vs 1.23M for full model
- **Speed**: Sub-millisecond inference for single gaze point
- **Accuracy**: Optimized specifically for gaze location
- **Simplicity**: No need for spatial reconstruction

### Future Extensions
- Integrate segmentation mask for object-aware features
- Add temporal consistency for video sequences
- Multi-task learning with object classification
- Uncertainty estimation for depth predictions

## Lightweight Gaze-Only Architecture (Latest Development)

### Overview
A highly efficient architecture designed specifically for 88×88 input, achieving **better accuracy with 71% fewer parameters** than the original RT-MonoDepth approach.

### Key Innovations

#### 1. **Lightweight Encoder Design**
- **3-Level Architecture**: 88×88 → 44×44 → 22×22 → 11×11
- **Simple Conv Blocks**: Conv3x3 → BatchNorm → ReLU (no depthwise separable)
- **Channel Progression**: 32 → 64 → 128 (vs RT-MonoDepth's 24→48→96→192)
- **No 4th Level**: Avoids tiny 5.5×5.5 feature maps that provide little value

#### 2. **Efficient Feature Extraction**
```python
# Scale gaze coordinates for each level
Level 1: gaze_x/2, gaze_y/2    # 44×44 feature map
Level 2: gaze_x/4, gaze_y/4    # 22×22 feature map  
Level 3: gaze_x/8, gaze_y/8    # 11×11 feature map

# Extract features only at gaze location using bilinear interpolation
```

#### 3. **Streamlined Decoder**
- **4-Layer MLP**: 192 → 128 → 64 → 32 → 1
- **Progressive Reduction**: Gradual refinement from features to depth
- **LayerNorm + Dropout**: Better regularization for point prediction

### Performance Results

#### Lightweight 3-Level Model (354K params):
- **MAE**: 0.4083m (40.8cm) - Excellent for monocular depth
- **Relative Error**: 17.9%
- **δ < 1.25**: 81.67% accuracy
- **Training Speed**: ~5:45 per epoch on 4 GPUs

#### Comparison with Original:
- **Original RT-MonoDepth**: 1.23M params, MAE ~0.53m
- **Lightweight Model**: 354K params, MAE 0.41m
- **Result**: 23% better accuracy with 71% fewer parameters!

### Training Commands

#### 3-Level Encoder (Recommended):
```bash
python train_lightweight_gaze.py \
    --data-root ./processed_data \
    --encoder-levels 3 \
    --base-channels 32 \
    --batch-size 128 \
    --lr 4e-4 \
    --epochs 30
```

#### 2-Level Encoder (Ultra-Lightweight):
```bash
python train_lightweight_gaze.py \
    --data-root ./processed_data \
    --encoder-levels 2 \
    --base-channels 32 \
    --batch-size 256 \
    --lr 5e-4 \
    --epochs 30
```

### Architecture Rationale

#### Why Simpler Works Better:
1. **Task-Architecture Match**: Single-point prediction doesn't need complex features
2. **Appropriate Receptive Fields**: 3 levels sufficient for 88×88 input
3. **No Overparameterization**: Reduces overfitting risk
4. **Efficient Information Flow**: Direct path from features to prediction

#### Design Principles:
- **Encoder**: Extract multi-scale features from entire image
- **Gaze Integration**: Sample features only at gaze location after encoding
- **Decoder**: Simple MLP for single value regression
- **Loss**: Gaze-specific without spatial regularization

### Key Insights
1. **Architecture complexity should match task complexity**
2. **Gaze provides strong prior - no need to search entire image**
3. **88×88 resolution constrains useful feature extraction depth**
4. **Bilinear interpolation enables smooth sub-pixel feature extraction**