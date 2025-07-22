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

#### Approach 1: Original RT-MonoDepth (Dense Depth Prediction)

##### Full Resolution (1408×1408)
```bash
# Single GPU
python train.py \
    --data-root ./processed_data \
    --epochs 20 \
    --batch-size 4 \
    --lr 1e-4 \
    --crop-size 1408 \
    --checkpoint-dir ./checkpoints/rtmonodepth_full_original \
    --log-dir ./logs/rtmonodepth_full_original

# Multi-GPU (2 GPUs example)
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --data-root ./processed_data \
    --epochs 20 \
    --batch-size 8 \
    --lr 1.5e-4 \
    --crop-size 1408 \
    --checkpoint-dir ./checkpoints/rtmonodepth_full_original \
    --log-dir ./logs/rtmonodepth_full_original
```
**Model**: Original RT-MonoDepth-S with 1.23M parameters (858K encoder + 376K decoder)  
**Results**: Dense depth map at 1408×1408, can evaluate at any pixel including gaze

##### Low Resolution (88×88)
```bash
# Single GPU
python train_lowres.py \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --checkpoint-dir ./checkpoints/rtmonodepth_88_original \
    --log-dir ./logs/rtmonodepth_88_original

# Multi-GPU (2 GPUs example)
CUDA_VISIBLE_DEVICES=2,3 python train_lowres.py \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --epochs 20 \
    --batch-size 64 \
    --lr 2e-4 \
    --checkpoint-dir ./checkpoints/rtmonodepth_88_original \
    --log-dir ./logs/rtmonodepth_88_original
```
**Model**: Same RT-MonoDepth-S architecture but trained on 88×88 images  
**Results**: 88×88 dense depth map, faster training and inference

##### Running Both Simultaneously on 4 GPUs
```bash
# Terminal 1: Full resolution on GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --data-root ./processed_data \
    --epochs 20 \
    --batch-size 8 \
    --lr 1.5e-4 \
    --crop-size 1408 \
    --checkpoint-dir ./checkpoints/rtmonodepth_full_original \
    --log-dir ./logs/rtmonodepth_full_original

# Terminal 2: Low resolution on GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 python train_lowres.py \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --epochs 20 \
    --batch-size 64 \
    --lr 2e-4 \
    --checkpoint-dir ./checkpoints/rtmonodepth_88_original \
    --log-dir ./logs/rtmonodepth_88_original
```

#### Approach 2: Gaze-Only Depth (Single Point Prediction)

#### Approach 3: Lightweight Gaze-Only (RECOMMENDED)
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
| RT-MonoDepth Full | 1.23M | 1408×1408 map | TBD | 20-50ms | Need full depth map |
| RT-MonoDepth 88×88 | 1.23M | 88×88 map | TBD | 3-5ms | Fast dense depth |
| Gaze-Only Original | 1.41M | Single point | ~0.53m | 2-3ms | Baseline gaze depth |
| **Lightweight Gaze** | **354K** | **Single point** | **0.41m** | **2-3ms** | **Best for gaze** |
| Multi-task Gaze | 377K | Single point | **0.394m** | 2-3ms | Improved accuracy |
| Dual-Resolution | 1.13M | Single point | ~0.363m | ~15ms | Maximum accuracy |

### 4. Evaluate

#### Universal Evaluation Script (NEW - For All Model Types)
```bash
# Evaluate any model with automatic configuration detection
python evaluate_flexible.py \
    --checkpoint <path_to_checkpoint> \
    --image-size <input_size> \
    --data-root ./processed_data \
    --save-results

# Examples:

# 88×88 Baseline Model
python evaluate_flexible.py \
    --checkpoint ./checkpoints/lightweight_gaze/level3_ch32/checkpoint_best.pth \
    --image-size 88 \
    --data-root ./processed_data \
    --save-results

# 88×88 Multi-task Model
python evaluate_flexible.py \
    --checkpoint ./checkpoints/multitask_88/size88_level3_ch32/checkpoint_best.pth \
    --image-size 88 \
    --data-root ./processed_data \
    --save-results

# 352×352 Model
python evaluate_flexible.py \
    --checkpoint ./checkpoints/flexible_gaze_352/size352_level4_ch32/checkpoint_best.pth \
    --image-size 352 \
    --data-root ./processed_data \
    --save-results

# Dual-Resolution Model (88×88 context + 96×96 patch)
python evaluate_flexible.py \
    --checkpoint ./checkpoints/dual_resolution/size88_level3_ch32/checkpoint_best.pth \
    --image-size 88 \
    --model-type dual \
    --data-root ./processed_data \
    --save-results \
    --batch-size 32
```

The evaluation script automatically:
- Detects model type from checkpoint path
- Loads appropriate architecture
- Computes comprehensive metrics (MAE, RMSE, abs_rel, threshold accuracies)
- Measures inference timing and FPS
- Saves results to JSON and predictions to NPZ


#### Expected Results (Updated with Latest Models)

| Model | Params | Latency | MAE | RMSE | abs_rel | δ < 1.25 | FPS |
|-------|--------|---------|-----|------|---------|----------|-----|
| **88×88 Baseline** | 354K | 2.54ms | 0.420m | 0.600m | 0.188 | 76.5% | 393 |
| **88×88 Multi-task** | 377K | 2.54ms | **0.394m** | 0.601m | **0.184** | **80.4%** | 394 |
| **Dual-resolution** | 1.13M | ~15ms | ~0.363m* | TBD | TBD | TBD | ~60 |
| Dense Low-Res | 914K | 2.43ms | 0.418m | 0.518m | - | 74.8% | - |
| Original Gaze | 1.41M | ~2ms | ~0.53m | ~0.65m | - | ~75% | - |

*Validation MAE from training

The evaluation will report:
- **Standard metrics**: MAE, RMSE, abs_rel, sq_rel, log_mae, δ accuracies
- **Error statistics**: Median, std, min/max errors, 95th percentile
- **Latency**: Inference time per frame and FPS
- **Training info**: Best validation metrics from checkpoint

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
# Evaluate any trained model
python evaluate_flexible.py \
    --checkpoint <checkpoint_path> \
    --image-size <size> \
    --data-root ./processed_data \
    --save-results

# Test the downsampling pipeline
python test_lowres_pipeline.py --data-root ./processed_data --visualize

# Visualize a single sample from the dataset
python lowres_dataset.py --data-root ./processed_data --scale-factor 16 --visualize

# Resume training from checkpoint
python train_lightweight_gaze.py \
    --data-root ./processed_data \
    --encoder-levels 3 \
    --resume ./checkpoints/lightweight_gaze/level3_ch32/checkpoint_latest.pth \
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

## Flexible Resolution Training (NEW - Addresses Spatial Ambiguity)

### Overview
A new flexible training system that supports variable image sizes and encoder levels to address the spatial ambiguity issues in the 88×88 model. The key insight: at 88×88 with 3-level encoding, each spatial location in the deepest feature map represents an 8×8 patch, causing ±4 pixel uncertainty in gaze localization.

### Supported Configurations

#### 1. **352×352 with 4-Level Encoder (RECOMMENDED)**
```bash
python train_flexible_gaze.py \
    --data-root ./processed_data \
    --image-size 352 \
    --encoder-levels 4 \
    --base-channels 32 \
    --batch-size 64 \
    --lr 2e-3 \
    --lr-scaling sqrt \
    --num-workers 4 \
    --epochs 30 \
    --checkpoint-dir ./checkpoints/gaze_352_4level
```
- **Spatial Precision**: 16× better than 88×88
- **Receptive Field**: ~112 pixels (4× larger)
- **Feature Maps**: 352→176→88→44→22
- **Gaze Uncertainty**: ±2 pixels in original space
- **Expected Results**: Significantly reduced RMSE and max error

#### 2. **176×176 with 3-Level Encoder (Balanced)**
```bash
python train_flexible_gaze.py \
    --data-root ./processed_data \
    --image-size 176 \
    --encoder-levels 3 \
    --base-channels 32 \
    --batch-size 128 \
    --lr 3e-3 \
    --lr-scaling sqrt \
    --num-workers 4 \
    --epochs 30 \
    --checkpoint-dir ./checkpoints/gaze_176_3level
```
- **Spatial Precision**: 4× better than 88×88
- **Memory Efficient**: Still fits large batches
- **Good compromise between accuracy and speed**

#### 3. **88×88 with 3-Level Encoder (Original)**
```bash
python train_flexible_gaze.py \
    --data-root ./processed_data \
    --image-size 88 \
    --encoder-levels 3 \
    --base-channels 32 \
    --batch-size 128 \
    --lr 4e-4 \
    --epochs 30 \
    --checkpoint-dir ./checkpoints/gaze_88_3level
```
- Equivalent to the lightweight model configuration
- Baseline for comparison

#### 4. **Full Resolution 1408×1408 (Memory Intensive)**
```bash
# WARNING: Use small batch size and num_workers=0 to avoid system crashes
python train_flexible_gaze.py \
    --data-root ./processed_data \
    --image-size 1408 \
    --encoder-levels 5 \
    --base-channels 48 \
    --batch-size 32 \
    --lr 1.4e-3 \
    --lr-scaling sqrt \
    --num-workers 0 \
    --epochs 30 \
    --checkpoint-dir ./checkpoints/gaze_1408_5level
```

### Architecture Details

The flexible encoder automatically adapts to the input size:
- **Channel Progression**: base_channels × (1.5^level)
- **Feature Extraction**: Multi-scale bilinear sampling at gaze location
- **MLP Decoder**: Adapts input dimension based on total features
- **Memory Scaling**: ~(image_size/88)² relative to baseline

### Why This Addresses High Variance

The 88×88 model's high RMSE (0.6m) and max error (4.83m) are primarily due to:
1. **Spatial Ambiguity**: Each feature represents 8×8 pixels
2. **Limited Context**: Small receptive field misses larger objects
3. **Quantization Error**: Coarse feature maps lose fine details

The 352×352 configuration solves these by:
1. **16× Better Precision**: Each feature represents 2×2 pixels
2. **Larger Context**: 112-pixel receptive field captures full objects
3. **Fine-Grained Features**: 22×22 final feature map vs 11×11

### Expected Improvements

Based on the spatial precision increase, we expect:
- **RMSE**: 0.6m → ~0.3-0.4m (30-50% reduction)
- **Max Error**: 4.83m → ~2-3m (40-60% reduction)
- **MAE**: 0.41m → ~0.35m (modest improvement)

### Training Tips

1. **Start with 352×352**: Best balance of accuracy and efficiency
2. **Use sqrt LR scaling**: Helps with larger batch sizes
3. **Monitor memory**: Larger images need more GPU/CPU RAM
4. **Gradual increases**: Try 176→352→704 if needed
5. **Keep num_workers low**: Prevents memory issues with large images

## Multi-Task Learning (Improved Performance)

### Overview
Multi-task learning forces the model to learn consistent features by predicting auxiliary patch statistics alongside the main depth prediction. This approach achieved **6.4% improvement** in MAE over the baseline.

### Training Command
```bash
python train_flexible_gaze.py \
    --data-root ./processed_data \
    --image-size 88 \
    --encoder-levels 3 \
    --base-channels 32 \
    --use-multi-task \
    --batch-size 128 \
    --lr 4e-4 \
    --epochs 30 \
    --checkpoint-dir ./checkpoints/multitask_88
```

### Architecture Details
The multi-task model predicts:
1. **Primary**: Depth at gaze location
2. **Auxiliary**: Patch statistics from 16×16 region around gaze:
   - Mean depth
   - Standard deviation
   - Gradient magnitude
   - Edge score
   - Depth bin classification (5 bins: 0-2m, 2-4m, 4-6m, 6-8m, 8m+)

### Loss Function
- **Total Loss** = depth_loss + 0.1×mean_loss + 0.1×std_loss + 0.05×gradient_loss + 0.05×edge_loss + 0.1×bin_loss
- Forces feature consistency across related tasks

## Dual-Resolution Architecture (Maximum Accuracy)

### Overview
Combines low-resolution context (88×88) with high-resolution patch (96×96) at gaze location for maximum accuracy. Achieves ~8.3% improvement over baseline but at 3× parameter cost.

### Training Command
```bash
python train_flexible_gaze.py \
    --data-root ./processed_data \
    --image-size 88 \
    --use-dual-resolution \
    --patch-size 96 \
    --patch-channels 32 \
    --encoder-levels 3 \
    --base-channels 32 \
    --batch-size 64 \
    --lr 2e-4 \
    --epochs 50 \
    --checkpoint-dir ./checkpoints/dual_resolution
```

### Architecture Components
1. **Context Encoder**: Processes full 88×88 image for scene understanding
2. **Patch Encoder**: Processes 96×96 high-res patch centered at gaze
3. **Feature Fusion**: Attention-based fusion of context and patch features
4. **Depth Predictor**: Final MLP for depth prediction

### Key Benefits
- **High-res details**: 96×96 patch provides fine-grained features at gaze
- **Context awareness**: 88×88 image provides scene understanding
- **Best of both**: Combines efficiency of low-res with accuracy of high-res