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

### 3. Train Model

#### Full Resolution (1408×1408)
```bash
# Single GPU
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

# Specific GPUs only
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --data-root ./processed_data \
    --epochs 20 \
    --batch-size 16 \
    --lr 2e-4 \
    --crop-size 1408
```
Note: 
- Single GPU: Use batch-size 4 for full resolution
- Multi-GPU: Can use batch-size 16 (4 per GPU) or higher
- Learning rate scales with batch size (2x batch → ~1.4-2x lr)

#### Low Resolution Training (88×88 with 16x downscaling)
```bash
# Efficient training with larger batch size
python train_lowres.py \
    --data-root ./processed_data \
    --lowres-scale 16 \
    --epochs 20 \
    --batch-size 32 \
    --lr 2e-4 \
    --crop-size 1408

# Other scale factors (2, 4, 8, 16)
python train_lowres.py \
    --data-root ./processed_data \
    --lowres-scale 8 \
    --epochs 20 \
    --batch-size 16 \
    --lr 2e-4
```

Benefits of low-resolution training:
- **256x faster** computation (16x16 = 256x fewer pixels)
- **8x larger batches** possible with same GPU memory
- **Faster convergence** for gaze-specific depth prediction
- **Ideal for real-time** applications where only gaze depth matters
- **Gaze-aware evaluation**: Computes metrics specifically at gaze locations

Note: The training scripts include a custom collate function to handle missing gaze data gracefully.

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