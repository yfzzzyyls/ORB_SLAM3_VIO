# Depth Prediction CNN with ADT Dataset

This project implements RT-MonoDepth-S for metric depth prediction using the Aria Digital Twin (ADT) dataset.

## Dataset Information

ADT provides:
- RGB images: 1408×1408 at 20Hz from camera-rgb (214-1)
- Depth maps: 1408×1408 synthetic ground truth depth
- Depth format: 16-bit uint millimeters (divide by 1000 for meters)
- Depth range: 0-7.6 meters (typical indoor scenes)
- Timestamp offset: Depth recording starts ~10-16 seconds after RGB

## Extracted Dataset

The dataset has been extracted with timestamp-based matching:
- **Train**: 7 sequences (~20,154 RGB-depth pairs)
- **Val**: 1 sequence (~2,881 pairs)
- **Test**: 2 sequences (~5,731 pairs)
- **Total**: ~28,766 matched RGB-depth pairs at full 20Hz
- **Quality**: 1-to-1 RGB-depth matching with 0.1ms average time difference
- **Coverage**: ~86% valid depth pixels per frame

## Quick Start

### 1. Setup Environment
```bash
# Use existing orbslam conda environment
source ~/miniconda3/bin/activate
conda activate orbslam
```

### 2. Extract Data (if not already done)
```bash
# Uses timestamp-based matching to handle RGB-depth time offset
python extract_dataset.py  # All defaults configured for full 20Hz extraction
```

### 3. Train Model
```bash
python train.py \
    --data-root ./processed_data \
    --epochs 20 \
    --batch-size 4 \
    --lr 1e-4 \
    --crop-size 1408
```
Note: Use batch-size 4 for full resolution. Increase to 8-16 if using smaller crops.

### 4. Evaluate
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data-root ./processed_data
```

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
- **Training**: ~20,000 frames at 20Hz (10x more than 2Hz subsampling)

## Data Pipeline

1. Load RGB (PNG) and depth (NPZ) from processed_data/
2. Convert depth: `depth_m = depth_uint16.float() / 1000.0`
3. Create valid mask: `valid = depth > 0`
4. Apply data augmentation (random horizontal flip)
5. Normalize RGB to [0, 1]
6. No cropping - use full 1408×1408 resolution

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