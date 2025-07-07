# Depth Prediction CNN with ADT Dataset

This project implements RT-MonoDepth-S for metric depth prediction using the Aria Digital Twin (ADT) dataset.

## Dataset Information

ADT provides:
- RGB images: 1408×1408 native, cropped to 1024×1024 in ATEK
- Depth maps: 1024×1024, pixel-aligned with RGB crop
- Depth format: 16-bit uint millimeters (divide by 1000 for meters)
- Accuracy: ~5mm error on 1024px crop

## Quick Start

### 1. Setup Environment
```bash
# Use existing orbslam conda environment
source ~/miniconda3/bin/activate
conda activate orbslam
```

### 2. Download ADT Data
We need ATEK cubercnn format (1024×1024 RGB + depth):
```bash
cd /home/external/ORB_SLAM3_AEA/depthCNN
python download_adt_cubercnn.py
```

### 3. Setup RT-MonoDepth-S
```bash
python setup_rtmonodepth.py
```

### 4. Train Model
```bash
python train_rtmonodepth.py \
    --data-path /mnt/ssd_ext/incSeg-data/adt/ATEK_cubercnn \
    --batch-size 4 \
    --epochs 25 \
    --lr 1e-4
```

### 5. Evaluate
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data-path /mnt/ssd_ext/incSeg-data/adt/ATEK_cubercnn
```

### 6. Export to TensorRT
```bash
python export_tensorrt.py \
    --checkpoint checkpoints/best_model.pth \
    --output models/rtmonodepth_fp16.trt
```

## Model Details

- **Architecture**: RT-MonoDepth-S (3M parameters)
- **Input**: 640×640 crops from 1024×1024 images
- **Loss**: 0.9×SILog + 0.1×L1
- **Target**: ~14ms inference @ 1024×1024 on embedded GPU

## Data Pipeline

1. Load ATEK cubercnn shards (1024×1024)
2. Convert depth: `depth_m = depth_uint16.float() / 1000.0`
3. Mask invalid pixels (depth == 0)
4. Random 640×640 crops for training
5. Keep intrinsics for each sample

## Notes

- ATEK cubercnn provides RGB at 1024×1024 (center crop from 1408×1408)
- Depth is already pixel-aligned with RGB
- Use provided intrinsics (recalculated for 1024px crop)
- For full 1408×1408: upsample depth bilinearly after masking zeros