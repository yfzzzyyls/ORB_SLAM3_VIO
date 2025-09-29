#!/bin/bash

# Training script for spatial-enhanced dual-resolution model with 44x44 output

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate orbslam

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Training parameters
DATA_ROOT="/mnt/ssd_ext/incSeg-data/processed_adt"
OUTPUT_DIR="./checkpoints/dual_resolution_spatial_44x44"
BATCH_SIZE=16
EPOCHS=50
LR=2e-4
NUM_WORKERS=4
MAX_TRAIN_SEQ=20
MAX_VAL_SEQ=2

echo "=== Training Spatial-Enhanced Dual-Resolution Model (44×44) ==="
echo "Architecture features:"
echo "  - Dual-resolution pathways: 88×88 context + 44×44 RGB patch"
echo "  - Spatial features: Full 11×11 feature maps (NO 5×5 extraction)"
echo "  - CNN decoder: Progressive upsampling 11→22→44"
echo "  - Skip connections: At 22×22 (from encoder) and 44×44 (from RGB)"
echo "  - Output: 44×44 depth patch (1,936 values)"
echo "  - Parameters: ~517K"
echo ""
echo "Training config:"
echo "  - Training sequences: ${MAX_TRAIN_SEQ}"
echo "  - Validation sequences: ${MAX_VAL_SEQ}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Learning rate: ${LR}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Expected MAE: < 0.40m"
echo ""
echo "Key improvements over 16×16 model:"
echo "  - 7.5x more dense output (1,936 vs 256 values)"
echo "  - No spatial bottleneck (full 11×11 features)"
echo "  - Multi-scale skip connections"
echo "  - More efficient (517K vs 591K parameters)"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Start training
python train_dual_resolution_spatial.py \
    --data-root $DATA_ROOT \
    --output-dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --max-train-sequences $MAX_TRAIN_SEQ \
    --max-val-sequences $MAX_VAL_SEQ \
    --base-channels 24 \
    --spatial-region-size 5 \
    --output-size 44 \
    --dropout 0.1 \
    --aux-weight 0.1 \
    --weight-decay 0.01 \
    --amp

echo ""
echo "Training completed! Check results in: $OUTPUT_DIR"