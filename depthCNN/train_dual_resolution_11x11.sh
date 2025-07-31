#!/bin/bash

# Training script for clean 11×11 dual-resolution model

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate orbslam

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Training parameters
DATA_ROOT="/mnt/ssd_ext/incSeg-data/processed_adt"
OUTPUT_DIR="./checkpoints/dual_resolution_11x11"
BATCH_SIZE=128  # Larger batch size for faster training
EPOCHS=100
LR=2.8e-4  # Scaled learning rate for batch size 128 (sqrt scaling)
NUM_WORKERS=8  # More workers for faster data loading
MAX_TRAIN_SEQ=20
MAX_VAL_SEQ=2

# Model configuration
CONTEXT_CHANNELS=32    # Base channels for context encoder (expands to 64→128→256)
PATCH_CHANNELS=32      # Base channels for patch encoder
DROPOUT=0.1            # Dropout after fusion
AUX_WEIGHT=0.1        # Weight for auxiliary losses

echo "=== Training Clean 11×11 Dual-Resolution Model ==="
echo "Architecture:"
echo "  - Context: 88×88 → 44×44 → 22×22 → 11×11 (64→128→256 channels)"
echo "  - Patch: 44×44 → 22×22 → 11×11 with cross-connections"
echo "  - Output: 11×11 depth with center weighting"
echo "  - Dropout: ${DROPOUT} after fusion"
echo "  - Auxiliary supervision weight: ${AUX_WEIGHT}"
echo ""
echo "Training config:"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Learning rate: ${LR}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Training sequences: ${MAX_TRAIN_SEQ}"
echo "  - Validation sequences: ${MAX_VAL_SEQ}"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python train_dual_resolution_11x11.py \
    --data-root $DATA_ROOT \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --max-train-sequences $MAX_TRAIN_SEQ \
    --max-val-sequences $MAX_VAL_SEQ \
    --context-channels $CONTEXT_CHANNELS \
    --patch-channels $PATCH_CHANNELS \
    --dropout $DROPOUT \
    --aux-weight $AUX_WEIGHT

echo "Training completed!"