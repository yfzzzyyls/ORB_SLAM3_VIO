#!/bin/bash

# Script to train spatial dual-resolution model with CoordConv
# Supports both single-GPU and multi-GPU training

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate orbslam

# Check if running with DDP
if [ -z "$1" ]; then
    echo "Usage: $0 <num_gpus>"
    echo "Example: $0 1  # Single GPU"
    echo "Example: $0 4  # 4 GPUs with DDP"
    exit 1
fi

NUM_GPUS=$1
DATA_ROOT="/mnt/ssd_ext/incSeg-data/processed_adt"
BATCH_SIZE=32  # Per GPU
EPOCHS=100
LR=1e-4
CHECKPOINT_DIR="./checkpoints/spatial_dual_coordconv"
LOG_DIR="./logs/spatial_dual_coordconv"

# Adjust batch size and learning rate based on number of GPUs
if [ $NUM_GPUS -eq 1 ]; then
    echo "Training on single GPU"
    TOTAL_BATCH_SIZE=$BATCH_SIZE
    
    python train_spatial_dual_resolution_coordconv.py \
        --data-root $DATA_ROOT \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --checkpoint-dir $CHECKPOINT_DIR \
        --log-dir $LOG_DIR \
        --num-workers 4 \
        --save-freq 5
        
elif [ $NUM_GPUS -gt 1 ]; then
    echo "Training on $NUM_GPUS GPUs with DDP"
    
    # Calculate total batch size
    TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
    
    # Scale learning rate (linear scaling)
    LR_SCALED=$(python -c "print($LR * $NUM_GPUS)")
    
    echo "Total batch size: $TOTAL_BATCH_SIZE"
    echo "Scaled learning rate: $LR_SCALED"
    
    # Run with torchrun (PyTorch's distributed launcher)
    torchrun --nproc_per_node=$NUM_GPUS \
        train_spatial_dual_resolution_coordconv.py \
        --data-root $DATA_ROOT \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR_SCALED \
        --checkpoint-dir $CHECKPOINT_DIR \
        --log-dir $LOG_DIR \
        --num-workers 4 \
        --save-freq 5
else
    echo "Invalid number of GPUs: $NUM_GPUS"
    exit 1
fi

echo "Training completed!"