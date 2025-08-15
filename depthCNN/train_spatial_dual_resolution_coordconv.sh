#!/bin/bash

# Script to train spatial dual-resolution model with CoordConv
# Now includes optimizations for breaking through plateau:
# - CosineAnnealingWarmRestarts scheduler
# - Stochastic Weight Averaging (SWA)
# - Test-Time Augmentation (TTA)
# - AdamW with proper weight decay

# Activate conda environment
source ~/miniconda3/bin/activate orbslam

# Create log directory if it doesn't exist
mkdir -p ./logs/spatial_dual_coordconv

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/spatial_dual_coordconv/training_${TIMESTAMP}.log"

# Check if running with DDP
if [ -z "$1" ]; then
    echo "Usage: $0 <num_gpus> [resume_checkpoint] [options]"
    echo "Example: $0 1  # Single GPU"
    echo "Example: $0 4  # 4 GPUs with DDP"
    echo "Example: $0 4 ./checkpoints/spatial_dual_coordconv/checkpoint_latest.pth  # Resume training"
    echo ""
    echo "Options (set as environment variables):"
    echo "  USE_SWA=1  # Enable Stochastic Weight Averaging"
    echo "  SCHEDULER=cosine_restarts  # or 'plateau', 'onecycle'"
    echo "  LR=5e-5  # Custom learning rate"
    exit 1
fi

NUM_GPUS=$1
DATA_ROOT="/mnt/ssd_ext/incSeg-data/processed_adt"
BATCH_SIZE=32  # Per GPU
EPOCHS=${EPOCHS:-100}  # Standard training epochs
LR=${LR:-1e-4}  # Standard learning rate for fresh training
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}  # Standard weight decay
CHECKPOINT_DIR="./checkpoints/spatial_dual_coordconv"
LOG_DIR="./logs/spatial_dual_coordconv"
SCHEDULER=${SCHEDULER:-cosine_restarts}  # Default to cosine warm restarts
USE_SWA=${USE_SWA:-0}  # Disable SWA for standard training

# Optional resume checkpoint (second argument)
RESUME_CHECKPOINT=$2

# Print optimization settings
echo "======================================"
echo "🚀 Training Configuration"
echo "======================================"
echo "• Scheduler: $SCHEDULER"
echo "• Learning rate: $LR"
echo "• Weight decay: $WEIGHT_DECAY"
echo "• Epochs: $EPOCHS"
echo "• Batch size per GPU: $BATCH_SIZE"

if [ "$USE_SWA" = "1" ]; then
    echo "• SWA: Enabled (starting at epoch 80)"
    SWA_ARGS="--use-swa --swa-start 80 --swa-lr 2e-5"
else
    echo "• SWA: Disabled"
    SWA_ARGS=""
fi
echo "• TTA: Auto-enabled after epoch 50"
echo "======================================"

# Adjust batch size and learning rate based on number of GPUs
if [ $NUM_GPUS -eq 1 ]; then
    echo "Training on single GPU"
    TOTAL_BATCH_SIZE=$BATCH_SIZE
    
    # Build command - set max sequences to None to use ALL available
    CMD="python train_spatial_dual_resolution_coordconv.py \
        --data-root $DATA_ROOT \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --scheduler $SCHEDULER \
        --checkpoint-dir $CHECKPOINT_DIR \
        --log-dir $LOG_DIR \
        --num-workers 4 \
        --save-freq 5 \
        --max-train-sequences 999 \
        --max-val-sequences 999 \
        $SWA_ARGS"
    
    # Add resume if provided
    if [ ! -z "$RESUME_CHECKPOINT" ]; then
        echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
        CMD="$CMD --resume $RESUME_CHECKPOINT"
    fi
    
    # Execute command and save output
    echo "Saving output to: $LOG_FILE"
    eval $CMD 2>&1 | tee $LOG_FILE
        
elif [ $NUM_GPUS -gt 1 ]; then
    echo "Training on $NUM_GPUS GPUs with DDP"
    
    # Calculate total batch size
    TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
    
    # Don't scale learning rate when resuming from plateau
    # Linear scaling is for training from scratch
    if [ -z "$RESUME_CHECKPOINT" ]; then
        # Scale learning rate (linear scaling) for fresh training
        LR_SCALED=$(python -c "print($LR * $NUM_GPUS)")
    else
        # Keep learning rate as-is when resuming
        LR_SCALED=$LR
    fi
    
    echo "Total batch size: $TOTAL_BATCH_SIZE"
    echo "Learning rate: $LR_SCALED"
    
    # Build command - set max sequences to 999 to use ALL available
    CMD="torchrun --nproc_per_node=$NUM_GPUS \
        train_spatial_dual_resolution_coordconv.py \
        --data-root $DATA_ROOT \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR_SCALED \
        --weight-decay $WEIGHT_DECAY \
        --scheduler $SCHEDULER \
        --checkpoint-dir $CHECKPOINT_DIR \
        --log-dir $LOG_DIR \
        --num-workers 4 \
        --save-freq 5 \
        --max-train-sequences 999 \
        --max-val-sequences 999 \
        $SWA_ARGS"
    
    # Add resume if provided
    if [ ! -z "$RESUME_CHECKPOINT" ]; then
        echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
        CMD="$CMD --resume $RESUME_CHECKPOINT"
    fi
    
    # Execute command and save output
    echo "Saving output to: $LOG_FILE"
    eval $CMD 2>&1 | tee $LOG_FILE
else
    echo "Invalid number of GPUs: $NUM_GPUS"
    exit 1
fi

echo "Training completed!"
echo "Terminal output saved to: $LOG_FILE"
echo "TensorBoard logs saved to: $LOG_DIR"
echo "Checkpoints saved to: $CHECKPOINT_DIR"