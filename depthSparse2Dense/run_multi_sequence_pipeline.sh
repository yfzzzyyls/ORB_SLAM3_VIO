#!/bin/bash
#
# Main pipeline for multi-sequence sparse-to-dense training
# Processes all ADT sequences in train folder
#

set -e  # Exit on error

# Configuration
TRAIN_DIR="/mnt/ssd_ext/incSeg-data/adt/train"
DURATION=""  # Empty means full sequence, set to e.g. "60" for 60 seconds
MERGED_DATA_DIR="merged_training_data"
MODEL_OUTPUT_DIR="trained_models_multi"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Multi-Sequence Sparse-to-Dense Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if train directory exists
if [ ! -d "$TRAIN_DIR" ]; then
    echo -e "${RED}Error: Training directory not found: $TRAIN_DIR${NC}"
    echo "Please run the sequence organization step first."
    exit 1
fi

# Count sequences
N_SEQUENCES=$(ls -d "$TRAIN_DIR"/* 2>/dev/null | wc -l)
N_TEST_SEQUENCES=$(ls -d /mnt/ssd_ext/incSeg-data/adt/test/* 2>/dev/null | wc -l)
echo -e "Found ${YELLOW}$N_SEQUENCES${NC} sequences in training directory"
echo -e "Found ${YELLOW}$N_TEST_SEQUENCES${NC} sequences in test directory"
echo ""

# Step 1: Convert VRS to TUM-VI
echo -e "${GREEN}Step 1: Converting VRS files to TUM-VI format${NC}"
echo "----------------------------------------"
if [ -n "$DURATION" ]; then
    ./batch_convert_to_tumvi.sh "$TRAIN_DIR" "$DURATION"
else
    ./batch_convert_to_tumvi.sh "$TRAIN_DIR"
fi
echo ""

# Step 2: Run SLAM on all sequences
echo -e "${GREEN}Step 2: Running ORB-SLAM3 on all sequences${NC}"
echo "----------------------------------------"
export SAVE_TRACKING=1
./batch_run_slam.sh tumvi_sequences
echo ""

# Step 3: Generate sparse depth maps
echo -e "${GREEN}Step 3: Processing SLAM outputs to sparse depth${NC}"
echo "----------------------------------------"
./batch_process_sparse_depth.sh
echo ""

# Step 4: Merge all sequences
echo -e "${GREEN}Step 4: Merging sequences into training dataset${NC}"
echo "----------------------------------------"
python merge_sequences.py \
    --pattern "sparse_depth_sequences/sparse_*" \
    --output_dir "$MERGED_DATA_DIR" \
    --train_ratio 0.9
echo ""

# Verify merged data
if [ ! -d "$MERGED_DATA_DIR" ] || [ ! -f "$MERGED_DATA_DIR/metadata/frames.json" ]; then
    echo -e "${RED}Error: Merged data not created properly${NC}"
    exit 1
fi

# Count merged frames
N_FRAMES=$(python -c "import json; print(len(json.load(open('$MERGED_DATA_DIR/metadata/frames.json'))))" 2>/dev/null || echo 0)
echo -e "Total merged frames: ${YELLOW}$N_FRAMES${NC}"
echo ""

# Step 5: Train the model
echo -e "${GREEN}Step 5: Training sparse-to-dense model${NC}"
echo "----------------------------------------"
echo -e "${YELLOW}Note: Training on 8 sequences with 90/10 train/val split${NC}"

# Check for existing checkpoint
RESUME_FLAG=""
if [ -d "$MODEL_OUTPUT_DIR" ] && [ -f "$MODEL_OUTPUT_DIR/checkpoint_last.pth" ]; then
    echo -e "${YELLOW}Found existing checkpoint, will resume training${NC}"
    RESUME_FLAG="--resume"
fi

# Determine batch size based on available GPUs
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
if [ "$N_GPUS" -gt 1 ]; then
    BATCH_SIZE=$((32 * N_GPUS))
    LEARNING_RATE="4e-4"
    echo -e "Using ${YELLOW}$N_GPUS GPUs${NC} with batch size $BATCH_SIZE"
else
    BATCH_SIZE=32
    LEARNING_RATE="2e-4"
    echo -e "Using ${YELLOW}single GPU${NC} with batch size $BATCH_SIZE"
fi

python train_sparse_to_dense.py \
    --data_dir "$MERGED_DATA_DIR" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --epochs 100 \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --model unet \
    $RESUME_FLAG

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  - Training sequences: $N_SEQUENCES (will be split 90/10 for train/val)"
echo "  - Test sequences: $N_TEST_SEQUENCES (held out for final evaluation)"
echo "  - Total training frames: $N_FRAMES"
echo "  - Model output: $MODEL_OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check tensorboard logs: tensorboard --logdir $MODEL_OUTPUT_DIR/tensorboard"
echo "  2. Process test sequences: ./process_test_sequences.sh"
echo "  3. Evaluate on test sequences: ./evaluate_on_test.sh"
echo "  4. Run inference: python inference_sparse_to_dense.py"