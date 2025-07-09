#!/bin/bash
#
# Process test sequences through SLAM and sparse depth generation
# This prepares test data for final model evaluation
#

set -e  # Exit on error

# Configuration
TEST_DIR="/mnt/ssd_ext/incSeg-data/adt/test"
TEST_OUTPUT_DIR="test_sequences_processed"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Processing Test Sequences${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}Error: Test directory not found: $TEST_DIR${NC}"
    exit 1
fi

# Count test sequences
N_SEQUENCES=$(ls -d "$TEST_DIR"/* 2>/dev/null | wc -l)
echo -e "Found ${YELLOW}$N_SEQUENCES${NC} test sequences"
echo ""

# Setup environment
source /home/external/miniconda3/bin/activate
conda activate orbslam
source ../setup_env.sh

# Step 1: Convert test VRS to TUM-VI
echo -e "${GREEN}Step 1: Converting test VRS files to TUM-VI format${NC}"
echo "----------------------------------------"
mkdir -p "$TEST_OUTPUT_DIR/tumvi"

for vrs_file in "$TEST_DIR"/*/ADT_*main_recording.vrs; do
    if [ ! -f "$vrs_file" ]; then
        continue
    fi
    
    seq_dir=$(dirname "$vrs_file")
    seq_name=$(basename "$seq_dir")
    output_dir="$TEST_OUTPUT_DIR/tumvi/$seq_name"
    
    echo "Converting $seq_name..."
    
    # Check if already converted
    if [ -d "$output_dir" ] && [ -f "$output_dir/mav0/timestamps.txt" ]; then
        echo "  Already converted, skipping..."
    else
        python ../aria_to_tumvi.py "$vrs_file" "$output_dir"
    fi
done
echo ""

# Step 2: Run SLAM on test sequences
echo -e "${GREEN}Step 2: Running SLAM on test sequences${NC}"
echo "----------------------------------------"
export SAVE_TRACKING=1

for tumvi_dir in "$TEST_OUTPUT_DIR"/tumvi/*; do
    if [ ! -d "$tumvi_dir" ]; then
        continue
    fi
    
    seq_name=$(basename "$tumvi_dir")
    echo "Processing $seq_name with SLAM..."
    
    # Check if already processed
    tracking_output="../results/test_tracking_data_${seq_name}"
    if [ -d "$tracking_output" ] && [ "$(ls -A $tracking_output 2>/dev/null | wc -l)" -gt 100 ]; then
        echo "  Already processed, skipping..."
        continue
    fi
    
    # Run SLAM from parent directory
    CURRENT_DIR=$(pwd)
    cd ..
    
    # Create custom output name for test sequences
    ./Examples/Monocular-Inertial/mono_inertial_tum_vi \
        Vocabulary/ORBvoc.txt \
        Examples/Monocular-Inertial/Aria2TUM-VI.yaml \
        "$CURRENT_DIR/$tumvi_dir/mav0" \
        "$CURRENT_DIR/$tumvi_dir/mav0/timestamps.txt" \
        "$CURRENT_DIR/$tumvi_dir/mav0/imu.csv" \
        "test_${seq_name}" || {
        echo "  ERROR: SLAM failed for $seq_name"
        cd "$CURRENT_DIR"
        continue
    }
    
    # Move tracking data to test directory
    if [ -d "results/tracking_data_test_${seq_name}" ]; then
        mv "results/tracking_data_test_${seq_name}" "results/test_tracking_data_${seq_name}"
    fi
    
    cd "$CURRENT_DIR"
done
echo ""

# Step 3: Generate sparse depth for test sequences
echo -e "${GREEN}Step 3: Generating sparse depth maps for test sequences${NC}"
echo "----------------------------------------"
mkdir -p "$TEST_OUTPUT_DIR/sparse_depth"

for tracking_dir in ../results/test_tracking_data_*; do
    if [ ! -d "$tracking_dir" ]; then
        continue
    fi
    
    seq_name=$(basename "$tracking_dir" | sed 's/test_tracking_data_//')
    tumvi_dir="$TEST_OUTPUT_DIR/tumvi/$seq_name"
    output_dir="$TEST_OUTPUT_DIR/sparse_depth/sparse_$seq_name"
    
    echo "Processing sparse depth for $seq_name..."
    
    # Check if already processed
    if [ -d "$output_dir" ] && [ -f "$output_dir/metadata/frames.json" ]; then
        n_sparse=$(ls "$output_dir/sparse_depth" 2>/dev/null | wc -l || echo 0)
        if [ "$n_sparse" -gt 100 ]; then
            echo "  Already processed ($n_sparse sparse depth maps), skipping..."
            continue
        fi
    fi
    
    python process_slam_to_sparse_depth_adt.py \
        "$tracking_dir" \
        "$tumvi_dir" \
        "$output_dir" || {
        echo "  ERROR: Sparse depth generation failed for $seq_name"
        continue
    }
done
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Processing Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Test data summary:"
for sparse_dir in "$TEST_OUTPUT_DIR"/sparse_depth/sparse_*; do
    if [ -d "$sparse_dir" ]; then
        seq_name=$(basename "$sparse_dir" | sed 's/sparse_//')
        n_sparse=$(ls "$sparse_dir/sparse_depth" 2>/dev/null | grep -c "\.npy$" | grep -v "_conf" || echo 0)
        echo "  $seq_name: $n_sparse sparse depth maps"
    fi
done
echo ""
echo "Test data location: $TEST_OUTPUT_DIR"
echo ""
echo "Next step: Run evaluation with trained model"
echo "  python evaluate_on_test.py --model_checkpoint trained_models_multi/checkpoint_best.pth"