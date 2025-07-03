#!/bin/bash

# Test the updated pipeline with a new sequence
echo "=== Testing Updated Pipeline with New Sequence ==="

# Set the VRS file as per user request
VRS_FILE="/mnt/ssd_ext/incSeg-data/aria_everyday/loc3_script4_seq3_rec1/AriaEverydayActivities_1.0.0_loc3_script4_seq3_rec1_main_recording.vrs"

echo "Sequence: loc3_script4_seq3_rec1"
echo ""

# Step 1: Convert VRS to TUM-VI format
echo "Step 1: Converting VRS to TUM-VI format..."
source /home/external/miniconda/bin/activate
conda activate orbslam
python aria_to_tumvi.py "$VRS_FILE" aria_tumvi_test --duration 30

# Check if dataset.yaml was created
if [ -f "aria_tumvi_test/dataset.yaml" ]; then
    echo ""
    echo "Dataset info:"
    grep -E "sequence_name:|duration:|num_images:" aria_tumvi_test/dataset.yaml
fi

# Step 2: Run ORB-SLAM3
echo ""
echo "Step 2: Running ORB-SLAM3..."
./run_orbslam3_aria_tumvi.sh aria_tumvi_test my_trajectory

# Step 3: Evaluate results
echo ""
echo "Step 3: Evaluating results..."
./evaluate_slam_clean.sh aria_tumvi_test my_trajectory

echo ""
echo "Pipeline complete! Check evaluation/ folder for results."