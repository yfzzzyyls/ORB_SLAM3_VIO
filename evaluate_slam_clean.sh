#!/bin/bash
# Complete SLAM evaluation with clean output

echo "=== Complete SLAM Evaluation Pipeline ==="

# Create evaluation directory
mkdir -p evaluation

# Clean old plots in evaluation directory
echo "Cleaning old evaluation files..."
rm -f evaluation/ate_plot.pdf evaluation/rpe_1s_plot.pdf evaluation/rpe_5s_plot.pdf 
rm -f evaluation/trajectory_comparison_3d.pdf evaluation/trajectory_comparison_top.pdf
rm -f evaluation/ate_results.zip evaluation/rpe_1s_results.zip evaluation/rpe_5s_results.zip

# Create ground truth directory
mkdir -p ground_truth_data

# Extract MPS ground truth from Aria dataset
echo "Extracting MPS closed-loop ground truth from Aria dataset..."

# Find which sequence was used (check for most recent VRS conversion)
if [ -d "aria_tumvi_test" ]; then
    # Try to find the sequence from dataset.yaml
    if [ -f "aria_tumvi_test/mav0/dataset.yaml" ]; then
        VRS_FILE=$(grep -A1 "vrs_file:" aria_tumvi_test/mav0/dataset.yaml | grep -v "vrs_file:" | xargs)
        SEQUENCE_DIR=$(dirname "$VRS_FILE")
        echo "Found sequence: $SEQUENCE_DIR"
    else
        # Default to loc1_script1_seq1_rec1
        SEQUENCE_DIR="/mnt/ssd_ext/incSeg-data/aria_everyday/loc1_script1_seq1_rec1"
        echo "Using default sequence: $SEQUENCE_DIR"
    fi
else
    SEQUENCE_DIR="/mnt/ssd_ext/incSeg-data/aria_everyday/loc1_script1_seq1_rec1"
    echo "Using default sequence: $SEQUENCE_DIR"
fi

# Get VRS start time from ORB-SLAM3 output
VRS_START_TIME_NS=$(head -1 results/f_my_trajectory.txt | awk '{print $1}' | cut -d'.' -f1)
echo "VRS start time: $VRS_START_TIME_NS ns"

# Extract MPS ground truth with timestamp alignment
python extract_mps_ground_truth.py "$SEQUENCE_DIR" --output-dir ground_truth_data --vrs-start-time-ns $VRS_START_TIME_NS

# Use the extracted MPS ground truth file
GROUND_TRUTH_FILE="ground_truth_data/mps_closed_loop_tum.txt"

# 1. Compute ATE (Absolute Trajectory Error)
echo -e "\n=== Computing ATE ==="
evo_ape tum "$GROUND_TRUTH_FILE" results/f_my_trajectory.txt \
    -va \
    --plot \
    --save_plot evaluation/ate_plot.pdf \
    --save_results evaluation/ate_results.zip

# 2. Compute RPE at 1 second (10 frames at 10Hz)
echo -e "\n=== Computing RPE at 1 second (10 frames) ==="
evo_rpe tum "$GROUND_TRUTH_FILE" results/f_my_trajectory.txt \
    -va \
    --delta 10 \
    --delta_unit f \
    --plot \
    --save_plot evaluation/rpe_1s_plot.pdf \
    --save_results evaluation/rpe_1s_results.zip

# 3. Compute RPE at 5 seconds (50 frames at 10Hz)
echo -e "\n=== Computing RPE at 5 seconds (50 frames) ==="
evo_rpe tum "$GROUND_TRUTH_FILE" results/f_my_trajectory.txt \
    -va \
    --delta 50 \
    --delta_unit f \
    --plot \
    --save_plot evaluation/rpe_5s_plot.pdf \
    --save_results evaluation/rpe_5s_results.zip

# 4. Plot trajectories comparison
echo -e "\n=== Plotting Trajectory Comparison ==="
# 3D view
evo_traj tum "$GROUND_TRUTH_FILE" results/f_my_trajectory.txt \
    --ref=ground_truth \
    -a \
    --plot --plot_mode xyz \
    --save_plot evaluation/trajectory_comparison_3d.pdf

# Top view
evo_traj tum "$GROUND_TRUTH_FILE" results/f_my_trajectory.txt \
    --ref=ground_truth \
    -a \
    --plot --plot_mode xy \
    --save_plot evaluation/trajectory_comparison_top.pdf

# 5. Generate summary report
echo -e "\n=== Summary Report ==="

# Create summary markdown file
cat > evaluation/slam_evaluation_summary.md << 'EOF'
# SLAM Evaluation Summary

## Dataset
- **Sequence**: Aria Everyday Activities
- **Date**: $(date)
- **IMU Rate**: 1000Hz (native rate)
- **Camera**: 10Hz monocular with 150° FOV fisheye

## Results
All metrics and visualizations are saved in this evaluation folder.

### Files Generated
- `ate_plot.pdf` - ATE visualization and error distribution
- `rpe_1s_plot.pdf` - RPE at 1 second intervals
- `rpe_5s_plot.pdf` - RPE at 5 second intervals
- `trajectory_comparison_3d.pdf` - 3D trajectory comparison
- `trajectory_comparison_top.pdf` - Top-down view comparison
- `ate_results.zip` - Raw ATE data
- `rpe_1s_results.zip` - Raw RPE 1s data
- `rpe_5s_results.zip` - Raw RPE 5s data

### Metrics Interpretation
- **ATE < 0.1m**: Excellent global consistency
- **RPE < 0.05m**: Excellent local accuracy
- **RPE 0.05-0.1m**: Good local accuracy

### Ground Truth Source
- Using MPS closed-loop SLAM trajectory from Aria dataset
- MPS provides high-accuracy ground truth with loop closure corrections
- Ground truth frequency: ~1000 Hz

Note: RPE measures relative motion error, not cumulative drift rate.
EOF

echo "Evaluation Complete!"
echo ""
echo "All results saved in evaluation/ folder:"
ls -la evaluation/*.pdf evaluation/*.zip 2>/dev/null | grep -E "(\\.pdf|\\.zip)"
echo ""
echo "View results: cd evaluation && evince ate_plot.pdf"