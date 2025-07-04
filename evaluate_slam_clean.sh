#!/bin/bash
# Complete SLAM evaluation with clean output
# Now supports evaluating any sequence

echo "=== Complete SLAM Evaluation Pipeline ==="

# Check arguments
if [ $# -eq 0 ]; then
    # Try to use last processed data
    if [ -f "results/last_processed_data_dir.txt" ] && [ -f "results/last_trajectory_name.txt" ]; then
        DATA_DIR=$(cat results/last_processed_data_dir.txt)
        TRAJECTORY_NAME=$(cat results/last_trajectory_name.txt)
        echo "Using last processed data: $DATA_DIR"
        echo "Trajectory name: $TRAJECTORY_NAME"
    else
        echo "Usage: $0 <data_directory> <trajectory_name>"
        echo "Example: $0 aria_tumvi_test my_trajectory"
        echo ""
        echo "Or run without arguments to evaluate the last processed sequence"
        exit 1
    fi
elif [ $# -eq 1 ]; then
    DATA_DIR=$1
    TRAJECTORY_NAME="my_trajectory"  # Default name
elif [ $# -eq 2 ]; then
    DATA_DIR=$1
    TRAJECTORY_NAME=$2
else
    echo "Usage: $0 <data_directory> <trajectory_name>"
    echo "Example: $0 aria_tumvi_test my_trajectory"
    exit 1
fi

# Check if trajectory file exists
TRAJECTORY_FILE="results/f_${TRAJECTORY_NAME}.txt"
if [ ! -f "$TRAJECTORY_FILE" ]; then
    echo "Error: Trajectory file not found: $TRAJECTORY_FILE"
    echo "Make sure you've run ORB-SLAM3 first!"
    exit 1
fi

# Create evaluation directory
mkdir -p evaluation

# Clean old plots in evaluation directory
echo "Cleaning old evaluation files..."
rm -f evaluation/ate_plot.pdf evaluation/rpe_1s_plot.pdf evaluation/rpe_5s_plot.pdf 
rm -f evaluation/trajectory_comparison_3d.pdf evaluation/trajectory_comparison_top.pdf
rm -f evaluation/ate_results.zip evaluation/rpe_1s_results.zip evaluation/rpe_5s_results.zip

# Create ground truth directory
mkdir -p ground_truth_data

# Extract sequence information from dataset.yaml
echo "Determining source sequence..."
if [ -f "$DATA_DIR/dataset.yaml" ]; then
    # Extract sequence name from dataset.yaml
    SEQUENCE_NAME=$(grep "sequence_name:" "$DATA_DIR/dataset.yaml" | cut -d':' -f2 | xargs)
    echo "Found sequence: $SEQUENCE_NAME"
    
    # Find the sequence directory
    SEQUENCE_DIR="/mnt/ssd_ext/incSeg-data/aria_everyday/$SEQUENCE_NAME"
    
    if [ ! -d "$SEQUENCE_DIR" ]; then
        echo "Error: Sequence directory not found: $SEQUENCE_DIR"
        exit 1
    fi
else
    echo "Warning: dataset.yaml not found in $DATA_DIR"
    echo "Cannot determine source sequence for ground truth extraction"
    exit 1
fi

# Get actual VRS start time from the VRS file using the dataset.yaml
echo "Getting VRS device timestamp..."
VRS_FILE=$(grep "vrs_file:" "$DATA_DIR/dataset.yaml" | cut -d':' -f2- | xargs)
if [ -f "$VRS_FILE" ]; then
    VRS_START_TIME_NS=$(source /home/external/miniconda3/bin/activate && conda activate orbslam && python -c "
from projectaria_tools.core import data_provider
provider = data_provider.create_vrs_data_provider('$VRS_FILE')
slam_left = provider.get_stream_id_from_label('camera-slam-left')
img_data = provider.get_image_data_by_index(slam_left, 0)
print(img_data[1].capture_timestamp_ns)
" 2>/dev/null)
    echo "VRS device start time: $VRS_START_TIME_NS ns"
else
    echo "Warning: Could not find VRS file, using relative timestamps"
    VRS_START_TIME_NS=""
fi

# Extract MPS ground truth
echo "Extracting MPS closed-loop ground truth from: $SEQUENCE_DIR"
if [ -n "$VRS_START_TIME_NS" ]; then
    python extract_mps_ground_truth.py "$SEQUENCE_DIR" --output-dir ground_truth_data --vrs-start-time-ns $VRS_START_TIME_NS
else
    python extract_mps_ground_truth.py "$SEQUENCE_DIR" --output-dir ground_truth_data
fi

# Check if ground truth was extracted
GROUND_TRUTH_FILE="ground_truth_data/mps_closed_loop_tum.txt"
if [ ! -f "$GROUND_TRUTH_FILE" ]; then
    echo "Error: Ground truth extraction failed!"
    echo "Make sure MPS data is available in: $SEQUENCE_DIR"
    exit 1
fi

# Align SLAM trajectory timestamps with MPS timestamps
echo -e "\n=== Aligning trajectories ==="
python align_trajectories.py "$TRAJECTORY_FILE" "$GROUND_TRUTH_FILE"
ALIGNED_TRAJECTORY="${TRAJECTORY_FILE%.txt}_aligned.txt"

if [ ! -f "$ALIGNED_TRAJECTORY" ]; then
    echo "Error: Trajectory alignment failed!"
    exit 1
fi

echo "Using aligned trajectory: $ALIGNED_TRAJECTORY"

# 1. Compute ATE (Absolute Trajectory Error)
echo -e "\n=== Computing ATE ==="
evo_ape tum "$GROUND_TRUTH_FILE" "$ALIGNED_TRAJECTORY" \
    --plot --plot_mode xy --save_plot evaluation/ate_plot.pdf \
    --save_results evaluation/ate_results.zip \
    --align --correct_scale

# 2. Compute RPE at 1 second (10 frames at 10Hz)
echo -e "\n=== Computing RPE at 1 second (10 frames) ==="
evo_rpe tum "$GROUND_TRUTH_FILE" "$ALIGNED_TRAJECTORY" \
    --plot --plot_mode xy --save_plot evaluation/rpe_1s_plot.pdf \
    --save_results evaluation/rpe_1s_results.zip \
    --align --correct_scale \
    --delta 10 --delta_unit f

# 3. Compute RPE at 5 seconds (50 frames at 10Hz)
echo -e "\n=== Computing RPE at 5 seconds (50 frames) ==="
evo_rpe tum "$GROUND_TRUTH_FILE" "$ALIGNED_TRAJECTORY" \
    --plot --plot_mode xy --save_plot evaluation/rpe_5s_plot.pdf \
    --save_results evaluation/rpe_5s_results.zip \
    --align --correct_scale \
    --delta 50 --delta_unit f

# 4. Create trajectory comparison plots
echo -e "\n=== Plotting Trajectory Comparison ==="
evo_traj tum "$GROUND_TRUTH_FILE" --ref "$ALIGNED_TRAJECTORY" \
    --plot --plot_mode xyz --save_plot evaluation/trajectory_comparison_3d.pdf 2>/dev/null || true

evo_traj tum "$GROUND_TRUTH_FILE" --ref "$ALIGNED_TRAJECTORY" \
    --plot --plot_mode xy --save_plot evaluation/trajectory_comparison_top.pdf 2>/dev/null || true

# Extract and display metrics
echo -e "\n=== Extracting Metrics ==="
cd evaluation

# Extract ATE metrics
if [ -f ate_results.zip ]; then
    unzip -q -o ate_results.zip
    if [ -f stats.json ]; then
        echo -e "\n📊 ATE (Absolute Trajectory Error):"
        python3 -c "
import json
with open('stats.json') as f:
    data = json.load(f)
    print(f'  RMSE:   {data[\"rmse\"]:.3f} m')
    print(f'  Mean:   {data[\"mean\"]:.3f} m')
    print(f'  Median: {data[\"median\"]:.3f} m')
    print(f'  Std:    {data[\"std\"]:.3f} m')
    print(f'  Min:    {data[\"min\"]:.3f} m')
    print(f'  Max:    {data[\"max\"]:.3f} m')
"
    fi
fi

# Extract 1s RPE metrics
if [ -f rpe_1s_results.zip ]; then
    unzip -q -o rpe_1s_results.zip
    if [ -f stats.json ]; then
        echo -e "\n📊 RPE at 1 second:"
        python3 -c "
import json
with open('stats.json') as f:
    data = json.load(f)
    print(f'  RMSE:   {data[\"rmse\"]:.3f} m/s')
    print(f'  Mean:   {data[\"mean\"]:.3f} m/s')
    print(f'  Median: {data[\"median\"]:.3f} m/s')
"
    fi
fi

# Extract 5s RPE metrics
if [ -f rpe_5s_results.zip ]; then
    unzip -q -o rpe_5s_results.zip
    if [ -f stats.json ]; then
        echo -e "\n📊 RPE at 5 seconds:"
        python3 -c "
import json
with open('stats.json') as f:
    data = json.load(f)
    print(f'  RMSE:   {data[\"rmse\"]:.3f} m/5s')
    print(f'  Mean:   {data[\"mean\"]:.3f} m/5s')
    print(f'  Median: {data[\"median\"]:.3f} m/5s')
"
    fi
fi

cd ..

# 5. Generate summary report
echo -e "\n=== Summary Report ==="
echo "Evaluation Complete!"
echo ""
echo "Sequence evaluated: $SEQUENCE_NAME"
echo "Trajectory name: $TRAJECTORY_NAME"
echo ""
echo "All results saved in evaluation/ folder:"
ls -la evaluation/*.pdf evaluation/*.zip

echo -e "\nView results: cd evaluation && evince ate_plot.pdf"