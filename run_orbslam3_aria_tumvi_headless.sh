#!/bin/bash

# Run ORB-SLAM3 on Aria data using TUM-VI format (headless version)
# Now supports automatic detection of converted data directory

echo "ORB-SLAM3 Aria TUM-VI Runner (Headless)"
echo "========================================"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_directory> [output_name]"
    echo "Example: $0 aria_tumvi_test test_trajectory"
    echo ""
    echo "The data_directory should contain the mav0 folder from aria_to_tumvi.py conversion"
    exit 1
fi

DATA_DIR=$1
OUTPUT_NAME=${2:-"aria_trajectory"}

# Setup environment
source /home/external/ORB_SLAM3_AEA/setup_env.sh

# Check if data exists
if [ ! -d "$DATA_DIR/mav0" ]; then
    echo "Error: Data directory not found: $DATA_DIR/mav0"
    echo "Make sure you've run aria_to_tumvi.py first!"
    exit 1
fi

# Set paths
MAV0_DIR="$DATA_DIR/mav0"
IMAGES_DIR="$MAV0_DIR/cam0/data"
TIMESTAMPS_FILE="$MAV0_DIR/timestamps.txt"
IMU_FILE="$MAV0_DIR/imu0/data.csv"

# Verify all required files exist
for path in "$IMAGES_DIR" "$TIMESTAMPS_FILE" "$IMU_FILE"; do
    if [ ! -e "$path" ]; then
        echo "Error: Required path not found: $path"
        exit 1
    fi
done

# Count data
NUM_IMAGES=$(ls $IMAGES_DIR/*.png 2>/dev/null | wc -l)
NUM_TIMESTAMPS=$(wc -l < $TIMESTAMPS_FILE)
NUM_IMU=$(tail -n +2 $IMU_FILE | wc -l)

echo "Data summary:"
echo "  Images: $NUM_IMAGES"
echo "  Timestamps: $NUM_TIMESTAMPS"
echo "  IMU samples: $NUM_IMU"

# Extract sequence info from dataset.yaml if available
if [ -f "$DATA_DIR/dataset.yaml" ]; then
    SEQUENCE_NAME=$(grep "sequence_name:" "$DATA_DIR/dataset.yaml" | cut -d':' -f2 | xargs)
    echo "  Sequence: $SEQUENCE_NAME"
fi
echo ""

# Create results directory
mkdir -p results

# Get absolute paths
ABS_DATA_DIR=$(realpath "$DATA_DIR")
ABS_IMAGES_DIR="$ABS_DATA_DIR/mav0/cam0/data"
ABS_TIMESTAMPS_FILE="$ABS_DATA_DIR/mav0/timestamps.txt"
ABS_IMU_FILE="$ABS_DATA_DIR/mav0/imu0/data.csv"

cd results

# Run ORB-SLAM3 without viewer
echo "Starting ORB-SLAM3 (headless)..."
echo "Output will be saved to: $OUTPUT_NAME"
echo ""

/home/external/ORB_SLAM3_AEA/Examples/Monocular-Inertial/mono_inertial_tum_vi_noviewer \
    /home/external/ORB_SLAM3_AEA/Vocabulary/ORBvoc.txt \
    /home/external/ORB_SLAM3_AEA/Examples/Monocular-Inertial/Aria2TUM-VI.yaml \
    "$ABS_IMAGES_DIR" \
    "$ABS_TIMESTAMPS_FILE" \
    "$ABS_IMU_FILE" \
    $OUTPUT_NAME

echo ""
echo "ORB-SLAM3 finished!"
echo ""

# Show results
if [ -f "f_${OUTPUT_NAME}.txt" ]; then
    echo "Frame trajectory saved to: results/f_${OUTPUT_NAME}.txt"
    echo "Number of frames tracked: $(wc -l < f_${OUTPUT_NAME}.txt)"
    echo ""
    echo "First 5 poses:"
    head -5 f_${OUTPUT_NAME}.txt
else
    echo "Warning: No frame trajectory file generated"
fi

if [ -f "kf_${OUTPUT_NAME}.txt" ]; then
    echo ""
    echo "Keyframe trajectory saved to: results/kf_${OUTPUT_NAME}.txt"
    echo "Number of keyframes: $(wc -l < kf_${OUTPUT_NAME}.txt)"
fi

# Save info about which data was processed
echo "$DATA_DIR" > last_processed_data_dir.txt
echo "$OUTPUT_NAME" > last_trajectory_name.txt

echo ""
echo "To evaluate this trajectory, run:"
echo "./evaluate_slam_clean.sh $DATA_DIR $OUTPUT_NAME"