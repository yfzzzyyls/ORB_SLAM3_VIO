#!/bin/bash

# Run ORB-SLAM3 on ADT data in TUM-VI format
# Works with output from processed_adt_to_tumvi.py

echo "ORB-SLAM3 Runner for ADT Dataset"
echo "================================"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <sequence_name|--all> [output_name]"
    echo "Examples:"
    echo "  $0 Apartment_release_clean_seq131_M1292            # Run single sequence"
    echo "  $0 Apartment_release_clean_seq131_M1292 my_test    # Run with custom output name"
    echo "  $0 --all                                           # Run all sequences"
    echo ""
    echo "Sequences should be in output_tumvi/ directory from processed_adt_to_tumvi.py"
    echo ""
    echo "Available sequences:"
    ls -1 output_tumvi/ 2>/dev/null | grep -v "^$" || echo "  No sequences found in output_tumvi/"
    exit 1
fi

# Handle --all option
if [ "$1" == "--all" ]; then
    echo "Running ORB-SLAM3 on all sequences..."
    echo ""
    
    # Get all sequences
    SEQUENCES=$(ls -1 output_tumvi/ 2>/dev/null | grep -v "^$")
    if [ -z "$SEQUENCES" ]; then
        echo "Error: No sequences found in output_tumvi/"
        exit 1
    fi
    
    # Run each sequence
    for SEQ in $SEQUENCES; do
        echo "=========================================="
        echo "Processing sequence: $SEQ"
        echo "=========================================="
        
        # Recursively call this script for each sequence
        $0 "$SEQ"
        
        echo ""
        echo "Finished $SEQ"
        echo ""
        sleep 2  # Brief pause between sequences
    done
    
    echo "All sequences processed!"
    exit 0
fi

SEQUENCE_NAME=$1
OUTPUT_NAME=${2:-"${SEQUENCE_NAME}_trajectory"}
DATA_DIR="output_tumvi/$SEQUENCE_NAME"

# Setup environment
# Assume user has already activated conda environment
if [ "$CONDA_DEFAULT_ENV" != "orbslam" ]; then
    echo "Warning: Not in orbslam conda environment!"
    echo "Please run: conda activate orbslam"
    echo "Continuing anyway..."
fi

# Setup environment (should already be done by setup_env.sh)
if [ -z "$Pangolin_DIR" ]; then
    echo "Setting up Pangolin paths..."
    export Pangolin_DIR=/home/external/Pangolin/build
    export CMAKE_PREFIX_PATH=/home/external/Pangolin/build:$CMAKE_PREFIX_PATH
    export LD_LIBRARY_PATH=/home/external/Pangolin/build:$LD_LIBRARY_PATH
fi

# Check if data exists
if [ ! -d "$DATA_DIR/mav0" ]; then
    echo "Error: Data directory not found: $DATA_DIR/mav0"
    echo "Make sure you've run processed_adt_to_tumvi.py first!"
    echo ""
    echo "Available sequences:"
    ls -1 output_tumvi/ 2>/dev/null | grep -v "^$"
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

# Extract info from dataset.yaml
if [ -f "$DATA_DIR/dataset.yaml" ]; then
    DURATION=$(grep "duration:" "$DATA_DIR/dataset.yaml" | cut -d':' -f2 | xargs)
    echo "  Sequence: $SEQUENCE_NAME"
    echo "  Duration: ${DURATION}s"
fi
echo ""

# Create results directory under parent
mkdir -p ../results

# Get absolute paths
ABS_DATA_DIR=$(realpath "$DATA_DIR")
ABS_IMAGES_DIR="$ABS_DATA_DIR/mav0/cam0/data"
ABS_TIMESTAMPS_FILE="$ABS_DATA_DIR/mav0/timestamps.txt"
ABS_IMU_FILE="$ABS_DATA_DIR/mav0/imu0/data.csv"

cd ../results

# Check if tracking data save is requested
if [ ! -z "$SAVE_TRACKING" ] || [ ! -z "$ORB_SLAM3_SAVE_TRACKING" ]; then
    export ORB_SLAM3_SAVE_TRACKING="tracking_data_${OUTPUT_NAME}"
    mkdir -p "$ORB_SLAM3_SAVE_TRACKING"
    echo "Tracking data will be saved to: results/$ORB_SLAM3_SAVE_TRACKING"
    echo ""
fi

# Run ORB-SLAM3 with Pangolin viewer
echo "Starting ORB-SLAM3..."
echo "Trajectory will be saved as: $OUTPUT_NAME"
echo ""
echo "Viewer controls:"
echo "  - Left mouse: Rotate view"
echo "  - Right mouse: Pan view" 
echo "  - Scroll: Zoom in/out"
echo "  - 's': Start/stop SLAM"
echo "  - 'r': Reset system"
echo "  - 'q': Quit"
echo ""

../Examples/Monocular-Inertial/mono_inertial_tum_vi \
    ../Vocabulary/ORBvoc.txt \
    ../Examples/Monocular-Inertial/Aria2TUM-VI.yaml \
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
echo "cd .. && ./evaluate_slam_clean.sh scripts/$DATA_DIR $OUTPUT_NAME"