#!/bin/bash
#
# Batch run ORB-SLAM3 on converted TUM-VI sequences
# Usage: ./batch_run_slam.sh <tumvi_base_dir>
#

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <tumvi_base_dir>"
    echo "Example: $0 tumvi_sequences"
    exit 1
fi

TUMVI_BASE_DIR="$1"
SLAM_RESULTS_DIR="slam_results"

# Create results directory
mkdir -p "$SLAM_RESULTS_DIR"

# Setup environment
echo "Setting up environment..."
# Check if already in orbslam environment
if [[ "$CONDA_DEFAULT_ENV" == "orbslam" ]]; then
    echo "Already in orbslam environment"
else
    # Initialize conda properly
    eval "$(/home/external/miniconda3/bin/conda shell.bash hook)"
    conda activate orbslam
fi
source ../setup_env.sh

# Enable tracking data saving
export SAVE_TRACKING=1

# Count sequences
TOTAL_SEQUENCES=$(find "$TUMVI_BASE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Found $TOTAL_SEQUENCES sequences to process"

# Process each sequence
COUNTER=0
for seq_dir in "$TUMVI_BASE_DIR"/*; do
    if [ ! -d "$seq_dir" ]; then
        continue
    fi
    
    COUNTER=$((COUNTER + 1))
    seq_name=$(basename "$seq_dir")
    
    echo "[$COUNTER/$TOTAL_SEQUENCES] Processing $seq_name..."
    
    # Check if already processed
    tracking_output="../results/tracking_data_${seq_name}"
    if [ -d "$tracking_output" ] && [ "$(ls -A $tracking_output 2>/dev/null | wc -l)" -gt 100 ]; then
        echo "  Already processed, skipping..."
        continue
    fi
    
    # Run SLAM
    echo "  Running ORB-SLAM3..."
    
    # Save current directory
    CURRENT_DIR=$(pwd)
    
    # Change to parent directory for correct paths
    cd ..
    
    # Run the SLAM script
    ./run_orbslam3_aria_tumvi.sh "$CURRENT_DIR/$seq_dir" "$seq_name" || {
        echo "  ERROR: SLAM failed for $seq_name"
        cd "$CURRENT_DIR"
        continue
    }
    
    # Return to original directory
    cd "$CURRENT_DIR"
    
    # Verify output
    if [ -d "$tracking_output" ]; then
        n_frames=$(ls "$tracking_output" | wc -l)
        echo "  SLAM completed: $n_frames tracking frames saved"
        
        # Copy trajectory files
        cp "../results/CameraTrajectory_${seq_name}_mono_inertial.txt" "$SLAM_RESULTS_DIR/" 2>/dev/null || true
        cp "../results/KeyFrameTrajectory_${seq_name}_mono_inertial.txt" "$SLAM_RESULTS_DIR/" 2>/dev/null || true
    else
        echo "  WARNING: No tracking data found for $seq_name"
    fi
done

echo ""
echo "SLAM processing complete!"
echo ""

# Summary
echo "Processing summary:"
for tracking_dir in ../results/tracking_data_*; do
    if [ -d "$tracking_dir" ]; then
        seq_name=$(basename "$tracking_dir" | sed 's/tracking_data_//')
        n_frames=$(ls "$tracking_dir" | wc -l)
        echo "  $seq_name: $n_frames tracking frames"
    fi
done