#!/bin/bash
#
# Batch process SLAM outputs to generate sparse depth maps
# Usage: ./batch_process_sparse_depth.sh [visualize]
#

set -e  # Exit on error

# Check arguments
VISUALIZE=""
if [ "$1" == "visualize" ]; then
    VISUALIZE="--visualize"
fi

SPARSE_OUTPUT_BASE="sparse_depth_sequences"

# Create output directory
mkdir -p "$SPARSE_OUTPUT_BASE"

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

# Find all tracking data directories
TRACKING_DIRS=(../results/tracking_data_*)
TOTAL_SEQUENCES=${#TRACKING_DIRS[@]}

if [ "$TOTAL_SEQUENCES" -eq 0 ]; then
    echo "No tracking data found in ../results/"
    exit 1
fi

echo "Found $TOTAL_SEQUENCES sequences to process"

# Process each sequence
COUNTER=0
for tracking_dir in "${TRACKING_DIRS[@]}"; do
    if [ ! -d "$tracking_dir" ]; then
        continue
    fi
    
    COUNTER=$((COUNTER + 1))
    
    # Extract sequence name
    seq_name=$(basename "$tracking_dir" | sed 's/tracking_data_//')
    tumvi_dir="tumvi_sequences/$seq_name"
    output_dir="$SPARSE_OUTPUT_BASE/sparse_$seq_name"
    
    echo "[$COUNTER/$TOTAL_SEQUENCES] Processing $seq_name..."
    
    # Check if already processed
    if [ -d "$output_dir" ] && [ -f "$output_dir/metadata/frames.json" ]; then
        n_sparse=$(ls "$output_dir/sparse_depth" 2>/dev/null | wc -l || echo 0)
        if [ "$n_sparse" -gt 100 ]; then
            echo "  Already processed ($n_sparse sparse depth maps), skipping..."
            continue
        fi
    fi
    
    # Check if TUM-VI data exists
    if [ ! -d "$tumvi_dir" ]; then
        echo "  ERROR: TUM-VI data not found at $tumvi_dir"
        continue
    fi
    
    # Process to sparse depth
    echo "  Generating sparse depth maps..."
    python process_slam_to_sparse_depth_adt.py \
        "$tracking_dir" \
        "$tumvi_dir" \
        "$output_dir" \
        $VISUALIZE || {
        echo "  ERROR: Sparse depth generation failed for $seq_name"
        continue
    }
    
    # Count output files
    if [ -d "$output_dir/sparse_depth" ]; then
        n_sparse=$(ls "$output_dir/sparse_depth" | grep -c "\.npy$" | grep -v "_conf" || echo 0)
        n_rgb=$(ls "$output_dir/rgb" 2>/dev/null | wc -l || echo 0)
        echo "  Generated: $n_sparse sparse depth maps, $n_rgb RGB images"
    fi
done

echo ""
echo "Sparse depth processing complete!"
echo ""

# Summary
echo "Processing summary:"
for sparse_dir in "$SPARSE_OUTPUT_BASE"/sparse_*; do
    if [ -d "$sparse_dir" ]; then
        seq_name=$(basename "$sparse_dir" | sed 's/sparse_//')
        n_sparse=$(ls "$sparse_dir/sparse_depth" 2>/dev/null | grep -c "\.npy$" | grep -v "_conf" || echo 0)
        metadata_file="$sparse_dir/metadata/frames.json"
        if [ -f "$metadata_file" ]; then
            n_frames=$(python -c "import json; print(len(json.load(open('$metadata_file'))))" 2>/dev/null || echo 0)
            echo "  $seq_name: $n_sparse sparse depth maps from $n_frames frames"
        else
            echo "  $seq_name: $n_sparse sparse depth maps (metadata missing)"
        fi
    fi
done

# Calculate total
TOTAL_SPARSE=0
for sparse_dir in "$SPARSE_OUTPUT_BASE"/sparse_*; do
    if [ -d "$sparse_dir/sparse_depth" ]; then
        n=$(ls "$sparse_dir/sparse_depth" | grep -c "\.npy$" | grep -v "_conf" || echo 0)
        TOTAL_SPARSE=$((TOTAL_SPARSE + n))
    fi
done

echo ""
echo "Total sparse depth maps generated: $TOTAL_SPARSE"