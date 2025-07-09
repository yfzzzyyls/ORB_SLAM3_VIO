#!/bin/bash
#
# Batch convert ADT VRS files to TUM-VI format
# Usage: ./batch_convert_to_tumvi.sh <input_dir> [duration]
#

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_dir> [duration_in_seconds]"
    echo "Example: $0 /mnt/ssd_ext/incSeg-data/adt/train 60"
    exit 1
fi

INPUT_DIR="$1"
DURATION="${2:-}"  # Optional duration, empty means full sequence
OUTPUT_BASE_DIR="tumvi_sequences"

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

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

# Count sequences
TOTAL_SEQUENCES=$(find "$INPUT_DIR" -name "*main_recording.vrs" | wc -l)
echo "Found $TOTAL_SEQUENCES sequences to convert"

# Process each VRS file
COUNTER=0
for vrs_file in "$INPUT_DIR"/*/ADT_*main_recording.vrs; do
    if [ ! -f "$vrs_file" ]; then
        echo "No VRS files found in $INPUT_DIR"
        exit 1
    fi
    
    COUNTER=$((COUNTER + 1))
    
    # Extract sequence name
    seq_dir=$(dirname "$vrs_file")
    seq_name=$(basename "$seq_dir")
    output_dir="$OUTPUT_BASE_DIR/$seq_name"
    
    echo "[$COUNTER/$TOTAL_SEQUENCES] Processing $seq_name..."
    
    # Check if already converted
    if [ -d "$output_dir" ] && [ -f "$output_dir/mav0/timestamps.txt" ]; then
        echo "  Already converted, skipping..."
        continue
    fi
    
    # Convert to TUM-VI format
    echo "  Converting VRS to TUM-VI format..."
    if [ -n "$DURATION" ]; then
        python ../aria_to_tumvi.py "$vrs_file" "$output_dir" --duration "$DURATION"
    else
        python ../aria_to_tumvi.py "$vrs_file" "$output_dir"
    fi
    
    # Verify conversion
    if [ ! -f "$output_dir/mav0/timestamps.txt" ]; then
        echo "  ERROR: Conversion failed for $seq_name"
        exit 1
    fi
    
    # Count frames
    n_frames=$(wc -l < "$output_dir/mav0/timestamps.txt")
    echo "  Converted successfully: $n_frames frames"
done

echo ""
echo "Conversion complete!"
echo "Output directory: $OUTPUT_BASE_DIR"
echo ""

# Summary
echo "Conversion summary:"
for seq_dir in "$OUTPUT_BASE_DIR"/*; do
    if [ -d "$seq_dir" ]; then
        seq_name=$(basename "$seq_dir")
        n_frames=$(wc -l < "$seq_dir/mav0/timestamps.txt" 2>/dev/null || echo "0")
        echo "  $seq_name: $n_frames frames"
    fi
done