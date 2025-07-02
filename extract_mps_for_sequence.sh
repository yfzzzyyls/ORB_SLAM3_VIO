#!/bin/bash
# Helper script to extract MPS ground truth data for a sequence

if [ $# -eq 0 ]; then
    echo "Usage: $0 <sequence_name>"
    echo "Example: $0 loc3_script4_seq3_rec1"
    exit 1
fi

SEQUENCE=$1
SEQUENCE_DIR="/mnt/ssd_ext/incSeg-data/aria_everyday/$SEQUENCE"

if [ ! -d "$SEQUENCE_DIR" ]; then
    echo "Error: Sequence directory not found: $SEQUENCE_DIR"
    exit 1
fi

# Check if MPS data already extracted
if [ -d "$SEQUENCE_DIR/mps/slam" ]; then
    echo "MPS data already extracted for $SEQUENCE"
    exit 0
fi

# Check for zip files
ZIP_COUNT=$(ls $SEQUENCE_DIR/*mps*.zip 2>/dev/null | wc -l)
if [ $ZIP_COUNT -eq 0 ]; then
    echo "No MPS zip files found in $SEQUENCE_DIR"
    exit 1
fi

echo "Extracting MPS data for sequence: $SEQUENCE"
echo "Found $ZIP_COUNT MPS zip files"

# Extract all MPS zip files
for zip_file in $SEQUENCE_DIR/*mps*.zip; do
    echo "Extracting: $(basename $zip_file)"
    unzip -o "$zip_file" -d "$SEQUENCE_DIR/" > /dev/null
done

echo "MPS extraction complete!"

# Verify extraction - check multiple possible locations
FOUND=false
for trajectory_path in \
    "$SEQUENCE_DIR/mps/slam/closed_loop_trajectory.csv" \
    "$SEQUENCE_DIR/slam/closed_loop_trajectory.csv" \
    "$SEQUENCE_DIR/closed_loop_trajectory.csv"; do
    if [ -f "$trajectory_path" ]; then
        echo "✓ Ground truth trajectory found at: $trajectory_path"
        POSE_COUNT=$(wc -l < "$trajectory_path")
        echo "  Contains $POSE_COUNT poses"
        FOUND=true
        break
    fi
done

if [ "$FOUND" = false ]; then
    echo "⚠️  Warning: closed_loop_trajectory.csv not found"
    echo "  Checked locations:"
    echo "    - $SEQUENCE_DIR/mps/slam/"
    echo "    - $SEQUENCE_DIR/slam/"
    echo "    - $SEQUENCE_DIR/"
fi