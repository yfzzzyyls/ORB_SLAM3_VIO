#!/bin/bash
# Setup environment for ORB-SLAM3 using conda environment

# Only activate conda if not already in the orbslam environment
if [ "$CONDA_DEFAULT_ENV" != "orbslam" ]; then
    source /home/external/miniconda3/bin/activate
    conda activate orbslam
fi

# Note: Pangolin must be built separately as it's not in conda
# Set Pangolin paths (needs to be rebuilt if deleted)
export Pangolin_DIR=/home/external/Pangolin/build
export CMAKE_PREFIX_PATH=/home/external/Pangolin/build:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/home/external/Pangolin/build:$LD_LIBRARY_PATH

echo "Environment configured:"
echo "  Conda env: $CONDA_DEFAULT_ENV (provides OpenCV, Python packages)"
echo "  Pangolin_DIR: $Pangolin_DIR (must be built from source)"
echo ""
echo "You can now:"
echo "  - Build ORB-SLAM3 with: ./build.sh"
echo "  - Run test sequence with: ./test_new_sequence.sh"