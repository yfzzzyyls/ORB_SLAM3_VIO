#!/bin/bash
# Setup environment for ORB-SLAM3 with local OpenCV

# Activate Python virtual environment
source ~/venv/py39/bin/activate

# Set OpenCV paths
export OpenCV_DIR=$HOME/venv/py39/opencv_cpp/lib64/cmake/opencv4
export PKG_CONFIG_PATH=$HOME/venv/py39/opencv_cpp/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HOME/venv/py39/opencv_cpp/lib64:$LD_LIBRARY_PATH

echo "Environment configured:"
echo "  Python venv: $VIRTUAL_ENV"
echo "  OpenCV_DIR: $OpenCV_DIR"
echo ""
echo "You can now build ORB-SLAM3 with: ./build.sh"