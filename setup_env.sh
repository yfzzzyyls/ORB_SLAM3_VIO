#!/bin/bash
# Setup environment for ORB-SLAM3 with local OpenCV and Pangolin

# Activate Python virtual environment
source ~/venv/py39/bin/activate

# Set OpenCV paths
export OpenCV_DIR=/home/external/opencv_install/lib64/cmake/opencv4
export PKG_CONFIG_PATH=/home/external/opencv_install/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/home/external/opencv_install/lib64:$LD_LIBRARY_PATH

# Set Pangolin paths
export Pangolin_DIR=/home/external/Pangolin/build
export CMAKE_PREFIX_PATH=/home/external/Pangolin/build:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/home/external/Pangolin/build:$LD_LIBRARY_PATH

# Set GLEW paths
export GLEW_ROOT=/home/external/glew_install/usr
export CMAKE_PREFIX_PATH=/home/external/glew_install/usr:$CMAKE_PREFIX_PATH
export PKG_CONFIG_PATH=/home/external/glew_install/usr/lib64/pkgconfig:$PKG_CONFIG_PATH

echo "Environment configured:"
echo "  Python venv: $VIRTUAL_ENV"
echo "  OpenCV_DIR: $OpenCV_DIR"
echo "  Pangolin_DIR: $Pangolin_DIR"
echo "  GLEW_ROOT: $GLEW_ROOT"
echo ""
echo "You can now build ORB-SLAM3 with: ./build.sh"