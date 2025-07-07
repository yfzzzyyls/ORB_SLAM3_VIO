#!/bin/bash
set -e

echo "Setting up build environment..."

# Set up Pangolin paths
export Pangolin_DIR=/home/external/Pangolin/build
export CMAKE_PREFIX_PATH=/home/external/Pangolin/build:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/home/external/Pangolin/build:$LD_LIBRARY_PATH

echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
if [ ! -f "ORBvoc.txt" ]; then
    tar -xf ORBvoc.txt.tar.gz
fi
cd ..

echo "Configuring and building ORB_SLAM3 ..."

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPangolin_DIR=/home/external/Pangolin/build
make -j4

echo "Build complete!"
