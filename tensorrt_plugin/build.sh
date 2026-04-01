#!/bin/bash

# Build script for FMHA TensorRT Plugin

set -e  # Exit on any error

echo "Building FMHA TensorRT Plugin..."

# Create build directory
mkdir -p build
cd build

# Configure with cmake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT=/usr/src/tensorrt \
    -DCUDA_TOOLKIT_ROOT_DIR=$(dirname $(dirname $(which nvcc)))

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the plugin
echo "Building the plugin..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"
echo "Plugin location: $(pwd)/lib/libfmha_plugin.so"

# Test if the library was created
if [ -f "lib/libfmha_plugin.so" ]; then
    echo "Plugin built successfully!"
    ls -lh lib/libfmha_plugin.so
else
    echo "Plugin build failed - library not found!"
    exit 1
fi

echo "Build process completed."