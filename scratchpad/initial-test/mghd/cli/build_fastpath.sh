#!/usr/bin/env bash
set -euo pipefail

echo "Building fastpath C API shared library..."

# Get script directory and navigate to c_api
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../fastpath/c_api"

# Create build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the library
echo "Building shared library..."
cmake --build . -j$(nproc)

# Check if library was created
if [ -f "libfastpath.so" ]; then
    echo "✅ Successfully built $(pwd)/libfastpath.so"
    ls -la libfastpath.so
else
    echo "❌ Failed to build libfastpath.so"
    exit 1
fi

echo "🎉 Build completed successfully!"
