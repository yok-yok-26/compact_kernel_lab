#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
BUILD_TYPE=${1:-Release}
BUILD_DIR=${ROOT}/build/${BUILD_TYPE,,}
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DDEBUG_CUDA_SYNC=$([[ "$BUILD_TYPE" == "Debug" ]] && echo ON || echo OFF)
