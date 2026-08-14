#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_TYPE=${1:-Release}
BUILD_DIR=${ROOT}/build/${BUILD_TYPE,,}
if [[ ! -d "$BUILD_DIR" ]]; then
  "$ROOT/scripts/configure.sh" "$BUILD_TYPE"
fi
cmake --build "$BUILD_DIR" -j"$(nproc)"
