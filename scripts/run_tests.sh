#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-baseline}
BUILD_TYPE=${BUILD_TYPE:-Debug}
LOG="$ROOT/reports/correctness/${MODE}_latest.log"
"$ROOT/scripts/build.sh" "$BUILD_TYPE"
mkdir -p "$(dirname "$LOG")"
"$ROOT/build/${BUILD_TYPE,,}/test_compact" --mode "$MODE" --log "$LOG"
echo "correctness_log=$LOG"
