#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MODE=${1:-baseline}
LOG="$ROOT/reports/synccheck/${MODE}_latest.log"
BUILD_TYPE=Debug "$ROOT/scripts/build.sh" Debug
mkdir -p "$(dirname "$LOG")"
"$CUDA_HOME/bin/compute-sanitizer" --tool synccheck --error-exitcode 1 --log-file "$LOG" "$ROOT/build/debug/test_compact" --mode "$MODE" --log "$ROOT/reports/correctness/${MODE}_synccheck_correctness.log"
echo "synccheck_log=$LOG"
