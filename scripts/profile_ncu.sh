#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MODE=${1:-v1}
BATCH_SIZE=${BATCH_SIZE:-8}
N=${N:-1048576}
THRESHOLD=${THRESHOLD:-0}
KEEP_PROB=${KEEP_PROB:-0.5}
BASE="$ROOT/reports/ncu/${MODE}_b${BATCH_SIZE}_n${N}_t${THRESHOLD}_latest"
CSV="$ROOT/reports/benchmark/${MODE}_b${BATCH_SIZE}_n${N}_t${THRESHOLD}_p${KEEP_PROB}_ncu_single.csv"
BUILD_TYPE=Release "$ROOT/scripts/build.sh" Release
mkdir -p "$ROOT/reports/ncu" "$ROOT/reports/benchmark"
"$CUDA_HOME/bin/ncu" --set full --force-overwrite --target-processes all \
  --export "$BASE" \
  "$ROOT/build/release/benchmark_compact" --mode "$MODE" --batch-size "$BATCH_SIZE" --n "$N" --threshold "$THRESHOLD" --keep-prob "$KEEP_PROB" --single-launch --csv "$CSV" | tee "${BASE}_raw.txt"
"$CUDA_HOME/bin/ncu" --import "${BASE}.ncu-rep" --page details > "${BASE}_details.txt" || true
echo "ncu_report=${BASE}.ncu-rep"
echo "ncu_raw=${BASE}_raw.txt"
echo "ncu_details=${BASE}_details.txt"
echo "single_launch_csv=$CSV"
