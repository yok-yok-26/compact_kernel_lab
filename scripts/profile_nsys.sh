#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-v1}
BATCH_SIZE=${BATCH_SIZE:-8}
N=${N:-1048576}
THRESHOLD=${THRESHOLD:-0}
KEEP_PROB=${KEEP_PROB:-0.5}
OUT="$ROOT/reports/nsys/${MODE}_b${BATCH_SIZE}_n${N}_t${THRESHOLD}_latest"
CSV="$ROOT/reports/benchmark/${MODE}_b${BATCH_SIZE}_n${N}_t${THRESHOLD}_p${KEEP_PROB}_nsys_single.csv"
BUILD_TYPE=Release "$ROOT/scripts/build.sh" Release
mkdir -p "$ROOT/reports/nsys" "$ROOT/reports/benchmark"
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt -o "$OUT" \
  "$ROOT/build/release/benchmark_compact" --mode "$MODE" --batch-size "$BATCH_SIZE" --n "$N" --threshold "$THRESHOLD" --keep-prob "$KEEP_PROB" --single-launch --csv "$CSV"
echo "nsys_report=${OUT}.nsys-rep"
echo "single_launch_csv=$CSV"
