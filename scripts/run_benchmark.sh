#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-baseline}
BATCH_SIZE=${BATCH_SIZE:-8}
N=${N:-1048576}
THRESHOLD=${THRESHOLD:-0}
KEEP_PROB=${KEEP_PROB:-0.5}
ITERS=${ITERS:-100}
BUILD_TYPE=Release "$ROOT/scripts/build.sh" Release
CSV="$ROOT/reports/benchmark/${MODE}_b${BATCH_SIZE}_n${N}_t${THRESHOLD}_p${KEEP_PROB}_latest.csv"
mkdir -p "$(dirname "$CSV")"
"$ROOT/build/release/benchmark_compact" --mode "$MODE" --batch-size "$BATCH_SIZE" --n "$N" --threshold "$THRESHOLD" --keep-prob "$KEEP_PROB" --iters "$ITERS" --csv "$CSV"
echo "benchmark_csv=$CSV"
