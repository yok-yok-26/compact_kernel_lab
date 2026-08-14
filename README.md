# compact_kernel_lab

CUDA learning lab for batched threshold-only stream compaction on an NVIDIA GeForce RTX 5070.

The project compares hand-written CUDA compact kernels against generic library baselines. The compact contract is:

```text
values[B, N]      int32 payload
keep_flags[B, N]  retained for API compatibility; ignored by the threshold-only contract
threshold         int32 scalar
predicate         values[b, i] >= threshold
output[B, N]      int32, compacted independently per batch row
out_counts[B]     int32, one compacted count per batch row
```

## Contents

```text
benchmark/                 benchmark driver
include/                   public CUDA/C++ interfaces and error checks
kernels/                   compact implementations v1..v5 and baselines
reference/                 CPU/Python reference helper
scripts/                   build, test, benchmark, sanitizer, and profiler helpers
tests/                     CUDA correctness test driver
reports/benchmark/         public benchmark CSV outputs
reports/trends/            public performance plots and plot source CSVs
reports/preview_images_latest/  selected public performance preview images
```

Analysis logs, raw profiler captures, sanitizer output, correctness logs, build products, and local debug files are intentionally excluded from the public repository.

## Requirements

Known development environment:

- NVIDIA GeForce RTX 5070
- CUDA Toolkit 12.8
- CMake 3.24 or newer
- C++17/CUDA17 compiler support
- Nsight Systems, Nsight Compute, and Compute Sanitizer are optional for local profiling and diagnostics

The default CUDA architecture in `CMakeLists.txt` is `120`, matching the RTX 5070 environment used for this lab.

## Build

From a fresh clone on a CUDA-capable Linux machine:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
```

The helper script is equivalent for the lab workflow:

```bash
./scripts/build.sh Release
```

For correctness/debug builds:

```bash
./scripts/build.sh Debug
```

## Verify

Run correctness tests for a selected mode:

```bash
./scripts/run_tests.sh baseline
./scripts/run_tests.sh library_cub_device_select
./scripts/run_tests.sh v1
./scripts/run_tests.sh v2
./scripts/run_tests.sh v3
./scripts/run_tests.sh v4
./scripts/run_tests.sh v5
```

Some implementations have restricted legal domains, such as vectorized kernels that require compatible `N` alignment. Illegal configurations are expected to be reported as skips by the harness, not correctness failures.

## Benchmark

Run a representative benchmark:

```bash
BATCH_SIZE=8 N=1048576 THRESHOLD=0 KEEP_PROB=0.5 ITERS=100 ./scripts/run_benchmark.sh v5
```

Compare against the generic CUB DeviceSelect baseline:

```bash
BATCH_SIZE=8 N=1048576 THRESHOLD=0 KEEP_PROB=0.5 ITERS=100 ./scripts/run_benchmark.sh library_cub_device_select
```

Benchmark CSV files are written under `reports/benchmark/`.

## Public Performance Artifacts

This repository keeps performance display artifacts only:

- benchmark CSV files under `reports/benchmark/`
- trend source CSV files and PNG/SVG charts under `reports/trends/`
- selected preview PNGs under `reports/preview_images_latest/`

Raw NCU/NSYS reports, sanitizer logs, correctness logs, debug output, roofline experiments, and analysis notes are not tracked.

## Profiling Helpers

Local profiling helpers are included for reproducibility:

```bash
./scripts/profile_nsys.sh v5
./scripts/profile_ncu.sh v5
./scripts/run_memcheck.sh v5
./scripts/run_racecheck.sh v5
./scripts/run_synccheck.sh v5
```

Profiler timings should be interpreted separately from ordinary Release benchmark timings. Release benchmark CSVs are the source for end-to-end latency comparisons; Nsight reports are diagnostic data for kernel-stage analysis.

## License

Apache License 2.0.
