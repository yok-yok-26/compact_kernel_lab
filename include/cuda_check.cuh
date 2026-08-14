#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
      std::fprintf(stderr, "CUDA_CHECK failed at %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_err), static_cast<int>(_err)); \
      std::exit(EXIT_FAILURE); \
    } \
  } while (0)

#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
