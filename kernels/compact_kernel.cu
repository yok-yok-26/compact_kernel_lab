#include "compact_kernel.cuh"
#include "cuda_check.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace {

struct KeepThresholdPredicate {
  int threshold;

  __host__ __device__ bool operator()(int value) const {
    return value >= threshold;
  }
};

__global__ void compact_simple_cuda_kernel(const int* values,
                                           int* output,
                                           int* out_counts,
                                           int batch_size,
                                           int n,
                                           int threshold) {
  int b = blockIdx.x;
  if (b >= batch_size) return;

  const int* batch_values = values + static_cast<std::size_t>(b) * n;
  int* batch_output = output + static_cast<std::size_t>(b) * n;
  int count = 0;
  for (int i = 0; i < n; ++i) {
    int value = batch_values[i];
    if (value >= threshold) {
      batch_output[count++] = value;
    }
  }
  out_counts[b] = count;
}

__global__ void store_count_kernel(int* out_counts, int batch_idx, int count) {
  out_counts[batch_idx] = count;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void compact_cub_blockscan_small_kernel(const int* values,
                                                   int* output,
                                                   int* out_counts,
                                                   int batch_size,
                                                   int n,
                                                   int threshold) {
  int b = blockIdx.x;
  if (b >= batch_size) return;

  using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int* batch_values = values + static_cast<std::size_t>(b) * n;
  int* batch_output = output + static_cast<std::size_t>(b) * n;
  int local_values[ITEMS_PER_THREAD];
  int local_keep[ITEMS_PER_THREAD];
  int local_count = 0;

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int idx = threadIdx.x * ITEMS_PER_THREAD + item;
    int value = 0;
    int keep = 0;
    if (idx < n) {
      value = batch_values[idx];
      keep = value >= threshold;
    }
    local_values[item] = value;
    local_keep[item] = keep;
    local_count += keep;
  }

  int thread_offset = 0;
  BlockScan(temp_storage).ExclusiveSum(local_count, thread_offset);

  int out_idx = thread_offset;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    if (local_keep[item]) {
      batch_output[out_idx++] = local_values[item];
    }
  }

  if (threadIdx.x == BLOCK_THREADS - 1) {
    out_counts[b] = thread_offset + local_count;
  }
}

}  // namespace

cudaError_t compact_baseline_launch(const int* values,
                                    const std::uint8_t* keep_flags,
                                    int* output,
                                    int* out_counts,
                                    int batch_size,
                                    int n,
                                    int threshold,
                                    cudaStream_t stream) {
  (void)keep_flags;
  if (batch_size < 0 || n < 0) {
    return cudaErrorInvalidValue;
  }
  if (batch_size == 0) {
    return cudaSuccess;
  }

  CUDA_CHECK(cudaMemsetAsync(out_counts, 0, static_cast<std::size_t>(batch_size) * sizeof(int), stream));
  if (n == 0) {
    return cudaSuccess;
  }

  compact_simple_cuda_kernel<<<batch_size, 1, 0, stream>>>(values, output, out_counts, batch_size, n, threshold);
  CUDA_KERNEL_CHECK();
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;
}

cudaError_t compact_library_cub_thrust_launch(const int* values,
                                              const std::uint8_t* keep_flags,
                                              int* output,
                                              int* out_counts,
                                              int batch_size,
                                              int n,
                                              int threshold,
                                              cudaStream_t stream) {
  (void)keep_flags;
  if (batch_size < 0 || n < 0) {
    return cudaErrorInvalidValue;
  }
  if (batch_size == 0) {
    return cudaSuccess;
  }

  CUDA_CHECK(cudaMemsetAsync(out_counts, 0, static_cast<std::size_t>(batch_size) * sizeof(int), stream));
  if (n == 0) {
    return cudaSuccess;
  }

  auto policy = thrust::cuda::par.on(stream);
  for (int b = 0; b < batch_size; ++b) {
    const int* batch_values = values + static_cast<std::size_t>(b) * n;
    int* batch_output = output + static_cast<std::size_t>(b) * n;
    int* end = thrust::copy_if(policy, batch_values, batch_values + n, batch_output, KeepThresholdPredicate{threshold});
    int kept = static_cast<int>(end - batch_output);
    store_count_kernel<<<1, 1, 0, stream>>>(out_counts, b, kept);
    CUDA_KERNEL_CHECK();
  }
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;
}

cudaError_t compact_library_cub_device_select_launch(const int* values,
                                                     const std::uint8_t* keep_flags,
                                                     int* output,
                                                     int* out_counts,
                                                     int batch_size,
                                                     int n,
                                                     int threshold,
                                                     cudaStream_t stream) {
  (void)keep_flags;
  if (batch_size < 0 || n < 0) {
    return cudaErrorInvalidValue;
  }
  if (batch_size == 0) {
    return cudaSuccess;
  }

  CUDA_CHECK(cudaMemsetAsync(out_counts, 0, static_cast<std::size_t>(batch_size) * sizeof(int), stream));
  if (n == 0) {
    return cudaSuccess;
  }

  static int cached_n = -1;
  static std::size_t required_bytes = 0;
  if (cached_n != n) {
    CUDA_CHECK(cub::DeviceSelect::If(nullptr,
                                     required_bytes,
                                     values,
                                     output,
                                     out_counts,
                                     n,
                                     KeepThresholdPredicate{threshold},
                                     stream));
    cached_n = n;
  }

  static std::vector<void*> temp_storages;
  static std::vector<std::size_t> temp_storage_bytes;

  const int old_size = static_cast<int>(temp_storages.size());
  if (old_size < batch_size) {
    temp_storages.resize(batch_size, nullptr);
    temp_storage_bytes.resize(batch_size, 0);
  }

  for (int b = 0; b < batch_size; ++b) {
    if (required_bytes > temp_storage_bytes[b]) {
      if (temp_storages[b] != nullptr) {
        CUDA_CHECK(cudaFree(temp_storages[b]));
      }
      CUDA_CHECK(cudaMalloc(&temp_storages[b], required_bytes));
      temp_storage_bytes[b] = required_bytes;
    }
  }

  for (int b = 0; b < batch_size; ++b) {
    const int* batch_values = values + static_cast<std::size_t>(b) * n;
    int* batch_output = output + static_cast<std::size_t>(b) * n;
    int* batch_count = out_counts + b;
    CUDA_CHECK(cub::DeviceSelect::If(temp_storages[b],
                                     temp_storage_bytes[b],
                                     batch_values,
                                     batch_output,
                                     batch_count,
                                     n,
                                     KeepThresholdPredicate{threshold},
                                     stream));
  }
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;
}
