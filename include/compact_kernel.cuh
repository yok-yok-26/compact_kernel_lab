#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Batched compact contract:
// values: int32 contiguous row-major [batch_size, n]
// keep_flags: retained for API compatibility; ignored by threshold-only compact
// threshold: keep elements with values[b, i] >= threshold
// output: int32 contiguous row-major [batch_size, n]; each batch writes compacted data to output[b, 0:out_counts[b])
// out_counts: int32 contiguous [batch_size], one count per batch
cudaError_t compact_baseline_launch(const int* values,
                                    const std::uint8_t* keep_flags,
                                    int* output,
                                    int* out_counts,
                                    int batch_size,
                                    int n,
                                    int threshold,
                                    cudaStream_t stream);

// Generic library baseline: Thrust/CUB composition.
cudaError_t compact_library_cub_thrust_launch(const int* values,
                                              const std::uint8_t* keep_flags,
                                              int* output,
                                              int* out_counts,
                                              int batch_size,
                                              int n,
                                              int threshold,
                                              cudaStream_t stream);

// Generic library baseline: CUB DeviceSelect::If with reusable temporary storage.
cudaError_t compact_library_cub_device_select_launch(const int* values,
                                                     const std::uint8_t* keep_flags,
                                                     int* output,
                                                     int* out_counts,
                                                     int batch_size,
                                                     int n,
                                                     int threshold,
                                                     cudaStream_t stream);

// User-owned v1 entrypoint. Implement this in kernels/compact_kernel_v1.cu.
cudaError_t compact_v1_launch(const int* values,
                               const std::uint8_t* keep_flags,
                               int* output,
                               int* out_counts,
                               int batch_size,
                               int n,
                               int threshold,
                               cudaStream_t stream);

// User-owned v2 entrypoint. Implement this in kernels/compact_kernel_v2.cu.
cudaError_t compact_v2_launch(const int* values,
                              const std::uint8_t* keep_flags,
                              int* output,
                              int* out_counts,
                              int batch_size,
                              int n,
                              int threshold,
                              cudaStream_t stream);

// User-owned v3 entrypoint. Implement this in kernels/compact_kernel_v3.cu.
cudaError_t compact_v3_launch(const int* values,
                              const std::uint8_t* keep_flags,
                              int* output,
                              int* out_counts,
                              int batch_size,
                              int n,
                              int threshold,
                              cudaStream_t stream);

// User-owned v4 entrypoint. Implement this in kernels/compact_kernel_v4.cu.
cudaError_t compact_v4_launch(const int* values,
                              const std::uint8_t* keep_flags,
                              int* output,
                              int* out_counts,
                              int batch_size,
                              int n,
                              int threshold,
                              cudaStream_t stream);

// User-owned v5 entrypoint. Implement this in kernels/compact_kernel_v5.cu.
cudaError_t compact_v5_launch(const int* values,
                              const std::uint8_t* keep_flags,
                              int* output,
                              int* out_counts,
                              int batch_size,
                              int n,
                              int threshold,
                              cudaStream_t stream);
