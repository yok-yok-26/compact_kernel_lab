#include "compact_kernel.cuh"
#include "cuda_check.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace {

template <typename T>
__device__ __forceinline__ T prefix_include_warp(unsigned mask, T val) {
  int lane_id = threadIdx.x & 31;
  T other;

  for (int stride = 1; stride < 32; stride <<= 1) {
    other = __shfl_up_sync(mask, val, stride);
    if (lane_id >= stride) {
      val += other;
    }
  }

  return val;
}

// TODO(silenceduke): Implement your batched compact optimization here.
// Keep condition: keep_flags[b, i] != 0 && values[b, i] >= threshold.
// Contract: one independent stable compact per batch row.
template<size_t SEMESIZE, size_t WARPSIZE>
__global__ void compact_v1_kernel(const int* values,
                                  int* output,
                                  int* out_counts,
                                  int batch_size,
                                  int n,
                                  int threshold) {
  int gtx = blockDim.x * blockIdx.x + threadIdx.x;
  int ibatch = blockIdx.y;
  if (ibatch >= batch_size) return;
  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  
  int val = 0;
  int flag = 0;
  if (gtx < n)
  {
    val = values[gtx + ibatch * n];
    if (val >= threshold)
    {
        flag = 1;
    }
  }
  
  unsigned mask = __activemask();
  int warp_flag = prefix_include_warp<int>(mask, flag);

  __shared__ int seme[SEMESIZE];
  if (lane_id == (WARPSIZE - 1))
  {
    seme[warp_id] = warp_flag;
  }
  __syncthreads();
  
  int acc_flag = 0;
  if (warp_id == 0)
  {
    if (lane_id < SEMESIZE) // 8192 256 --> 32
    {
        acc_flag = seme[lane_id];
    }
    __syncwarp();

    acc_flag = prefix_include_warp<int>(mask, acc_flag) - acc_flag;

    if (lane_id == (SEMESIZE - 1))
    {
        out_counts[blockIdx.x + gridDim.x * ibatch] = seme[lane_id] + acc_flag;
    }

    if (lane_id < SEMESIZE)
    {
        seme[lane_id] = acc_flag;
    }
  }
  __syncthreads();
  
  
  if (flag)
  {
    output[
        (seme[warp_id] + warp_flag - flag) + (blockIdx.x + gridDim.x * ibatch) * blockDim.x
    ] = val;
  }
  
}



__global__ void simple_warp_prefix_exclude(int* in, int* out, int batch_size, int ACTIVENUM){

    int lane_id = threadIdx.x & 31;
    int ibatch = blockIdx.x;
    if (ibatch >= batch_size) return;

    int val = 0;
    if (lane_id < ACTIVENUM)
    {
        val = in[lane_id + ibatch * ACTIVENUM];
    }

    unsigned mask = __activemask();

    val = prefix_include_warp<int>(mask, val) - val;
    
    if (lane_id < ACTIVENUM)
    {
        out[lane_id + ibatch * ACTIVENUM] = val;
    }
}


/// @brief 汇总结果 v1
/// @param in 要求长度<=1024
/// @return 
template <size_t WARPSIZE>
__global__ void simple_warp_prefix_exclude_expand(int* in, int* out, int batch_size, int ACTIVENUM){

    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int gid = lane_id + 32 * warp_id;
    int ibatch = blockIdx.x;
    if (ibatch >= batch_size) return;

    int val = 0;
    if ((gid) < ACTIVENUM)
    {
        val = in[gid + ibatch * ACTIVENUM];
    }

    unsigned mask = __activemask();
    __shared__ int seme[WARPSIZE];
    int warp_val = prefix_include_warp<int>(mask, val);
    
    if ((warp_id == 0) && (lane_id >= (blockDim.x / WARPSIZE)))
    {
      seme[lane_id] = 0;
    }
    
    if (lane_id == (WARPSIZE - 1))
    {
      seme[warp_id] = warp_val;
    }
    __syncthreads();

    int acc_val = 0;
    if (warp_id == 0)
    {
      acc_val = seme[lane_id];
      __syncwarp();
      acc_val = prefix_include_warp<int>(mask, acc_val) - acc_val;
      seme[lane_id] = acc_val;
    }

    __syncthreads();
    
    if (gid < ACTIVENUM)
    {
        out[gid + ibatch * ACTIVENUM] = seme[warp_id] + warp_val - val;
    }
}



/// @brief 汇总结果 v2
/// @param in 要求长度 没有
/// @return 
template <size_t WARPSIZE>
__global__ void simple_warp_prefix_exclude_expand_v2(int* in, int* out, int batch_size, int ACTIVENUM){

  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int gid;
  int tx = threadIdx.x;
  int val, old_acc_val = 0;
  int ibatch = blockIdx.x;
  if (ibatch >= batch_size) return;
  __shared__ int seme[WARPSIZE];
  unsigned mask = __activemask();



  for (size_t iloop = 0; iloop < ACTIVENUM; iloop+=blockDim.x)
  {
    gid = iloop + tx;
    if ((gid) < ACTIVENUM)
    {
        val = in[gid + ibatch * ACTIVENUM];
    }
    else
    {
      val = 0;
    }
    
    int warp_val = prefix_include_warp<int>(mask, val);
    if ((warp_id == 0) && (lane_id >= (blockDim.x / WARPSIZE)))
    {
      seme[lane_id] = 0;
    }
    
    if (lane_id == (WARPSIZE - 1))
    {
      seme[warp_id] = warp_val;
    }
    __syncthreads();

    int acc_val = 0;
    if (warp_id == 0)
    {
      acc_val = seme[lane_id];
      __syncwarp();

      acc_val = prefix_include_warp<int>(mask, acc_val) - acc_val;
      seme[lane_id] = acc_val;
    }
    __syncthreads();

    if (gid < ACTIVENUM)
    {
        out[gid + ibatch * ACTIVENUM] = seme[warp_id] + warp_val - val + old_acc_val;
    }

    old_acc_val += seme[WARPSIZE - 1];
    __syncthreads();
  }
  
}




__global__ void integrate_block(
    int* tmp_counts, int* tmpput, int* offset_counts, int* out_counts, int* output, int n, int batch_size
){
  int tx = threadIdx.x, bx = blockIdx.x;
  int ibatch = blockIdx.y;
  if (ibatch >= batch_size) return;
  int ACTIVENUM = gridDim.x;
  
  if (threadIdx.x < tmp_counts[bx + gridDim.x * ibatch]){
    output[(offset_counts[bx + ibatch * gridDim.x] + tx) + ibatch * n] = 
      tmpput[(tx + blockDim.x * bx) + ibatch * blockDim.x * gridDim.x];
  }

  if (tx == 0)
  {
    out_counts[ibatch] = 
        offset_counts[(gridDim.x - 1) + ibatch * gridDim.x] + tmp_counts[(gridDim.x - 1) + ibatch * gridDim.x];
  }
  

}


}  // namespace

static cudaError_t compact_v1_workspace_launch(const int* values,
                                                const std::uint8_t* keep_flags,
                                                int* tmpput,
                                                int* tmp_counts,
                                                int* offset_counts,
                                                int* output,
                                                int* out_counts,
                                                int batch_size,
                                                int n,
                                                int threshold,
                                                cudaStream_t stream) {
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

  // TODO(silenceduke): Choose the grid/block/smem strategy for batched compact.
  // The placeholder intentionally does no work, so v1 correctness should fail until implemented.
  constexpr int block = 256;
  dim3 grid((n + block - 1) / block, batch_size);
  compact_v1_kernel<block / 32, 32><<<grid, block, 0, stream>>>(
    values, tmpput, tmp_counts, batch_size, n, threshold
  );

  if (grid.x <= 1024)
  {
    simple_warp_prefix_exclude_expand<32><<<batch_size, ((grid.x + 31) / 32 * 32), 0, stream>>>(
      tmp_counts, offset_counts, batch_size, grid.x
    );
  }
  else
  {
    simple_warp_prefix_exclude_expand_v2<32><<<batch_size, 256>>>(
      tmp_counts, offset_counts, batch_size, grid.x
    );
  }

  integrate_block<<<grid, block, 0, stream>>>(
    tmp_counts, tmpput, offset_counts, out_counts, output, n, batch_size
  );


  CUDA_KERNEL_CHECK();
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;
}


cudaError_t compact_v1_launch(const int* values,
                               const std::uint8_t* keep_flags,
                               int* output,
                               int* out_counts,
                               int batch_size,
                               int n,
                               int threshold,
                               cudaStream_t stream) {
  if (batch_size < 0 || n < 0) {
    return cudaErrorInvalidValue;
  }
  if (batch_size == 0) {
    return cudaSuccess;
  }

  constexpr int block = 256;
  int grid_x = (n + block - 1) / block;
  std::size_t tmp_output_elems = static_cast<std::size_t>(batch_size) * std::max(1, grid_x) * block;
  std::size_t count_stride = static_cast<std::size_t>(std::max(grid_x, block / 32));
  std::size_t tmp_count_elems = static_cast<std::size_t>(batch_size) * count_stride;

  int* tmpput = nullptr;
  int* tmp_counts = nullptr;
  int* offset_counts = nullptr;
  CUDA_CHECK(cudaMallocAsync(&tmpput, std::max<std::size_t>(1, tmp_output_elems) * sizeof(int), stream));
  CUDA_CHECK(cudaMallocAsync(&tmp_counts, std::max<std::size_t>(1, tmp_count_elems) * sizeof(int), stream));
  CUDA_CHECK(cudaMallocAsync(&offset_counts, std::max<std::size_t>(1, tmp_count_elems) * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(tmp_counts, 0, std::max<std::size_t>(1, tmp_count_elems) * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(offset_counts, 0, std::max<std::size_t>(1, tmp_count_elems) * sizeof(int), stream));

  cudaError_t err = compact_v1_workspace_launch(values,
                                                keep_flags,
                                                tmpput,
                                                tmp_counts,
                                                offset_counts,
                                                output,
                                                out_counts,
                                                batch_size,
                                                n,
                                                threshold,
                                                stream);
  CUDA_CHECK(cudaFreeAsync(offset_counts, stream));
  CUDA_CHECK(cudaFreeAsync(tmp_counts, stream));
  CUDA_CHECK(cudaFreeAsync(tmpput, stream));
  return err;
}
