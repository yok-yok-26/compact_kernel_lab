#include "compact_kernel.cuh"
#include "cuda_check.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace {

static bool compact_v5_debug_n32_enabled(int n) {
  const char* env = std::getenv("COMPACT_V5_DEBUG_N32");
  return env != nullptr && env[0] != '\0' && n == 32;
}

static void compact_v5_dump_device_ints(const char* label,
                                        const int* ptr,
                                        std::size_t elems,
                                        std::size_t stride,
                                        int batch_size,
                                        cudaStream_t stream) {
  if (ptr == nullptr || elems == 0) return;
  std::vector<int> host(elems, 0);
  CUDA_CHECK(cudaMemcpyAsync(host.data(), ptr, elems * sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::fprintf(stderr, "[v5-debug] %s elems=%zu stride=%zu batch_size=%d\n", label, elems, stride, batch_size);
  for (int b = 0; b < batch_size; ++b) {
    std::size_t begin = static_cast<std::size_t>(b) * stride;
    std::size_t end = std::min(begin + stride, elems);
    std::fprintf(stderr, "[v5-debug] %s batch=%d:", label, b);
    for (std::size_t i = begin; i < end; ++i) {
      std::fprintf(stderr, " %d", host[i]);
    }
    std::fprintf(stderr, "\n");
  }
}

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

template <typename T>
__device__ __forceinline__ T reduce_sum_warp(unsigned mask, T val) {
  int lane_id = threadIdx.x & 31;

  for (int stride = 16; stride > 0; stride >>= 1) {
    val += __shfl_down_sync(mask, val, stride);
  }

  return val;
}





// TODO(silenceduke): 1个thread负责4个数据
// 
template<size_t SEMESIZE, size_t WARPSIZE, size_t TILESIZE = 16>
__global__ void compact_v5_kernel(int* values,
                                  int* out_counts,
                                  int batch_size,
                                  int n,
                                  int threshold) {
  if ((n & 3) != 0) return;
  int gtx = (blockDim.x * blockIdx.x + threadIdx.x) * TILESIZE;
  int ibatch = blockIdx.y;
  if (ibatch >= batch_size) return;
  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  
  int4 local_val[4]{
    {INT_MIN,INT_MIN,INT_MIN,INT_MIN},
    {INT_MIN,INT_MIN,INT_MIN,INT_MIN},
    {INT_MIN,INT_MIN,INT_MIN,INT_MIN},
    {INT_MIN,INT_MIN,INT_MIN,INT_MIN}
  };
  int local_cnt = 0;

  if (gtx < n) local_val[0] = *reinterpret_cast<int4*>(&values[gtx + ibatch * n]);
  if (gtx + 4 < n) local_val[1] = *reinterpret_cast<int4*>(&values[gtx + 4 + ibatch * n]);
  if (gtx + 8 < n) local_val[2] = *reinterpret_cast<int4*>(&values[gtx + 8 + ibatch * n]);
  if (gtx + 12 < n) local_val[3] = *reinterpret_cast<int4*>(&values[gtx + 12 + ibatch * n]);

  local_cnt = 
  (local_val[0].x >= threshold) + (local_val[0].y >= threshold) + 
  (local_val[0].z >= threshold) + (local_val[0].w >= threshold) + 
  (local_val[1].x >= threshold) + (local_val[1].y >= threshold) + 
  (local_val[1].z >= threshold) + (local_val[1].w >= threshold) + 
  (local_val[2].x >= threshold) + (local_val[2].y >= threshold) + 
  (local_val[2].z >= threshold) + (local_val[2].w >= threshold) + 
  (local_val[3].x >= threshold) + (local_val[3].y >= threshold) + 
  (local_val[3].z >= threshold) + (local_val[3].w >= threshold);

  
  unsigned all = __activemask();
  int warp_cnt = reduce_sum_warp(all, local_cnt);

  __shared__ int seme[SEMESIZE];
  if (lane_id == 0)
  {
    seme[warp_id] = warp_cnt;
  }
  __syncthreads();

  if (warp_id == 0)
  {
    if (lane_id < SEMESIZE)
    {
      warp_cnt = seme[lane_id];
    }
    else
    {
      warp_cnt = 0;
    }
    int block_cnt = prefix_include_warp<int>(all, warp_cnt);

    if (lane_id == (WARPSIZE - 1))
    {
      out_counts[gridDim.x * ibatch + blockIdx.x] = block_cnt;
    }
    
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
__global__ void simple_warp_prefix_exclude_expand(
  int* in, int* out, int* out_counts, int batch_size, int ACTIVENUM
){

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

    int acc_val_tmp = 0, acc_val = 0;
    if (warp_id == 0)
    {
      acc_val = seme[lane_id];
      __syncwarp();

      acc_val_tmp = prefix_include_warp<int>(mask, acc_val);
      acc_val = acc_val_tmp - acc_val;
      seme[lane_id] = acc_val;

      if (lane_id == (WARPSIZE - 1))
      {
        out_counts[ibatch] = acc_val_tmp;
      }
      
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
__global__ void simple_warp_prefix_exclude_expand_v2(
  int* in, int* out, int* out_counts, int batch_size, int ACTIVENUM
){

  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int gid;
  int tx = threadIdx.x;
  int val, old_acc_val = 0, old_acc_val_tmp = 0;
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

    int acc_val = 0, acc_val_tmp = 0;
    if (warp_id == 0)
    {
      acc_val = seme[lane_id];
      __syncwarp();

      acc_val_tmp = prefix_include_warp<int>(mask, acc_val);
      acc_val = acc_val_tmp - acc_val;
      seme[lane_id] = acc_val;

      if (lane_id == (WARPSIZE - 1))
      {
        old_acc_val_tmp += acc_val_tmp;
      }
    }
    __syncthreads();

    if (gid < ACTIVENUM)
    {
        out[gid + ibatch * ACTIVENUM] = seme[warp_id] + warp_val - val + old_acc_val;
    }

    old_acc_val += seme[WARPSIZE - 1];
    __syncthreads();
  }

  if ((warp_id == 0) && (lane_id == (WARPSIZE - 1)))
  {
    out_counts[ibatch] = old_acc_val;
  }
  
}





template<size_t SEMESIZE, size_t WARPSIZE, size_t TILESIZE = 16>
__global__ void integrate_block(
    int* input, int* offset_counts, int* output, 
    int n, int batch_size, int threshold
){
  if ((n & 3) != 0) return;
  int gtx = (blockDim.x * blockIdx.x + threadIdx.x) * TILESIZE;
  int ibatch = blockIdx.y;
  if (ibatch >= batch_size) return;
  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int local_out_count = offset_counts[blockIdx.x + ibatch * gridDim.x];
  __shared__ int seme_compact[SEMESIZE * WARPSIZE * TILESIZE];

  // 初始化方式必须关注!!!!!
  int4 local_val[4]{
    {INT_MIN, INT_MIN, INT_MIN, INT_MIN},
    {INT_MIN, INT_MIN, INT_MIN, INT_MIN},
    {INT_MIN, INT_MIN, INT_MIN, INT_MIN},
    {INT_MIN, INT_MIN, INT_MIN, INT_MIN}
  };
  int local_cnt = 0;
  if (gtx < n) local_val[0] = *reinterpret_cast<int4*>(&input[gtx + ibatch * n]);
  if (gtx + 4 < n) local_val[1] = *reinterpret_cast<int4*>(&input[gtx + 4 + ibatch * n]);
  if (gtx + 8 < n) local_val[2] = *reinterpret_cast<int4*>(&input[gtx + 8 + ibatch * n]);
  if (gtx + 12 < n) local_val[3] = *reinterpret_cast<int4*>(&input[gtx + 12 + ibatch * n]);

  local_cnt = 
  (local_val[0].x >= threshold) + (local_val[0].y >= threshold) + 
  (local_val[0].z >= threshold) + (local_val[0].w >= threshold) + 
  (local_val[1].x >= threshold) + (local_val[1].y >= threshold) + 
  (local_val[1].z >= threshold) + (local_val[1].w >= threshold) + 
  (local_val[2].x >= threshold) + (local_val[2].y >= threshold) + 
  (local_val[2].z >= threshold) + (local_val[2].w >= threshold) + 
  (local_val[3].x >= threshold) + (local_val[3].y >= threshold) + 
  (local_val[3].z >= threshold) + (local_val[3].w >= threshold);
  
  
  unsigned mask = __activemask();
  int warp_flag = prefix_include_warp<int>(mask, local_cnt);


  __shared__ int seme[SEMESIZE];
  if (lane_id == (WARPSIZE - 1))
  {
    seme[warp_id] = warp_flag;
  }
  __syncthreads();
  
  int acc_flag = 0, acc_flag_old = 0;
  __shared__ int acc_flag_old_shared;
  if (warp_id == 0)
  {
    if (lane_id < SEMESIZE)
    {
        acc_flag = seme[lane_id];
    }
    __syncwarp();

    acc_flag_old = prefix_include_warp<int>(mask, acc_flag);
    acc_flag = acc_flag_old - acc_flag;
    if (lane_id == (SEMESIZE - 1)){
        acc_flag_old_shared = acc_flag_old;
    }

    if (lane_id < SEMESIZE)
    {
        seme[lane_id] = acc_flag;
    }
  }
  __syncthreads();

  
  int local_idx = 0;
  if (local_val[0].x >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[0].x;
    local_idx++;
  }
  if (local_val[0].y >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[0].y;
    local_idx++;
  }
  if (local_val[0].z >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[0].z;
    local_idx++;
  }
  if (local_val[0].w >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[0].w;
    local_idx++;
  }
  if (local_val[1].x >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[1].x;
    local_idx++;
  }
  if (local_val[1].y >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[1].y;
    local_idx++;
  }
  if (local_val[1].z >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[1].z;
    local_idx++;
  }
  if (local_val[1].w >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[1].w;
    local_idx++;
  }
  if (local_val[2].x >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[2].x;
    local_idx++;
  }
  if (local_val[2].y >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[2].y;
    local_idx++;
  }
  if (local_val[2].z >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[2].z;
    local_idx++;
  }
  if (local_val[2].w >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[2].w;
    local_idx++;
  }
  if (local_val[3].x >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[3].x;
    local_idx++;
  }
  if (local_val[3].y >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[3].y;
    local_idx++;
  }
  if (local_val[3].z >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[3].z;
    local_idx++;
  }
  if (local_val[3].w >= threshold) {
    seme_compact[seme[warp_id] + local_idx + warp_flag - local_cnt] = local_val[3].w;
    local_idx++;
  }
  __syncthreads();



  int block_cnt = acc_flag_old_shared;
  for (int idx = threadIdx.x; idx < block_cnt; idx += blockDim.x)
  {
    if (idx < block_cnt)
    {
        output[
            (idx + local_out_count) + 
            n * ibatch
        ] = seme_compact[idx];
    }
  }


}


}  // namespace

static cudaError_t compact_v5_workspace_launch(const int* values,
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
  // The placeholder intentionally does no work, so v2 correctness should fail until implemented.
  constexpr int block = 256;
  constexpr int tilesize = 16;
  dim3 grid(((n + tilesize - 1) / tilesize + block - 1) / block, batch_size);
  const bool debug_n32 = compact_v5_debug_n32_enabled(n);
  if (debug_n32) {
    std::fprintf(stderr,
                 "[v5-debug] begin batch_size=%d n=%d threshold=%d block=%d tilesize=%d grid=(%u,%u,%u)\n",
                 batch_size,
                 n,
                 threshold,
                 block,
                 tilesize,
                 grid.x,
                 grid.y,
                 grid.z);
    compact_v5_dump_device_ints("input_values",
                                values,
                                static_cast<std::size_t>(batch_size) * n,
                                n,
                                batch_size,
                                stream);
  }

  compact_v5_kernel<block / 32, 32><<<grid, block, 0, stream>>>(
    const_cast<int*>(values), tmp_counts, batch_size, n, threshold
  );
  if (debug_n32) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    compact_v5_dump_device_ints("after_stage1_tmp_counts",
                                tmp_counts,
                                static_cast<std::size_t>(batch_size) * grid.x,
                                grid.x,
                                batch_size,
                                stream);
  }

  if (grid.x <= 1024)
  {
    simple_warp_prefix_exclude_expand<32><<<batch_size, ((grid.x + 31) / 32 * 32), 0, stream>>>(
      tmp_counts, offset_counts, out_counts, batch_size, grid.x
    );
  }
  else
  {
    simple_warp_prefix_exclude_expand_v2<32><<<batch_size, 256>>>(
      tmp_counts, offset_counts, out_counts, batch_size, grid.x
    );
  }

  if (debug_n32) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    compact_v5_dump_device_ints("after_stage2_offset_counts",
                                offset_counts,
                                static_cast<std::size_t>(batch_size) * grid.x,
                                grid.x,
                                batch_size,
                                stream);
    compact_v5_dump_device_ints("after_stage2_out_counts",
                                out_counts,
                                static_cast<std::size_t>(batch_size),
                                1,
                                batch_size,
                                stream);
    CUDA_CHECK(cudaMemsetAsync(tmpput, 0x7f, static_cast<std::size_t>(batch_size) * block * sizeof(int), stream));
    integrate_block<block / 32, 32><<<grid, block, 0, stream>>>(
      const_cast<int*>(values), offset_counts, tmpput, n, batch_size, threshold
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    compact_v5_dump_device_ints("after_stage3_debug_output_first_256_per_batch",
                                tmpput,
                                static_cast<std::size_t>(batch_size) * block,
                                block,
                                batch_size,
                                stream);
  } else {
    integrate_block<block / 32, 32><<<grid, block, 0, stream>>>(
      const_cast<int*>(values), offset_counts, output, n, batch_size, threshold
    );
  }

  CUDA_KERNEL_CHECK();
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;
}


cudaError_t compact_v5_launch(const int* values,
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

  cudaError_t err = compact_v5_workspace_launch(values,
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
