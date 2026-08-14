#include "compact_kernel.cuh"
#include "cuda_check.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

static cudaError_t launch_mode(const std::string& mode,
                               const int* d_values,
                               const std::uint8_t* d_flags,
                               int* d_out,
                               int* d_counts,
                               int batch_size,
                               int n,
                               int threshold,
                               cudaStream_t stream) {
  if (mode == "baseline" || mode == "simple_cuda") return compact_baseline_launch(d_values, d_flags, d_out, d_counts, batch_size, n, threshold, stream);
  if (mode == "library" || mode == "library_cub_thrust") return compact_library_cub_thrust_launch(d_values, d_flags, d_out, d_counts, batch_size, n, threshold, stream);
  if (mode == "library_cub_device_select" || mode == "library_cub") return compact_library_cub_device_select_launch(d_values, d_flags, d_out, d_counts, batch_size, n, threshold, stream);
  if (mode == "v1" || mode == "user") return compact_v1_launch(d_values, d_flags, d_out, d_counts, batch_size, n, threshold, stream);
  if (mode == "v2") return compact_v2_launch(d_values, d_flags, d_out, d_counts, batch_size, n, threshold, stream);
  if (mode == "v3") return compact_v3_launch(d_values, d_flags, d_out, d_counts, batch_size, n, threshold, stream);
  if (mode == "v4") return compact_v4_launch(d_values, d_flags, d_out, d_counts, batch_size, n, threshold, stream);
  if (mode == "v5") return compact_v5_launch(d_values, d_flags, d_out, d_counts, batch_size, n, threshold, stream);
  return cudaErrorInvalidValue;
}

int main(int argc, char** argv) {
  std::string mode = "baseline";
  std::string csv = "reports/benchmark/latest.csv";
  int batch_size = 8;
  int n = 1 << 20;
  int threshold = 0;
  int iters = 100;
  int warmup = 10;
  double keep_prob = 0.5;
  bool single_launch = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
    else if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
    else if (arg == "--threshold" && i + 1 < argc) threshold = std::stoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
    else if (arg == "--keep-prob" && i + 1 < argc) keep_prob = std::stod(argv[++i]);
    else if (arg == "--csv" && i + 1 < argc) csv = argv[++i];
    else if (arg == "--single-launch") single_launch = true;
  }
  if (batch_size < 0 || n < 0) {
    std::fprintf(stderr, "batch_size and n must be non-negative\n");
    return 2;
  }
  if (single_launch) {
    warmup = 0;
    iters = 1;
  }

  const int total = batch_size * n;
  std::mt19937 rng(20260526);
  std::uniform_int_distribution<int> value_dist(threshold - 64, threshold + 64);
  std::vector<int> values(std::max(1, total));
  std::vector<std::uint8_t> flags(std::max(1, total));
  std::vector<int> expected_counts(std::max(1, batch_size), 0);
  for (int b = 0; b < batch_size; ++b) {
    std::bernoulli_distribution keep_dist(std::min(0.99, std::max(0.01, keep_prob + 0.05 * ((b % 5) - 2))));
    for (int i = 0; i < n; ++i) {
      int idx = b * n + i;
      values[idx] = value_dist(rng);
      flags[idx] = static_cast<std::uint8_t>(keep_dist(rng));
      expected_counts[b] += values[idx] >= threshold;
    }
  }

  int *d_values = nullptr, *d_out = nullptr, *d_counts = nullptr;
  std::uint8_t* d_flags = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, std::max(1, total) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_flags, std::max(1, total) * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_out, std::max(1, total) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counts, std::max(1, batch_size) * sizeof(int)));
  if (total > 0) {
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flags, flags.data(), total * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
  }

  for (int i = 0; i < warmup; ++i) CUDA_CHECK(launch_mode(mode, d_values, d_flags, d_out, d_counts, batch_size, n, threshold, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) CUDA_CHECK(launch_mode(mode, d_values, d_flags, d_out, d_counts, batch_size, n, threshold, 0));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));

  std::vector<int> counts(std::max(1, batch_size), 0);
  if (batch_size > 0) CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
  long long total_count = 0;
  for (int c : counts) total_count += c;

  double ms = ms_total / std::max(1, iters);
  double read_gb = static_cast<double>(total) * (sizeof(int) + sizeof(std::uint8_t)) / 1e9;
  double write_gb = static_cast<double>(total_count) * sizeof(int) / 1e9;
  double count_gb = static_cast<double>(batch_size) * sizeof(int) / 1e9;
  double logical_gb = read_gb + write_gb + count_gb;
  double gbps = logical_gb / (ms / 1000.0);

  std::ofstream out(csv);
  out << "mode,batch_size,n,total_elements,threshold,keep_prob,total_count,iters,avg_ms,logical_gb,logical_gbps,single_launch\n";
  out << mode << "," << batch_size << "," << n << "," << total << "," << threshold << "," << keep_prob << "," << total_count
      << "," << iters << "," << ms << "," << logical_gb << "," << gbps << "," << (single_launch ? 1 : 0) << "\n";
  std::printf("mode=%s batch_size=%d n=%d total=%d threshold=%d keep_prob=%.3f total_count=%lld avg_ms=%.6f logical_gbps=%.3f csv=%s\n",
              mode.c_str(), batch_size, n, total, threshold, keep_prob, total_count, ms, gbps, csv.c_str());

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_flags));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_counts));
  return 0;
}
