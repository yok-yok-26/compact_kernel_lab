#include "compact_kernel.cuh"
#include "cuda_check.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

struct CaseData {
  std::string name;
  int batch_size = 0;
  int n = 0;
  int threshold = 0;
  std::vector<int> values;
  std::vector<std::uint8_t> flags;
};

static std::vector<std::vector<int>> cpu_compact_batched(const CaseData& tc) {
  std::vector<std::vector<int>> out(tc.batch_size);
  for (int b = 0; b < tc.batch_size; ++b) {
    for (int i = 0; i < tc.n; ++i) {
      int idx = b * tc.n + i;
      if (tc.values[idx] >= tc.threshold) out[b].push_back(tc.values[idx]);
    }
  }
  return out;
}

static CaseData make_case(const std::string& pattern, int batch_size, int n, int threshold, std::mt19937& rng) {
  CaseData tc;
  tc.name = pattern + "_b" + std::to_string(batch_size) + "_n" + std::to_string(n) + "_t" + std::to_string(threshold);
  tc.batch_size = batch_size;
  tc.n = n;
  tc.threshold = threshold;
  tc.values.resize(static_cast<std::size_t>(batch_size) * n);
  tc.flags.resize(static_cast<std::size_t>(batch_size) * n);
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < n; ++i) {
      int idx = b * n + i;
      int centered = (i % 97) - 48;
      tc.values[idx] = centered + (b % 3) - 1;
      if (pattern == "all_keep") {
        tc.flags[idx] = 1;
      } else if (pattern == "none_keep") {
        tc.flags[idx] = 0;
      } else if (pattern == "alternating") {
        tc.flags[idx] = static_cast<std::uint8_t>((i + b) & 1);
      } else if (pattern == "sparse_impulse") {
        tc.flags[idx] = static_cast<std::uint8_t>(n > 0 && i == (b * 17 + n / 2) % n);
        if (tc.flags[idx] != 0) tc.values[idx] = threshold + 3;
      } else if (pattern == "threshold_boundary") {
        tc.flags[idx] = 1;
        tc.values[idx] = threshold + ((i % 5) - 2);
      } else if (pattern == "batch_skew") {
        tc.flags[idx] = static_cast<std::uint8_t>((b % 3 == 0) ? 1 : ((b % 3 == 1) ? 0 : (i % 7 == 0)));
      } else {
        tc.flags[idx] = static_cast<std::uint8_t>((rng() % 100) < static_cast<unsigned>(13 + (b * 29) % 75));
        tc.values[idx] = static_cast<int>(rng() % 129) - 64;
      }
    }
  }
  return tc;
}

static std::vector<CaseData> make_cases() {
  std::vector<int> sizes = {0, 1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 4097, 8191, 8192, 8193, 131072, 1048576};
  std::vector<int> batches = {0, 1, 2, 3, 7, 8};
  std::vector<int> thresholds = {-17, 0, 23};
  std::vector<std::string> patterns = {"all_keep", "none_keep", "alternating", "sparse_impulse", "threshold_boundary", "batch_skew", "random"};
  std::vector<CaseData> cases;
  std::mt19937 rng(20260526);
  for (int threshold : thresholds) {
    for (int batch_size : batches) {
      for (int n : sizes) {
        for (const auto& pattern : patterns) {
          cases.push_back(make_case(pattern, batch_size, n, threshold, rng));
        }
      }
    }
  }
  return cases;
}

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

static bool requires_n_mod_4(const std::string& mode) {
  return mode == "v3" || mode == "v4" || mode == "v5";
}

int main(int argc, char** argv) {
  std::string mode = "baseline";
  std::string log_path = "reports/correctness/latest.log";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (arg == "--log" && i + 1 < argc) log_path = argv[++i];
  }

  std::ofstream log(log_path);
  if (!log) {
    std::fprintf(stderr, "failed to open log path: %s\n", log_path.c_str());
    return 2;
  }
  log << "mode=" << mode << "\n";
  log << "contract=int32 values[B,N], threshold int32, int32 output[B,N], int32 out_counts[B]\n";
  log << "predicate=values[b,i] >= threshold\n";
  if (requires_n_mod_4(mode)) {
    log << "mode_contract=" << mode << " supports n % 4 == 0 only; other n values are skipped\n";
  }

  int failures = 0;
  int skipped = 0;
  for (const auto& tc : make_cases()) {
    if (requires_n_mod_4(mode) && (tc.n % 4) != 0) {
      log << "SKIP case=" << tc.name << " threshold=" << tc.threshold << " batch_size=" << tc.batch_size << " n=" << tc.n
          << " reason=" << mode << "_requires_n_mod_4_eq_0\n";
      ++skipped;
      continue;
    }
    const int total = tc.batch_size * tc.n;
    const auto expected = cpu_compact_batched(tc);
    int *d_values = nullptr, *d_out = nullptr, *d_counts = nullptr;
    std::uint8_t* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_values, std::max(1, total) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags, std::max(1, total) * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max(1, total) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, std::max(1, tc.batch_size) * sizeof(int)));
    if (total > 0) {
      CUDA_CHECK(cudaMemcpy(d_values, tc.values.data(), total * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_flags, tc.flags.data(), total * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_out, 0x7f, std::max(1, total) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counts, 0x7f, std::max(1, tc.batch_size) * sizeof(int)));

    log << "RUN case=" << tc.name << " threshold=" << tc.threshold << " batch_size=" << tc.batch_size << " n=" << tc.n << "\n";
    log.flush();
    cudaError_t err = launch_mode(mode, d_values, d_flags, d_out, d_counts, tc.batch_size, tc.n, tc.threshold, 0);
    if (err != cudaSuccess) {
      log << "FAIL case=" << tc.name << " launch_error=" << cudaGetErrorString(err) << "\n";
      ++failures;
    } else {
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<int> got_counts(std::max(1, tc.batch_size), -1);
      if (tc.batch_size > 0) CUDA_CHECK(cudaMemcpy(got_counts.data(), d_counts, tc.batch_size * sizeof(int), cudaMemcpyDeviceToHost));
      std::vector<int> got_output(std::max(1, total));
      if (total > 0) CUDA_CHECK(cudaMemcpy(got_output.data(), d_out, total * sizeof(int), cudaMemcpyDeviceToHost));

      bool ok = true;
      int bad_batch = -1;
      int worst_idx = -1;
      int expected_count = 0;
      int got_count = 0;
      int expected_value = 0;
      int got_value = 0;
      for (int b = 0; b < tc.batch_size && ok; ++b) {
        expected_count = static_cast<int>(expected[b].size());
        got_count = got_counts[b];
        if (got_count != expected_count) {
          ok = false;
          bad_batch = b;
          worst_idx = -1;
          break;
        }
        for (int i = 0; i < expected_count; ++i) {
          int got = got_output[b * tc.n + i];
          if (got != expected[b][i]) {
            ok = false;
            bad_batch = b;
            worst_idx = i;
            expected_value = expected[b][i];
            got_value = got;
            break;
          }
        }
      }

      if (!ok) {
        log << "FAIL case=" << tc.name << " threshold=" << tc.threshold << " batch_size=" << tc.batch_size << " n=" << tc.n
            << " bad_batch=" << bad_batch << " expected_count=" << expected_count
            << " got_count=" << got_count << " worst_idx=" << worst_idx;
        if (worst_idx >= 0) log << " expected=" << expected_value << " got=" << got_value;
        log << "\n";
        ++failures;
      } else {
        int total_kept = 0;
        for (const auto& batch : expected) total_kept += static_cast<int>(batch.size());
        log << "PASS case=" << tc.name << " threshold=" << tc.threshold << " batch_size=" << tc.batch_size << " n=" << tc.n
            << " total_count=" << total_kept << "\n";
      }
    }
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_counts));
  }
  log << "summary failures=" << failures << " skipped=" << skipped << "\n";
  return failures == 0 ? 0 : 1;
}
