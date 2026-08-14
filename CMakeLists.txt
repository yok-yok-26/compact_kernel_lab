cmake_minimum_required(VERSION 3.24)
project(compact_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures")

option(DEBUG_CUDA_SYNC "Synchronize after kernel launches in debug builds" OFF)

add_library(compact_kernels STATIC
  kernels/compact_kernel.cu
  kernels/compact_kernel_v1.cu
  kernels/compact_kernel_v2.cu
  kernels/compact_kernel_v3.cu
  kernels/compact_kernel_v4.cu
  kernels/compact_kernel_v5.cu
)
target_include_directories(compact_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(compact_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
)
target_compile_definitions(compact_kernels PUBLIC
  $<$<BOOL:${DEBUG_CUDA_SYNC}>:DEBUG_CUDA_SYNC>
)

add_executable(test_compact tests/test_compact.cu)
target_link_libraries(test_compact PRIVATE compact_kernels)
target_include_directories(test_compact PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(benchmark_compact benchmark/benchmark_compact.cu)
target_link_libraries(benchmark_compact PRIVATE compact_kernels)
target_include_directories(benchmark_compact PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
