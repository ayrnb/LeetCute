#pragma once
#include <cuda_runtime.h>

// 函数指针类型定义（可选）
template <unsigned int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void gemm_baseline_gpu(float*, float*, float*, int, int, int);