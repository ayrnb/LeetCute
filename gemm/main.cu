
#include <stdio.h>
#include <sys/time.h> 
#include <cuda.h>
#include "my_gemm_baseline.cu"
#define THREAD_PER_BLOCK 1024

bool check_result(const float* ref, const float* test, int m, int n, const char* name="",float epsilon = 1e-6f) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float a = ref[i * n + j];
            float b = test[i * n + j];
            float diff = fabsf(a - b);
            if (diff > epsilon) {
                printf("%s Mismatch at (%d,%d): ref = %f, test = %f, diff = %f\n",name,i, j, a, b, diff);
                return false;
            }
        }
    }
    // printf("All values matched within tolerance.\n");
    return true;
}



double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    const int m = 2048, n = 2048, k = 2048;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    // 检查环境变量
    const char* enable_cpu_opt = getenv("ENABLE_CPU_OPT");
    bool run_cpu_opt = (enable_cpu_opt != NULL && strcmp(enable_cpu_opt, "1") == 0);

    // 分配内存（根据是否运行CPU优化决定分配多少）
    float *A = (float*)malloc(mem_size_A);
    float *B = (float*)malloc(mem_size_B);
    float *C_ref = (float*)malloc(mem_size_C);
    float *C_gpu = (float*)malloc(mem_size_C);
    float *C_v1 = run_cpu_opt ? (float*)malloc(mem_size_C) : nullptr;
    float *C_v2 = run_cpu_opt ? (float*)malloc(mem_size_C) : nullptr;
    float *C_v3 = run_cpu_opt ? (float*)malloc(mem_size_C) : nullptr;

    // 初始化矩阵（使用1.0填充）
    for(int i = 0; i < m * k; i++) A[i] = 1.0f;
    for(int i = 0; i < k * n; i++) B[i] = 1.0f;
    memset(C_ref, 0, mem_size_C);

    // 基准CPU计算
    double start = get_time();
    for(int i = 0; i < m; i++) {
        for(int l = 0; l < k; l++) {
            float a_val = A[i * k + l];
            for(int j = 0; j < n; j++) {
                C_ref[i * n + j] += a_val * B[l * n + j];
            }
        }
    }
    printf("CPU time: %.3f s\n", get_time() - start);

    // GPU计算部分
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, mem_size_A);
    cudaMalloc(&d_B, mem_size_B);
    cudaMalloc(&d_C, mem_size_C);
    cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    
    // 基准GPU核函数
    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    
    cudaEventRecord(start_event);
    gemm_baseline_gpu<16,16><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaEventRecord(end_event);
    cudaEventSynchronize(end_event);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start_event, end_event);
    printf("GPU baseline: %.6f s\n", elapsed / 1000);
    cudaMemcpy(C_gpu, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    bool ok = check_result(C_ref, C_gpu, m, n, "baseline", 1e-4);

    // 共享内存版本（复用d_C）
    cudaMemset(d_C, 0, mem_size_C);  // 清零
    
    cudaEventRecord(start_event);
    gemm_share_v1_gpu<16,16><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaEventRecord(end_event);
    cudaEventSynchronize(end_event);
    
    cudaEventElapsedTime(&elapsed, start_event, end_event);
    printf("GPU shared: %.6f s\n", elapsed / 1000);
    cudaMemcpy(C_gpu, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    ok &= check_result(C_ref, C_gpu, m, n, "shared", 1e-4);

    // 添加CPU优化版本（仅在环境变量启用时运行）
    if(run_cpu_opt) {
        memcpy(C_v1, C_ref, mem_size_C);
        memcpy(C_v2, C_ref, mem_size_C);
        memcpy(C_v3, C_ref, mem_size_C);

        // v1优化版本
        start = get_time();
        for (int l = 0; l < k; l++) {
            for (int i = 0; i < m; i++) {
                float a_val = A[i * k + l];
                for (int j = 0; j < n; j++) {
                    C_v1[i * n + j] += a_val * B[l * n + j];
                }
            }
        }
        printf("CPU v1 time: %.3f s\n", get_time() - start);

        // v2优化版本（指针优化）
        start = get_time();
        for (int l = 0; l < k; l++) {
            const float* b_row = &B[l * n];
            for (int i = 0; i < m; i++) {
                float a_val = A[i * k + l];
                float* c_row = &C_v2[i * n];
                for (int j = 0; j < n; j++) {
                    c_row[j] += a_val * b_row[j];
                }
            }
        }
        printf("CPU v2 time: %.3f s\n", get_time() - start);

        // v3优化版本（循环展开）
        start = get_time();
        for (int l = 0; l < k; l++) {
            const float* b_row = &B[l * n];
            for (int i = 0; i < m; i++) {
                float a_val = A[i * k + l];
                float* c_row = &C_v3[i * n];
                for (int j = 0; j < n; j+=4) {
                    c_row[j+0] += a_val * b_row[j+0];
                    c_row[j+1] += a_val * b_row[j+1];
                    c_row[j+2] += a_val * b_row[j+2];
                    c_row[j+3] += a_val * b_row[j+3];
                }
            }
        }
        printf("CPU v3 time: %.3f s\n", get_time() - start);
    }

    // 清理资源
    free(A); free(B); free(C_ref); free(C_gpu);
    if(run_cpu_opt) {
        free(C_v1); free(C_v2); free(C_v3);
    }
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start_event); cudaEventDestroy(end_event);
}