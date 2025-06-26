
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

int main(){
    // printf("hello world\n");
    int m=2048;
    int n=2048;
    int k=2048; 
    const size_t mem_size_A=m*k*sizeof(float);
    const size_t mem_size_B=k*n*sizeof(float);
    const size_t mem_size_C=m*n*sizeof(float);

    // 分配内存
    float *A=(float*)malloc(mem_size_A);
    float *B=(float*)malloc(mem_size_B);
    float *C_ref = (float*)malloc(mem_size_C);
    float *C_v1 = (float*)malloc(mem_size_C);
    float *C_v2 = (float*)malloc(mem_size_C);
    float *C_v3 = (float*)malloc(mem_size_C);
    float *C_gpu=(float*)malloc(mem_size_C);
    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A,mem_size_A);
    cudaMalloc(&d_B,mem_size_B);
    cudaMalloc(&d_C,mem_size_C);

    // 初始化A B矩阵
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            A[i * k + j] = 1.0f;
        }
       
    }
    for(int i=0;i<k;i++){
        for(int j=0;j<n;j++){
            B[i * n + j]=1.0f;
        }
    }
    // 清零 C
    memset(C_ref, 0, mem_size_C);
    memcpy(C_v1, C_ref, mem_size_C);
    memcpy(C_v2, C_ref, mem_size_C);

    // 计算A*B=C_ref
    double start_time=get_time();
    for(int i=0;i<m;i++){
        for(int l=0;l<k;l++){
            for(int j=0;j<n;j++){
                C_ref[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
    double end_time=get_time();
    double time=double(end_time-start_time);
    printf("v0 time: %.3f s\n", time);  
    start_time=get_time();
    for (int l = 0; l < k; l++) {
        for (int i = 0; i < m; i++) {
            float a_val = A[i * k + l];  
            for (int j = 0; j < n; j++) {
                C_v1[i * n + j] += a_val * B[l * n + j];
            }
        }
    }
    end_time=get_time();
    time=double(end_time-start_time);
    printf("v1 time: %.3f s\n", time);
    // 减少地址计算
    start_time=get_time();
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
    end_time=get_time();
    time=double(end_time-start_time);
    printf("v2 time: %.3f s\n", time);

    //内存循环展开计算
    start_time=get_time();
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
    end_time=get_time();
    time=double(end_time-start_time);
    printf("v3 time: %.3f s\n", time);

    bool ok1 = check_result(C_ref, C_v1, m, n,"v1");
    bool ok2 = check_result(C_ref, C_v2, m, n,"v2");
    bool ok3 = check_result(C_ref, C_v3, m, n,"v3");
    
    //cpu 还可以使用SSE/AVX和OpenMP等tile优化技术进一步提高计算效率，这里我们直接在GPU上进行
    cudaMalloc(&d_A,mem_size_A);
    cudaMemcpy(d_A,A,mem_size_A,cudaMemcpyHostToDevice);
    cudaMalloc(&d_B,mem_size_B);
    cudaMemcpy(d_B,B,mem_size_B,cudaMemcpyHostToDevice);
    cudaMalloc(&d_C,mem_size_C);
    dim3 block(16,16);
    dim3 grid((m+block.x-1)/16,(n+block.y-1)/16);
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,0);
    gemm_baseline_gpu<16,16><<<grid,block>>>(d_A,d_B,d_C,m,n,k);
    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time,start,end);
    printf("v4 time: %.6f s\n", gpu_time/1000);
    cudaMemcpy(C_gpu,d_C,mem_size_C,cudaMemcpyDeviceToHost);

    bool ok4 = check_result(C_ref, C_gpu, m, n,"gpu",1e-4);
    free(A);
    free(B);
    free(C_ref);
    free(C_v1);
    free(C_v2);
    free(C_v3);
    free(C_gpu);

    
}