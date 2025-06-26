
#include <stdio.h>
#include <sys/time.h> 
#include <cuda.h>
#include <cuda_runtime.h>
#define THREAD_PER_BLOCK 1024


template <unsigned int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void gemm_baseline_gpu(float *A,float *B,float *C,int m,int n,int k){
    int tx=threadIdx.x+blockIdx.x*blockDim.x;
    int ty=threadIdx.y+blockIdx.y*blockDim.y;
    float sum=0.f;
    if(tx<m&&ty<n){
        for(int i=0;i<k;i++){
            sum+=A[tx*k+i]*B[ty*k+i];
        }
        C[tx*n+ty]=sum;
    }

}

