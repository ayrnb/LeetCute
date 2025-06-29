
#include "gemm_kernel.h"


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

template __global__ void gemm_baseline_gpu<16, 16>(float*, float*, float*, int, int, int);
template __global__ void gemm_baseline_gpu<32, 2>(float*, float*, float*, int, int, int);