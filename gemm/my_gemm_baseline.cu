
#include "gemm_kernel.h"

// READ 2MNK
// WRITE MN

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

//先不做tile 直接把每一行加载到share memory

template <unsigned int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void gemm_share_v1_gpu(float *A,float *B,float *C,int m,int n,int k){

    __shared__ float A_share[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_share[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx=threadIdx.x+blockIdx.x*blockDim.x;
    int ty=threadIdx.y+blockIdx.y*blockDim.y;
    float sum=0.f;
    int width =(k+BLOCK_SIZE-1)/BLOCK_SIZE;
    for(int i=0;i<width;i++){
        if(tx<m&&threadIdx.y+i*BLOCK_SIZE<k){
            A_share[threadIdx.x][threadIdx.y]=A[tx*k+threadIdx.y+i*BLOCK_SIZE];
        }else{
            A_share[threadIdx.x][threadIdx.y]=0.f;
        }
        if(ty<n&&threadIdx.x+i*BLOCK_SIZE<k){
            B_share[threadIdx.x][threadIdx.y]=B[i*BLOCK_SIZE*n+threadIdx.x*n+threadIdx.y];
        }else{
            B_share[threadIdx.x][threadIdx.y]=0.f;
        }
        __syncthreads();
        for(int s=0;s<BLOCK_SIZE;s++){
            sum+= A_share[threadIdx.x][s]* B_share[s][threadIdx.y];
        }
    }
    C[n*tx+ty]=sum;

}


template __global__ void gemm_baseline_gpu<16, 16>(float*, float*, float*, int, int, int);
template __global__ void gemm_baseline_gpu<32, 2>(float*, float*, float*, int, int, int);
template __global__ void gemm_share_v1_gpu<16, 16>(float*, float*, float*, int, int, int);
