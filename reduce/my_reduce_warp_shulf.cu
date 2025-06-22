#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 128


template <unsigned int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void reduce_warp_shulf(float *input,float *output,int n){
    float sum=0.f;
    int tid=threadIdx.x;
    int index=tid + blockIdx.x*NUM_PER_THREAD* blockDim.x;
    for(int i=0; i<NUM_PER_THREAD; i++){
        sum += input[index+i*BLOCK_SIZE];
    }

    sum+=__shfl_down_sync(0xffffffff,sum,16);
    sum+=__shfl_down_sync(0xffffffff,sum,8);
    sum+=__shfl_down_sync(0xffffffff,sum,4);
    sum+=__shfl_down_sync(0xffffffff,sum,2);
    sum+=__shfl_down_sync(0xffffffff,sum,1);
    __shared__ float warpSums[32];
    const int lane_id=tid%warpSize;
    const int warp_id=tid/warpSize;

    if(lane_id==0){
        warpSums[warp_id]=sum;
    }
    __syncthreads();
    if(warp_id==0){
        sum=((lane_id<blockDim.x/32)?warpSums[lane_id]:0.f);
        sum+=__shfl_down_sync(0xffffffff,sum,16);
        sum+=__shfl_down_sync(0xffffffff,sum,8);
        sum+=__shfl_down_sync(0xffffffff,sum,4);
        sum+=__shfl_down_sync(0xffffffff,sum,2);
        sum+=__shfl_down_sync(0xffffffff,sum,1);
    }

    
    if(threadIdx.x==0)output[blockIdx.x]=sum;

}
bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(abs(out[i]-res[i])>0.05)
            return false;
    }
    return true;
}


int main(){

    const int N=32*1024*1024;

    float* input=(float*)malloc(N*sizeof(float));
    float* d_input;
    cudaMalloc((void **)&d_input,N*sizeof(float));
    const int num_blocks = 1024;
    const int NUM_PER_BLOCK = N / num_blocks;
    const int NUM_PER_THREAD = NUM_PER_BLOCK/THREAD_PER_BLOCK;
    float* output=(float*)malloc(num_blocks*sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output,num_blocks*sizeof(float));
    float *result=(float*)malloc(num_blocks*sizeof(float));

    for(int i=0;i<N;i++){
        input[i]=1;
    }
    //CPU 
    for(int i=0;i<num_blocks;i++){
        int cur=0;
        for(int j=0;j<NUM_PER_BLOCK;j++){
            cur+=input[i*NUM_PER_BLOCK+j];
        }
        result[i]=cur;
    }

    //gpu warp divergence
    cudaMemcpy(d_input,input,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    reduce_warp_shulf<THREAD_PER_BLOCK, NUM_PER_THREAD><<<num_blocks,THREAD_PER_BLOCK>>>(d_input,d_output,N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaMemcpy(output,d_output,num_blocks*sizeof(float),cudaMemcpyDeviceToHost);

    float time;
    cudaEventElapsedTime(&time,start,stop);
    cudaMemcpy(output,d_output,num_blocks*sizeof(float),cudaMemcpyDeviceToHost);
    printf("baseline time: %f\n",time);
    if(check(output,result,num_blocks)){
        printf("baseline test pass\n");
    }
    else {
        for(int i=0;i<num_blocks;i++){
            printf("gpu result: %f\n",output[i]);
        }
    }
  
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;

    


}