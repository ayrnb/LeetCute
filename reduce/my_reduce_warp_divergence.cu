#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128



__global__ void reduce_warp_divergence(float *input,float *output,int N){
    __shared__ float shared[THREADS_PER_BLOCK];
    float* input_begin=input + blockIdx.x * blockDim.x;
    shared[threadIdx.x]=input_begin[threadIdx.x];
    __syncthreads();
    for(int i=1;i<blockDim.x;i*=2){

        if(threadIdx.x<(blockDim.x/(2*i))){
            int index=threadIdx.x*2*i;
            shared[index]+=shared[index+i];
        }
        __syncthreads();
    }
    if(threadIdx.x==0)output[blockIdx.x]=shared[0];

}
bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(abs(out[i]-res[i])>0.05)
            return false;
    }
    return true;
}


int main(){

    int N=32*1024*1024;
    int num_blocks=N/THREADS_PER_BLOCK;

    float* input=(float*)malloc(N*sizeof(float));
    float* d_input;
    cudaMalloc((void **)&d_input,N*sizeof(float));
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
        for(int j=0;j<THREADS_PER_BLOCK;j++){
            cur+=input[i*THREADS_PER_BLOCK+j];
        }
        result[i]=cur;
    }

    //gpu warp divergence
    cudaMemcpy(d_input,input,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    reduce_warp_divergence<<<num_blocks,THREADS_PER_BLOCK>>>(d_input,d_output,N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaMemcpy(output,d_output,N,cudaMemcpyHostToDevice);

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