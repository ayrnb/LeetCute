#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK 128

__global__ void reduce_baseline(float *input,float *output,int N){
    float* index_block= input+blockIdx.x*blockDim.x;
    for(int i=1;i<blockDim.x;i*=2){
        if(threadIdx.x%(i*2)==0){
            index_block[threadIdx.x]+=index_block[threadIdx.x+i];   
        }
        __syncthreads();
    }
    if(threadIdx.x==0) output[blockIdx.x]=index_block[0];

}


bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(abs(out[i]-res[i])>0.05)
            return false;
    }
    return true;
}

int main(){

    const int  N=32*1024*1024;
    float *input=(float*)malloc(N*sizeof(float));
    float *d_input;
    cudaMalloc(&d_input,N*sizeof(float));
    int block_num=N/THREADS_PER_BLOCK;
    float *output=(float*)malloc(block_num*sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output,block_num*sizeof(float));
    float *resutl=(float*)malloc(block_num*sizeof(float));
    for(int i=0;i<N;i++){
        input[i]=2.0*(float)drand48()-1.0;
    }
    //cpu 计算
    time_t start_time=clock();
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREADS_PER_BLOCK;j++){
            cur+=input[i*THREADS_PER_BLOCK+j];
            }
        resutl[i]=cur;
        // printf("cpu result: %f\n",resutl[i]);
    }
    time_t end_time=clock();
    printf("cpu time: %f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);
    //gpu 计算
    cudaMemcpy(d_input,input,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    reduce_baseline<<<block_num,THREADS_PER_BLOCK>>>(d_input,d_output,N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time,start,stop);
    cudaMemcpy(output,d_output,block_num*sizeof(float),cudaMemcpyDeviceToHost);
    printf("baseline time: %f\n",time);
    if(check(output,resutl,block_num)){
        printf("baseline test pass\n");
    }
    else {
        for(int i=0;i<block_num;i++){
            printf("gpu result: %f\n",output[i]);
        }
    };
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
    



    


