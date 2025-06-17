#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256



__global__ void reduce_use_idle_thread(float *input,float *output,int N){
    __shared__ float shared[THREADS_PER_BLOCK];
    int  i=threadIdx.x + blockIdx.x* blockDim.x*2;
    shared[threadIdx.x]=input[threadIdx.x]+input[i+blockDim.x];
    __syncthreads();
    for(int i=blockDim.x/2;i>0;i>>=1){

        if(threadIdx.x<i){
            shared[threadIdx.x]+=shared[threadIdx.x+i];
        }
        __syncthreads();
    }
    if(threadIdx.x==0)output[blockIdx.x]=shared[0];

}
bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(abs(out[i]-res[i])>0.05){
            printf("test failed; i: %d out: %f res: %f\n",i,out[i],res[i]);
            break;

        }
    }
    printf("test pass");
    return true;
}


int main(){

    int N=32*1024*1024;
    int num_blocks=N/THREADS_PER_BLOCK/2;

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
    int block_dim=THREADS_PER_BLOCK*2;
    for(int i=0;i<num_blocks;i++){
        int cur=0;
        for(int j=0;j<block_dim;j++){
            cur+=input[i*block_dim+j];
        }
        result[i]=cur;
    }

    //gpu warp divergence
    cudaMemcpy(d_input,input,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    reduce_use_idle_thread<<<num_blocks,THREADS_PER_BLOCK>>>(d_input,d_output,N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaMemcpy(output,d_output,N,cudaMemcpyHostToDevice);

    float time;
    cudaEventElapsedTime(&time,start,stop);
    cudaMemcpy(output,d_output,num_blocks*sizeof(float),cudaMemcpyDeviceToHost);
    printf("baseline time: %f\n",time);
    check(output,result,num_blocks);

  
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;

    


}