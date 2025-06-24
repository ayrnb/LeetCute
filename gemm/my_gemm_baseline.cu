
#include <stdio.h>

bool check_result(const float* ref, const float* test, int m, int n, float epsilon = 1e-6f) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float a = ref[i * n + j];
            float b = test[i * n + j];
            float diff = fabsf(a - b);
            if (diff > epsilon) {
                printf("Mismatch at (%d,%d): ref = %f, test = %f, diff = %f\n", i, j, a, b, diff);
                return false;
            }
        }
    }
    printf("All values matched within tolerance.\n");
    return true;
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
    float *C=(float*)malloc(mem_size_C);
    // 初始化A B矩阵
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            A[i * k + j]=1;
        }
       
    }
    for(int i=0;i<k;i++){
        for(int j=0;j<n;j++){
            B[i * n + j]=1;
        }
    }
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C[i * n + j]=0;
        }
    }
    //计算A*B=C
    time_t start_time=clock();
    for(int i=0;i<m;i++){
        for(int l=0;l<k;l++){
            for(int j=0;j<n;j++){
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
    time_t end_time=clock();
    double time=double(end_time-start_time)/CLOCKS_PER_SEC;
    printf("v0 time: %f\n", time);
    start_time=clock();
    for (int l = 0; l < k; l++) {
        for (int i = 0; i < m; i++) {
            float a_val = A[i * k + l];  
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_val * B[l * n + j];
            }
        }
    }
    end_time=clock();
    time=double(end_time-start_time)/CLOCKS_PER_SEC;
    printf("v1 time: %f\n", time);
    // 减少地址计算
    start_time=clock();
    for (int l = 0; l < k; l++) {
        const float* b_row = &B[l * n];
        for (int i = 0; i < m; i++) {
            float a_val = A[i * k + l];
            float* c_row = &C[i * n];
            for (int j = 0; j < n; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
    end_time=clock();
    time=double(end_time-start_time)/CLOCKS_PER_SEC;
    printf("v2 time: %f\n", time);

   

    
}