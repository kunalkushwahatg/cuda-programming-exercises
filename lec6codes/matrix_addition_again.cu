#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 512 // size in one dir that means matrix size is NxN
#define THREADS_PER_BLOCK 256

__global__ void matrixAddition(int *a , int *b, int *c){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N){
        c[i * N + j] = a[i * N + j] + b[i * N + j];
    }

}


int main(){
    int a[N][N] , b[N][N] , c[N][N];
    int *dev_a , *dev_b , *dev_c;

    int size = int(N*N)*sizeof(int);

    cudaMalloc((void**)&dev_a , size);
    cudaMalloc((void**)&dev_b , size);
    cudaMalloc((void**)&dev_c , size);

    for(int i = 0 ; i < N ; i++){
        for(int j = 0 ; j < N ; j++){
            a[i][j] = i+j;
            b[i][j] = i-j;
        }
    }

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(THREADS_PER_BLOCK ,THREADS_PER_BLOCK);
    int dim_grid = int((N + THREADS_PER_BLOCK - 1)  / THREADS_PER_BLOCK);
    dim3 dimGrid(dim_grid ,dim_grid);

    matrixAddition<<<dimGrid , dimBlock>>>(dev_a,dev_b,dev_c);
    
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    

    return 0;
}
