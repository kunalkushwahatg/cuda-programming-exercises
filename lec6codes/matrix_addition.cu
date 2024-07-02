#define N 512
#define BLOCK_DIM 512

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void matrixAddition(int *a, int *b, int *c);

int main(){

    int a[N][N], b[N][N], c[N][N];
    int *dev_a, *dev_b, *dev_c;

    int size = N * N * sizeof(int);

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            a[i][j] = 1;
            b[i][j] = 2;
        }
    }

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);

    matrixAddition<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);
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

__global__ void matrixAddition(int *a, int *b, int *c){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N){
        c[i * N + j] = a[i * N + j] + b[i * N + j];
    }

}