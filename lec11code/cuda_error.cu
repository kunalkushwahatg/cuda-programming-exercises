#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>


__global__ void K(int *x){
    *x = 0;
    printf("x: %d\n", *x); 
}

int main(){
    int *x = NULL;
    printf("-----------------------------------------without cudaMalloc--------------------------------------------------------\n");
    K<<<1, 1>>>(x);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    printf("error: %d , %s , %s\n", error, cudaGetErrorName(error), cudaGetErrorString(error));

    printf("-----------------------------------------with cudaMalloc--------------------------------------------------------\n");
    cudaMalloc(&x, sizeof(int));
    K<<<1, 1>>>(x);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError(); 
    printf("error: %d , %s , %s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
    return 0;
}

//it will give error: 77, cudaErrorIllegalAddress, an illegal memory access was encountered 
//because we are trying to write to a NULL pointer
//we also need to allocate memory to the pointer x