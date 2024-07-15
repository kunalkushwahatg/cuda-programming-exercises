#include <stdio.h>

#define n 10000

__device__ unsigned wlsize;
__device__ int worklist[n];

__global__ void k1(){
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    worklist[atomicInc(&wlsize, n)] = id;
}

__global__ void k2(){
    printf("number of elements in worklist: %d\n", wlsize);
}

int main(){
    //cudaMemset is used to set the memory to zero 
    cudaMemset(&wlsize, 0, sizeof(unsigned));

    //cureently the worklist is empty
    //wlsize is the number of elements in the worklist

    k1<<<100, 100>>>();
    cudaDeviceSynchronize();

    k2<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
        



}