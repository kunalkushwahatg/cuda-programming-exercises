#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>



/*
__device__: Use this qualifier when you are sure that the function will only be called
 from within other device or kernel functions 
and does not need to be executed on the host.
__host__ __device__: Use this qualifier when you want the 
flexibility to call the function from both host and device code.
 This can be useful for utility functions that perform simple operations and can be used in both contexts. 

*/









__host__ __device__ void fun(int *counter){
    ++(*counter);
}

__global__ void k1(int *counter){
    fun(counter);
    printf("counter: %d\n", *counter);
}

int main(){
    int *counter;
    cudaHostAlloc(&counter, sizeof(int), cudaHostAllocDefault);
    *counter = 0;
    k1<<<10, 1>>>(counter);
    cudaDeviceSynchronize();
    fun(counter);
    printf("counter: %d\n", *counter);
    return 0;
}