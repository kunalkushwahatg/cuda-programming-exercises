#include <stdio.h>
#include <cuda.h>
#define N 100

//workflow of the code
//1. Allocate memory on the device
//2. Launch the kernel function
//3. Copy the result from device to host memory
//4. Print the result
//5. Free the allocated device memory



// Kernel function that performs addition
__global__ void add(int *a){
    a[threadIdx.x] = threadIdx.x * threadIdx.x;
}



int main(){
    //interger array of size N and void pointer is declared
    int a[N] , *da;
    int i;

    //cudaMalloc is used to allocate memory on the device where &da is the pointer to the device memory and n is the size of the memory to be allocated
    cudaMalloc(&da, N * sizeof(int));
    
    //add<<<1, N>>> is used to launch the kernel function with 1 block of N threads
    add<<<1,N>>>(da);

    //cudaMemcpy is used to copy the result from device to host memory
    cudaMemcpy(a, da, N * sizeof(int), cudaMemcpyDeviceToHost);

    //free the cuda reset
    cudaFree(da);

    //printing the result
    for(i=0; i<N; i++)
        printf("%d ", a[i]);
    return 0;
}