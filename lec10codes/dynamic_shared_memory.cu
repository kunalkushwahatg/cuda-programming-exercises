#include <stdio.h>

//by using extern keyword, we can define shared memory size in the kernel call
//dynamic shared memory allocation helps to define shared memory size in the kernel call and hence we can 
//  use different sizes of shared memory in the same kernel

__global__ void myKernel(int numElements) {
 
    extern __shared__ float sharedMem[];

    int idx = threadIdx.x;

    if (idx < numElements) {
        sharedMem[idx] = idx * 2.0f;
    }


    __syncthreads();

    if (idx < numElements) {
        printf("Index: %d, Value: %f\n", idx, sharedMem[idx]);
    }
}

int main() {
    int numElements = 10;
    int sharedMemSize = numElements * sizeof(float);


    myKernel<<<1, numElements, sharedMemSize>>>(numElements);

    cudaDeviceSynchronize();

    return 0;
}
