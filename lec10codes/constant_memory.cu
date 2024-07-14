/*
    constant memory is a read-only memory that is cached and shared among all threads in a block.
    constant memory is optimized for read-only access and provides fast access to data with spatial locality.
    constant memory is limited in size (64 KB) and is cached for fast access.
    constant memory is declared using the __constant__ keyword and can be accessed by all threads in a block.
    
    To use constant memory, you need to:
    1. Declare a constant memory variable using the __constant__ keyword.
    2. Copy data to the constant memory using cudaMemcpyToSymbol.
    3. Access the constant memory in the kernel using the variable name.

    //define constant memory variable in global scope
    __constant__ float constData[SIZE];

    //copy data to constant memory
    cudaMemcpyToSymbol(constData, hostData, SIZE * sizeof(float));

    //access constant memory in the kernel
    float value = constData[idx];

    fast ,  readonly memory ,  global access for all threads in a block , 64kb size 
*/

#include <stdio.h>

#define SIZE 10

__constant__ float constData[SIZE];

__global__ void myKernel(int numElements) {
 
    int idx = threadIdx.x;

    if (idx < numElements) {
        float value = constData[idx];
        printf("Index: %d, Value: %f\n", idx, value);
    }
}

int main() {
    float hostData[SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    cudaMemcpyToSymbol(constData, hostData, SIZE * sizeof(float));

    myKernel<<<1, SIZE>>>(SIZE);

    cudaDeviceSynchronize();

    return 0;
}