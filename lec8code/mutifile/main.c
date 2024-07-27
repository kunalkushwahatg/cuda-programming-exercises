// main.c
#include <stdio.h>
#include <stdlib.h> // Include this header for malloc, free, etc.

extern void myKernel(float *d_data, int N); // Declaration of CUDA kernel

int main() {
    int N = 1024;
    float *h_data, *d_data;

    // Allocate memory on host
    h_data = (float*)malloc(N * sizeof(float));

    // Allocate memory on device
    cudaMalloc(&d_data, N * sizeof(float));

    // Launch kernel from separate file
    myKernel<<<1, 256>>>(d_data, N);

    // Copy data from device to host and do further processing

    // Free memory
    cudaFree(d_data);
    free(h_data);

    return 0;
}

//gcc -c main.c -o main.o
//nvcc main.o kernel.o -o my_program
//./my_program

