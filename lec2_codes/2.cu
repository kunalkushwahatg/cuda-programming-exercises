#include <stdio.h>
#include <cuda_runtime.h> // Includes the CUDA runtime library

// Kernel function that performs addition
__global__ void add(int a, int b, int *c){
    *c = a + b;  // Adds a and b, stores result in the memory location pointed to by c
}

int main(void){
    int c;        // Variable to store the result on the host
    int *dc;      // Pointer to an integer for device memory this variavle is for gpu
    
    // Allocate memory on the device
    //you're essentially saying, "Here's where my pointer is stored; please update it to point to the newly allocated memory."
    cudaMalloc((void **)&dc, sizeof(int));   //(void **) makes the dc from a int pointer to a void pointer

    // Launch the add kernel with 1 block of 1 thread
    add<<<1, 1>>>(2, 7, dc);
    
    // Copy the result from device to host memory
    cudaMemcpy(&c, dc, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print the result
    printf("2 + 7 = %d\n", c);

    // Free the allocated device memory
    cudaFree(dc);

    return 0;  // Return 0 indicating successful execution
}
