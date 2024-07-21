//we can write cuda using cpp code also just like c 
//here i will add two values and we will also talk about nvprofileer

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void add(int *a, int *b, int *c){
    *c = *a + *b;
}

int main(){
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    //allocate memory on device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    //initialize a and b
    a = 2;
    b = 7;

    //copy a and b to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    //launch add() kernel on GPU
    add<<<1, 1>>>(d_a, d_b, d_c);

    //copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "result: " << c << std::endl;

    //print the error
    cudaError_t error = cudaGetLastError();
    std::cout << "error: " << error << " , " << cudaGetErrorName(error) << " , " << cudaGetErrorString(error) << std::endl;


    //free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

//to compile this code use the following command
//nvcc cudaincpp.cu -o cudaincpp

//to profile this code use the following command
//nvprof ./cudaincpp

//profile tells about the time taken by each function and the memory used by each function
