#include <stdio.h>  
#include <cuda_runtime.h>


__global__ void childKernel(int father){
    printf("Parent %d child %d\n", father, threadIdx.x);
}

__global__ void parentKernel(){
    childKernel<<<1, 5>>>(threadIdx.x);
}

int main(){
    parentKernel<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}

/*

to run the code :
nvcc -arch=sm_35 -rdc=true inheritence.cu -o inheritence -lcudadevrt

requires  a compute capability of 3.5 or higher

you can run it in google colab


---------------------------------------------OUTPUT---------------------------------------------
Parent 0 child 0
Parent 0 child 1
Parent 0 child 2
....more

pareent kernel is assosiated with a parent grid and child kernel is assosiated with a child grid
parent and child  may execute asynchronously
a parent grid si not completed until all child grids are completed
global memory is shared by both parent and child kernels
but they have distinct local memory and registers and shared memory

---------------------------------------------Muli gpu---------------------------------------------
for one by one : firstly 0 device then 1 device
cudaSetDevice(0);
k1<<<1,1>>>();
cudaSetDevice(1);
k2<<<1,1>>>();
cudaMecpy()

parallelization perspective :
cudaSetDevice(0);
k1<<<1,1>>>();
cudaMemcpyAsync()
cudaSetDevice(1);
k2<<<1,1>>>();
cudaMemcpyAsync()


*/