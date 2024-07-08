// kernel.cu
#include <cuda_runtime.h>

__global__ void myKernel(float *d_data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        d_data[tid] *= 2.0f;
    }
}

//nvcc -c kernel.cu -o kernel.o
