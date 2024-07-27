
/*
MEMORY COALESCING:
Combining multiple memory accesses into a single 
transaction when threads access contiguous memory locations.
Reduces the number of memory transactions.

MEMORY COALESCED : result[tid] = data[tid];
NON MEMORY COALESCED :  result[tid] = data[tid * 2];

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void coalescedAccess(int *data, int *result, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        result[tid] = data[tid];
    }
}

__global__ void nonCoalescedAccess(int *data, int *result, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        result[tid] = data[tid * 2];
    }
}

int main() {
    int N = 1 << 20; // 1M elements
    int *h_data = (int*)malloc(N * sizeof(int));
    int *h_result = (int*)malloc(N * sizeof(int));
    int *d_data, *d_result;

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Timing coalesced access
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    coalescedAccess<<<numBlocks, blockSize>>>(d_data, d_result, N);
    cudaDeviceSynchronize(); // Ensure kernel is complete
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Coalesced access time: %f seconds\n", cpu_time_used);

    // Timing non-coalesced access
    start = clock();
    nonCoalescedAccess<<<numBlocks, blockSize>>>(d_data, d_result, N / 2); // Adjust size for non-coalesced
    cudaDeviceSynchronize(); // Ensure kernel is complete
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Non-coalesced access time: %f seconds\n", cpu_time_used);

    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    free(h_result);

    return 0;
}
