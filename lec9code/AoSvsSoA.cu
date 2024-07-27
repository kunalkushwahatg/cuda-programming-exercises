// AoS (Array of Structures): Data is organized as an array where each element is a structure containing multiple fields.
// SoA (Structure of Arrays): Data is organized as separate arrays for each field or member of a structure.
 #include <stdio.h>
#include <cuda.h>

#define N 1024

struct AoS {
    float x;
    float y;
    float z;
} *h_AoS;

struct SoA {
    float *x;
    float *y;
    float *z;
} h_SoA;

__global__ void kernel_AoS(struct AoS *d_AoS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_AoS[idx].x = d_AoS[idx].x + 10;
    d_AoS[idx].y = d_AoS[idx].y + 10;
    d_AoS[idx].z = d_AoS[idx].z + 10;
}

__global__ void kernel_SoA(struct SoA d_SoA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_SoA.x[idx] = d_SoA.x[idx] + 10;
    d_SoA.y[idx] = d_SoA.y[idx] + 10;
    d_SoA.z[idx] = d_SoA.z[idx] + 10;
}

int main() {
    struct AoS *d_AoS;
    struct SoA d_SoA;

    h_AoS = (struct AoS *)malloc(N * sizeof(struct AoS));
    cudaMalloc(&d_AoS, N * sizeof(struct AoS));

    h_SoA.x = (float *)malloc(N * sizeof(float));
    h_SoA.y = (float *)malloc(N * sizeof(float));
    h_SoA.z = (float *)malloc(N * sizeof(float));

    cudaMalloc(&d_SoA.x, N * sizeof(float));
    cudaMalloc(&d_SoA.y, N * sizeof(float));
    cudaMalloc(&d_SoA.z, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_AoS[i].x = i;
        h_AoS[i].y = i;
        h_AoS[i].z = i;

        h_SoA.x[i] = i;
        h_SoA.y[i] = i;
        h_SoA.z[i] = i;
    }

    cudaMemcpy(d_AoS, h_AoS, N * sizeof(struct AoS), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SoA.x, h_SoA.x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SoA.y, h_SoA.y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SoA.z, h_SoA.z, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (int)ceil((float)N / blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure AoS kernel execution time
    cudaEventRecord(start);
    kernel_AoS<<<gridSize, blockSize>>>(d_AoS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msecTotalAoS = 0.0f;
    cudaEventElapsedTime(&msecTotalAoS, start, stop);

    // Measure SoA kernel execution time
    cudaEventRecord(start);
    kernel_SoA<<<gridSize, blockSize>>>(d_SoA);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msecTotalSoA = 0.0f;
    cudaEventElapsedTime(&msecTotalSoA, start, stop);

    printf("AoS: %f ms\n", msecTotalAoS);
    printf("SoA: %f ms\n", msecTotalSoA);

    cudaMemcpy(h_AoS, d_AoS, N * sizeof(struct AoS), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_SoA.x, d_SoA.x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_SoA.y, d_SoA.y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_SoA.z, d_SoA.z, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_AoS);
    cudaFree(d_SoA.x);
    cudaFree(d_SoA.y);
    cudaFree(d_SoA.z);
    free(h_AoS);
    free(h_SoA.x);
    free(h_SoA.y);
    free(h_SoA.z);

    return 0;

//output 
// AoS: 572.338196 ms
// SoA: 0.051072 ms


}
