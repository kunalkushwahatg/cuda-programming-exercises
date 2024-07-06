
    /* 

        The overview of wha twe are going to do here is :
        - width variable is width of the matrix
        - create the column and row variables which are stored inside registers that are only accessed by their respective threads 
        - create shared memory tiles for the matrices a and b that is tileA and tileB which are defined in shared memory that are only 
          acessed by their respective blocks.
        - create a value variable that will store the result of the multiplication of the two matrices and this value variable will 
          be acessed by all the threads in the block.VAlue is stored in memory called register  memory 
        - check if the column and row values are inside the grid
        - for(int m = 0 ; m < (N + TILE_WIDTH - 1)/TILE_WIDTH ; m++ ) runs loop through every tile.
        - tileA[threadIdx.y][threadIdx.x] = a[ width*row + (TILE_WIDTH*m + threadIdx.x) ] by this code tileA which is 2d matrix 
          is filled by accesing the rowwise vector A .We will be accessing the rows of tile A.
        - tileB[threadIdx.y][threadIdx.x] = b[ (m*TILE_WIDTH + threadIdx.y)*width + column ] by this code tileB which is 2d matrix 
          is filled by accesing the rowwise vector B .We will be accessing the columns of tile B.
        - __syncthreads(); will make sure alloction of tileA and tileB in every block is completed.
        - for (int k = 0; k < TILE_WIDTH; ++k)  by this code every block in GPU will go through every element of their respective block.
        - value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x]; will calculate the row vs column multiplication of and in every thread 
          wil have diffrent value of var value but in diffrent block the same threads the value of 
    
    */

   /*
   ---------------------------------------------chat gpt rephrased version----------------------------------------------
    CUDA Kernel for Matrix Multiplication

    Overview:
    - `width`: Width of the square matrices A, B, and C.
    - Registers `column` and `row` store thread-specific indices within the matrix, accessed only by their respective threads.
    - Shared memory tiles `tileA` and `tileB` hold sub-matrices of A and B respectively, accessible exclusively within each block.
    - `value`: Register memory storing partial multiplication results, accessed by all threads within the block.
    - Checks ensure threads operate within matrix boundaries.
    - Iterates through each tile (m, m) in the matrices, accommodating for TILE_WIDTH boundaries.
    - `tileA[threadIdx.y][threadIdx.x]` accesses row-wise elements of A to fill `tileA`.
    - `tileB[threadIdx.y][threadIdx.x]` accesses column-wise elements of B to fill `tileB`.
    - `__syncthreads()` ensures synchronization of `tileA` and `tileB` allocation across all threads in the block.
    - Each block processes all elements within its TILE_WIDTH block in parallel for efficient matrix multiplication.
    - `value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x]` calculates the product of corresponding elements in `tileA` and `tileB`.
    - Each thread accumulates its partial result in `value`, ensuring accurate matrix multiplication across multiple blocks.

    Parameters:
    - `a`, `b`: Input matrices A and B stored in global memory.
    - `c`: Output matrix C stored in global memory.
    - `width`: Width of the matrices A, B, and C (assumed to be the same).

    Note: This kernel assumes square matrices A, B, and C of size `width x width`.
*/



#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 4 // size in one dir that means matrix size is NxN
#define THREADS_PER_BLOCK 2 
#define TILE_WIDTH 2

__global__ void matrixMultiplication(int *a, int *b, int *c, int width) {
    // i and j values inside the grid
    int column = blockIdx.x * blockDim.x + threadIdx.x; // column 
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row

    // tiles are only accessed by their respective blocks so we need to declare them as shared memory
    __shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tileB[TILE_WIDTH][TILE_WIDTH];

    // stored inside register memory 
    int value = 0;

    if (column < width && row < width) {
        for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; m++) { // going through every tile
            tileA[threadIdx.y][threadIdx.x] = a[width * row + (TILE_WIDTH * m + threadIdx.x)];
            tileB[threadIdx.y][threadIdx.x] = b[(m * TILE_WIDTH + threadIdx.y) * width + column];
            __syncthreads();
            // Compute partial results
            for (int k = 0; k < TILE_WIDTH; ++k) {
                value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            }
            __syncthreads();
        }
        c[width * row + column] = value;  
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int main() {

    /*
        if we would have allocated the memory like a[n][n] then it would have given memroy error 
    */

    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    // to calculate the time taken by the code
    clock_t start, end;
    double cpu_time_used;

    int size = N * N * sizeof(int);

    // Allocate memory on the host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return -1;
    }

    checkCudaError(cudaMalloc((void**)&dev_a, size), "cudaMalloc dev_a");
    checkCudaError(cudaMalloc((void**)&dev_b, size), "cudaMalloc dev_b");
    checkCudaError(cudaMalloc((void**)&dev_c, size), "cudaMalloc dev_c");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    printf("\n -------------------THE MATRIX A IS -------------------\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", a[i * N + j]);
        }
        printf("\n");
    }

    printf("\n -------------------THE MATRIX B IS -------------------\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", b[i * N + j]);
        }
        printf("\n");
    }

    checkCudaError(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice), "cudaMemcpy a to dev_a");
    checkCudaError(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice), "cudaMemcpy b to dev_b");

    dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    int dim_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 dimGrid(dim_grid, dim_grid);

    start = clock();

    matrixMultiplication<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    checkCudaError(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy dev_c to c");

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time used: %f seconds\n", cpu_time_used);

    checkCudaError(cudaFree(dev_a), "cudaFree dev_a");
    checkCudaError(cudaFree(dev_b), "cudaFree dev_b");
    checkCudaError(cudaFree(dev_c), "cudaFree dev_c");

    printf("\n -------------------THE MATRIX C IS -------------------\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i * N + j]);
        }
        printf("\n");
    }

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}


