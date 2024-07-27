//to run the code in host (CPU) we need to use the following code



//-------------------------------ERRORs-------------------------------------


#include <stdio.h>

__host__ __device__ void kernel(*counter){
    printf("Hello from the CPU\n");
    ++(*counter);
}

int main(){
    int *counter;

    cudaHostAlloc((void**)&counter, sizeof(int), cudaHostAllocDefault);

    *counter = 0;
    printf("Counter: %d\n", *counter);

    kernel<<<1,1>>>(&counter);
    cudaDeviceSynchronize();
    printf("Counter: %d\n", counter);
    return 0;
}
