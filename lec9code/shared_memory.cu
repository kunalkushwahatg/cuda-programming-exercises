#include <stdio.h>
#include <cuda.h>

#define N 1024

__global__ void kernel(){
    __shared__ unsigned s;
    if(threadIdx.x == 0){
        printf("assigning 0 to s\n");
        s = 0;
    }
    if(threadIdx.x == 1){
        printf("incrementing s by 1\n");
        s =+ 1;
    }
    if (threadIdx.x == 100){
        printf("incrementing s by 100\n");
        s =+ 100;
    }
    if (threadIdx.x == 1){
        printf("s = %d\n", s);
    }
}

int main(){
    for (size_t i = 0; i < 10; i++)
    {
        kernel<<<1, 1024>>>();
        cudaDeviceSynchronize();
    }
    

    return 0;
}