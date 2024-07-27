
//__global__ inidates that code inside it run on gpu
#include <cuda.h>
#include <stdio.h>
__global__ void kernel1(void){
    printf("hello from GPU 1\n");
}
__global__ void kernel2(void){
    printf("hello from GPU 2 \n");
}

int main(){
    // block , threads  1 block = n threads 
    kernel1<<<1,1>>>();
    kernel2<<<1,1>>>();
     //calls the gpu kernal to execute and run synchronously
    printf("hello from cpu \n");
    cudaDeviceSynchronize();
    return 0;
}

