//with the help of this code we will show that threads are randomly and executed not sequentially 
#include <stdio.h>

#define BLOCKSIZE 10

__global__ void kernel(){

    __shared__ char str[BLOCKSIZE];

    str[threadIdx.x] = threadIdx.x + 'A';
    __syncthreads();

    if(threadIdx.x == 0){
        printf("BlOCK %d: %s\n", blockIdx.x, str);
    }

}

int main(){

    kernel<<<BLOCKSIZE, 26>>>();
    cudaDeviceSynchronize();
    

    return 0;
}