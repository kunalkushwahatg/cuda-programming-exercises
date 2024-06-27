#include <stdio.h>
#include <cuda_runtime.h> // Add missing import

// Kernel function that performs addition
__global__ void add(int *f , int *g, int *h){
    *h = *f + *g;  // Adds a and b, stores result in the memory location pointed to by c
}

int main(){
    int *da , *db , *dc;        // Variable to store the result on the host
    int a = 2 , b = 7;
    int c;

    //da db and dc are pointers to cpu memory now
    cudaMalloc((void**)&da , sizeof(int));
    cudaMalloc((void**)&db , sizeof(int));
    cudaMalloc((void**)&dc , sizeof(int));
    //da db and dc are pointers to gpu memory now

    //copying the values of a and b to the gpu memory
    cudaMemcpy(da , &a , sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(db , &b , sizeof(int) , cudaMemcpyHostToDevice);

    //passisng the values of a ie da  and b ie db to the gpu memory and performing the addition
    add<<<1,1>>>(da, db , dc);

    //copying the result from gpu memory to cpu memory
    cudaMemcpy(&c , dc , sizeof(int) , cudaMemcpyDeviceToHost);

    //printing the result
    printf("%d + %d = %d\n",a,b,c);


    return 0;
}