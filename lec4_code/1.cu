#include <stdio.h>

__global__ void add(int *a , int *b , int *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
    printf("%d \t",c[index]);
}

#define N (20*20)
#define THREADS_PER_BLOCK 5

int main(){
    int *a , *b , *c; //host copies of a , b, c
    int *dev_a , *dev_b , *dev_c; //device copies of a , b ,c 

    int size = N*sizeof(int);

    //allocate memory in gpu 
    cudaMalloc((void**)&dev_a,size);
    cudaMalloc((void**)&dev_b,size);
    cudaMalloc((void**)&dev_c,size);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    *a = 2;
    *b = 7;

    a[1] = 3;
    b[1] = 5;

    cudaMemcpy(dev_a ,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b ,b,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c ,c,size,cudaMemcpyHostToDevice);

    add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_a,dev_b,dev_c);

    cudaMemcpy(c,dev_c,size,cudaMemcpyDeviceToHost);

    printf("sum=%d",*c);
    free(a);free(b);free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    return 0;
}