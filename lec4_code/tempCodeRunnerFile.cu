#include <stdio.h>

#define N 4
#define THREADS_PER_BLOCK 4

__global__ void add(int *a , int *b , int *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    //declare memory in shared memory
    //data shared among all threads in a block
    //data is not shared among blocks
    __shared__ int temp[N];

    temp[index] = a[index] * b[index];

    //make sure that all earlier values in diffrent threads are executed before the next value is executed
    __syncthreads();
    if(index==0){
        int sum = 0;
        for(int i=0 ; i<N ; i++){
            sum += temp[i];
        }
        printf("\n %d \n ",sum);

        *c = sum;
    }
     
}



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

    printf("Enter the first array\n");
    for(int i=0 ; i<4 ; i++){
        scanf("%d",&a[i]);
    }


    printf("Enter the second array\n");
    for(int i=0 ; i<4 ; i++){
        scanf("%d",&b[i]);
    }

    cudaMemcpy(dev_a ,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b ,b,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c ,c,size,cudaMemcpyHostToDevice);

    add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_a,dev_b,dev_c);

    cudaMemcpy(c,dev_c,size,cudaMemcpyDeviceToHost);

    printf("dot product is = %d",*c);

    free(a);free(b);free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    return 0;
}