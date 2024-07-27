#include <stdio.h>

#define N 8
#define THREADS_PER_BLOCK 4

/*
race condition : race condition occurs when program behaviour depend upon relative timing of two or more event sequences
lets say a code in device memory is like *c += sum what actally happens here is 
read(reads 0) - > compute(sum = 0+1) -> write(writes 1) 
but if two diffrent blocks are executing the same code then the following can happen
block1 reads 0 -> block2 reads 0 -> block1 computes sum = 0+1 -> block2 computes sum = 0+1 -> block1 writes 
1 -> block2 writes 1 
*/

/*
ATOMIC OPERATIONS : atomic operations are operations that are executed as a single unit of 
                    work without the possibility of interruption. It ensures that read commpute and write are executed
                    at sime sime withocut any interruption of other blocks.

atomicAdd : atomicAdd is a function that adds the value of the second argument to the value pointed by the first argument
            and stores the result in the first argument
            atomicAdd(int* address , int val)
            atomicAdd is a atomic operation and is used to avoid
atomicCAS : atomicCAS is a function that compares the value of the first argument with the value of the second argument
            if the values are equal then the value of the third argument is stored in the first argument
            atomicCAS(int* address , int compare , int val)

atomicSub : atomicSub is a function that subtracts the value of the second argument from the value pointed by the first argument
            and stores the result in the first argument
            atomicSub(int* address , int val)

atomicMin : atomicMin is a function that compares the value of the first argument with the value of the second argument
            if the value of the first argument is greater than the value of the second argument then the value of the second argument is stored in the first argument
            atomicMin(int* address , int val)

atomicMax : atomicMax is a function that compares the value of the first argument with the value of the second argument
            if the value of the first argument is less than the value of the second argument then the value of the second argument is stored in the first argument
            atomicMax(int* address , int val)

atomicInc : atomicInc is a function that increments the value of the first argument by 1 and stores the result in the first argument
            atomicInc(int* address , int val)

atomicDec : atomicDec is a function that decrements the value of the first argument by 1 and stores the result in the first argument
            atomicDec(int* address , int val)

for more info visit https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions


*/

__global__ void add(int *a , int *b , int *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    //declare memory in shared memory
    //data shared among all threads in a block
    //data is not shared among blocks
    __shared__ int temp[THREADS_PER_BLOCK];

    temp[threadIdx.x] = a[index] * b[index];

    //make sure that all earlier values in diffrent threads are executed before the next value is executed
    __syncthreads();
    if(index==0){
        int sum = 0;
        for(int i=0 ; i<THREADS_PER_BLOCK ; i++){
            sum += temp[i];
        }
        printf("\n %d \n ",sum);

        atomicAdd(c,sum);
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
    for(int i=0 ; i<N ; i++){
        scanf("%d",&a[i]);
    }


    printf("Enter the second array\n");
    for(int i=0 ; i<N ; i++){
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