#include <stdio.h>
#define BLOCKSIZE 36

__global__ void dkernel(){
    __shared__ int data[BLOCKSIZE];
    data[threadIdx.x] = int(threadIdx.x);
    if (threadIdx.x == 0){
        for(int i=0 ; i<BLOCKSIZE; i++){
            printf("%d ",data[i]);
        }
    }
}

int main(){

    //total memory = 64kb
    // cudaFuncCachePreferNone       /**< Default function cache configuration, no preference */
    // cudaFuncCachePreferShared  /**< Prefer larger shared memory and smaller L1 cache  */
    // cudaFuncCachePreferL1      /**< Prefer larger L1 cache and smaller shared memory */
    // cudaFuncCachePreferEqual     /**< Prefer equal size L1 cache and shared memory */


    cudaFuncSetCacheConfig(dkernel,cudaFuncCachePreferL1); //atllocate more size for l1 
    dkernel<<<1,BLOCKSIZE>>>();
    cudaDeviceSynchronize();

    return 0;
}