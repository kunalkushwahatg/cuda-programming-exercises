#include <stdio.h>
#include <cuda.h>

/*
Global Memory
    Speed: Relatively slow, high latency.
    Memory: Large capacity.
    Code: Accessible by all threads.
    Extra Functionalities: No caching by default.

Texture Memory
    Speed: Faster than global due to caching.
    Memory: Read-only for kernels.
    Code: Optimized for 2D spatial locality.
    Extra Functionalities: Supports linear interpolation.

Constant Memory
    Speed: Cached, faster than global.
    Memory: Limited to 64KB.
    Code: Read-only for kernels.
    Extra Functionalities: Broadcasts data efficiently.

L1 Cache
    Speed: Very fast.
    Memory: Configurable size (up to 48KB).
    Code: Caches global/local memory.
    Extra Functionalities: Shared with shared memory.

Shared Memory
    Speed: Fast, low latency.
    Memory: Limited, up to 96KB per block.
    Code: Accessible by threads in a block.
    Extra Functionalities: Ideal for cooperation between threads.

Registers
    Speed: Fastest memory type.
    Memory: Limited per thread.
    Code: Private to each thread.
    Extra Functionalities: Directly mapped to hardware registers.

BANDWIDTH : 
Bandwidth is the data transfer rate, measured in bytes per second,
 between memory and processing units in a computer system.

Registers > Shared Memory > L1 Cache > Texture Memory > Constant Memory > Global Memory

LATENCY : 
Latency is the time delay between initiating a data transfer 
        request and the start of the actual data transfer.

Registers < Shared Memory < L1 Cache < Texture Memory < Constant Memory < Global Memory


LOCALITY : 
Locality refers to the use of data elements within relatively close storage locations in a computer system. 
 
Temporal Locality:  Repeated access to the same memory locations over short time periods.
Spatial Locality: Accessing memory locations that are close to each other.




*/








__device__ unsigned dfun(unsigned id){
    printf("%d\n",id);
    if(id>10 && id<15) return dfun(id+1);
    else return 0;
}

__global__ void dkernel(unsigned n){
    dfun(n);
}

#define BLOCKSIZE 256
int main(){
    unsigned n = 11;
    dkernel<<<1,BLOCKSIZE>>>(n);
    cudaDeviceSynchronize();

    return 0;
}
