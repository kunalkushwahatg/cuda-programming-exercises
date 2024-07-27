//dynamic parallelism

/*
useful in senarios involving recursive algorithms neseted parallelism
algorithm uses hierarchical data structures



---------------------------------------------Muli gpu---------------------------------------------
for one by one : firstly 0 device then 1 device
cudaSetDevice(0);
k1<<<1,1>>>();
cudaSetDevice(1);
k2<<<1,1>>>();
cudaMecpy()

parallelization perspective :
cudaSetDevice(0);
k1<<<1,1>>>();
cudaMemcpyAsync()
cudaSetDevice(1);
k2<<<1,1>>>();
cudaMemcpyAsync()


some apis are:
cudaGeetDeviceCount() - returns the number of devices
cudaDeviceCanAccessPeer(&) - it means one device cannot acess the memory of another device
cudaDeviceEnablePeerAccess() - 
        it means one device can access the memory of another device
        max 8 devices can be accessed by one device it needs 64 bit pointer        


------------------------------------------Enumerate Devices------------------------------------------ 
int deviceCount;
cudaGetDeviceCount(&deviceCount);
for(int i=0;i<deviceCount;i++){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,i);
    printf("Device %d has compute capability %d.%d.\n",i,prop.major,prop.minor);
}


---------------------------------------------Wrap Voting---------------------------------------------
__all(predicate) - returns true if all threads in the block evaluate predicate to true
In CUDA, __all() is an intrinsic function used within a kernel to perform a warp-wide reduction of a predicate.
 It checks if the predicate is true for all threads in a warp and returns a boolean result.
  Specifically, __all(predicate) will return true if the predicate is true for every thread in the warp and false otherwise.


__any(predicate) - returns true if any thread in the block evaluates predicate to true
__ballot(predicate) - returns a 32-bit integer in which each bit is set to the result of the corresponding thread in the block evaluating predicate





*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void k1(){
    unsigned val = __all(threadIdx.x<100);
    if (threadIdx.x % 32 == 0){
        printf("Thread %d: %d\n",threadIdx.x,val);
    }
}
int main(){
    k1<<<1,128>>>();
    cudaDeviceSynchronize();
    return 0;
}

//output

// Thread 0: 1
// Thread 32: 1
// Thread 64: 1
// Thread 96: 0


//if we replace __all with __ballot then output will be
