/*

-------------------------------------------------MAKEFILE-------------------------------------------------

MakeFile : A Makefile is a special file used by the make tool to automate the process of compiling and linking programs.
It simplifies building complex projects by specifying how to compile and link the source code into an executable program


-------------------------------------------------CUDA ERRORS-------------------------------------------------

cudaSucess = 0 : No errors
cudaErrorMissingConfiguration = 1
cudaErrorMemoryAllocation = 2
cudaErrorInitializationError = 3
cudaErrorLaunchFailure = 4
cudaErrorPriorLaunchFailure = 5
cudaErrorLaunchTimeout = 6
cudaErrorLaunchOutOfResources = 7
cudaErrorInvalidDeviceFunction = 8
cudaErrorInvalidConfiguration = 9
cudaErrorInvalidDevice = 10
cudaErrorInvalidValue = 11
cudaErrorInvalidPitchValue = 12

-------------------------------------------------CUDA DEBUGGING-------------------------------------------------

cuda-gdb

generate debugging information  
-nvcc -g -G file.cu -o file
-disable optimization

run with cuda-gdb
-cuda-gdb ./file
-run



due to lot of threads , cuda-gdb works with a focous(current thread)


nvcc -g -G lec13codes/cuda_errors.cu -o lec13codes/cuda_errors
cuda-gdb ./lec13codes/cuda_errors


-------------------------------------------------CUDA PROFILING-------------------------------------------------


Profiling  : Measure the indicators of performance 
-time taken by the kernel
-time taken by the memory transfer
-memory usage
-number of chache misses
-degress of divergence 
-degree of coalescing

types of profiler :
nvprof : command line profiler
nvvp,nsight : visual profiler


to use profiler :
run code : nvcc -G -lineinfo -o my_cuda_app my_cuda_app.cu
run profiler : nvprof ./my_cuda_app
advance profiling : nnvprof --print-gpu-trace ./my_cuda_app

The NSight Systems (nsys) profiler is a performance analysis tool developed by NVIDIA.
It is designed to provide comprehensive profiling capabilities for applications running on NVIDIA GPUs, including detailed insights into both CPU and GPU performance.
The primary purpose of NSight Systems is to help developers understand and optimize the performance of their applications by identifying bottlenecks, inefficiencies, and areas for improvement.

https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#:~:text=NVIDIA%20Nsight%20Compute%20is%20an,compare%20results%20within%20the%20tool.

you can profile part of code and also save the profile data in a file csv

more info available at nvidea website



*/

#include <stdio.h>
#include <cuda_runtime.h>

//hello world prininting kernel

__global__ void K(){
    printf("Hello World from thread %d\n", threadIdx.x);
}

int main(){
    K<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}

