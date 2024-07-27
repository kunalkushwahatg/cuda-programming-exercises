//Memory banks are divisions within the shared memory that allow multiple threads to access memory simultaneously without interference.

//A bank conflict occurs when multiple threads in the same warp (a group of 32 threads) access the same memory bank simultaneously.
// This forces the accesses to be serialized, reducing parallelism and performance.
//To avoid bank conflicts, threads should access different memory banks or structure their access patterns to minimize conflicts.
//Shared memory is divided into 32 banks, so bank conflicts can occur when multiple threads access the same bank.
//The number of banks is determined by the hardware and cannot be changed.

//these types of memory accloactions causes bank conflicts
//1. Non-coalesced access patterns: When threads access memory in a non-coalesced pattern, it can lead to bank conflicts.
    //non-coalesced pattern is when threads access memory in a non-sequential order, such as accessing every other element in an array.
    //eg. data[threadIdx.x] = int(threadIdx.x);
//2. Strided access patterns: When threads access memory with a fixed stride, it can lead to bank conflicts.
    //strided pattern is when threads access memory with a fixed stride, such as accessing every 2nd, 3rd, or 4th element in an array.
    //eg. sharedMem[idx] = idx * 2.0f

//To avoid bank conflicts, threads should access different memory banks or structure their access patterns to minimize conflicts.
//For example, using a coalesced access pattern where threads access memory in a sequential order can help avoid bank conflicts.



//Texture Memory : Texture memory is a read-only memory that is optimized for 2D spatial locality and can be accessed by threads in a warp.
//Texture memory is cached and provides fast access to data with spatial locality, such as images or textures.
//define : texture<float, 2, cudaReadModeElementType> texRef;
//in main: cudaBindTexture(0, texRef, data, size);
//in kernel: float value = tex2D(texRef, x, y);

//how to use
//1. Define a texture reference variable with the desired data type, dimensions, and read mode . 
    //eg. texture<float, 2, cudaReadModeElementType> texRef;
//2. Bind the texture reference to a device memory array using cudaBindTexture.
    //eg. cudaBindTexture(0, texRef, data, size);
    
//3. Access the texture memory in the kernel using the tex2D function. 
    //eg. float value = tex2D(texRef, x, y);


