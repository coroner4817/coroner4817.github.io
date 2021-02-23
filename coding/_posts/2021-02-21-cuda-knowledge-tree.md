---
layout: post
title: CUDA Knowledge Tree
date: 2021-02-21 06:07 -0500
description: >
  Knowledge tree of CUDA
image:
  path: "/assets/img/blog/Nvidia_CUDA_Logo.jpg"
related_posts: []
---

CUDA knowledge tree. Took notes from official documents.

* toc
{:toc .large-only}


### CUDA Version and HW (Jetson nano)
- Architecture: Maxwell (Jetson nano)
- CUDA library version: nvcc --version, CUDA 10.2
- CUDA Compute Capability: 5.3
- Kernel Driver compatibility: >= 440.33
- CUDA core (streaming processor): 1 streaming multiprocessor (SM), 128 CUDA core per SM, each core can exec a warp at a time
- Processor SoC: Tegra X1
- Embedded Sys: Jetson
- JetPack SDK: 4.4, including OS, ML CV libraries and dev tools, installed on Tegra board
- L4T: Linux for Tegra, Custom Linux dist included in JetPack
  - dpkg-query --show nvidia-l4t-core
  - cat /etc/nv_tegra_release
- Resources:
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x
  - https://en.wikipedia.org/wiki/CUDA
  - https://en.wikipedia.org/wiki/Nvidia_Jetson
  - https://en.wikipedia.org/wiki/Tegra
  - https://elinux.org/Jetson
  - https://elinux.org/Jetson_Nano
- Maxwell Physical Limitation (Sample: deviceQuery)
  - CUDA Driver Version / Runtime Version          10.2 / 10.2
  - CUDA Capability Major/Minor version number:    5.3
  - Total amount of global memory:                 3964 MBytes (4156682240 bytes)
  - ( 1) Multiprocessors, (128) CUDA Cores/MP:     128 CUDA Cores
  - GPU Max Clock rate:                            922 MHz (0.92 GHz)
  - Memory Clock rate:                             13 Mhz
  - Memory Bus Width:                              64-bit
  - L2 Cache Size:                                 262144 bytes
  - Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  - Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  - Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  - Total amount of constant memory:               65536 bytes
  - Total amount of shared memory per block:       49152 bytes
  - Total number of registers available per block: 32768
  - Warp size:                                     32
  - Maximum number of threads per multiprocessor:  2048
  - Maximum number of threads per block:           1024
  - Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  - Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
- https://docs.nvidia.com/cuda/maxwell-compatibility-guide/index.html
- https://docs.nvidia.com/deploy/cuda-compatibility/index.html
- https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html

### PTX, Compiler and Streaming Multiprocessor
- cubin: CUDA program binary
- PTX: Parallel Thread Execution, defines a GPU architecture independent virtual machine and ISA (PTX instruction set)
  - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  - Handcraft PTX: https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html
  - ISA: PTX Instruction Set Architecture, machine independent.
  - PTX program are translated to target HW Instruction Set at install time or application load time
  - CUDA code -> IR -> PTX program(ISA) -> Code gen -> Target HW instruction set
  - CTA: Cooperative Thread Array, array of parallel executed threads, aka block, max size 1024
  - Warp: maximal subset of threads from a single CTA that can be executed on a streaming processor, such that each thread executes the same instruction at the same time.
  - Warp size is always 32, no matter how many CUDA core SM contains
  - Threads within a warp also called lanes. Each theard is a lane
  - Memory Hierarchy:
    - Thread: private local memory
    - Block: shared memory
    - Global memory, constant
    - Other memory: texture, surface, notice that texture and surface memory are cache memory, which means the write before read in the same kernel call is not valid when read it
    - Host memory and device memory: copy through the Direct Memory Access engine
    - On-chip SM shared memory: 32-bit register per Scalar Processor (aka streaming processor)
  - Streaming Processor and SM
    - SM contains multiple Scalar Processor and shared memory within the SM. Also contain texture cache, etc
    - Zero overhead thread scheduling
    - Fast Barrier synchronization
    - lightweight thread creation
    - SIMT: Single-instruction, multiple-thread
      - Each thread is mapped to 1 Scalar Processor and each thread executed with its own register state
      - SIMT unit schedule the thread and split consecutive threads within the CTA into warps
      - At each instruction issue time, SIMT unit select a warp
      - A warp execute 1 common instruction of all threads at a time. But if there is code path branching, warp will hold other threads and wait for diverged threads finish. Eventually converge all threads. Different warp are executing independently
      - SIMD: Single Instruction, Multiple Data: a single instruction controls multiple processing elements.
      - SIMT vs SIMD: SIMD vector organizations expose the SIMD width to the software, whereas SIMT instructions specify the execution and branching behavior of a single thread
      - SIMD is dive into data element level parallelism, SIMT is kernel level parallelism
  - Independent Thread Scheduling: starting from Volta arch, threads can sync even the code path is diverge. Threads can now diverge and reconverge at sub-warp granularity.
- Compile command:
  - /usr/local/cuda/bin/nvcc
      -gencode=arch=compute_20,code=sm_20
      -gencode=arch=compute_30,code=sm_30
      -gencode=arch=compute_35,code=sm_35
      -gencode=arch=compute_50,code=sm_50
      -gencode=arch=compute_52,code=sm_52
      -gencode=arch=compute_52,code=compute_52
      -O2 -o mykernel.o -c mykernel.cu
  - compute_XX refers to a PTX-ISA version and sm_XX refers to a HW dependent cubin version
  - -gencode=arch=compute_35,code=sm_35 means: generate compute_35 PTX code, then compile to sm_35 arch binary. Notice that PTX code is not perserved
  - The above command compile binaries to compatibility up to 5.2
  - -gencode=arch=compute_52,code=compute_52: the last flag generate the PTX for future architecture to compile to future cubin, PTX code is perserved
  - PTX is cross HW generation, cubin only for targeting HW
  - nvcc -ptx will generate PTX assembly code
  - binary(PTX) compatibility is backward compatible
  - Start from CUDA 10, we have forward compatibility, which means we can use the old HW but upgrade the CUDA libraries and driver to 11, then we can run CUDA 11 program.
- Tuning CUDA Application for Maxwell
  - Maxwell Streaming Processor is also called SMM


### CUDA Programming High-level Guide
- Find ways to parallelize sequential code,
- Minimize data transfers between the host and the device,
- Adjust kernel launch configuration to maximize device utilization,
- Ensure global memory accesses are coalesced,
- Minimize redundant accesses to global memory whenever possible,
- Avoid long sequences of diverged execution by threads within the same warp.
- CUDA Occupancy Calculator: https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html


### CUDA C Programming Guide
- CUDA provide abstraction of: a hierarchy of thread groups, shared memories, and barrier synchronization
- Basics:
  - Function specifiers:
    - \_\_global\_\_: running on device, called on host, so that we pass host custom struct to the kernel
    - \_\_device\_\_: running on device called on device
    - \_\_host\_\_: runnint on host, can be omitted. Or can be used together with \_\_device\_\_, which means code will compiled and run for both host and device
    - \_\_noinline\_\_, \_\_forceinline\_\_
  - Variable specifiers:
    - \_\_device\_\_: device variable
    - \_\_shared\_\_: shared memory of block
    - \_\_constant\_\_: global constant device variable
    - \_\_managed\_\_: map memory between host and device, so both can directly access it
    - \_\_restrict\_\_: avoid pointer aliasing
  - kernel <<< GridSize, BlockSize, SharedMem, Stream >>> (args...)
    - GridSize: how many block in a grid
    - BlockSize: how many thread in a block
  - Templates with CUDA: https://developer.nvidia.com/blog/cplusplus-11-in-cuda-variadic-templates/
  - Max threads per block is 1024
  - Two way of launching kernel in 1D shape: M = 2014
    - <<< (N + M - 1) / M, M >>>
    - <<< (N / M + 1), M >>>
  - Usually launched threads is more than need, so need to use indexing to fence the kernel execution
  - Indexing in a 3D grid and 3D block:
    - first dim3 is the block size in a grid, second dim3 is the thread size in a block. So we can understand the whole kernel launch in a single grid
    - xxxDim is the dimenson of itself (shape of children), xxxIdx is the index of self in parent space
    - int i = (blockDim.x * blockDim.y * blockDim.z) *
          (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) +
          blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  - shared memory:
    - static shared memory: \_\_shared\_\_ int s[64]; Fix sized at compile time
    - Dynamic shared memory: Size determined at runtime
      - config memory size using the third parameter of the kernel launch and declare variable like: extern \_\_shared\_\_ int s[];
  - Barrier Sync:
    - block sync: \_\_syncthreads()
    - device sync: cudaDeviceSync()
  - Memory Model:
    - Registers: stack variables are placed in registers
    - Thread local: DRAM, slow. stack variables that is either dynamic array or variables too large to fit in register
    - Block shared memory: on chip, user manage-able L1 cache, maximum 48k per block
    - Global Memory: DRAM
      - Read-only cache
    - Read-only:
      - Const momery
      - Texture buffer
- Complier: nvcc, based on LLVM as back end
  - Can compiler both host code and device code
  - Device code will be compiled into assembly(PTX) or cubin binary
  - Just-in-time Compilation: the PTX code will be compiled at application loading time to the target machine instruction set. So this is why when compile PTX code we need to set different compute compatibility
  - Runtime Compilation: NVRTC
  - cubin, arch specific binary code. If cubin is generated for X.y, then it can only run on X.z which z>y
  - cuobjdump show the content of cubin
  - PTX code is much flexible, since it is an abstraction. we can cross compile a PTX of difference compute compatibility and then compile it to any other higher compatibility arch
  - If not compiled to cubin, then application need to load the PTX and JIT compile. For running on future arch, we must generate PTX code using flag like: -gencode=arch=compute_75,code=compute_75. In the future it can be JIT compile to 8.6 cubin during load time on 8.6 arch
  - CUDA C macro to determine arch: \_\_CUDA_ARCH\_\_
  - flag can be shorted to: -arch=compute_70
- CUDA Runtime Init
  - Runtime context is initialized on the first CUDA kernel call. cudaDeviceReset() destroy the context
- Device Memory
  - Device memory is Linear Memory layout
  - cudaMalloc*: cudaMallocHost, cudaMalloc, cudaMallocPitch (for 2D array), cudaMalloc3D
  - cudaMemcpy*: cudaMemcpy, cudaMemcpyAsync, cudaMemcpy2D, cudaMemcpy3D, cudaMemcpyToSymbol
  - L1/L2 cache: cache the global memory for fast access
    - cudaDeviceSetLimit: config the L2 cache persisting size
    - When using CUDA stream or CUDA graph kernel, we can config to make specified global memory more easy to be cached in L2
    - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#L2_simple_example
- Shared Memory
  - Improvements on the Matmul examples: (Sample: MatrixMul)
    - Standard implementation without using shared memory
      - Issue with this approach is that we have to read the same row or column from the global memory multiple times. A's row need to be loaded everytime multiply with B's column. And global memory is slow
    - Optimized memory fetch using shared memory
      - Divided the matrix into small square sub-matrix, size same as the 2D block size. If needed, padding 0s.
      - Declare two 2D shared memory arrays, each size same as the 2D block size. Each thread take responsbility of reading 1 element from both A and B to the according shared memory.
      - _syncthreads()
      - Now we can take advantage of the fast read speed of the shared memory
      - Calc the matmul of the sub-matrix and sum up to get the final result
      - _syncthreads() to make sure all threads has write sub-result to the shared memory variable, then iterate to the next sub-matrix
    - https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
- Page-locked Host memory (Sample: SimpleZeroCopy)
  - cudaHostAlloc(), cudaHostRegister()
  - Lock the memory to the physical memory, not swap to virtual memory (hard drive space), so that guarentee the app has sufficient memory allocated
  - Page lock host memory enables mapping host momery across device, eliminating copy to device
  - After cudaDeviceSynchronize(), we can directly read device output from host mapped memory
- Async Concurrent Exec (stream)
  - Cuda stream
    - Maximum number of kernel can be launched on a device is based on Compute Compatibility
    - Stream is a sequence of commands, will be exec in order. But different stream will be exec out of order
    - cudaStreamCreate(), cudaStreamDestroy()
    - stream priority: cudaStreamCreateWithPriority
  - Explicit Sync stream
    - cudaDeviceSynchronize(sync all stream), cudaStreamSynchronize(sync a single stream), cudaStreamWaitEvent(wait for a singal, for sync stream), cudaStreamQuery()
  - Implicit Sync
    - Two commands from different streams cannot run concurrently if any one of the following operations is issued in-between them by the host thread:
      - a page-locked host memory allocation,
      - a device memory allocation,
      - a device memory set,
      - a memory copy between two addresses to the same device memory,
      - ...
  - Callback Functions:
    - Function specificer: CUDART_CB
    - fire during stream at host side: cudaLaunchHostFunc()
    - cross-stream sync can be done through cudaEventRecord() signal

  - CUDA Graph:
    - created through explicit API or stream capture
    - Graph nodes structure can be updated after created
    - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs

  - Event:
    - cudaEvent_t: a signal that can be used for sync across stream

- Multi-device System
  - cudaSetDevice
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system

- Unified Virtual Address Space
  - CUDA malloc host memory or the device memory is using the unified address space

- IPC
  - Share GPU device memory pointers across process
  - Only support on Linux

- Error Checking
  - Check host API returned error code. Using Macro to write a small wrapper
  - cudaPeekAtLastError(), get error for kernel launuching
  - always checking for error at each cuda API call

- Access Texture and Surface Memory
  - CUDA can read from the Texture and Surface HW buffer, instead of reading from global memory
  - This is not OpenGL interop, which is OpenGL read CUDA output
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory

- Graphics Interoperability
  - Enable CUDA to read data written by OpenGL or Direct3D, or to enable CUDA to write data for consumption by OpenGL or Direct3D
  - cudaGraphicsGLRegisterBuffer(), cudaGraphicsMapResources(), cudaGraphicsResourceGetMappedPointer(), cudaGraphicsUnmapResources()
  - Basic flow is:
    - create an opengl interop buffer for CUDA to fill in
    - use cudaGraphicsGLRegisterBuffer to register memory mapping
    - During rendering loop, map and acquire the OpenGL buffer in cuda via: cudaGraphicsMapResources and cudaGraphicsResourceGetMappedPointer
    - Launch CUDA kernel to fill in the buffer
    - cudaGraphicsUnmapResources, unbind buffer
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#opengl-interoperability

- Version and Compatibility
  - CUDA is backward compatibile, not forward compatible
  - backward compatibile: App compiled againist old driver API will continue to working with new driver API
  - Compute Compatibility sometimes is not enforced on versioning

- Hardware Implementation
  - multithreaded Streaming Multiprocessors (SM)
  - SIMT arch has no branch prediction or speculative execution
  - Although 32 threads within a warp are sync on instruction, one thread can still branching itself and has own register state. Which lead to waiting of other thread, which is a bad thing
  - Threads within a block are always consecutive
  - Warp scheduler
  - Best practice is to make all thread within a warp under the same code path
  - Start from Volta, CUDA has the Independent Thread Scheduling, which not limited by the warp execution limitation. GPU exec at thread granularity. Scheduler will group active thread from same warp into SIMT unit. So no more bounding by the code path diverge
  - atomic instruction are serialized within a warp
  - Total number of warp need per block: ceil(threads / 32)

- Performance Guide
  - Maximize performance:
    - Maximize parallel execution to achieve maximum utilization;
    - Optimize memory usage to achieve maximum memory throughput;
    - Optimize instruction usage to achieve maximum instruction throughput.
  - Maximize Utilization
    - Application level
      - Use stream
    - Device level
      - break the algo into multiple kernel and execute concurrently
    - Multiprocessor level (Sample: simpleOccupancy)
      - More efficiently schedule the warp without too many resource waiting
      - cudaOccupancyMaxPotentialBlockSize(): automatically determine the block size, faster than dev assign block size
      - Occupancy Calculator
  - Maximize Memory Throughput
    - Minimize the data transfer between host and device
    - Minimize the data transfer between global memory, via maximuzing using of on-chip memory: shared memory, L1/L2 cache, texture cache, constant cache
    - shared memory: user managed cache
    - Shared memory workflow:
      - Load data from device memory to shared memory,
      - Synchronize with all the other threads of the block so that each thread can safely read shared memory locations that were populated by different threads,
      - Process the data in shared memory,
      - Synchronize again if necessary to make sure that shared memory has been updated with the results,
      - Write the results back to device memory.
    - Data transfer between device and host
      - Try move more code to device
      - Batching small data transfer into large data tf always better
      - Use Page-lock host memory to avoid dev calling memCpy (but it still does memcpy in background)
      - If CPU and GPU are sharing physcially memory, then definitely use page lock, because there is no memcpy at all
    - Device memory access: memory layout
      - Global Memory
        - Variable are always align to 32/64/128 bytes
        - Warp will coalesces all the memory access needed, so if the memory are consecutive, then better
        - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x
        - Padding data if necessary, e.g. padding the 2D array
        - Alignment struct:
          struct \_\_align\_\_(8) {
              float x;
              float y;
          };
        - 2D array indexing: BaseAddress + stride * ty + tx
          - For these accesses to be fully coalesced, both the width of the thread block and the width of the array must be a multiple of the warp size.
          - cudaMallocPitch() can help us padding data
      - Local memory
        - Local memory is on device mem, so same slow as global memory, this is the threadlocal memory
        - data will be placed on thread local memory by compiler:
          - Arrays for which it cannot determine that they are indexed with constant quantities,
          - Large structures or arrays that would consume too much register space,
          - Any variable if the kernel uses more registers than available
        - We can determine if a variable is on local memory through PTX code
      - Shared memory
        - onchip memory, much faster than local and global memory
        - Shared memory is divided into equally-sized memory called bank, which can be access simultaneously. memory r/w in to different banks can be concurrently.
        - Bank Conflict: if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized
          - However, if 2 threads request the same address within a bank, there is no back conflict.
          - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x\_\_examples-of-strided-shared-memory-accesses
        - cudaDeviceSetSharedMemConfig(): we can config bank size to avoid conflict
      - Constant Memory
        - On device memory with constant cache
      - Texture and surface memory
        - device memory with texture cache
        - Texture cache is optimized for 2D data
    - Maximize Instruction throughput
      - Use intrinsic functions and atomic functions
      - User single precision rather than double
      - Avoid branching
      - reduce instructions, especially sync operations
      - use built in functions in: math_functions.h, device_functions.h
      - Some nvcc compiler flags: -ftz=true, prec div=false, -prec-sqrt=false
      - Intrinsic functions: \_\_fdividef(), rsqrtf()
      - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions
      - Single precision and half precision instructions
      - Use bit operations: & | >> << when possible
    - Control Flow
      - Avoid branching, due to code path diverge and warp need to wait
      - controlling condition should be written so as to minimize the number of divergent warps
      - If all thread under a warp will behave the same on a branching condition, then there is no code diverge
      - #pragma unroll: above a loop in kernel can force unrolling
      - \_\_syncthreads() can also impact performance
    - Warmup kernel
      - instanciate a dummy kernel launch before the actual executing. This is because the first kernel launch always invlove context setup. This is critical for benchmarking
      - Warmup should comes with a cudaDeviceSynchronize()

### Other topics
- Built-in features
  - Data Types
    - int2, float4, char1..., created with API like: make_int2
  - Variables
    - gridDim, blockIdx, blockDim, threadIdx, warpSize=32
  - Memory Fence functions
    - \_\_threadfence(), \_\_threadfence_block(), \_\_threadfence_system(): make sure the write operand before this call have occured. Important for weakly-ordered memory model. This more for global variable write and read
    - Alternatively, we can use atomic functions to avoid data racing
    - Compare with \_\_syncthreads(): \_\_threadfence() only affect the memory operations. It is not sync point for all threads to wait
  - Sync functions
    - \_\_syncthreads(), \_\_syncthreas_count(), \_\_syncwarp()...
    - somtimes \_\_syncwarp() is enough instead of \_\_syncthreads, This is base on how we launch the kernel
  - Matchematical Functions
    - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix
  - Texture functions (Sample:simpleTexture)
    - access the texture in the kernel for read and write. The texture is stored in the texture cache, which is optimized for 2D data
    - using this we can speed up image processing and graphic interop, etc
    - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-functions
  - Surface Functions
    - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-functions
  - Global Memory access functions
    - Read-only cache load: \_\_ldg()
    - Load and store global memory: \_\_ldcg()..., \_\_stwb()...
  - Time functions
    - clock(), clock64()
    - \_\_nanosleep()
  - Atomic Functions: read-modify-write atomic operation in kernel
    - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    - Type of Atomic
      - System wise Atomic: atomicAdd_system (between CPU and GPU)
      - Device wise Atomic: atomicAdd
      - Block wise Atomic: atomicAdd_block
    - Any atomic operation can be implemented with compare and swap function atomicCAS()
    - Math atomic functions
    - Bitwise atomic functions
  - Address space predicate functions
    - Determine where is memory stored at
  - Compiler Optimization Hint Functions
    - \_\_builtin_assume_aligned(): hint compiler that the data's alignment
    - \_\_builtin_assume(): add some condition hint to compiler
    - \_\_builtin_expect(): hint compiler for branching
  - Warp broadcasting
    - \_\_shfl_sync
    - https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
  - Alternative Floating Point
    - \_\_nv_bfloat16, tf32
  - Assert: assert
  - Trap: \_\_trap(), abort the kernel
  - malloc: we can malloc per thread memory in kernel
- Cooperative Groups (Sample:simpleCooperativeGroups)
  - Give developer more contorl of the kernel granularity
- CUDA dynamic parallelism: fork kernel inside a kernel (Sample:cdpSimplePrint)

- Virtual Memory Management
  - Allow application resize a malloced memory through memory mapping and unmapping

- Math functions:
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions

- nvcc
  - CUDA device code is not just C98. Since nvcc front end compiler can also compile host C++ code, it can support the C++ features in device code. So we can see C++ features in device code like: auto, template, lambda...
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support

- CUDA driver API
  - we can do CUDA typed computation direct using low level driver API, instead of CUDA SDK
  - (sample:vectorAddDrv)

-Unified Memory Programming
  - By using memory mapping, both CPU and GPU can see the same memory address for a varible. So this hidden the memory copy cross host and device, or multi-GPU

Misc:
  - https://en.wikipedia.org/wiki/Sorting_network


### CUDA Best Practices Guide
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- Search for "Note:[High|Medium|Low] Priority" in the article
- Summary: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#recommendations-and-best-practices-appendix
- APOD: Assess, Parallelize, Optimize, Deploy
  - Assess: find the hotspot code path
  - Parallelize: Implement a baseline algorithm, can use supported SDK libraries
  - Optimize: A lot of iteration and profiling. Optimization can be at any level of the pipeline, even at HW level. Always first implement the higher priority optimization method before work on lower priority optimization
  - Deploy: Depoly to production in patches to minimize risk
- Heterogeneous Computing: each processing unit is leveraged to do the kind of work it does best: sequential work on the host and parallel work on the device
- What should be running on CUDA: 
  - Should consider the overhead of memory copying across processing unit
  - Memory should be consecutive if possible
  - Use shared mem or Texture cache if possible
  - Use some hint/pattern, so that GPU can coalesce memory operations into 1

- Profiling
  - gprof: profiling CPU programs
  - Programming Scaling: Define different application profile, which we need to rely on to determine how to parallelism
    - Strong Scaling: Program size is fixed, how number of processor speed up the program
    - Weak Scaling: Program size will grow to fill the number of processors

- Parallelizing
  - First check GPU-optimized libraries like cuBLAS, cuFFT, Thrust
  - Thrust is very useful for fast prototyping CUDA program
  - Or can choose to use parallel compiler like OpenAcc or OpenMP to parallel loop with compiler hint

- Testing
  - Verfication: write unit test with reference data to avoid numerical accuracy issue
  - We can use specify a device functions as \_\_host\_\_ \_\_device\_\_, so that we can unit test it on CPU

- Debugging
  - CUDA-GDB 
  - NVIDIA Nsight

- Numerical Accuracy
  - when comparison, always add epsilon. 

- Performance Metrics
  - Use CPU timer with cudaDeviceSynchronize, cudaStreamSynchronize, cudaEventSynchronize. These stall call might reduce performance
  - Use GPU timer: via cudaEventRecord to signaling
  - Memory Bandwidth:  
    - Bandwidth to access global memory, we can calculate a theoretical value using HW spec
    - Effective bandwidth: bandwidth calc using runtime data
  - Memory Optimization
    - Data transfer between Host and device: bandwidth is much less than GPU memory bandwidth. So we should avoid memcpy if possible
    - Also when memcpy cross devices, we can batch small memcpy to a single memcpy
    - Use page-lock pinned memory mapping between host and device
    - Use Async memory transfer, when use cudaMemcpyAsync, need to use with stream together
    - Also with Async kernel launch, we can do task on CPU at the meanwhile.
    - Overlapping Computing: We can have multiple streams and they can parallel both CPU memcpyToDevice and GPU kernel launch tasks
    - Zero-copy: Via memory mapping, we can avoid memcpy, but also means memory cannot be cached at the GPU side
    - Unified Virtual Addressing: a new way of zero-copying
    - Different Memory Attributes: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces\_\_salient-features-device-memory
    - Coalecse Global Memory access pattern
      - GPU will coalecse adjacent memory access
      - GPU will padding access non-aligned memory 
      - Misaligned memory access: transform the thread ID so that ID and the task is not really matched
        - offset: int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
        - stride: int xid = (blockIdx.x*blockDim.x + threadIdx.x)*stride;
    - L2 Cache: config persistence of data in L2 cache
    - Shared Memory:
      - helping to coalesce or eliminate redundant access to global memory
      - Example: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab
    - Async memCpy from global memory to shared memory
      - memcpy_async
    - Thread local memory: aka off-chip global memory
    - Texture Memory: read-only
    - Constant Memory: read-only
    - Registers

- Exection Configuration Optimization
  - Occupancy: the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps
  - Rules to launch kernel:
    - The number of threads per block should be a multiple of 32 threads
    - Threads per block should be a multiple of warp size to avoid wasting computation on under-populated warps and to facilitate coalescing.
    - A minimum of 64 threads per block should be used, and only if there are multiple concurrent blocks per multiprocessor.
    - Between 128 and 256 threads per block is a good initial range for experimentation with different block sizes.
    - Use several smaller thread blocks rather than one large thread block per multiprocessor if latency affects performance. This is particularly beneficial to kernels that frequently call \_\_syncthreads().

  - Multiple GPU Context:
    - Context created when first kernel launch
    - Only 1 context can be exec at a time
    - cuCtxPushCurrent(), cuCtxPopCurrent()...
  
- Instruction Optimization
  - Single-precision floats provide the best performance
  - Use shift operations to avoid expensive division and modulo calculations.
  - Use signed integers rather than unsigned integers as loop counters.
  - Use rsqrtf for square root
  - Avoid automatic conversion of doubles to floats.
  - https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#exponentiation-small-fractions
  - use CUDA math library
  - Minimize the use of global memory. Prefer shared memory access where possible.


- Control Flow
  - Avoid different execution paths within the same warp
  - Avoid (if, switch, do, for, while)
  - Make it easy for the compiler to use branch predication in lieu of loops or control statements.
  - Use #pragma unroll

- Deploy CUDA: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#deploying-cuda-applications
- CUDA upgrade: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#cuda-compatibility-and-upgrades
- Deployment: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#preparing-for-deployment


### Other notes
- Other HPC framework
  - OpenCL: standard for all CPU and GPU
  - Thread level: OpenMP, TBB
  - Vector level: SIMD
- CUDA dev tools:
  - nvcc
  - ptxas: assembly code
  - cuobjdump
  - nvidia-smi
  - nsight
  - nvvp
  - nvprof
- CUDA princples:
  - Maximum Computing intensity
  - Decrease time on memory operations
  - Coalesce global memory access
  - Access adjcent memory and avoid bank conflict
  - avoid thread divergence branching
  - Avoid inbalance workload across threads
  - Pick good algorithms
  - Cache-aware memory efficiency
  - arch specific optimization
  - instructions level optimization
  - Use low precision data like GEMM
- Optimization ideas:
  - Tile: break big problem size into small tiles. Computing individually, then converge
  - Each tile is a block
  - Template:
    - Each thread in the block copy data to shared memory
    - \_\_syncthreads()
    - Read data and operations
    - Force Coalesce
- How to Coalesce threads' result:
  - If doing primitive binary merge like merge sort, then eventually the first thread has too much workload
  - We can merge threads that has a certain stride, so the work is more balance


### Resources
- CUDA toolkit documentation (https://docs.nvidia.com/cuda/)
- Best Practices Guide CUDA (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- Programming guide CUDA (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- CUDA samples (https://docs.nvidia.com/cuda/cuda-samples/index.html)
- CUDA libraries and 3rd party lib: https://developer.nvidia.com/gpu-accelerated-libraries
- https://docs.nvidia.com/cuda/cuda-runtime-api/
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- https://docs.nvidia.com/deploy/cuda-compatibility/index.html
- https://nvidia.github.io/libcudacxx/