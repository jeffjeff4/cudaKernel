#include <cuda_runtime.h>

//method0
//error
/*
inline __device__ __host__ unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1) / b;}

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define STRIDE_FACTOR 8
#define BLOCK_SIZE STRIDE_FACTOR * THREADS_PER_BLOCK

__global__ void init_output(float* output) {
    *output = 0.0f;
}

__device__ void warp_reduce(volatile float* smem, unsigned int tid) {
    smem[tid] += smem[tid+32];
    smem[tid] += smem[tid+16];
    smem[tid] += smem[tid+8];
    smem[tid] += smem[tid+4];
    smem[tid] += smem[tid+2];
    smem[tid] += smem[tid+1];
}

// Make sure to delete the old __device__ void warp_reduce function, it's not needed.

__global__ void reduction_kernel(const float* input, float* output, int N) {
    // 1. Use 'double' for precision and declare as __shared__ to fix the crash
    __shared__ double smem[THREADS_PER_BLOCK];
    
    auto tid = threadIdx.x;
    auto block_start = blockIdx.x * BLOCK_SIZE;
    
    // 2. Use 'double' for the local sum
    double sum = 0.0;

    // 3. Load data and sum into the 'double' variable
    if (block_start+tid < N) {
        sum = (double)input[block_start + tid];
    }
    for (int i=1; i<STRIDE_FACTOR; ++i) {
        auto idx = block_start + i * THREADS_PER_BLOCK + tid;
        if (idx<N) {
            sum += (double)input[idx];
        }
    }
    
    // 4. Write the high-precision sum to shared memory
    smem[tid] = sum;
    __syncthreads();
    
    // 5. Use the single, safe, correct reduction loop.
    //    This loop has no race conditions and replaces 'warp_reduce'.
    for (int stride = THREADS_PER_BLOCK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // 6. The final block sum is in smem[0]. Add it to the output.
    if (tid == 0) {
        atomicAdd(output, (float)smem[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 8 - 1) / (threadsPerBlock * 8);
    //cudaMemset(output, 0, sizeof(float));
    init_output<<<1, 1>>>(output);

    reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

    cudaDeviceSynchronize();
}
//*/


//method1
/*
__global__ void reduce(const float* input, float* output, int N) {
    __shared__ float smem[256];
    
    int local_tid = threadIdx.x;

    float local_sum = 0.0;
    for (int f = 0; f < 8; f++) {
        int global_idx = local_tid + f * blockDim.x + blockIdx.x * 8 * blockDim.x;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    smem[local_tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            smem[local_tid] += smem[local_tid + s];
        }
        __syncthreads();
    }
    
    if (local_tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 8 - 1) / (threadsPerBlock * 8);

    reduce<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
*/

//method2
//wrong, never run to generate results
// ------------------------------------------------------------
//  CUDA sum-reduction (N up to 2^31-1, any float values)
//  - exact (no atomic-float rounding)
//  - works on every GPU (SM 3.0+)
//  - single kernel launch
// ------------------------------------------------------------

// ------------------------------------------------------------
//  CUDA sum-reduction – exact, single-kernel, works on CPU/GPU
// ------------------------------------------------------------

// ------------------------------------------------------------
//  CUDA sum-reduction – exact, single-kernel, works on CPU/GPU
// ------------------------------------------------------------

// ------------------------------------------------------------
//  CUDA sum-reduction – exact, fast, no hang, N ≤ 2^31-1
// ------------------------------------------------------------
/*
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

inline __device__ __host__ unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

#define WARP_SIZE        32
#define THREADS_PER_BLOCK 256
#define STRIDE_FACTOR    8
#define BLOCK_SIZE       (STRIDE_FACTOR * THREADS_PER_BLOCK)  // 2048

// ---------- float <-> two 32-bit ints (device) ----------
__device__ __forceinline__ void float_to_bits(float f, uint32_t* hi, uint32_t* lo) {
    uint32_t u = __float_as_uint(f);
    *hi = u >> 16;
    *lo = (u << 16) | 0x8000u;
}

__device__ __forceinline__ float bits_to_float(uint32_t hi, uint32_t lo) {
    uint32_t u = (hi << 16) | (lo & 0xFFFFu);
    return __uint_as_float(u);
}

// ---------- host version ----------
static inline float bits_to_float_host(uint32_t hi, uint32_t lo) {
    uint32_t u = (hi << 16) | (lo & 0xFFFFu);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------- warp reduction ----------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------- kernel ----------
__global__ void reduction_kernel(const float* input,
                                 uint32_t* out_hi,
                                 uint32_t* out_lo,
                                 int N)
{
    __shared__ float smem[WARP_SIZE];

    int tid  = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;
    int block_start = blockIdx.x * BLOCK_SIZE;

    float sum = 0.0f;

    // Load STRIDE_FACTOR elements per thread
    #pragma unroll
    for (int i = 0; i < STRIDE_FACTOR; ++i) {
        int idx = block_start + i * THREADS_PER_BLOCK + tid;
        if (idx < N) sum += input[idx];
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) smem[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float warp_sum = (lane < (THREADS_PER_BLOCK / WARP_SIZE)) ? smem[lane] : 0.0f;
        warp_sum = warp_reduce_sum(warp_sum);
        if (lane == 0) {
            uint32_t hi, lo;
            float_to_bits(warp_sum, &hi, &lo);
            atomicAdd(out_hi, hi);
            atomicAdd(out_lo, lo);
        }
    }
}

// ------------------------------------------------------------
//  Host wrapper – FIXED grid size, no hang
// ------------------------------------------------------------
extern "C" void solve(const float* input, float* output, int N)
{
    uint32_t *d_hi = nullptr, *d_lo = nullptr;
    cudaMalloc(&d_hi, sizeof(uint32_t));
    cudaMalloc(&d_lo, sizeof(uint32_t));
    cudaMemset(d_hi, 0, sizeof(uint32_t));
    cudaMemset(d_lo, 0, sizeof(uint32_t));

    // FIXED: Use 64-bit math to avoid overflow
    unsigned long long elements_per_block = BLOCK_SIZE;
    unsigned long long num_blocks = (static_cast<unsigned long long>(N) + elements_per_block - 1) / elements_per_block;

    // Clamp to int (CUDA grid size is int)
    int blocks = (num_blocks > INT_MAX) ? INT_MAX : static_cast<int>(num_blocks);

    reduction_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, d_hi, d_lo, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // Optional: print error
        return;
    }

    uint32_t h_hi, h_lo;
    cudaMemcpy(&h_hi, d_hi, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_lo, d_lo, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    *output = bits_to_float_host(h_hi, h_lo);

    cudaFree(d_hi);
    cudaFree(d_lo);
}
*/


// ------------------------------------------------------------
//  CUDA sum-reduction – exact, fast, no hang, N ≤ 2^31-1
// ------------------------------------------------------------
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

inline __device__ __host__ unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

#define WARP_SIZE        32
#define THREADS_PER_BLOCK 256
#define STRIDE_FACTOR    8
#define BLOCK_SIZE       (STRIDE_FACTOR * THREADS_PER_BLOCK)  // 2048

// ---------- float <-> two 32-bit ints (device) ----------
__device__ __forceinline__ void float_to_bits(float f, uint32_t* hi, uint32_t* lo) {
    uint32_t u = __float_as_uint(f);
    *hi = u >> 16;
    *lo = (u << 16) | 0x8000u;
}

__device__ __forceinline__ float bits_to_float(uint32_t hi, uint32_t lo) {
    uint32_t u = (hi << 16) | (lo & 0xFFFFu);
    return __uint_as_float(u);
}

// ---------- host version ----------
static inline float bits_to_float_host(uint32_t hi, uint32_t lo) {
    uint32_t u = (hi << 16) | (lo & 0xFFFFu);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------- warp reduction ----------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------- kernel ----------
__global__ void reduction_kernel(const float* input,
                                 uint32_t* out_hi,
                                 uint32_t* out_lo,
                                 int N)
{
    __shared__ float smem[WARP_SIZE];

    int tid  = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;
    int block_start = blockIdx.x * BLOCK_SIZE;

    float sum = 0.0f;

    // Load STRIDE_FACTOR elements per thread
    #pragma unroll
    for (int i = 0; i < STRIDE_FACTOR; ++i) {
        int idx = block_start + i * THREADS_PER_BLOCK + tid;
        if (idx < N) sum += input[idx];
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) smem[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float warp_sum = (lane < (THREADS_PER_BLOCK / WARP_SIZE)) ? smem[lane] : 0.0f;
        warp_sum = warp_reduce_sum(warp_sum);
        if (lane == 0) {
            uint32_t hi, lo;
            float_to_bits(warp_sum, &hi, &lo);
            atomicAdd(out_hi, hi);
            atomicAdd(out_lo, lo);
        }
    }
}

// ------------------------------------------------------------
//  Host wrapper – FIXED grid size, no hang
// ------------------------------------------------------------
extern "C" void solve(const float* input, float* output, int N)
{
    uint32_t *d_hi = nullptr, *d_lo = nullptr;
    cudaMalloc(&d_hi, sizeof(uint32_t));
    cudaMalloc(&d_lo, sizeof(uint32_t));
    cudaMemset(d_hi, 0, sizeof(uint32_t));
    cudaMemset(d_lo, 0, sizeof(uint32_t));

    // FIXED: Use 64-bit math to avoid overflow
    unsigned long long elements_per_block = BLOCK_SIZE;
    unsigned long long num_blocks = (static_cast<unsigned long long>(N) + elements_per_block - 1) / elements_per_block;

    // Clamp to int (CUDA grid size is int)
    int blocks = (num_blocks > INT_MAX) ? INT_MAX : static_cast<int>(num_blocks);

    reduction_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, d_hi, d_lo, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // Optional: print error
        return;
    }

    uint32_t h_hi, h_lo;
    cudaMemcpy(&h_hi, d_hi, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_lo, d_lo, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    *output = bits_to_float_host(h_hi, h_lo);

    cudaFree(d_hi);
    cudaFree(d_lo);
}