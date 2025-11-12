//method0
/*
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff
#define BLOCK 32

__global__ void matrixVecMultiply(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i=tid; i<N; i+=BLOCK) {
        sum += A[row*N +i] * x[i];
    }
    for (int i = BLOCK/2; i>0; i>>=1) {
        sum += __shfl_down_sync(FULL_MASK, sum, i);
    }
    if (tid == 0) {
        y[row] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    matrixVecMultiply<<<M, BLOCK>>>(A, x, y, M, N, nnz);
} 
*/

//---------------------------------------------------------------------------------------------------
//method1
///*
#include <cuda_runtime.h>

template<int WARP_SIZE = 32>
__forceinline__ __device__ float warpReduceSum(float val) {
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
        
__global__ void matrixVecMultiply(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int N, int nnz) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.x + blockDim.x * ty;

    int bx = blockIdx.x;
    const int WARP_SIZE = 32;
    int lane = tid % WARP_SIZE;
    int m = blockDim.y * bx + ty;
    const int pack_size = 4;
    int pack_num = N / pack_size;
    int pack_off = pack_size * pack_num;

    float4* a_f4_ptr = (float4*)(A + m * N);
    float4* x_f4_ptr = (float4*)x;

    if (m < M) {
        float val = 0.0f;

        #pragma unroll
        for (int i = tx; i < pack_num; i += blockDim.x) {
            float4 a_float4 = *(a_f4_ptr + i);
            float4 x_float4 = *(x_f4_ptr + i);

            val += a_float4.x * x_float4.x + 
                   a_float4.y * x_float4.y + 
                   a_float4.z * x_float4.z + 
                   a_float4.w * x_float4.w;
        }

        #pragma unroll
        for (int i = pack_off + tx; i < N; i += blockDim.x) {
            val += A[m * N + i] * x[i];
        }
        val = warpReduceSum<WARP_SIZE>(val);
        if (lane == 0) {
            y[m] = val;
        }
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    dim3 grid((M+3) / 4);
    dim3 block(32, 4);
    matrixVecMultiply<<<grid, block>>>(A, x, y, M, N, nnz);
} 
//*/