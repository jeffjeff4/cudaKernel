#include <cuda_runtime.h>

__global__ void dotKernel(const float* A, const float* B, float* result, int N) {
    __shared__ extern float local_sum[];
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= N) {
        return;
    }
    float tmp = A[tid] * B[tid];
    local_sum[0] = 0.0;
    __syncthreads();

    atomicAdd(result, tmp);
    __syncthreads();

    if (threadIdx.x != 0) {
        atomicAdd(result, local_sum[0]);
    }
}


__global__ void naiveDotKernel(const float* A, const float* B, float* result, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= N) {
        return;
    }
    float tmp = A[tid] * B[tid];
    atomicAdd(result, tmp);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    const int num_threads = 32;
    const int num_blocks = (N + num_threads - 1) / num_threads;

    //naiveDotKernel<<<num_blocks, num_threads, sizeof(float)>>>(A, B, result, N);
    dotKernel<<<num_blocks, num_threads, sizeof(float)>>>(A, B, result, N);

}