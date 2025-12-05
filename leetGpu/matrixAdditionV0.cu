#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    const auto tid = threadIdx.x;
    const auto id = blockDim.x * blockIdx.x + tid;

    if (id >= N) {
        return;
    }
    C[id] = A[id] + B[id];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N*N);
    // cudaDeviceSynchronize();
}