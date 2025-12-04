#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N/2) {
        float a = input[idx];

        input[idx] = input[N-1-idx];
        input[N-1-idx] = a;
    }

}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}