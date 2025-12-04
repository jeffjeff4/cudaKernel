#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float tmp0 = 0.0f;
    float tmp1 = 0.0f;

    for (int i=tid; i < halfN; i+=(gridDim.x * blockDim.x) ) {
        tmp0 = input[i];
        tmp1 = input[i+halfN];
        tmp0 = tmp0 / (1.0f + __expf(-tmp0));
        tmp0 *= tmp1;
        output[i] = tmp0;
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}