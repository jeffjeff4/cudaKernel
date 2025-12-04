#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int x = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int sum = 0;
    if (x<N) {
        int t0 = input[x+0];
        int t1 = input[x+1];
        int t2 = input[x+2];
        int t3 = input[x+3];
    
        if (t0==K) ++sum;
        if (t1==K) ++sum;
        if (t2==K) ++sum;
        if (t3==K) ++sum;
        
        if (sum>0) {
            atomicAdd(output, sum);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}