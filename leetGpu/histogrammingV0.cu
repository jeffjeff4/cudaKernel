#include <cuda_runtime.h>

const int threads = 512;
const int coarse_factor = 16;

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        sdata[i] = 0;
    }
    __syncthreads();

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(&sdata[input[i]], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], sdata[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int blocks = (N + threads * coarse_factor - 1) / (threads * coarse_factor);
    size_t sdata_size = num_bins * sizeof(int);
    histogram_kernel<<<blocks, threads, sdata_size>>>(input, histogram, N, num_bins);

}
