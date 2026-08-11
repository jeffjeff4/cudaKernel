#include <cuda_runtime.h>

static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int ELEMENTS_PER_THREAD = 8;

__global__ void segmented_reduce_kernel(const float* __restrict__ input,
                                       float* __restrict__ partial_sums,
                                       int N) {
    __shared__ float shared_sum[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD;

    float sum = 0.0f;
    int idx = block_start + tid;

    // Each thread loads up to ELEMENTS_PER_THREAD elements from the segment.
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (idx < N) {
            sum += input[idx];
        }
        idx += blockDim.x;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

__global__ void final_reduce_kernel(const float* __restrict__ partial_sums,
                                    float* __restrict__ output,
                                    int num_partial) {
    __shared__ float shared_sum[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    float sum = 0.0f;
    int idx = tid;

    while (idx < num_partial) {
        sum += partial_sums[idx];
        idx += blockDim.x;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_sum[0];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) {
        return;
    }

    int num_blocks = (N + THREADS_PER_BLOCK * ELEMENTS_PER_THREAD - 1) /
                     (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD);

    float* d_partial = nullptr;
    cudaMalloc(&d_partial, num_blocks * sizeof(float));

    segmented_reduce_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(input,
                                                               d_partial,
                                                               N);

    int final_threads = THREADS_PER_BLOCK;
    int final_blocks = 1;
    final_reduce_kernel<<<final_blocks, final_threads>>>(d_partial,
                                                         output,
                                                         num_blocks);

    cudaFree(d_partial);
}
