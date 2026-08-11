#include <cuda_runtime.h>

static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int WARP_SIZE = 32;
static constexpr int ELEMENTS_PER_THREAD = 8;
static constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

__device__ __forceinline__ float warp_reduce(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void segmented_warp_reduce_kernel(const float* __restrict__ input,
                                             float* __restrict__ partial_sums,
                                             int N) {
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int block_start = blockIdx.x * THREADS_PER_BLOCK * ELEMENTS_PER_THREAD;
    int idx = block_start + tid;

    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (idx < N) {
            local_sum += input[idx];
        }
        idx += THREADS_PER_BLOCK;
    }

    float warp_sum = warp_reduce(local_sum);

    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (lane_id == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < WARPS_PER_BLOCK; ++i) {
            block_sum += warp_sums[i];
        }
        partial_sums[blockIdx.x] = block_sum;
    }
}

__global__ void final_tree_reduce_kernel(const float* __restrict__ partial_sums,
                                         float* __restrict__ output,
                                         int num_partial) {
    __shared__ float shared_sum[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    float sum = 0.0f;
    int idx = tid;
    int stride = blockDim.x;

    while (idx < num_partial) {
        sum += partial_sums[idx];
        idx += stride;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_sum[0];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (output == nullptr) {
        return;
    }

    if (N <= 0) {
        *output = 0.0f;
        return;
    }

    int num_blocks = (N + THREADS_PER_BLOCK * ELEMENTS_PER_THREAD - 1) /
                     (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD);
    if (num_blocks < 1) {
        num_blocks = 1;
    }

    float* d_partial = nullptr;
    cudaMalloc(&d_partial, num_blocks * sizeof(float));

    segmented_warp_reduce_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(input,
                                                                    d_partial,
                                                                    N);

    final_tree_reduce_kernel<<<1, THREADS_PER_BLOCK>>>(d_partial,
                                                       output,
                                                       num_blocks);

    cudaFree(d_partial);
}
