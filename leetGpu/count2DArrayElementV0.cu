#include <cuda_runtime.h>

#define ELEMENTS_PER_THREAD 8 // 每个线程处理的连续元素数

//method0
__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    __shared__ int block_sum[256];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    block_sum[tid] = 0;

    int base_x = (blockDim.x * blockIdx.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_cnt = 0;

    if (y<N) {
        for (int i=0; i<ELEMENTS_PER_THREAD; ++i) {
            int x = base_x + i;
            if (x < M) {
                int val = input[y*M +x];
                if (val==K) {
                    ++local_cnt;
                }
            }
        }

        block_sum[tid] = local_cnt;
        __syncthreads();

        for (int stride = (blockDim.x * blockDim.y) / 2; stride>0; stride>>=1) {
            if (tid < stride) {
                block_sum[tid] += block_sum[tid + stride];
            }
            __syncthreads();
        }

        //wrong
        //atomicAdd(output, block_sum[0]);
        //__syncthreads();

        //correct
        if (tid == 0) {
            atomicAdd(output, block_sum[0]);
        }

    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}