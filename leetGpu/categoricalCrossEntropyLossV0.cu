#include <cuda_runtime.h>

#define BLOCK_THREADS 32
#define COARSE 2
#define THREAD_RESP (BLOCK_THREADS / COARSE)
#define CUDART_INF_F __int_as_float(0x7f800000)


__global__ void ccel_kernel_unaligned(const float* logits, const int* true_labels, float* loss, int N, int C) {
    __shared__ float shmem[COARSE][BLOCK_THREADS+1];
    __shared__ float log_label;
    __shared__ float sum_val;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int base_col = ty * BLOCK_THREADS * THREAD_RESP;
    int col = base_col + tx;
    int row = bx * C;
    if (base_col > C) {
        return;
    }
    int true_label = true_labels[bx];
    sum_val = 0.0f;

    float cur_val = 0.0f;
    float local_log_label = -CUDART_INF_F;
    float thread_sum = 0.0f;
    int mini = min(C, base_col + BLOCK_THREADS * THREAD_RESP);

    #pragma unroll
    for (int i=col; i<mini; i+=BLOCK_THREADS) {
        cur_val = logits[row+i];
        if (true_label==i) {
            local_log_label = cur_val;
        }
        thread_sum += exp(cur_val);
    }
    shmem[ty][tx] = thread_sum;

    if (local_log_label != -CUDART_INF_F) {
        log_label = local_log_label;
    }

    #pragma unroll
    for (int s = BLOCK_THREADS/2;  s>=1; s/=2) {
        __syncthreads();
        if (tx<s) {
            shmem[ty][tx] += shmem[ty][tx+s];
        }
    }
    if (tx==0) {
        atomicAdd(&sum_val, shmem[ty][tx]);
    }

    __syncthreads();
    if (tx==0 && ty==0) {
        float val = (log(sum_val) - log_label) / N;
        atomicAdd(loss, val);
    }
}


// logits, true_labels, loss are device pointers
extern "C" 
void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    cudaMemset(loss, 0, sizeof(float));
    dim3 blocks(BLOCK_THREADS, COARSE);
    ccel_kernel_unaligned<<<N, blocks>>>(logits, true_labels, loss, N, C);
    cudaDeviceSynchronize();
}