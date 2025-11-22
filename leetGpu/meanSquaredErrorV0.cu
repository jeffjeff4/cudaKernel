#include <cuda_runtime.h>

const int threads = 128;
const int warp_size = 32;
const int coarse_factor = 4;

__global__ void meanSquaredError(const float* predictions, const float* targets, float* mse, int N) {
    __shared__ float sdata[threads / warp_size];

    int idx = blockIdx.x * blockDim.x * coarse_factor + threadIdx.x;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    float val = 0.0f;
    for (int i=0; i<coarse_factor; ++i) {
        int local_idx = idx + i * blockDim.x;
        if (local_idx < N) {
            float delta = predictions[local_idx] - targets[local_idx];
            val += delta * delta;
        }
    }

    for (int offset=warp_size / 2; offset>0; offset/=2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id==0) sdata[warp_id] = val;
    __syncthreads();

    if (warp_id==0) {
        if (threadIdx.x < threads / warp_size) {
            val = sdata[threadIdx.x];
        } else {
            val = 0.0f;
        }
        for (int offset=warp_size/2; offset>0; offset/=2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        };
        if (lane_id==0) atomicAdd(mse, val/N);
    }
} 


// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    int blocks = (N + threads * coarse_factor - 1) / (threads * coarse_factor);
    meanSquaredError<<<blocks, threads>>>(predictions, targets, mse, N); 
}
