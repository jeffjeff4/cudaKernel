#include <cuda_runtime.h>

const int warp_size = 32;
const int tile_dim = 32;
const int coarse_factor = 4;

__global__ void matrix_transpose(const float* input, float* output, int N, int d) {
    __shared__ float tile[tile_dim][tile_dim+1];

    int row_start = blockIdx.y * tile_dim + threadIdx.y;
    int col = blockIdx.x * tile_dim + threadIdx.x;

    #pragma unroll
    int step = tile_dim / coarse_factor;
    for (int i=0; i<tile_dim; i+= step) {
        int row = row_start + i;
        tile[threadIdx.y + i][threadIdx.x] = 0.0f;
        if (row<N and col<d) {
            tile[threadIdx.y + i][threadIdx.x] = input[row*d + col];
        }
    }
    __syncthreads();

    row_start = blockIdx.x * tile_dim + threadIdx.y;
    col = blockIdx.y * tile_dim + threadIdx.x;

    if (col<N) {
        #pragma unroll
        for (int i = 0; i < tile_dim; i += tile_dim / coarse_factor) {
            int row = row_start + i;
            if (row<d) {
                output[row*N + col] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}


__global__ void matrix_multiply(const float* A, const float* B, float* C, int M, int N, int d) {
    __shared__ float A_s[tile_dim][tile_dim];
    __shared__ float B_s[tile_dim][tile_dim];

    int row = blockIdx.y * tile_dim + threadIdx.y;
    int col = blockIdx.x * tile_dim + threadIdx.x;

    float sum = 0.0f;
    for (int tile_start = 0; tile_start<d; tile_start+=tile_dim) {
        int a_row = row;
        int a_col = tile_start + threadIdx.x;
        A_s[threadIdx.y][threadIdx.x] = 0.0f;

        if (a_row<M and a_col<d) {
            A_s[threadIdx.y][threadIdx.x] = A[a_row*d + a_col];
        }

        int b_row = tile_start + threadIdx.y;
        int b_col = col;
        B_s[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row<d and b_col<N) {
            B_s[threadIdx.y][threadIdx.x] = B[b_row*N + b_col];
        }
        __syncthreads();

        #pragma unroll
        for (int k=0; k<tile_dim; ++k) {
            sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
        }
        __syncthreads();

        if (row<M and col<N) {
            C[row*N + col] = sum;
        }
    }
}


__global__ void softmax(float* scores, int M, int N, int d) {
    extern __shared__ float sdata[];
    int sdata_len = (blockDim.x + warp_size - 1) / warp_size;

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    int row = blockIdx.x;
    int col = threadIdx.x;

    float val = -INFINITY;
    if (row<M and col<N) {
        val = scores[row*N + col];
    }
    for (int offset=warp_size/2; offset>0; offset /=2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    sdata[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = -INFINITY;
        if (lane_id < sdata_len) {
            val = sdata[lane_id];
        }
        for (int offset=warp_size/2; offset>0; offset /=2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        sdata[0] = val;
    }
    __syncthreads();
    float max = sdata[0];
    __syncthreads();

    float numerator = __expf((scores[row*N + col] - max) / sqrtf(d));

    val = 0.0f;
    if (row<M and col<N) {
        val = numerator;
    }
    for (int offset=warp_size/2; offset>0; offset /=2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    sdata[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = 0.0f;
        if (lane_id < sdata_len) {
            val = sdata[lane_id];
        }
        for (int offset=warp_size/2; offset>0; offset /=2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        sdata[0] = val;
    }
    __syncthreads();
    float denominator = sdata[0];

    scores[row*N + col] = numerator / denominator;
}

//this code works
//removed this cmd:
//    if (warp_id == 0) { }
/*
__global__ void softmax(float* scores, int M, int N, int d) {
    extern __shared__ float sdata[];
    int sdata_len = (blockDim.x + warp_size - 1) / warp_size;

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    int row = blockIdx.x;
    int col = threadIdx.x;

    float val = -INFINITY;
    if (row<M and col<N) {
        val = scores[row*N + col];
    }
    for (int offset=warp_size/2; offset>0; offset /=2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    sdata[warp_id] = val;
    __syncthreads();
    val = -INFINITY;
        if (lane_id < sdata_len) {
            val = sdata[lane_id];
        }
        for (int offset=warp_size/2; offset>0; offset /=2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    sdata[0] = val;

    __syncthreads();
    float max = sdata[0];
    __syncthreads();

    float numerator = __expf((scores[row*N + col] - max) / sqrtf(d));

    val = 0.0f;
    if (row<M and col<N) {
        val = numerator;
    }
    for (int offset=warp_size/2; offset>0; offset /=2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    sdata[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = 0.0f;
        if (lane_id < sdata_len) {
            val = sdata[lane_id];
        }
        for (int offset=warp_size/2; offset>0; offset /=2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        sdata[0] = val;
    }
    __syncthreads();
    float denominator = sdata[0];

    scores[row*N + col] = numerator / denominator;
}
*/


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *KT, * scores;
    cudaMalloc(&KT, d * N * sizeof(float));
    cudaMalloc(&scores, M * N * sizeof(float));
    
    dim3 blockDim1(tile_dim, tile_dim / coarse_factor);
    dim3 gridDim1((d + tile_dim - 1) / tile_dim, (N + tile_dim - 1) / tile_dim);
    matrix_transpose<<<gridDim1, blockDim1>>>(K, KT, N, d);
    cudaDeviceSynchronize();

    dim3 blockDim2(tile_dim, tile_dim);
    dim3 gridDim2((N + tile_dim - 1) / tile_dim, (M + tile_dim - 1) / tile_dim);
    matrix_multiply<<<gridDim2, blockDim2>>>(Q, KT, scores, M, N, d);
    cudaDeviceSynchronize();

    dim3 blockDim3(N);
    dim3 gridDim3(M);
    size_t sdata_size = ((blockDim3.x + warp_size - 1) / warp_size) * sizeof(float);
    softmax<<<gridDim3, blockDim3, sdata_size>>>(scores, M, N, d);
    cudaDeviceSynchronize();

    dim3 blockDim4(tile_dim, tile_dim);
    dim3 gridDim4((d + tile_dim - 1) / tile_dim, (M + tile_dim - 1) / tile_dim);
    matrix_multiply<<<gridDim4, blockDim4>>>(scores, V, output, M, d, N);

    cudaFree(KT);
    cudaFree(scores);
}