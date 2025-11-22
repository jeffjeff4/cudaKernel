#include <cuda_runtime.h>
#define MAX_KERNEL_FLATTEND_SIZE 441
__constant__ float const_kernel[MAX_KERNEL_FLATTEND_SIZE];

#define TILE_DIM_W 32
#define TILE_DIM_H 8

__global__ void convolutionKernel(const float* input, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    extern __shared__ float sh_tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int k_radius_x = kernel_cols >> 1;
    const int k_radius_y = kernel_rows >> 1;

    const int out_col = blockIdx.x * TILE_DIM_W + tx;
    const int out_row = blockIdx.y * TILE_DIM_H + ty;

    const int sh_tile_width = TILE_DIM_W + kernel_cols - 1;
    const int sh_tile_height = TILE_DIM_H + kernel_rows - 1;

    const int in_tile_start_row = blockIdx.y * TILE_DIM_H - k_radius_y;
    const int in_tile_start_col = blockIdx.x * TILE_DIM_W - k_radius_x;

    for (int y=ty; y<sh_tile_height; y+= TILE_DIM_H) {
        for (int x=tx; x<sh_tile_width; x+=TILE_DIM_W) {
            const int cur_row = in_tile_start_row + y;
            const int cur_col = in_tile_start_col + x;

            if (cur_row >= 0 and cur_row<input_rows and cur_col>=0 and cur_col<input_cols) {
                sh_tile[y*sh_tile_width+x] = input[cur_row*input_cols+cur_col];
            } else {
                sh_tile[y*sh_tile_width+x] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (out_row<input_rows and out_col<input_cols) {
        float sum = 0.0f;
        const float* sh_tile_row_ptr = &sh_tile[ty*sh_tile_width+tx];

        #pragma unroll
        for (int i=0; i<kernel_rows; ++i) {
            #pragma unroll
            for (int j=0; j<kernel_rows; ++j) {
                sum += sh_tile_row_ptr[j] * const_kernel[i*kernel_cols+j];
            }
            sh_tile_row_ptr += sh_tile_width;
        }

        output[out_row*input_cols + out_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    cudaMemcpyToSymbol(const_kernel, kernel, kernel_rows * kernel_cols * sizeof(float));
    dim3 block_dim(TILE_DIM_W, TILE_DIM_H);
    dim3 grid_dim((input_cols + TILE_DIM_W - 1) / TILE_DIM_W,
                  (input_rows + TILE_DIM_H - 1) / TILE_DIM_H);
    size_t sh_mem_size = (block_dim.x + kernel_cols - 1) * (block_dim.y + kernel_rows - 1) * sizeof(float);
    convolutionKernel<<<grid_dim, block_dim, sh_mem_size>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
