#include <cuda_runtime.h>
__constant__ float c_kernel[1024];
//__constant__ int DIM = 16;
const int DIM = 16;

//method0
/*
__global__ void convolution2d(const float* input, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    __shared__ float s_data[2034];
    int local_row = threadIdx.x, local_col = threadIdx.y;

    int global_row = threadIdx.x + blockDim.x * blockIdx.x;
    int global_col = threadIdx.y + blockDim.y * blockIdx.y;

    int tile_start_x = blockDim.x * blockIdx.x;
    int tile_start_y = blockDim.y * blockIdx.y;

    int tile_size_x = blockDim.x + kernel_rows-1;
    int tile_size_y = blockDim.y + kernel_cols-1;
    
    int output_size_x = input_rows - kernel_rows + 1;
    int output_size_y = input_cols - kernel_cols + 1;

    for (int i = local_row; i < tile_size_x; i += blockDim.x) {
        for (int j = local_col; j < tile_size_y; j += blockDim.y) {
            if (tile_start_x + i < input_rows and tile_start_y + j < input_cols) {
                s_data[i * tile_size_y + j] = input[(tile_start_x + i) * input_cols + (tile_start_y + j)];
            } else {
                s_data[i * tile_size_y + j] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (global_row < output_size_x and global_col < output_size_y) {
        float sum = 0.0f;
        for (int i=0; i<kernel_rows; ++i) {
            for (int j=0; j<kernel_cols; ++j) {
                sum += c_kernel[i*kernel_cols + j] * s_data[(local_row+i) * tile_size_y + local_col +j];
            }
        }
        output[global_row * output_size_y + global_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    cudaMemcpyToSymbol(c_kernel, kernel, kernel_rows * kernel_cols * sizeof(float));
    dim3 threadsPerBlock(DIM, DIM);
    dim3 blocksPerGrid((input_rows+DIM-1) / DIM, (input_cols+DIM-1) / DIM);
    convolution2d<<<blocksPerGrid, threadsPerBlock>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
*/


//method1
//this is my code
///*
__global__ void convolution2d(const float* input, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    __shared__ float s_data[2034];
    int local_row = threadIdx.y, local_col = threadIdx.x;

    int global_col = threadIdx.x + blockDim.y * blockIdx.x;
    int global_row = threadIdx.y + blockDim.x * blockIdx.y;

    int tile_start_x = blockDim.x * blockIdx.x;
    int tile_start_y = blockDim.y * blockIdx.y;

    int tile_size_x = blockDim.x + kernel_cols - 1;
    int tile_size_y = blockDim.y + kernel_rows - 1;
    
    int output_size_x = input_cols - kernel_cols + 1;
    int output_size_y = input_rows - kernel_rows + 1;

    for (int i = local_row; i < tile_size_y; i += blockDim.y) {
        for (int j = local_col; j < tile_size_x; j += blockDim.x) {
            if (tile_start_y + i < input_rows and tile_start_x + j < input_cols) {
                s_data[i * tile_size_x + j] = input[(tile_start_y + i) * input_cols + (tile_start_x + j)];
            } else {
                s_data[i * tile_size_x + j] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (global_row < output_size_y and global_col < output_size_x) {
        float sum = 0.0f;
        for (int i=0; i<kernel_rows; ++i) {
            for (int j=0; j<kernel_cols; ++j) {
                sum += c_kernel[i*kernel_cols + j] * s_data[(local_row+i) * tile_size_x + local_col +j];
            }
        }
        output[global_row * output_size_x + global_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    cudaMemcpyToSymbol(c_kernel, kernel, kernel_rows * kernel_cols * sizeof(float));
    dim3 threadsPerBlock(DIM, DIM);
    //dim3 blocksPerGrid((input_rows+DIM-1) / DIM, (input_cols+DIM-1) / DIM);
    dim3 blocksPerGrid((input_cols+DIM-1) / DIM, (input_rows+DIM-1) / DIM);
    convolution2d<<<blocksPerGrid, threadsPerBlock>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
//*/