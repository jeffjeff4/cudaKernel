#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared_kernel(float* odata, const float* idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // 避免 bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width)
        odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

void transpose_shared(torch::Tensor input, torch::Tensor output) {
    const int width = input.size(1);
    const int height = input.size(0);

    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    transpose_shared_kernel<<<gridDim, blockDim>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        width,
        height
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_shared", &transpose_shared, "Shared Memory Transpose");
}
