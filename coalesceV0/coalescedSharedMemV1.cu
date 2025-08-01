// block size
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared_memory(float* odata, const float* idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // col
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // row

    // transpose block index
    int width_in = width;
    int height_in = height;

    // ---- 1. Coalesced read from idata to shared memory ----
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];

    __syncthreads();

    // transpose thread indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // 注意交换 x/y 以完成转置
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // ---- 2. Coalesced write from shared memory to odata ----
    if (x < height && y < width)
        odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
}
