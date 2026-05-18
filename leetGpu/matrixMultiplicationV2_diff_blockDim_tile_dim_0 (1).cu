// =============================================================================
//  matrixMultiplicationV2_diff_blockDim_tile_dim_0.cu
//
//  Fix: blockDim.y = 8, TILE_DIM = 16  (blockDim.y != TILE_DIM)
//
//  Original bug:
//    1. row = blockIdx.y * blockDim.y + threadIdx.y
//       With blockDim.y=8, this gives wrong global row (e.g. row=13 instead
//       of row=21 for ty=5, by=1). Block by=1 covers rows 16..31 but the
//       formula lands in rows 8..15.
//
//    2. Only one sh_a[ty][tx] and sh_b[ty][tx] slot was written per thread.
//       With 8 threads covering a 16-row tile, rows 8..15 of sh_a/sh_b were
//       never written (uninitialized shared memory).
//
//  Root cause:
//    blockDim.y (8) controls how many threads exist in Y per block.
//    TILE_DIM   (16) controls how many matrix rows each block's tile covers.
//    When blockDim.y < TILE_DIM, each thread must cover multiple rows.
//    ROWS_PER_THREAD = TILE_DIM / blockDim.y = 16 / 8 = 2.
//
//  Fixes applied:
//    1. threadsPerBlock changed to (TILE_DIM, BLOCK_DIM_Y) = (16, 8)
//    2. Grid still based on TILE_DIM (not BLOCK_DIM_Y) -- unchanged.
//    3. Each thread loads ROWS_PER_THREAD=2 rows into sh_a and sh_b.
//    4. Each thread accumulates ROWS_PER_THREAD=2 partial sums.
//    5. Each thread writes ROWS_PER_THREAD=2 output elements to C.
//    6. All row index calculations use TILE_DIM (not blockDim.y).
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// -------------------------------------------------------
// TILE_DIM  : size of the shared memory tile (16 x 16)
// BLOCK_DIM_Y: threads in the Y direction per block (8)
//              NOTE: BLOCK_DIM_Y < TILE_DIM intentionally
// ROWS_PER_THREAD: how many tile rows each thread handles
//              = TILE_DIM / BLOCK_DIM_Y = 16 / 8 = 2
// -------------------------------------------------------
#define TILE_DIM      16
#define BLOCK_DIM_Y    8
#define ROWS_PER_THREAD (TILE_DIM / BLOCK_DIM_Y)   // = 2
#define SIZE         512

// =============================================================================
//  KERNEL
// =============================================================================
__global__ void matmul_kernel_fixed(const float* A, const float* B, float* C,
                                     int M, int N, int K)
{
    // -------------------------------------------------------------------------
    // Shared memory tile: always TILE_DIM x TILE_DIM = 16 x 16
    // Unchanged from original -- the tile SIZE does not change.
    // -------------------------------------------------------------------------
    __shared__ float sh_a[TILE_DIM][TILE_DIM];
    __shared__ float sh_b[TILE_DIM][TILE_DIM];

    // -------------------------------------------------------------------------
    // Thread indices
    // tx : 0..TILE_DIM-1  = 0..15   (blockDim.x = TILE_DIM = 16)
    // ty : 0..BLOCK_DIM_Y-1 = 0..7  (blockDim.y = BLOCK_DIM_Y = 8)
    // -------------------------------------------------------------------------
    int tx = threadIdx.x;   // 0..15
    int ty = threadIdx.y;   // 0..7   (only 8 threads in Y!)

    // -------------------------------------------------------------------------
    // Output column for this thread (same formula, blockDim.x = TILE_DIM = 16
    // so these are equal, but we write TILE_DIM explicitly for clarity)
    // -------------------------------------------------------------------------
    int col = blockIdx.x * TILE_DIM + tx;

    // -------------------------------------------------------------------------
    // FIX 1: Accumulators -- one per output row this thread handles.
    //
    // Original:  float sum = 0.0f;           (1 output per thread)
    // Fixed:     float sum[ROWS_PER_THREAD]  (2 outputs per thread)
    //
    // Thread ty=5 in block by=1 handles:
    //   local_row r=0: row index inside tile = ty + 0*8 = 5  -> global row 21
    //   local_row r=1: row index inside tile = ty + 1*8 = 13 -> global row 29
    // -------------------------------------------------------------------------
    float sum[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        sum[r] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // K-tile loop: slide along the shared (N) dimension
    // Grid is based on TILE_DIM, so numTiles uses TILE_DIM -- unchanged.
    // -------------------------------------------------------------------------
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int tile = 0; tile < numTiles; ++tile) {

        // ---------------------------------------------------------------------
        // FIX 2: Load A tile -- each thread loads ROWS_PER_THREAD=2 rows.
        //
        // Original (WRONG when BLOCK_DIM_Y != TILE_DIM):
        //   int a_row = blockIdx.y * TILE_DIM + threadIdx.y;  // 1 row only
        //   sh_a[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        //   BUG: sh_a rows 8..15 are never written! (only 8 threads in Y)
        //
        // Fixed: loop over ROWS_PER_THREAD, using local_row = ty + r*BLOCK_DIM_Y
        //   r=0: local_row = ty + 0 = ty     (covers sh_a rows  0..7)
        //   r=1: local_row = ty + 8          (covers sh_a rows  8..15)
        //
        // KEY: a_row uses TILE_DIM (16), NOT blockDim.y (8).
        //   blockIdx.y * TILE_DIM = block's starting M-row in global A.
        //   For by=1: starts at row 16, not row 8.
        // ---------------------------------------------------------------------
        int a_col = tile * TILE_DIM + tx;   // K-column in global A

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            // local_row: position inside the 16-row shared tile
            //   r=0, ty=5 -> local_row = 5      (sh_a row 5)
            //   r=1, ty=5 -> local_row = 5+8=13 (sh_a row 13)
            int local_row = ty + r * BLOCK_DIM_Y;

            // a_row: actual row in global matrix A
            //   by=1, local_row=5  -> a_row = 1*16 + 5  = 21
            //   by=1, local_row=13 -> a_row = 1*16 + 13 = 29
            //   (NOT by*8+ty=13 which was the original bug!)
            int a_row = blockIdx.y * TILE_DIM + local_row;

            sh_a[local_row][tx] = (a_row < M && a_col < N)
                                    ? A[a_row * N + a_col]
                                    : 0.0f;
        }

        // ---------------------------------------------------------------------
        // FIX 3: Load B tile -- each thread loads ROWS_PER_THREAD=2 rows.
        //
        // Original (WRONG when BLOCK_DIM_Y != TILE_DIM):
        //   int b_row = tile * TILE_DIM + threadIdx.y;  // 1 row only
        //   sh_b[threadIdx.y][threadIdx.x] = B[b_row * K + b_col];
        //   BUG: sh_b rows 8..15 never written!
        //
        // Fixed: same loop pattern as A loading.
        //   b_row is the K-dimension row of B.
        //   r=0: b_row = tile*16 + ty      (K-rows 0..7 of this tile)
        //   r=1: b_row = tile*16 + ty + 8  (K-rows 8..15 of this tile)
        // ---------------------------------------------------------------------
        int b_col = blockIdx.x * TILE_DIM + tx;   // N-column in global B

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            int local_row = ty + r * BLOCK_DIM_Y;

            // b_row: K-row in global matrix B
            //   tile=1, local_row=5  -> b_row = 1*16 + 5  = 21
            //   tile=1, local_row=13 -> b_row = 1*16 + 13 = 29
            int b_row = tile * TILE_DIM + local_row;

            sh_b[local_row][tx] = (b_row < N && b_col < K)
                                    ? B[b_row * K + b_col]
                                    : 0.0f;
        }

        // All TILE_DIM x TILE_DIM = 16x16 elements of sh_a and sh_b are now
        // filled. The 8 threads in Y each wrote 2 rows, covering all 16 rows.
        __syncthreads();

        // ---------------------------------------------------------------------
        // FIX 4: Compute -- accumulate into sum[r] for each row this thread owns.
        //
        // Original:
        //   for (int k=0; k<TILE_DIM; k++)
        //       sum += sh_a[ty][k] * sh_b[k][tx];  // 1 output row
        //
        // Fixed: inner loop over ROWS_PER_THREAD
        //   sum[0] accumulates the dot product for local_row=ty
        //   sum[1] accumulates the dot product for local_row=ty+8
        //
        // sh_a[local_row][k] * sh_b[k][tx]:
        //   local_row=5 , k=3: sh_a[5][3]  * sh_b[3][tx]
        //   local_row=13, k=3: sh_a[13][3] * sh_b[3][tx]
        // ---------------------------------------------------------------------
        for (int k = 0; k < TILE_DIM; ++k) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                int local_row = ty + r * BLOCK_DIM_Y;
                sum[r] += sh_a[local_row][k] * sh_b[k][tx];
            }
        }

        __syncthreads();

    } // end tile loop

    // -------------------------------------------------------------------------
    // FIX 5: Write back -- write ROWS_PER_THREAD=2 results per thread.
    //
    // Original (WRONG):
    //   int row = blockIdx.y * blockDim.y + threadIdx.y;  // blockDim.y=8!
    //   C[row * K + col] = sum;
    //   BUG: by=1, ty=5 -> row = 1*8+5 = 13  (WRONG, should be 21!)
    //        Block by=1 covers rows 16..31, not rows 8..15!
    //
    // Fixed: row = blockIdx.y * TILE_DIM + local_row
    //   by=1, local_row=5  -> row = 1*16+5  = 21  CORRECT
    //   by=1, local_row=13 -> row = 1*16+13 = 29  CORRECT
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int local_row = ty + r * BLOCK_DIM_Y;

        // KEY FIX: use TILE_DIM (16) not blockDim.y (8)
        int row = blockIdx.y * TILE_DIM + local_row;

        if (row < M && col < K) {
            C[row * K + col] = sum[r];
        }
    }
}


// =============================================================================
//  CPU Reference
// =============================================================================
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k) {
                s += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = s;
        }
    }
}

// =============================================================================
//  Verification
// =============================================================================
bool verify_results(const float* cpu_res, const float* gpu_res, int size) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(cpu_res[i] - gpu_res[i]) > 1e-3f) {
            std::cout << "Mismatch at [" << i << "]: CPU=" << cpu_res[i]
                      << "  GPU=" << gpu_res[i] << std::endl;
            return false;
        }
    }
    return true;
}

// =============================================================================
//  MAIN
// =============================================================================
int main() {
    const int M = SIZE;   // 512
    const int N = SIZE;   // 512
    const int K = SIZE;   // 512

    // -------------------------------------------------------------------------
    // Verify that TILE_DIM is divisible by BLOCK_DIM_Y
    // This is required: each thread loads exactly TILE_DIM/BLOCK_DIM_Y rows.
    // -------------------------------------------------------------------------
    static_assert(TILE_DIM % BLOCK_DIM_Y == 0,
        "TILE_DIM must be divisible by BLOCK_DIM_Y");

    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    std::vector<float> h_A(M * N), h_B(N * K), h_C_gpu(M * K), h_C_cpu(M * K);

    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * K; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // FIX 6: Launch configuration
    //
    // Original:  dim3 threadsPerBlock(TILE_DIM, TILE_DIM)    = (16, 16)
    // Fixed:     dim3 threadsPerBlock(TILE_DIM, BLOCK_DIM_Y) = (16,  8)
    //
    // Grid is STILL based on TILE_DIM (not BLOCK_DIM_Y!).
    //
    // Original grid was: ceil(K/TILE_DIM) x ceil(M/TILE_DIM)
    //   = ceil(512/16) x ceil(512/16) = 32 x 32 = 1024 blocks
    //
    // Fixed grid: SAME formula, SAME result.
    //   If we used BLOCK_DIM_Y in the grid: ceil(512/8) = 64 blocks in Y
    //   --> 64 * 16 = 1024 rows covered, but M=512 needs only 32 tiles!
    //   --> Each tile would cover only 8 rows instead of 16 -- WRONG.
    //
    // The grid must match the TILE size, not the thread count.
    // -------------------------------------------------------------------------
    dim3 threadsPerBlock(TILE_DIM, BLOCK_DIM_Y);           // (16, 8) -- FIXED
    dim3 blocksPerGrid((K + TILE_DIM - 1) / TILE_DIM,      // x: 32 blocks
                       (M + TILE_DIM - 1) / TILE_DIM);     // y: 32 blocks
                       // NOTE: grid uses TILE_DIM (16), NOT BLOCK_DIM_Y (8)!

    std::cout << "Config: TILE_DIM=" << TILE_DIM
              << "  BLOCK_DIM_Y=" << BLOCK_DIM_Y
              << "  ROWS_PER_THREAD=" << ROWS_PER_THREAD << std::endl;
    std::cout << "Threads per block: (" << threadsPerBlock.x
              << ", " << threadsPerBlock.y << ")" << std::endl;
    std::cout << "Blocks per grid:   (" << blocksPerGrid.x
              << ", " << blocksPerGrid.y << ")" << std::endl;

    // Warm-up
    matmul_kernel_fixed<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // GPU Performance
    const int iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        matmul_kernel_fixed<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_gpu_time = milliseconds / iterations;

    // CPU Reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    auto cpu_stop  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_stop - cpu_start;

    // Verification
    cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    bool match = verify_results(h_C_cpu.data(), h_C_gpu.data(), M * K);

    std::cout << "\nMatrix Dimensions: " << M << "x" << N
              << " * " << N << "x" << K << std::endl;
    std::cout << "Verification: " << (match ? "PASS" : "FAIL") << std::endl;
    std::cout << "GPU Avg Time: " << avg_gpu_time        << " ms" << std::endl;
    std::cout << "CPU Time:     " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "Speedup:      " << cpu_duration.count() / avg_gpu_time << "x" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
