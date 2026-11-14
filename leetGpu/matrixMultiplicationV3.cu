#include <cuda_runtime.h>

#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;
constexpr int BLOCK_SIZE = 256;
constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X;
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X;
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X;
constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_Y;
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y;
constexpr int THREAD_C_WARP_X = 8;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X;
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X;

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col0 = blockIdx.x * BLOCK_N;
    int row0 = blockIdx.y * BLOCK_M;
    int tidx = threadIdx.x;
    __shared__ float sA[2][BLOCK_K][BLOCK_M];
    __shared__ float sB[2][BLOCK_K][BLOCK_N];
    int tAx = tidx & (THREAD_A_LAYOUT_X - 1);
    int tAy = tidx / THREAD_A_LAYOUT_X;
    int tBx = tidx & (THREAD_B_LAYOUT_X - 1);
    int tBy = tidx / THREAD_B_LAYOUT_X;
    
    int warpId = tidx >> 5;
    int laneId = tidx & 31;
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1);
    int warpy = warpId / THREAD_C_WARP_DIM_X;
    int lanex = (laneId & 15) >> 1;
    int laney = ((laneId >> 4) << 1) + (laneId & 1); 
    int tCx = warpx * THREAD_C_WARP_X + lanex;
    int tCy = warpy * THREAD_C_WARP_Y + laney;
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};
    float tCsA[2][THREAD_C_Y_TILE_SIZE];
    float tCsB[2][THREAD_C_X_TILE_SIZE];
    int bufferId = 0;
    # pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f; 
    }
    # pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = i + tBy;
        # pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f; 
        }
    }
    __syncthreads();
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        # pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; ++ tk) {
            if (tk > 0) {
                # pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm ++) {
                    # pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn ++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }
            if (tk < BLOCK_K) {
                # pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm ++) {
                    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
                    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }
                # pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn ++) {
                    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2;
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
                }
            }
        }
        
        if (k < K) {
            int c = k + tAx;
            # pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f; 
            }
            # pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
                # pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f; 
                }
            }
            __syncthreads();
        }
        bufferId ^= 1;
    } 
    # pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i ++) {
        int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
        # pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j ++) {
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((K + BLOCK_N - 1) / BLOCK_N,
                       (M + BLOCK_M - 1) / BLOCK_M);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}



/*
// matmul_v0.cu
//
// =====================================================================
// PROBLEM STATEMENT
// =====================================================================
// Compute C = A @ B, where A is (M x K), B is (K x N), C is (M x N).
// All three matrices are stored in ROW-MAJOR format in GPU global memory.
//
// This file is self-contained: it includes the GPU kernel itself, a
// plain C++ CPU reference implementation (used only to verify the GPU
// kernel's results -- never used to do the "real" computation), a set
// of correctness test cases spanning tiny/ragged/large matrix shapes,
// and a performance-benchmarking harness with configurable warm-up and
// timed run counts.
//
// =====================================================================
// A NOTE ON THE KERNEL ITSELF, AND THREE BUGS FIXED IN `solve()`
// =====================================================================
// The __global__ kernel below (`matrix_multiplication_kernel`) is used
// exactly as supplied: a hand-tuned SGEMM kernel using two-level
// shared-memory double buffering, register tiling (the `acc[][]`
// per-thread array), float4-vectorized shared-memory loads, and a
// warp-aware thread layout for computing C.
//
// The ORIGINAL `solve()` wrapper that was supplied alongside this
// kernel had three bugs, all fixed here, explained individually below
// so nothing is silently changed without comment:
//
//   1. `extern C void solve(...)` is not valid C++ -- the linkage
//      specifier must be a string literal. Fixed to `extern "C"`.
//
//   2. The kernel launch passed arguments in the order (A, B, C, M, K,
//      N), but the kernel's own parameter list is (A, B, C, M, N, K).
//      That swap means the kernel's "N" parameter would have silently
//      received the real K, and vice versa. Fixed to pass (M, N, K) in
//      the kernel's actual declared order.
//
//   3. The grid's x-dimension (which tiles across C's COLUMNS, i.e.
//      the N dimension, since `col0 = blockIdx.x * BLOCK_N`) was sized
//      using `(K + BLOCK_N - 1) / BLOCK_N` instead of
//      `(N + BLOCK_N - 1) / BLOCK_N`. For any non-square matrix where
//      N != K, this either leaves part of C's columns never computed
//      (if N > K) or launches wasted blocks (if N < K) -- fixed to use
//      N.
//
// Bugs 2 and 3 compound: fixing only one without the other still
// produces wrong results for non-square, non-K==N matrices. Both are
// fixed together here. Every non-square test case below (e.g.
// 257x129x63) specifically exercises this fix.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// =====================================================================
// Perf-testing knobs (requirement #4): one constant controls how many
// warm-up (untimed) executions happen before measurement begins, and a
// separate constant controls how many timed executions are averaged
// over for each test case.
// =====================================================================
constexpr int NUM_WARMUP_RUNS = 5;
constexpr int NUM_TIMED_RUNS = 20;

// =====================================================================
// CUDA error-checking helper. Every CUDA API call in this file is
// wrapped with this -- fail loudly and immediately instead of silently
// continuing with an invalid device state.
// =====================================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t cuda_check_err__ = (call);                               \
        if (cuda_check_err__ != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(cuda_check_err__));                   \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// =====================================================================
// The GPU kernel and its tiling constants -- supplied as-is.
// =====================================================================
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;
constexpr int BLOCK_SIZE = 256;

constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X;
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X;
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X;

constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_Y;
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y;

constexpr int THREAD_C_WARP_X = 8;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X;
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X;

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                              float *C, int M, int N, int K) {
    int col0 = blockIdx.x * BLOCK_N;
    int row0 = blockIdx.y * BLOCK_M;
    int tidx = threadIdx.x;

    __shared__ float sA[2][BLOCK_K][BLOCK_M];
    __shared__ float sB[2][BLOCK_K][BLOCK_N];

    int tAx = tidx & (THREAD_A_LAYOUT_X - 1);
    int tAy = tidx / THREAD_A_LAYOUT_X;
    int tBx = tidx & (THREAD_B_LAYOUT_X - 1);
    int tBy = tidx / THREAD_B_LAYOUT_X;

    int warpId = tidx >> 5;
    int laneId = tidx & 31;
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1);
    int warpy = warpId / THREAD_C_WARP_DIM_X;
    int lanex = (laneId & 15) >> 1;
    int laney = ((laneId >> 4) << 1) + (laneId & 1);
    int tCx = warpx * THREAD_C_WARP_X + lanex;
    int tCy = warpy * THREAD_C_WARP_Y + laney;

    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};
    float tCsA[2][THREAD_C_Y_TILE_SIZE];
    float tCsB[2][THREAD_C_X_TILE_SIZE];
    int bufferId = 0;

#pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] =
            (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = i + tBy;
#pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
    }
    __syncthreads();

    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
#pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; ++tk) {
            if (tk > 0) {
#pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
#pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }
            if (tk < BLOCK_K) {
#pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm++) {
                    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
                    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }
#pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn++) {
                    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2;
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
                }
            }
        }
        if (k < K) {
            int c = k + tAx;
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] =
                    (r < M && c < K) ? A[r * K + c] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
#pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c2 = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] =
                        (r < K && c2 < N) ? B[r * N + c2] : 0.0f;
                }
            }
            __syncthreads();
        }
        bufferId ^= 1;
    }

#pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
#pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU).
//
// See the "three bugs fixed" note at the top of this file: linkage
// specifier quoted, kernel launch argument order corrected to (M, N,
// K), and the grid's x-dimension now sized from N instead of K.
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                       int K) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// =====================================================================
// Requirement #3: plain C++ CPU baseline, used ONLY to verify the GPU
// kernel's output -- never used to perform the "real" computation.
//
// Loop order is deliberately i-k-j (not the more obvious i-j-k): with
// row-major storage, this keeps the innermost loop walking B and C one
// full contiguous ROW at a time, which is dramatically more
// cache-friendly on a CPU than the strided access an i-j-k ordering
// would cause. This is what keeps the correctness-test sizes below
// (up to 1000x777x333) finishing in a reasonable amount of time.
// =====================================================================
void cpu_matmul_reference(const std::vector<float> &A,
                           const std::vector<float> &B, std::vector<float> &C,
                           int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < M; ++i) {
        float *c_row = &C[static_cast<size_t>(i) * N];
        for (int k = 0; k < K; ++k) {
            float a_val = A[static_cast<size_t>(i) * K + k];
            const float *b_row = &B[static_cast<size_t>(k) * N];
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

// =====================================================================
// Requirement #2: test cases spanning tiny, ragged-edged, rectangular,
// and large matrix shapes.
//
// CORRECTION, left visible rather than silently fixed: an earlier
// version of this file set `verify_correctness = false` for the three
// largest sizes (1024/2048/4096 cubed), assuming the CPU reference
// above would be "impractically slow" at that scale. That assumption
// was checked, not just asserted, by compiling cpu_matmul_reference()
// standalone and timing it for real with -O3:
//
//     M=N=K=1024: 0.092 sec (23.4 GFLOPS)
//     M=N=K=2048: 0.705 sec (24.4 GFLOPS)
//     M=N=K=4096: 9.422 sec (14.6 GFLOPS)
//
// GCC's -O3 auto-vectorizes the cache-friendly i-k-j inner loop well
// enough that even 4096^3 finishes in under 10 seconds -- there was no
// real reason to skip CPU verification at any of these sizes, so all
// test cases below now have `verify_correctness = true`.
// =====================================================================
struct TestCase {
    int M;
    int N;
    int K;
    const char *name;
    bool verify_correctness;
};

const std::vector<TestCase> test_cases = {
    {1, 1, 1, "Single element matrices (1x1 x 1x1)", true},
    {16, 16, 16, "Tiny matrices, smaller than one BLOCK_K chunk", true},
    {31, 31, 31, "Smaller than one BLOCK_M x BLOCK_N tile in every dimension", true},
    {128, 128, 128, "Exactly one BLOCK_M x BLOCK_N tile, exactly one BLOCK_K chunk", true},
    {129, 128, 128, "One extra row (M not a multiple of BLOCK_M)", true},
    {128, 129, 128, "One extra column (N not a multiple of BLOCK_N)", true},
    {128, 128, 129, "One extra reduction step (K not a multiple of BLOCK_K)", true},
    {257, 129, 63, "Rectangular, non-square, partially-tiled matrices (exercises the N/K grid-size fix)", true},
    {512, 512, 512, "Larger square matrices", true},
    {1000, 777, 333, "Deliberately awkward, non-power-of-two sizes", true},
    {1024, 1024, 1024, "Performance benchmark size, now also CPU-verified", true},
    {2048, 2048, 2048, "Performance benchmark size, now also CPU-verified", true},
    {4096, 4096, 4096, "Performance benchmark size, now also CPU-verified", true},
};

// =====================================================================
// Requirement #5/#6: perf counters (latency statistics + achieved
// throughput), collected per test case and summarized across all test
// cases at the end.
// =====================================================================
struct PerfStats {
    float min_ms = 0.0f;
    float max_ms = 0.0f;
    float mean_ms = 0.0f;
    float median_ms = 0.0f;
    double tflops = 0.0;
};

PerfStats compute_perf_stats(std::vector<float> latencies_ms, long long total_flops) {
    PerfStats stats;
    std::sort(latencies_ms.begin(), latencies_ms.end());

    stats.min_ms = latencies_ms.front();
    stats.max_ms = latencies_ms.back();

    float sum_ms = 0.0f;
    for (float v : latencies_ms) sum_ms += v;
    stats.mean_ms = sum_ms / static_cast<float>(latencies_ms.size());

    size_t mid = latencies_ms.size() / 2;
    stats.median_ms = (latencies_ms.size() % 2 == 0)
                           ? 0.5f * (latencies_ms[mid - 1] + latencies_ms[mid])
                           : latencies_ms[mid];

    stats.tflops = static_cast<double>(total_flops) / (stats.mean_ms / 1000.0) / 1e12;
    return stats;
}

struct CorrectnessResult {
    bool checked = false;
    bool passed = true;
    float max_abs_error = 0.0f;
};

struct TestCaseResult {
    const TestCase *test_case = nullptr;
    CorrectnessResult correctness;
    PerfStats perf;
};

// =====================================================================
// Runs one test case end to end: allocate device memory, fill random
// inputs (requirement #2), optionally verify against the CPU reference
// (requirement #3), then run the configured number of warm-up and
// timed executions (requirement #4) to collect latency/throughput
// (requirement #5).
// =====================================================================
TestCaseResult run_test_case(const TestCase &tc, std::mt19937 &rng) {
    const int M = tc.M, N = tc.N, K = tc.K;
    const size_t size_A = static_cast<size_t>(M) * K;
    const size_t size_B = static_cast<size_t>(K) * N;
    const size_t size_C = static_cast<size_t>(M) * N;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h_A(size_A), h_B(size_B), h_C_gpu(size_C);
    for (auto &v : h_A) v = dist(rng);
    for (auto &v : h_B) v = dist(rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));

    TestCaseResult result;
    result.test_case = &tc;

    // ---- Correctness, against the plain C++ CPU reference ----
    if (tc.verify_correctness) {
        std::vector<float> h_C_cpu(size_C);
        cpu_matmul_reference(h_A, h_B, h_C_cpu, M, N, K);

        solve(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

        float max_abs_error = 0.0f;
        for (size_t i = 0; i < size_C; ++i) {
            max_abs_error = std::max(max_abs_error, std::fabs(h_C_gpu[i] - h_C_cpu[i]));
        }
        // Same tolerance formula used throughout this project's Triton
        // test suites: float32 accumulates more rounding error the
        // longer the K-reduction is, so the tolerance scales with K.
        const float tolerance = 1e-2f + 1e-3f * static_cast<float>(K);

        result.correctness.checked = true;
        result.correctness.max_abs_error = max_abs_error;
        result.correctness.passed = max_abs_error < tolerance;
    }

    // ---- Performance: warm-up runs (untimed), then timed runs ----
    for (int i = 0; i < NUM_WARMUP_RUNS; ++i) {
        solve(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> latencies_ms;
    latencies_ms.reserve(NUM_TIMED_RUNS);
    for (int i = 0; i < NUM_TIMED_RUNS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        solve(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        latencies_ms.push_back(elapsed_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const long long total_flops = 2LL * M * N * K;
    result.perf = compute_perf_stats(latencies_ms, total_flops);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

void print_separator() {
    printf("----------------------------------------------------------------------\n");
}

int main() {
    printf("======================================================================\n");
    printf("CUDA Matrix Multiplication (matmul_v0.cu) -- Correctness + Performance\n");
    printf("======================================================================\n");
    printf("NUM_WARMUP_RUNS = %d, NUM_TIMED_RUNS = %d (per test case)\n", NUM_WARMUP_RUNS, NUM_TIMED_RUNS);

    std::mt19937 rng(0); // fixed seed: reproducible across runs

    std::vector<TestCaseResult> results;
    results.reserve(test_cases.size());

    int correctness_checked_count = 0;
    int correctness_passed_count = 0;

    for (const TestCase &tc : test_cases) {
        print_separator();
        printf("Test: %s\n", tc.name);
        printf("Shapes: A=(%d,%d)  B=(%d,%d)  C=(%d,%d)\n", tc.M, tc.K, tc.K, tc.N, tc.M, tc.N);

        TestCaseResult r = run_test_case(tc, rng);

        if (r.correctness.checked) {
            correctness_checked_count++;
            if (r.correctness.passed) correctness_passed_count++;
            printf("Correctness: max abs error = %.6f -> %s\n", r.correctness.max_abs_error,
                   r.correctness.passed ? "PASS" : "FAIL");
        } else {
            printf("Correctness: skipped (CPU reference impractically slow at this size)\n");
        }

        printf("Latency (ms): min=%.4f  median=%.4f  mean=%.4f  max=%.4f\n",
               r.perf.min_ms, r.perf.median_ms, r.perf.mean_ms, r.perf.max_ms);
        printf("Throughput: %.2f TFLOPS (computed from mean latency)\n", r.perf.tflops);

        results.push_back(r);
    }

    print_separator();
    printf("======================================================================\n");
    printf("Summary\n");
    printf("======================================================================\n");
    printf("%-10s %-8s %-8s %-8s %10s %10s %10s\n", "M", "N", "K", "Correct?", "Mean(ms)", "TFLOPS", "MaxErr");
    for (const TestCaseResult &r : results) {
        const TestCase &tc = *r.test_case;
        if (r.correctness.checked) {
            printf("%-10d %-8d %-8d %-8s %10.4f %10.2f %10.6f\n", tc.M, tc.N, tc.K,
                   r.correctness.passed ? "PASS" : "FAIL", r.perf.mean_ms, r.perf.tflops,
                   r.correctness.max_abs_error);
        } else {
            printf("%-10d %-8d %-8d %-8s %10.4f %10.2f %10s\n", tc.M, tc.N, tc.K, "SKIP",
                   r.perf.mean_ms, r.perf.tflops, "n/a");
        }
    }

    print_separator();
    printf("Correctness: %d / %d checked test cases passed\n", correctness_passed_count, correctness_checked_count);

    double sum_tflops = 0.0;
    for (const TestCaseResult &r : results) sum_tflops += r.perf.tflops;
    printf("Average throughput across all %zu test cases: %.2f TFLOPS\n", results.size(),
           sum_tflops / static_cast<double>(results.size()));
    print_separator();

    bool all_passed = (correctness_checked_count == correctness_passed_count);
    printf("%s\n", all_passed ? "ALL CORRECTNESS TESTS PASSED!" : "SOME CORRECTNESS TESTS FAILED.");
    printf("======================================================================\n");

    return all_passed ? 0 : 1;
}

*/

/*
// matmul_v1.cu
//
// =====================================================================
// PROBLEM STATEMENT (identical to matmul_v0.cu)
// =====================================================================
// Compute C = A @ B, where A is (M x K), B is (K x N), C is (M x N).
// All three matrices are stored in ROW-MAJOR format in GPU global memory.
//
// =====================================================================
// WHAT THIS FILE CHANGES RELATIVE TO matmul_v0.cu, AND WHY
// =====================================================================
// matmul_v0.cu stores A TRANSPOSED in shared memory (sA[k][m]) and B
// NOT transposed (sB[k][n]). This file answers a direct follow-up
// question: is the opposite possible -- A NOT transposed, B
// transposed instead?
//
// Short answer: yes, it is possible, and it is implemented below. But
// it is important to be precise about what that choice actually costs,
// rather than just calling it "a different but equally valid option":
//
//   WHY A gets transposed in v0: the compute loop needs, for one fixed
//   k at a time, a contiguous COLUMN of A (all m values). A's own
//   row-major storage makes a ROW of A contiguous (fixed m, varying
//   k) -- the wrong shape for what's needed -- so v0 transposes A on
//   the way into shared memory to make that column contiguous, which
//   is what makes the FLOAT4 vectorized register-load for A possible.
//
//   WHY B does NOT get transposed in v0: the compute loop needs, for
//   that same fixed k, a contiguous ROW of B (all n values). B's own
//   row-major storage ALREADY makes a row of B contiguous (fixed k,
//   varying n) -- exactly the shape needed, with zero rearranging.
//   There is nothing to gain by transposing B in v0.
//
// So flipping the choice (A natural, B transposed) is NOT a symmetric
// swap. It is strictly worse for vectorization, on both sides at once:
//
//   - A, left in its natural [m][k] shape in shared memory: grabbing
//     "all m for fixed k" is now a STRIDED read (stride = BLOCK_K),
//     same problem v0 specifically transposes to avoid.
//   - B, forced into a transposed [n][k] shape: grabbing "all n for
//     fixed k" is now ALSO a strided read (stride = BLOCK_K) -- a
//     brand new problem that did not exist in v0, since B's natural
//     layout was already the right shape.
//
// Concretely, that means the per-k register-staging step (loading
// tCsA/tCsB out of shared memory, once per k, inside the hottest loop
// of the whole kernel) can no longer use FLOAT4 vectorized loads for
// EITHER operand here -- both become 4 separate scalar loads. In
// matmul_v0.cu, only a non-transposed A would have had that problem;
// here, both A and B have it.
//
// Everything else -- the outer-product accumulate step, the warp-aware
// thread layout, the two-level double buffering, the final masked
// store to C, the solve() bug fixes -- is unchanged from matmul_v0.cu.
// The bank-conflict-avoidance XOR swizzle used on A's indices in v0 is
// dropped here for clarity: swizzling only matters for the vectorized
// access pattern it was designed around, and that access pattern no
// longer exists in this version.
//
// Verification test cases and the performance-testing harness
// (warm-up/timed run counts, perf counters, printed summary) are kept
// IDENTICAL to matmul_v0.cu, as requested, so the two files' printed
// results can be directly compared line for line.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

// NOTE: FLOAT4 is unused in this file's kernel (see the "what changed"
// note above -- both operands' register-staging reads are scalar now),
// but is kept defined for anyone diffing this file against matmul_v0.cu.
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// =====================================================================
// Perf-testing knobs -- identical to matmul_v0.cu.
// =====================================================================
constexpr int NUM_WARMUP_RUNS = 5;
constexpr int NUM_TIMED_RUNS = 20;

// =====================================================================
// CUDA error-checking helper -- identical to matmul_v0.cu.
// =====================================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t cuda_check_err__ = (call);                               \
        if (cuda_check_err__ != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(cuda_check_err__));                   \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// =====================================================================
// Tiling constants -- identical to matmul_v0.cu. None of these needed
// to change: this variant only changes shared-memory LAYOUT and the
// register-staging READ pattern, not the thread-to-data assignment.
// =====================================================================
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;
constexpr int BLOCK_SIZE = 256;

constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X;
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X;
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X;

constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_Y;
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y;

constexpr int THREAD_C_WARP_X = 8;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X;
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X;

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                              float *C, int M, int N, int K) {
    int col0 = blockIdx.x * BLOCK_N;
    int row0 = blockIdx.y * BLOCK_M;
    int tidx = threadIdx.x;

    // --- THE CORE CHANGE: shared-memory shapes are flipped relative
    // to matmul_v0.cu. ---
    //   v0: sA[2][BLOCK_K][BLOCK_M]  (A TRANSPOSED: k outer, m inner)
    //       sB[2][BLOCK_K][BLOCK_N]  (B natural:    k outer, n inner)
    //   v1: sA[2][BLOCK_M][BLOCK_K]  (A natural:    m outer, k inner)
    //       sB[2][BLOCK_N][BLOCK_K]  (B TRANSPOSED: n outer, k inner)
    __shared__ float sA[2][BLOCK_M][BLOCK_K];
    __shared__ float sB[2][BLOCK_N][BLOCK_K];

    int tAx = tidx & (THREAD_A_LAYOUT_X - 1);
    int tAy = tidx / THREAD_A_LAYOUT_X;
    int tBx = tidx & (THREAD_B_LAYOUT_X - 1);
    int tBy = tidx / THREAD_B_LAYOUT_X;

    int warpId = tidx >> 5;
    int laneId = tidx & 31;
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1);
    int warpy = warpId / THREAD_C_WARP_DIM_X;
    int lanex = (laneId & 15) >> 1;
    int laney = ((laneId >> 4) << 1) + (laneId & 1);
    int tCx = warpx * THREAD_C_WARP_X + lanex;
    int tCy = warpy * THREAD_C_WARP_Y + laney;

    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};
    float tCsA[2][THREAD_C_Y_TILE_SIZE];
    float tCsB[2][THREAD_C_X_TILE_SIZE];
    int bufferId = 0;

    // ---- Prologue load of A: natural [m][k] layout, no transpose,
    // no swizzle (there is no vectorized access pattern left to
    // protect with a swizzle in this version). ----
#pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        sA[0][i + tAy][tAx] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
    }
    // ---- Prologue load of B: TRANSPOSED [n][k] layout (flipped from
    // B's own natural [k][n] row-major order). ----
#pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = i + tBy;
#pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][j + tBx][i + tBy] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
    }
    __syncthreads();

    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
#pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; ++tk) {
            if (tk > 0) {
#pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
#pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }
            if (tk < BLOCK_K) {
                // ---- Register-staging read for A: fixed tk, sweep m.
                // sA is now [m][k] (k innermost), so 4 consecutive
                // m-values are NOT contiguous -- 4 separate scalar
                // reads replace the single FLOAT4 read matmul_v0.cu
                // used here. ----
#pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm++) {
                    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
                    tCsA[tk & 1][(tm << 2) + 0] = sA[bufferId][r + 0][tk];
                    tCsA[tk & 1][(tm << 2) + 1] = sA[bufferId][r + 1][tk];
                    tCsA[tk & 1][(tm << 2) + 2] = sA[bufferId][r + 2][tk];
                    tCsA[tk & 1][(tm << 2) + 3] = sA[bufferId][r + 3][tk];
                }
                // ---- Register-staging read for B: fixed tk, sweep n.
                // sB is now [n][k] (k innermost), so this is ALSO 4
                // scalar reads -- in matmul_v0.cu this read was a
                // single FLOAT4 (B's natural layout already matched).
                // ----
#pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn++) {
                    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2;
                    tCsB[tk & 1][(tn << 2) + 0] = sB[bufferId][c + 0][tk];
                    tCsB[tk & 1][(tn << 2) + 1] = sB[bufferId][c + 1][tk];
                    tCsB[tk & 1][(tn << 2) + 2] = sB[bufferId][c + 2][tk];
                    tCsB[tk & 1][(tn << 2) + 3] = sB[bufferId][c + 3][tk];
                }
            }
        }
        if (k < K) {
            int c = k + tAx;
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                sA[bufferId ^ 1][i + tAy][tAx] = (r < M && c < K) ? A[r * K + c] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
#pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c2 = col0 + j + tBx;
                    sB[bufferId ^ 1][j + tBx][i + tBy] =
                        (r < K && c2 < N) ? B[r * N + c2] : 0.0f;
                }
            }
            __syncthreads();
        }
        bufferId ^= 1;
    }

    // ---- Final write-back to C: UNCHANGED from matmul_v0.cu. acc[][]
    // holds the same values regardless of how sA/sB were laid out
    // internally, so this step needs no modification at all. ----
#pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
#pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU).
// Identical to matmul_v0.cu's (already bug-fixed) solve(): proper
// extern "C" quoting, kernel launched with (M, N, K) in that order,
// and the grid's x-dimension sized from N.
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                       int K) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// =====================================================================
// Plain C++ CPU baseline -- identical to matmul_v0.cu.
// =====================================================================
void cpu_matmul_reference(const std::vector<float> &A,
                           const std::vector<float> &B, std::vector<float> &C,
                           int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < M; ++i) {
        float *c_row = &C[static_cast<size_t>(i) * N];
        for (int k = 0; k < K; ++k) {
            float a_val = A[static_cast<size_t>(i) * K + k];
            const float *b_row = &B[static_cast<size_t>(k) * N];
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

// =====================================================================
// Test cases -- kept IDENTICAL to matmul_v0.cu, as requested, so the
// two files' results are directly comparable size-for-size.
// =====================================================================
struct TestCase {
    int M;
    int N;
    int K;
    const char *name;
    bool verify_correctness;
};

const std::vector<TestCase> test_cases = {
    {1, 1, 1, "Single element matrices (1x1 x 1x1)", true},
    {16, 16, 16, "Tiny matrices, smaller than one BLOCK_K chunk", true},
    {31, 31, 31, "Smaller than one BLOCK_M x BLOCK_N tile in every dimension", true},
    {128, 128, 128, "Exactly one BLOCK_M x BLOCK_N tile, exactly one BLOCK_K chunk", true},
    {129, 128, 128, "One extra row (M not a multiple of BLOCK_M)", true},
    {128, 129, 128, "One extra column (N not a multiple of BLOCK_N)", true},
    {128, 128, 129, "One extra reduction step (K not a multiple of BLOCK_K)", true},
    {257, 129, 63, "Rectangular, non-square, partially-tiled matrices (exercises the N/K grid-size fix)", true},
    {512, 512, 512, "Larger square matrices", true},
    {1000, 777, 333, "Deliberately awkward, non-power-of-two sizes", true},
    {1024, 1024, 1024, "Performance benchmark size, now also CPU-verified", true},
    {2048, 2048, 2048, "Performance benchmark size, now also CPU-verified", true},
    {4096, 4096, 4096, "Performance benchmark size, now also CPU-verified", true},
};

// =====================================================================
// Perf counters -- identical to matmul_v0.cu.
// =====================================================================
struct PerfStats {
    float min_ms = 0.0f;
    float max_ms = 0.0f;
    float mean_ms = 0.0f;
    float median_ms = 0.0f;
    double tflops = 0.0;
};

PerfStats compute_perf_stats(std::vector<float> latencies_ms, long long total_flops) {
    PerfStats stats;
    std::sort(latencies_ms.begin(), latencies_ms.end());

    stats.min_ms = latencies_ms.front();
    stats.max_ms = latencies_ms.back();

    float sum_ms = 0.0f;
    for (float v : latencies_ms) sum_ms += v;
    stats.mean_ms = sum_ms / static_cast<float>(latencies_ms.size());

    size_t mid = latencies_ms.size() / 2;
    stats.median_ms = (latencies_ms.size() % 2 == 0)
                           ? 0.5f * (latencies_ms[mid - 1] + latencies_ms[mid])
                           : latencies_ms[mid];

    stats.tflops = static_cast<double>(total_flops) / (stats.mean_ms / 1000.0) / 1e12;
    return stats;
}

struct CorrectnessResult {
    bool checked = false;
    bool passed = true;
    float max_abs_error = 0.0f;
};

struct TestCaseResult {
    const TestCase *test_case = nullptr;
    CorrectnessResult correctness;
    PerfStats perf;
};

// =====================================================================
// Runs one test case end to end -- identical structure to matmul_v0.cu.
// =====================================================================
TestCaseResult run_test_case(const TestCase &tc, std::mt19937 &rng) {
    const int M = tc.M, N = tc.N, K = tc.K;
    const size_t size_A = static_cast<size_t>(M) * K;
    const size_t size_B = static_cast<size_t>(K) * N;
    const size_t size_C = static_cast<size_t>(M) * N;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h_A(size_A), h_B(size_B), h_C_gpu(size_C);
    for (auto &v : h_A) v = dist(rng);
    for (auto &v : h_B) v = dist(rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));

    TestCaseResult result;
    result.test_case = &tc;

    if (tc.verify_correctness) {
        std::vector<float> h_C_cpu(size_C);
        cpu_matmul_reference(h_A, h_B, h_C_cpu, M, N, K);

        solve(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

        float max_abs_error = 0.0f;
        for (size_t i = 0; i < size_C; ++i) {
            max_abs_error = std::max(max_abs_error, std::fabs(h_C_gpu[i] - h_C_cpu[i]));
        }
        const float tolerance = 1e-2f + 1e-3f * static_cast<float>(K);

        result.correctness.checked = true;
        result.correctness.max_abs_error = max_abs_error;
        result.correctness.passed = max_abs_error < tolerance;
    }

    for (int i = 0; i < NUM_WARMUP_RUNS; ++i) {
        solve(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> latencies_ms;
    latencies_ms.reserve(NUM_TIMED_RUNS);
    for (int i = 0; i < NUM_TIMED_RUNS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        solve(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        latencies_ms.push_back(elapsed_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const long long total_flops = 2LL * M * N * K;
    result.perf = compute_perf_stats(latencies_ms, total_flops);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

void print_separator() {
    printf("----------------------------------------------------------------------\n");
}

int main() {
    printf("======================================================================\n");
    printf("CUDA Matrix Multiplication (matmul_v1.cu) -- A natural, B transposed\n");
    printf("======================================================================\n");
    printf("NUM_WARMUP_RUNS = %d, NUM_TIMED_RUNS = %d (per test case)\n", NUM_WARMUP_RUNS, NUM_TIMED_RUNS);

    std::mt19937 rng(0);

    std::vector<TestCaseResult> results;
    results.reserve(test_cases.size());

    int correctness_checked_count = 0;
    int correctness_passed_count = 0;

    for (const TestCase &tc : test_cases) {
        print_separator();
        printf("Test: %s\n", tc.name);
        printf("Shapes: A=(%d,%d)  B=(%d,%d)  C=(%d,%d)\n", tc.M, tc.K, tc.K, tc.N, tc.M, tc.N);

        TestCaseResult r = run_test_case(tc, rng);

        if (r.correctness.checked) {
            correctness_checked_count++;
            if (r.correctness.passed) correctness_passed_count++;
            printf("Correctness: max abs error = %.6f -> %s\n", r.correctness.max_abs_error,
                   r.correctness.passed ? "PASS" : "FAIL");
        } else {
            printf("Correctness: skipped\n");
        }

        printf("Latency (ms): min=%.4f  median=%.4f  mean=%.4f  max=%.4f\n",
               r.perf.min_ms, r.perf.median_ms, r.perf.mean_ms, r.perf.max_ms);
        printf("Throughput: %.2f TFLOPS (computed from mean latency)\n", r.perf.tflops);

        results.push_back(r);
    }

    print_separator();
    printf("======================================================================\n");
    printf("Summary\n");
    printf("======================================================================\n");
    printf("%-10s %-8s %-8s %-8s %10s %10s %10s\n", "M", "N", "K", "Correct?", "Mean(ms)", "TFLOPS", "MaxErr");
    for (const TestCaseResult &r : results) {
        const TestCase &tc = *r.test_case;
        if (r.correctness.checked) {
            printf("%-10d %-8d %-8d %-8s %10.4f %10.2f %10.6f\n", tc.M, tc.N, tc.K,
                   r.correctness.passed ? "PASS" : "FAIL", r.perf.mean_ms, r.perf.tflops,
                   r.correctness.max_abs_error);
        } else {
            printf("%-10d %-8d %-8d %-8s %10.4f %10.2f %10s\n", tc.M, tc.N, tc.K, "SKIP",
                   r.perf.mean_ms, r.perf.tflops, "n/a");
        }
    }

    print_separator();
    printf("Correctness: %d / %d checked test cases passed\n", correctness_passed_count, correctness_checked_count);

    double sum_tflops = 0.0;
    for (const TestCaseResult &r : results) sum_tflops += r.perf.tflops;
    printf("Average throughput across all %zu test cases: %.2f TFLOPS\n", results.size(),
           sum_tflops / static_cast<double>(results.size()));
    print_separator();

    bool all_passed = (correctness_checked_count == correctness_passed_count);
    printf("%s\n", all_passed ? "ALL CORRECTNESS TESTS PASSED!" : "SOME CORRECTNESS TESTS FAILED.");
    printf("======================================================================\n");

    return all_passed ? 0 : 1;
}

*/

/*
// matmul_v2_bfix.cu
//
// =====================================================================
// INVESTIGATION FILE -- not a general-purpose deliverable, built to
// empirically test a specific hypothesis raised while explaining
// matmul_v2.cu: does B's shared-memory write pattern have a real,
// measurable 8-way bank conflict, and does fixing it actually help?
// =====================================================================
//
// THE HYPOTHESIS: matmul_v2.cu stores B transposed as sB[2][BLOCK_N][BLOCK_K]
// = sB[2][128][8]. Its write is `sB[0][j + tBx][i + tBy] = ...`, i.e. flat
// address = (j+tBx)*8 + (i+tBy). Within one warp, tBy is constant (warp-wide,
// since THREAD_B_LAYOUT_X=32 means tBy=tidx/32 is the same for all 32
// threads in a warp) and tBx sweeps the full 0..31. Address = tBx*8 + const.
// Since 8*4=32 (an exact multiple of the bank count), this collapses into
// only 4 distinct banks, each hit by 8 threads at once -- an 8-way bank
// conflict, with NO swizzle ever added to fix it.
//
// THE FIX, derived and verified by hand first: XOR the k-index with a mask
// built from the bits of n that `*8 mod 32` throws away:
//
//     mask = (n >> 2) & 7
//     sB[0][n][k ^ mask] = <value for real (n, k)>        (write)
//     value = sB[bufferId][n][k ^ mask]                    (read, same mask)
//
// Verified by hand for warp 0, n=0..31: without the fix, banks collapse to
// {0,8,16,24} (8-way conflict each). With the fix, all 32 threads land on
// all 32 different banks, zero conflict.
//
// IMPORTANT CAVEAT: this mask is NOT always a multiple of 4 (it can be any
// value 0..7), so it does NOT preserve "4 consecutive stay consecutive"
// the way matmul_v0.cu's `tAx << 2` swizzle did. That means B's row-caching
// read can no longer use FLOAT4 safely here -- it is downgraded to 8
// scalar reads for this experiment. A is completely untouched (still
// FLOAT4, still conflict-free, as established).
//
// This isolates the real question cleanly: does removing B's bank conflict
// (at the cost of losing B's FLOAT4 read) produce a net win over
// matmul_v2.cu, matched test case for test case?
//
// Verification test cases and the performance-testing harness are kept
// IDENTICAL to matmul_v2.cu.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

constexpr int NUM_WARMUP_RUNS = 5;
constexpr int NUM_TIMED_RUNS = 20;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t cuda_check_err__ = (call);                               \
        if (cuda_check_err__ != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(cuda_check_err__));                   \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;
constexpr int BLOCK_SIZE = 256;

constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X;
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X;
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X;

constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_Y;
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y;

constexpr int THREAD_C_WARP_X = 8;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X;
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X;

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                              float *C, int M, int N, int K) {
    int col0 = blockIdx.x * BLOCK_N;
    int row0 = blockIdx.y * BLOCK_M;
    int tidx = threadIdx.x;

    __shared__ float sA[2][BLOCK_M][BLOCK_K]; // A natural:    m outer, k inner (unchanged)
    __shared__ float sB[2][BLOCK_N][BLOCK_K]; // B transposed: n outer, k inner (now swizzled)

    int tAx = tidx & (THREAD_A_LAYOUT_X - 1);
    int tAy = tidx / THREAD_A_LAYOUT_X;
    int tBx = tidx & (THREAD_B_LAYOUT_X - 1);
    int tBy = tidx / THREAD_B_LAYOUT_X;

    int warpId = tidx >> 5;
    int laneId = tidx & 31;
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1);
    int warpy = warpId / THREAD_C_WARP_DIM_X;
    int lanex = (laneId & 15) >> 1;
    int laney = ((laneId >> 4) << 1) + (laneId & 1);
    int tCx = warpx * THREAD_C_WARP_X + lanex;
    int tCy = warpy * THREAD_C_WARP_Y + laney;

    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};
    float regA[THREAD_C_Y_TILE_SIZE][BLOCK_K];
    float regB[THREAD_C_X_TILE_SIZE][BLOCK_K];
    int bufferId = 0;

    // ---- Prologue load of A: unchanged, still FLOAT4-eligible, still
    // conflict-free (see the investigation note at the top of this file). ----
#pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        sA[0][i + tAy][tAx] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
    }
    // ---- Prologue load of B: THE FIX. Same values, same real (n,k)
    // positions, but written to a bank-scrambled physical slot. ----
#pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = i + tBy;
#pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            int n = j + tBx;
            int kk = i + tBy;
            int mask = (n >> 2) & 7;
            sB[0][n][kk ^ mask] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
    }
    __syncthreads();

    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // ---- Stage A: unchanged, still FLOAT4. ----
#pragma unroll
        for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
            int m = tCy + tm * THREAD_C_LAYOUT_Y;
            FLOAT4(regA[tm][0]) = FLOAT4(sA[bufferId][m][0]);
            FLOAT4(regA[tm][4]) = FLOAT4(sA[bufferId][m][4]);
        }
        // ---- Stage B: THE FIX's read side. Same mask as the write,
        // applied per-element -- 8 scalar reads instead of 2 FLOAT4
        // reads, because the mask is not always a multiple of 4, so
        // "4 consecutive" is no longer guaranteed to stay consecutive
        // after the scramble. ----
#pragma unroll
        for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
            int n = tCx + tn * THREAD_C_LAYOUT_X;
            int mask = (n >> 2) & 7;
            regB[tn][0] = sB[bufferId][n][0 ^ mask];
            regB[tn][1] = sB[bufferId][n][1 ^ mask];
            regB[tn][2] = sB[bufferId][n][2 ^ mask];
            regB[tn][3] = sB[bufferId][n][3 ^ mask];
            regB[tn][4] = sB[bufferId][n][4 ^ mask];
            regB[tn][5] = sB[bufferId][n][5 ^ mask];
            regB[tn][6] = sB[bufferId][n][6 ^ mask];
            regB[tn][7] = sB[bufferId][n][7 ^ mask];
        }

        // ---- Dot-product accumulation: unchanged from matmul_v2.cu. ----
#pragma unroll
        for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
#pragma unroll
            for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                float partial = 0.0f;
#pragma unroll
                for (int kk = 0; kk < BLOCK_K; kk++) {
                    partial += regA[tm][kk] * regB[tn][kk];
                }
                acc[tm][tn] += partial;
            }
        }

        if (k < K) {
            int c = k + tAx;
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                sA[bufferId ^ 1][i + tAy][tAx] = (r < M && c < K) ? A[r * K + c] : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
#pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c2 = col0 + j + tBx;
                    int n = j + tBx;
                    int kk2 = i + tBy;
                    int mask = (n >> 2) & 7;
                    sB[bufferId ^ 1][n][kk2 ^ mask] =
                        (r < K && c2 < N) ? B[r * N + c2] : 0.0f;
                }
            }
            __syncthreads();
        }
        bufferId ^= 1;
    }

#pragma unroll
    for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
        int r = row0 + tCy + tm * THREAD_C_LAYOUT_Y;
#pragma unroll
        for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
            int c = col0 + tCx + tn * THREAD_C_LAYOUT_X;
            if (r < M && c < N) C[r * N + c] = acc[tm][tn];
        }
    }
}

extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                       int K) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cpu_matmul_reference(const std::vector<float> &A,
                           const std::vector<float> &B, std::vector<float> &C,
                           int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < M; ++i) {
        float *c_row = &C[static_cast<size_t>(i) * N];
        for (int k = 0; k < K; ++k) {
            float a_val = A[static_cast<size_t>(i) * K + k];
            const float *b_row = &B[static_cast<size_t>(k) * N];
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

struct TestCase {
    int M;
    int N;
    int K;
    const char *name;
    bool verify_correctness;
};

const std::vector<TestCase> test_cases = {
    {1, 1, 1, "Single element matrices (1x1 x 1x1)", true},
    {16, 16, 16, "Tiny matrices, smaller than one BLOCK_K chunk", true},
    {31, 31, 31, "Smaller than one BLOCK_M x BLOCK_N tile in every dimension", true},
    {128, 128, 128, "Exactly one BLOCK_M x BLOCK_N tile, exactly one BLOCK_K chunk", true},
    {129, 128, 128, "One extra row (M not a multiple of BLOCK_M)", true},
    {128, 129, 128, "One extra column (N not a multiple of BLOCK_N)", true},
    {128, 128, 129, "One extra reduction step (K not a multiple of BLOCK_K)", true},
    {257, 129, 63, "Rectangular, non-square, partially-tiled matrices (exercises the N/K grid-size fix)", true},
    {512, 512, 512, "Larger square matrices", true},
    {1000, 777, 333, "Deliberately awkward, non-power-of-two sizes", true},
    {1024, 1024, 1024, "Performance benchmark size, now also CPU-verified", true},
    {2048, 2048, 2048, "Performance benchmark size, now also CPU-verified", true},
    {4096, 4096, 4096, "Performance benchmark size, now also CPU-verified", true},
};

struct PerfStats {
    float min_ms = 0.0f;
    float max_ms = 0.0f;
    float mean_ms = 0.0f;
    float median_ms = 0.0f;
    double tflops = 0.0;
};

PerfStats compute_perf_stats(std::vector<float> latencies_ms, long long total_flops) {
    PerfStats stats;
    std::sort(latencies_ms.begin(), latencies_ms.end());
    stats.min_ms = latencies_ms.front();
    stats.max_ms = latencies_ms.back();
    float sum_ms = 0.0f;
    for (float v : latencies_ms) sum_ms += v;
    stats.mean_ms = sum_ms / static_cast<float>(latencies_ms.size());
    size_t mid = latencies_ms.size() / 2;
    stats.median_ms = (latencies_ms.size() % 2 == 0)
                           ? 0.5f * (latencies_ms[mid - 1] + latencies_ms[mid])
                           : latencies_ms[mid];
    stats.tflops = static_cast<double>(total_flops) / (stats.mean_ms / 1000.0) / 1e12;
    return stats;
}

struct CorrectnessResult {
    bool checked = false;
    bool passed = true;
    float max_abs_error = 0.0f;
};

struct TestCaseResult {
    const TestCase *test_case = nullptr;
    CorrectnessResult correctness;
    PerfStats perf;
};

TestCaseResult run_test_case(const TestCase &tc, std::mt19937 &rng) {
    const int M = tc.M, N = tc.N, K = tc.K;
    const size_t size_A = static_cast<size_t>(M) * K;
    const size_t size_B = static_cast<size_t>(K) * N;
    const size_t size_C = static_cast<size_t>(M) * N;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h_A(size_A), h_B(size_B), h_C_gpu(size_C);
    for (auto &v : h_A) v = dist(rng);
    for (auto &v : h_B) v = dist(rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));

    TestCaseResult result;
    result.test_case = &tc;

    if (tc.verify_correctness) {
        std::vector<float> h_C_cpu(size_C);
        cpu_matmul_reference(h_A, h_B, h_C_cpu, M, N, K);
        solve(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

        float max_abs_error = 0.0f;
        for (size_t i = 0; i < size_C; ++i) {
            max_abs_error = std::max(max_abs_error, std::fabs(h_C_gpu[i] - h_C_cpu[i]));
        }
        const float tolerance = 1e-2f + 1e-3f * static_cast<float>(K);
        result.correctness.checked = true;
        result.correctness.max_abs_error = max_abs_error;
        result.correctness.passed = max_abs_error < tolerance;
    }

    for (int i = 0; i < NUM_WARMUP_RUNS; ++i) {
        solve(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> latencies_ms;
    latencies_ms.reserve(NUM_TIMED_RUNS);
    for (int i = 0; i < NUM_TIMED_RUNS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        solve(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        latencies_ms.push_back(elapsed_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const long long total_flops = 2LL * M * N * K;
    result.perf = compute_perf_stats(latencies_ms, total_flops);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return result;
}

void print_separator() {
    printf("----------------------------------------------------------------------\n");
}

int main() {
    printf("======================================================================\n");
    printf("CUDA Matrix Multiplication (matmul_v2_bfix.cu) -- B bank-conflict fix\n");
    printf("======================================================================\n");
    printf("NUM_WARMUP_RUNS = %d, NUM_TIMED_RUNS = %d (per test case)\n", NUM_WARMUP_RUNS, NUM_TIMED_RUNS);

    std::mt19937 rng(0);
    std::vector<TestCaseResult> results;
    results.reserve(test_cases.size());

    int correctness_checked_count = 0;
    int correctness_passed_count = 0;

    for (const TestCase &tc : test_cases) {
        print_separator();
        printf("Test: %s\n", tc.name);
        printf("Shapes: A=(%d,%d)  B=(%d,%d)  C=(%d,%d)\n", tc.M, tc.K, tc.K, tc.N, tc.M, tc.N);

        TestCaseResult r = run_test_case(tc, rng);

        if (r.correctness.checked) {
            correctness_checked_count++;
            if (r.correctness.passed) correctness_passed_count++;
            printf("Correctness: max abs error = %.6f -> %s\n", r.correctness.max_abs_error,
                   r.correctness.passed ? "PASS" : "FAIL");
        } else {
            printf("Correctness: skipped\n");
        }

        printf("Latency (ms): min=%.4f  median=%.4f  mean=%.4f  max=%.4f\n",
               r.perf.min_ms, r.perf.median_ms, r.perf.mean_ms, r.perf.max_ms);
        printf("Throughput: %.2f TFLOPS (computed from mean latency)\n", r.perf.tflops);

        results.push_back(r);
    }

    print_separator();
    printf("Summary\n");
    printf("%-10s %-8s %-8s %-8s %10s %10s %10s\n", "M", "N", "K", "Correct?", "Mean(ms)", "TFLOPS", "MaxErr");
    for (const TestCaseResult &r : results) {
        const TestCase &tc = *r.test_case;
        printf("%-10d %-8d %-8d %-8s %10.4f %10.2f %10.6f\n", tc.M, tc.N, tc.K,
               r.correctness.passed ? "PASS" : "FAIL", r.perf.mean_ms, r.perf.tflops,
               r.correctness.max_abs_error);
    }

    print_separator();
    printf("Correctness: %d / %d checked test cases passed\n", correctness_passed_count, correctness_checked_count);
    bool all_passed = (correctness_checked_count == correctness_passed_count);
    printf("%s\n", all_passed ? "ALL CORRECTNESS TESTS PASSED!" : "SOME CORRECTNESS TESTS FAILED.");
    return all_passed ? 0 : 1;
}

*/

/*
"""
simulation_v1.py

A line-by-line, EXECUTION-ORDER Python re-implementation of
matmul_v0.cu's `matrix_multiplication_kernel` for exactly ONE thread
(TIDX), with no real matrix data anywhere -- every single operation
the real kernel performs (a shared-memory write, a register load, an
accumulate, a reload, a final C write) is logged as ONE ROW in a
SINGLE, continuous, unified table, in the EXACT sequence the real
CUDA source code executes them.

This is deliberately NOT split into separate per-phase tables (unlike
simulation_v0.py) -- the whole point here is to let you watch the
chain

        global A / B  -->  sA / sB  -->  tCsA / tCsB  -->  acc  -->  C

unfold as one continuous, in-order log, exactly mirroring the control
flow of the real __global__ function: every `if`, every nested `for`,
in the same order, with nothing reordered or grouped for presentation.

Only INDEX values are tracked (array positions, flat offsets, loop
counters, buffer ids) -- never real floating point matrix values,
since there is no actual A/B/C data here at all.

PRESET VALUES (same as simulation_v0.py):
    M=256 N=256 K=16
    BLOCK_M=128 BLOCK_N=128 BLOCK_K=8 BLOCK_SIZE=256
    THREAD_A_LAYOUT_X=8   THREAD_A_LAYOUT_Y=32
    THREAD_B_LAYOUT_X=32  THREAD_B_LAYOUT_Y=8
    THREAD_C_LAYOUT_X=16  THREAD_C_LAYOUT_Y=16
    THREAD_C_X_TILE_SIZE=8  THREAD_C_Y_TILE_SIZE=8
    THREAD_C_WARP_X=8  THREAD_C_WARP_Y=4  THREAD_C_WARP_DIM_X=2
    tidx = 13

blockIdx is not given, so block (0, 0) is assumed (row0=0, col0=0).
"""

# =====================================================================
# PRESET CONSTANTS
# =====================================================================
M, N, K = 256, 256, 16
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 8
BLOCK_SIZE = 256

THREAD_A_LAYOUT_X = 8
THREAD_A_LAYOUT_Y = 32
THREAD_B_LAYOUT_X = 32
THREAD_B_LAYOUT_Y = 8
THREAD_C_LAYOUT_X = 16
THREAD_C_LAYOUT_Y = 16
THREAD_C_X_TILE_SIZE = 8
THREAD_C_Y_TILE_SIZE = 8
THREAD_C_WARP_X = 8
THREAD_C_WARP_Y = 4
THREAD_C_WARP_DIM_X = 2

TIDX = 13
BLOCK_IDX_X = 0
BLOCK_IDX_Y = 0

row0 = BLOCK_IDX_Y * BLOCK_M
col0 = BLOCK_IDX_X * BLOCK_N

# =====================================================================
# The single, unified execution log. Every column that could possibly
# matter for ANY phase is included on EVERY row; whichever ones do not
# apply to a given row's phase are simply left as "-".
# =====================================================================
LOG_COLUMNS = [
    "step", "phase", "k", "tk", "i", "j", "tm", "tn",
    "tk-1", "(tk-1)&1", "tk&1", "bufferId",
    "r", "c", "swizzle_term", "swizzled_idx",
    "source", "destination", "in_bounds",
]
log = []
_step_counter = [0]


def log_row(phase, **kwargs):
    _step_counter[0] += 1
    row = {col: "-" for col in LOG_COLUMNS}
    row["step"] = _step_counter[0]
    row["phase"] = phase
    row.update(kwargs)
    log.append(row)


def print_unified_table():
    widths = {
        col: max(len(col), max((len(str(r[col])) for r in log), default=0))
        for col in LOG_COLUMNS
    }
    header = " | ".join(col.ljust(widths[col]) for col in LOG_COLUMNS)
    print(header)
    print("-" * len(header))
    for r in log:
        print(" | ".join(str(r[col]).ljust(widths[col]) for col in LOG_COLUMNS))


# =====================================================================
# Fixed, per-thread index values -- computed ONCE, printed as a short
# preamble (not part of the iteration log, since they never change).
# =====================================================================
tAx = TIDX & (THREAD_A_LAYOUT_X - 1)
tAy = TIDX // THREAD_A_LAYOUT_X
tBx = TIDX & (THREAD_B_LAYOUT_X - 1)
tBy = TIDX // THREAD_B_LAYOUT_X

warpId = TIDX >> 5
laneId = TIDX & 31
warpx = warpId & (THREAD_C_WARP_DIM_X - 1)
warpy = warpId // THREAD_C_WARP_DIM_X
lanex = (laneId & 15) >> 1
laney = ((laneId >> 4) << 1) + (laneId & 1)
tCx = warpx * THREAD_C_WARP_X + lanex
tCy = warpy * THREAD_C_WARP_Y + laney

print("=" * 100)
print("FIXED PER-THREAD VALUES  (tidx=%d, block=(%d,%d))" % (TIDX, BLOCK_IDX_X, BLOCK_IDX_Y))
print("=" * 100)
print(
    "row0=%d col0=%d tAx=%d tAy=%d tBx=%d tBy=%d warpId=%d laneId=%d "
    "warpx=%d warpy=%d lanex=%d laney=%d tCx=%d tCy=%d"
    % (row0, col0, tAx, tAy, tBx, tBy, warpId, laneId, warpx, warpy, lanex, laney, tCx, tCy)
)
print()

# =====================================================================
# PROLOGUE A load:
#   for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
#       int r = row0 + i + tAy;
#       sA[0][tAx][(i + tAy) ^ (tAx << 2)] =
#           (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
#   }
# =====================================================================
for i in range(0, BLOCK_M, THREAD_A_LAYOUT_Y):
    r = row0 + i + tAy
    swizzled = (i + tAy) ^ (tAx << 2)
    in_bounds = (r < M) and (tAx < K)
    source = "A[%d][%d]" % (r, tAx) if in_bounds else "ZERO_PAD"
    log_row(
        "PROLOGUE_LOAD_A",
        i=i, r=r, swizzle_term=tAx << 2, swizzled_idx=swizzled,
        source=source, destination="sA[0][%d][%d]" % (tAx, swizzled),
        in_bounds=in_bounds,
    )

# =====================================================================
# PROLOGUE B load:
#   for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
#       int r = i + tBy;
#       for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
#           int c = col0 + j + tBx;
#           sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
#       }
#   }
# =====================================================================
for i in range(0, BLOCK_K, THREAD_B_LAYOUT_Y):
    r = i + tBy
    for j in range(0, BLOCK_N, THREAD_B_LAYOUT_X):
        c = col0 + j + tBx
        in_bounds = (r < K) and (c < N)
        source = "B[%d][%d]" % (r, c) if in_bounds else "ZERO_PAD"
        log_row(
            "PROLOGUE_LOAD_B",
            i=i, j=j, r=r, c=c,
            source=source, destination="sB[0][%d][%d]" % (i + tBy, j + tBx),
            in_bounds=in_bounds,
        )

log_row("SYNCTHREADS")

# =====================================================================
# MAIN LOOP:
#   for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
#       for (int tk = 0; tk < BLOCK_K + 1; ++tk) {
#           if (tk > 0) { ...compute... }
#           if (tk < BLOCK_K) { ...register-load... }
#       }
#       if (k < K) { ...reload next chunk... }
#       bufferId ^= 1;
#   }
# =====================================================================
bufferId = 0

for k in range(BLOCK_K, K + BLOCK_K, BLOCK_K):
    for tk in range(0, BLOCK_K + 1):

        # ---- COMPUTE runs FIRST in the real code, when tk > 0 ----
        if tk > 0:
            buf = (tk - 1) & 1
            for tm in range(THREAD_C_Y_TILE_SIZE):
                for tn in range(THREAD_C_X_TILE_SIZE):
                    log_row(
                        "COMPUTE",
                        k=k, tk=tk, tm=tm, tn=tn,
                        **{"tk-1": tk - 1, "(tk-1)&1": buf},
                        bufferId=bufferId,
                        source="tCsA[%d][%d]*tCsB[%d][%d]" % (buf, tm, buf, tn),
                        destination="acc[%d][%d]" % (tm, tn),
                    )

        # ---- LOAD (register-staging) runs SECOND, when tk < BLOCK_K ----
        if tk < BLOCK_K:
            for tm in range(0, THREAD_C_Y_TILE_SIZE >> 2):
                r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2
                swizzled = r ^ (tk << 2)
                log_row(
                    "LOAD_A_REGISTER",
                    k=k, tk=tk, tm=tm,
                    **{"tk&1": tk & 1},
                    bufferId=bufferId,
                    r=r, swizzle_term=tk << 2, swizzled_idx=swizzled,
                    source="sA[%d][%d][%d]" % (bufferId, tk, swizzled),
                    destination="tCsA[%d][%d:%d]" % (tk & 1, tm << 2, (tm << 2) + 3),
                )
            for tn in range(0, THREAD_C_X_TILE_SIZE >> 2):
                c = (tCx + tn * THREAD_C_LAYOUT_X) << 2
                log_row(
                    "LOAD_B_REGISTER",
                    k=k, tk=tk, tn=tn,
                    **{"tk&1": tk & 1},
                    bufferId=bufferId,
                    c=c,
                    source="sB[%d][%d][%d]" % (bufferId, tk, c),
                    destination="tCsB[%d][%d:%d]" % (tk & 1, tn << 2, (tn << 2) + 3),
                )

    # ---- Mid-loop reload of the NEXT chunk into the OTHER buffer ----
    if k < K:
        c_col = k + tAx
        for i in range(0, BLOCK_M, THREAD_A_LAYOUT_Y):
            r = row0 + i + tAy
            swizzled = (i + tAy) ^ (tAx << 2)
            in_bounds = (r < M) and (c_col < K)
            source = "A[%d][%d]" % (r, c_col) if in_bounds else "ZERO_PAD"
            log_row(
                "RELOAD_A",
                k=k, i=i, r=r, c=c_col, swizzle_term=tAx << 2, swizzled_idx=swizzled,
                bufferId=bufferId ^ 1,
                source=source, destination="sA[%d][%d][%d]" % (bufferId ^ 1, tAx, swizzled),
                in_bounds=in_bounds,
            )
        for i in range(0, BLOCK_K, THREAD_B_LAYOUT_Y):
            r = k + i + tBy
            for j in range(0, BLOCK_N, THREAD_B_LAYOUT_X):
                c2 = col0 + j + tBx
                in_bounds = (r < K) and (c2 < N)
                source = "B[%d][%d]" % (r, c2) if in_bounds else "ZERO_PAD"
                log_row(
                    "RELOAD_B",
                    k=k, i=i, j=j, r=r, c=c2,
                    bufferId=bufferId ^ 1,
                    source=source, destination="sB[%d][%d][%d]" % (bufferId ^ 1, i + tBy, j + tBx),
                    in_bounds=in_bounds,
                )
        log_row("SYNCTHREADS", k=k)

    bufferId ^= 1
    log_row("BUFFER_FLIP", bufferId=bufferId)

# =====================================================================
# EPILOGUE:
#   for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
#       int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
#       for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
#           int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
#           if (r < M && c < N) C[r * N + c] = acc[i][j];
#       }
#   }
# =====================================================================
for i in range(THREAD_C_Y_TILE_SIZE):
    r = row0 + (tCy << 2) + (i & ~3) * THREAD_C_LAYOUT_Y + (i & 3)
    for j in range(THREAD_C_X_TILE_SIZE):
        c = col0 + (tCx << 2) + (j & ~3) * THREAD_C_LAYOUT_X + (j & 3)
        in_bounds = (r < M) and (c < N)
        log_row(
            "EPILOGUE_WRITE",
            i=i, j=j, r=r, c=c,
            source="acc[%d][%d]" % (i, j),
            destination="C[%d][%d]" % (r, c) if in_bounds else "SKIPPED (out of bounds)",
            in_bounds=in_bounds,
        )

# =====================================================================
# Print the ENTIRE execution log as ONE unified table.
# =====================================================================
print("=" * 100)
print("UNIFIED EXECUTION LOG -- %d total logged operations, in real kernel order" % len(log))
print("=" * 100)
print_unified_table()

*/


/*
"""
simulation_v2.py

A REVERSE-LINEAGE Python simulation of matmul_v0.cu's
`matrix_multiplication_kernel`: given ONE target element of C (e.g.
C[4][24]), this script discovers and prints the COMPLETE chain of
grid/block/thread identities and index values that produce it --

    A, B  -->  sA, sB  -->  tCsA, tCsB  -->  acc  -->  C[target_row][target_col]

-- answering exactly:
  1) which (block, tidx) LOADED each relevant element of A and B into
     shared memory (sA, sB) -- this is a DIFFERENT set of threads than
     the one that owns the final output, because loading sA/sB is a
     COOPERATIVE, whole-block operation using the tAx/tAy/tBx/tBy
     thread mapping, completely independent of the tCx/tCy mapping
     used for compute and the final write.
  2) which (block, tidx) copied the relevant sA/sB values into its own
     private tCsA/tCsB registers -- this IS the single "owning" thread
     (the one whose epilogue loop writes the target C element).
  3) how that SAME owning thread's acc[tm][tn] accumulator builds up,
     one real k-value at a time, from those registers.

No real A/B/C data is used anywhere -- only index values, exactly as
in simulation_v0.py / simulation_v1.py. This file follows the EXACT
same preset constants as those two files.

IMPORTANT NOTE ON ORDERING: the real kernel interleaves compute(tk)
and load(tk) so that compute(tk) actually executes ONE TK ITERATION
AFTER the load that staged its data (see matmul_v0.md Part 24 / the
(tk-1)&1 trick, also reproduced faithfully in simulation_v1.py). This
script instead groups "the load that stages real k-value's data" with
"the compute step that consumes that exact same data" into a SINGLE
row, because the goal here is to show DATA LINEAGE (which k-value's
A/B contributes to which multiply-add), not literal instruction
ordering. simulation_v1.py already covers the literal, unregrouped,
tk-by-tk instruction order if that is what you need instead.

PRESET VALUES (same as simulation_v0.py / simulation_v1.py):
    M=256 N=256 K=16
    BLOCK_M=128 BLOCK_N=128 BLOCK_K=8 BLOCK_SIZE=256
    THREAD_A_LAYOUT_X=8   THREAD_A_LAYOUT_Y=32
    THREAD_B_LAYOUT_X=32  THREAD_B_LAYOUT_Y=8
    THREAD_C_LAYOUT_X=16  THREAD_C_LAYOUT_Y=16
    THREAD_C_X_TILE_SIZE=8  THREAD_C_Y_TILE_SIZE=8
    THREAD_C_WARP_X=8  THREAD_C_WARP_Y=4  THREAD_C_WARP_DIM_X=2
"""

# =====================================================================
# PRESET CONSTANTS (identical to simulation_v0.py / simulation_v1.py)
# =====================================================================
M, N, K = 256, 256, 16
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 8
BLOCK_SIZE = 256

THREAD_A_LAYOUT_X = 8
THREAD_A_LAYOUT_Y = 32
THREAD_B_LAYOUT_X = 32
THREAD_B_LAYOUT_Y = 8
THREAD_C_LAYOUT_X = 16
THREAD_C_LAYOUT_Y = 16
THREAD_C_X_TILE_SIZE = 8
THREAD_C_Y_TILE_SIZE = 8
THREAD_C_WARP_X = 8
THREAD_C_WARP_Y = 4
THREAD_C_WARP_DIM_X = 2

# =====================================================================
# THE TARGET: which single element of C are we fully tracing?
# =====================================================================
TARGET_C_ROW = 4
TARGET_C_COL = 24


def thread_identity(tidx):
    """Every fixed per-thread index value, computed exactly as the
    real kernel's opening lines do. Returns a dict."""
    tAx = tidx & (THREAD_A_LAYOUT_X - 1)
    tAy = tidx // THREAD_A_LAYOUT_X
    tBx = tidx & (THREAD_B_LAYOUT_X - 1)
    tBy = tidx // THREAD_B_LAYOUT_X

    warpId = tidx >> 5
    laneId = tidx & 31
    warpx = warpId & (THREAD_C_WARP_DIM_X - 1)
    warpy = warpId // THREAD_C_WARP_DIM_X
    lanex = (laneId & 15) >> 1
    laney = ((laneId >> 4) << 1) + (laneId & 1)
    tCx = warpx * THREAD_C_WARP_X + lanex
    tCy = warpy * THREAD_C_WARP_Y + laney

    return dict(tidx=tidx, tAx=tAx, tAy=tAy, tBx=tBx, tBy=tBy,
                warpId=warpId, laneId=laneId, warpx=warpx, warpy=warpy,
                lanex=lanex, laney=laney, tCx=tCx, tCy=tCy)


def find_consumer(row0, col0):
    """
    Brute-force search, over every possible thread (tidx) AND every
    (i, j) position in its own epilogue write-back loop, for the ONE
    thread/position whose real C write lands exactly on
    (TARGET_C_ROW, TARGET_C_COL). This directly re-uses the identical
    epilogue formula from matmul_v0.cu -- no algebra is hand-inverted,
    everything is found by exhaustive, guaranteed-correct search.
    """
    for tidx in range(BLOCK_SIZE):
        ident = thread_identity(tidx)
        tCx, tCy = ident["tCx"], ident["tCy"]
        for i in range(THREAD_C_Y_TILE_SIZE):
            r = row0 + (tCy << 2) + (i & ~3) * THREAD_C_LAYOUT_Y + (i & 3)
            if r != TARGET_C_ROW:
                continue
            for j in range(THREAD_C_X_TILE_SIZE):
                c = col0 + (tCx << 2) + (j & ~3) * THREAD_C_LAYOUT_X + (j & 3)
                if c == TARGET_C_COL:
                    return tidx, i, j
    return None


def find_A_writer(target_tk, target_swizzled_idx):
    """
    Brute-force search for the ONE thread (and ONE i-iteration of its
    own prologue/reload loop) that writes sA[*][target_tk][target_swizzled_idx].
    Re-uses the exact write formula from matmul_v0.cu's A-load code.
    """
    for tidx in range(BLOCK_SIZE):
        tAx = tidx & (THREAD_A_LAYOUT_X - 1)
        if tAx != target_tk:
            continue
        tAy = tidx // THREAD_A_LAYOUT_X
        for i in range(0, BLOCK_M, THREAD_A_LAYOUT_Y):
            swizzled = (i + tAy) ^ (tAx << 2)
            if swizzled == target_swizzled_idx:
                return tidx, i, tAy
    return None


def find_B_writer(target_tk, target_n_idx):
    """
    Brute-force search for the ONE thread (and ONE j-iteration) that
    writes sB[*][target_tk][target_n_idx]. Re-uses the exact write
    formula from matmul_v0.cu's B-load code.
    """
    for tidx in range(BLOCK_SIZE):
        tBy = tidx // THREAD_B_LAYOUT_X
        for i in range(0, BLOCK_K, THREAD_B_LAYOUT_Y):  # i=0 only, since BLOCK_K==THREAD_B_LAYOUT_Y here
            if i + tBy != target_tk:
                continue
            tBx = tidx & (THREAD_B_LAYOUT_X - 1)
            for j in range(0, BLOCK_N, THREAD_B_LAYOUT_X):
                if j + tBx == target_n_idx:
                    return tidx, j
    return None


def print_table(headers, rows):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))


# =====================================================================
# STEP 1: which grid block owns the target C element?
# =====================================================================
block_idx_y = TARGET_C_ROW // BLOCK_M
block_idx_x = TARGET_C_COL // BLOCK_N
row0 = block_idx_y * BLOCK_M
col0 = block_idx_x * BLOCK_N

print("=" * 100)
print("TARGET: C[%d][%d]" % (TARGET_C_ROW, TARGET_C_COL))
print("=" * 100)
print("STEP 1: which grid block owns this element?")
print("    blockIdx.y = %d // %d = %d" % (TARGET_C_ROW, BLOCK_M, block_idx_y))
print("    blockIdx.x = %d // %d = %d" % (TARGET_C_COL, BLOCK_N, block_idx_x))
print("    row0 = %d,  col0 = %d" % (row0, col0))
print()

# =====================================================================
# STEP 2: which thread (tidx), and which (tm, tn) accumulator cell,
# owns this element? (brute-force search over the real epilogue formula)
# =====================================================================
consumer_tidx, TM, TN = find_consumer(row0, col0)
consumer = thread_identity(consumer_tidx)

print("STEP 2: which thread, and which acc[tm][tn] cell, owns this element?")
print("    tidx = %d  (found by exhaustively matching the epilogue formula)" % consumer_tidx)
print("    tm = %d, tn = %d  ->  this thread's acc[%d][%d] IS C[%d][%d]"
      % (TM, TN, TM, TN, TARGET_C_ROW, TARGET_C_COL))
print("    This thread's fixed identity: tAx=%d tAy=%d tBx=%d tBy=%d tCx=%d tCy=%d"
      % (consumer["tAx"], consumer["tAy"], consumer["tBx"], consumer["tBy"],
         consumer["tCx"], consumer["tCy"]))
print()
print("    NOTE: tAx/tAy/tBx/tBy above describe this SAME thread's role")
print("    whenever it cooperatively helps load shared memory (which is a")
print("    DIFFERENT job than owning C[%d][%d] -- see the A_writer_tidx /"
      % (TARGET_C_ROW, TARGET_C_COL))
print("    B_writer_tidx columns below: they are usually OTHER threads.")
print()

TM_REG, TM_SUB = TM // 4, TM % 4
TN_REG, TN_SUB = TN // 4, TN % 4

# =====================================================================
# STEP 3: walk every REAL k-value (0..K-1) this accumulator cell sums
# over, discovering for each one: which thread loaded A's element into
# sA, which thread loaded B's element into sB, where the consumer
# thread re-staged them into tCsA/tCsB, and the resulting multiply-add.
# =====================================================================
rows = []
step = 0

for k_outer in range(BLOCK_K, K + BLOCK_K, BLOCK_K):
    is_first_chunk = (k_outer == BLOCK_K)
    load_k_offset = 0 if is_first_chunk else (k_outer - BLOCK_K)
    load_phase = "PROLOGUE" if is_first_chunk else ("RELOAD@k_outer=%d" % (k_outer - BLOCK_K))
    buf = 0 if is_first_chunk else 1  # which sA/sB buffer this chunk's data lives in

    for tk in range(0, BLOCK_K):
        step += 1
        real_k = load_k_offset + tk  # the TRUE mathematical k-index, 0..K-1

        # ---- Where does the consumer's register-load read FROM in sA/sB? ----
        r_reg = (consumer["tCy"] + TM_REG * THREAD_C_LAYOUT_Y) << 2
        sA_swizzled_idx = (r_reg ^ (tk << 2)) + TM_SUB
        sB_idx = ((consumer["tCx"] + TN_REG * THREAD_C_LAYOUT_X) << 2) + TN_SUB

        # ---- Who originally WROTE that sA/sB slot? ----
        a_writer_tidx, a_i, a_writer_tAy = find_A_writer(tk, sA_swizzled_idx)
        b_writer_tidx, b_j = find_B_writer(tk, sB_idx)

        # ---- What REAL A/B element did that writer thread read? ----
        if is_first_chunk:
            a_real_row = row0 + a_i + a_writer_tAy
            a_real_col = tk  # prologue reads A[r*K + tAx], tAx == tk here
            b_real_row = tk  # prologue reads B[r*N + c], r == i+tBy == tk here
            b_real_col = col0 + b_j + (b_writer_tidx & (THREAD_B_LAYOUT_X - 1))
        else:
            a_real_row = row0 + a_i + a_writer_tAy
            a_real_col = load_k_offset + tk  # reload reads A[r*K + c], c = k_outer_prev + tAx
            b_real_row = load_k_offset + tk  # reload reads B[r*N + c2], r = k_outer_prev + i + tBy
            b_real_col = col0 + b_j + (b_writer_tidx & (THREAD_B_LAYOUT_X - 1))

        rows.append([
            step, real_k, load_phase,
            a_writer_tidx, "A[%d][%d]" % (a_real_row, a_real_col),
            "sA[%d][%d][%d]" % (buf, tk, sA_swizzled_idx),
            b_writer_tidx, "B[%d][%d]" % (b_real_row, b_real_col),
            "sB[%d][%d][%d]" % (buf, tk, sB_idx),
            consumer_tidx,
            "tCsA[%d][%d]" % (tk & 1, TM),
            "tCsB[%d][%d]" % (tk & 1, TN),
            "tCsA[%d][%d]*tCsB[%d][%d]" % (tk & 1, TM, tk & 1, TN),
        ])

print("STEP 3: the complete lineage, one row per REAL k-value (0..%d),"
      % (K - 1))
print("in the exact order these contributions sum together:")
print()
print_table(
    [
        "step", "real_k", "load_phase",
        "A_writer_tidx", "A_element", "sA_target",
        "B_writer_tidx", "B_element", "sB_target",
        "consumer_tidx", "tCsA_slot", "tCsB_slot", "multiply_term",
    ],
    rows,
)

# =====================================================================
# STEP 4: the final accumulation and write-back.
# =====================================================================
print()
print("STEP 4: the final result.")
print("    acc[%d][%d] = sum of all %d multiply_term values shown above"
      % (TM, TN, len(rows)))
print("    (i.e. sum over real_k=0..%d of A[row][real_k] * B[real_k][col])"
      % (K - 1))
print("    EPILOGUE_WRITE: thread tidx=%d writes acc[%d][%d] -> C[%d][%d]"
      % (consumer_tidx, TM, TN, TARGET_C_ROW, TARGET_C_COL))
print()
print("Simulation complete.")

*/

/*
"""
matmul_v2_bfix_simulation_v0.py

A line-by-line, EXECUTION-ORDER Python re-implementation of
matmul_v2_bfix.cu's `matrix_multiplication_kernel`, for exactly ONE
thread (TIDX), following the exact same style as simulation_v1.py
(one continuous, unified table, in real code order, A[][]/B[][]
format, index values only -- never real matrix data).

WHY THIS FILE'S STRUCTURE DIFFERS FROM simulation_v1.py'S, EVEN THOUGH
THE PRESENTATION STYLE IS IDENTICAL: matmul_v2_bfix.cu is a genuinely
different kernel from matmul_v0.cu:

  1. A is stored NATURAL (sA[BLOCK_M][BLOCK_K], m outer, k inner) with
     NO swizzle at all -- simpler than matmul_v0.cu's A.
  2. B is stored TRANSPOSED (sB[BLOCK_N][BLOCK_K], n outer, k inner)
     WITH a mask-based swizzle (`mask = (n >> 2) & 7`, XORed into the
     k-index) to avoid the bank conflict investigated in
     matmul_v0.md Parts 18-20.
  3. There is NO register-level software pipelining / no `tk` loop at
     all. Instead, this kernel caches a thread's ENTIRE BLOCK_K-long
     row of A and B (all 8 k-values at once) into regA/regB, then
     immediately does a full BLOCK_K-length dot-product reduction.
     The direct analog of matmul_v0.cu's per-`tk` accumulate step is
     this file's per-`kk` term inside that dot-product.
  4. The epilogue has no "groups of 4" bit-trick (no `i & 3` / `i &
     (~3)`) -- tm/tn map straight to output rows/columns.

PRESET VALUES (as given):
    M=256 N=256 K=16
    BLOCK_M=128 BLOCK_N=128 BLOCK_K=8 BLOCK_SIZE=256
    THREAD_A_LAYOUT_X=8   THREAD_A_LAYOUT_Y=32
    THREAD_B_LAYOUT_X=32  THREAD_B_LAYOUT_Y=8
    THREAD_C_LAYOUT_X=16  THREAD_C_LAYOUT_Y=16
    THREAD_C_X_TILE_SIZE=8  THREAD_C_Y_TILE_SIZE=8
    THREAD_C_WARP_X=8  THREAD_C_WARP_Y=4  THREAD_C_WARP_DIM_X=2
    tidx = 13

blockIdx is not given, so block (0, 0) is assumed (row0=0, col0=0).
"""

# =====================================================================
# PRESET CONSTANTS (as given)
# =====================================================================
M, N, K = 256, 256, 16
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 8
BLOCK_SIZE = 256

THREAD_A_LAYOUT_X = 8
THREAD_A_LAYOUT_Y = 32
THREAD_B_LAYOUT_X = 32
THREAD_B_LAYOUT_Y = 8
THREAD_C_LAYOUT_X = 16
THREAD_C_LAYOUT_Y = 16
THREAD_C_X_TILE_SIZE = 8
THREAD_C_Y_TILE_SIZE = 8
THREAD_C_WARP_X = 8
THREAD_C_WARP_Y = 4
THREAD_C_WARP_DIM_X = 2

TIDX = 13
BLOCK_IDX_X = 0
BLOCK_IDX_Y = 0

row0 = BLOCK_IDX_Y * BLOCK_M
col0 = BLOCK_IDX_X * BLOCK_N

# =====================================================================
# The single, unified execution log -- every column that could matter
# for ANY phase, with "-" for whichever do not apply to a given row.
# =====================================================================
LOG_COLUMNS = [
    "step", "phase", "k", "i", "j", "tm", "tn", "kk",
    "n", "mask", "bufferId", "r", "c",
    "source", "destination", "in_bounds",
]
log = []
_step_counter = [0]


def log_row(phase, **kwargs):
    _step_counter[0] += 1
    row = {col: "-" for col in LOG_COLUMNS}
    row["step"] = _step_counter[0]
    row["phase"] = phase
    row.update(kwargs)
    log.append(row)


def print_unified_table():
    widths = {
        col: max(len(col), max((len(str(r[col])) for r in log), default=0))
        for col in LOG_COLUMNS
    }
    header = " | ".join(col.ljust(widths[col]) for col in LOG_COLUMNS)
    print(header)
    print("-" * len(header))
    for r in log:
        print(" | ".join(str(r[col]).ljust(widths[col]) for col in LOG_COLUMNS))


# =====================================================================
# Fixed, per-thread index values -- computed ONCE, printed as a short
# preamble (identical formulas to matmul_v0.cu -- this part of the
# kernel is byte-for-byte unchanged between the two files).
# =====================================================================
tAx = TIDX & (THREAD_A_LAYOUT_X - 1)
tAy = TIDX // THREAD_A_LAYOUT_X
tBx = TIDX & (THREAD_B_LAYOUT_X - 1)
tBy = TIDX // THREAD_B_LAYOUT_X

warpId = TIDX >> 5
laneId = TIDX & 31
warpx = warpId & (THREAD_C_WARP_DIM_X - 1)
warpy = warpId // THREAD_C_WARP_DIM_X
lanex = (laneId & 15) >> 1
laney = ((laneId >> 4) << 1) + (laneId & 1)
tCx = warpx * THREAD_C_WARP_X + lanex
tCy = warpy * THREAD_C_WARP_Y + laney

print("=" * 100)
print("FIXED PER-THREAD VALUES  (tidx=%d, block=(%d,%d))" % (TIDX, BLOCK_IDX_X, BLOCK_IDX_Y))
print("=" * 100)
print(
    "row0=%d col0=%d tAx=%d tAy=%d tBx=%d tBy=%d warpId=%d laneId=%d "
    "warpx=%d warpy=%d lanex=%d laney=%d tCx=%d tCy=%d"
    % (row0, col0, tAx, tAy, tBx, tBy, warpId, laneId, warpx, warpy, lanex, laney, tCx, tCy)
)
print()

# =====================================================================
# PROLOGUE A load (NO swizzle at all -- the whole point of Part 17):
#   for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
#       int r = row0 + i + tAy;
#       sA[0][i + tAy][tAx] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
#   }
# =====================================================================
for i in range(0, BLOCK_M, THREAD_A_LAYOUT_Y):
    r = row0 + i + tAy
    in_bounds = (r < M) and (tAx < K)
    source = "A[%d][%d]" % (r, tAx) if in_bounds else "ZERO_PAD"
    log_row(
        "PROLOGUE_LOAD_A",
        i=i, r=r, bufferId=0,
        source=source, destination="sA[0][%d][%d]" % (i + tAy, tAx),
        in_bounds=in_bounds,
    )

# =====================================================================
# PROLOGUE B load (THE FIX -- mask-based swizzle, Parts 18-20):
#   for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
#       int r = i + tBy;
#       for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
#           int c = col0 + j + tBx;
#           int n = j + tBx;
#           int kk = i + tBy;
#           int mask = (n >> 2) & 7;
#           sB[0][n][kk ^ mask] = (r < K && c < N) ? B[r * N + c] : 0.0f;
#       }
#   }
# =====================================================================
for i in range(0, BLOCK_K, THREAD_B_LAYOUT_Y):
    r = i + tBy
    for j in range(0, BLOCK_N, THREAD_B_LAYOUT_X):
        c = col0 + j + tBx
        n = j + tBx
        kk = i + tBy
        mask = (n >> 2) & 7
        in_bounds = (r < K) and (c < N)
        source = "B[%d][%d]" % (r, c) if in_bounds else "ZERO_PAD"
        log_row(
            "PROLOGUE_LOAD_B",
            i=i, j=j, kk=kk, n=n, mask=mask, bufferId=0, r=r, c=c,
            source=source, destination="sB[0][%d][%d]" % (n, kk ^ mask),
            in_bounds=in_bounds,
        )

log_row("SYNCTHREADS")

# =====================================================================
# MAIN LOOP:
#   for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
#       ...stage regA (FLOAT4, no swizzle)...
#       ...stage regB (8 scalar reads, mask-descrambled)...
#       ...dot-product accumulate...
#       if (k < K) { ...reload next chunk... }
#       bufferId ^= 1;
#   }
# =====================================================================
bufferId = 0

for k in range(BLOCK_K, K + BLOCK_K, BLOCK_K):

    # ---- Stage A into regA: unchanged, still FLOAT4-eligible, no swizzle.
    #   for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
    #       int m = tCy + tm * THREAD_C_LAYOUT_Y;
    #       FLOAT4(regA[tm][0]) = FLOAT4(sA[bufferId][m][0]);
    #       FLOAT4(regA[tm][4]) = FLOAT4(sA[bufferId][m][4]);
    #   }
    # Logged per-kk (per real k-value within this chunk) for direct,
    # element-level comparability with the mask-swizzled B stage below.
    # ----
    for tm in range(THREAD_C_Y_TILE_SIZE):
        m = tCy + tm * THREAD_C_LAYOUT_Y
        for kk in range(BLOCK_K):
            log_row(
                "STAGE_A_REGISTER",
                k=k, tm=tm, kk=kk, bufferId=bufferId,
                source="sA[%d][%d][%d]" % (bufferId, m, kk),
                destination="regA[%d][%d]" % (tm, kk),
            )

    # ---- Stage B into regB: THE FIX's read side -- 8 scalar reads,
    # each individually un-scrambled with the SAME mask used to write it.
    #   for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
    #       int n = tCx + tn * THREAD_C_LAYOUT_X;
    #       int mask = (n >> 2) & 7;
    #       regB[tn][kk] = sB[bufferId][n][kk ^ mask];   (for kk = 0..7)
    #   }
    # ----
    for tn in range(THREAD_C_X_TILE_SIZE):
        n = tCx + tn * THREAD_C_LAYOUT_X
        mask = (n >> 2) & 7
        for kk in range(BLOCK_K):
            log_row(
                "STAGE_B_REGISTER",
                k=k, tn=tn, kk=kk, n=n, mask=mask, bufferId=bufferId,
                source="sB[%d][%d][%d]" % (bufferId, n, kk ^ mask),
                destination="regB[%d][%d]" % (tn, kk),
            )

    # ---- Dot-product accumulation: unchanged from matmul_v2.cu.
    #   for (tm...) for (tn...) {
    #       float partial = 0.0f;
    #       for (kk...) partial += regA[tm][kk] * regB[tn][kk];
    #       acc[tm][tn] += partial;
    #   }
    # Logged per (tm, tn, kk) term -- this IS the direct analog of
    # matmul_v0.cu's per-tk compute step: each kk here corresponds to
    # exactly one real k-value's contribution to the final sum.
    # ----
    for tm in range(THREAD_C_Y_TILE_SIZE):
        for tn in range(THREAD_C_X_TILE_SIZE):
            for kk in range(BLOCK_K):
                log_row(
                    "COMPUTE",
                    k=k, tm=tm, tn=tn, kk=kk, bufferId=bufferId,
                    source="regA[%d][%d]*regB[%d][%d]" % (tm, kk, tn, kk),
                    destination="acc[%d][%d] (partial += ...)" % (tm, tn),
                )

    # ---- Mid-loop reload of the NEXT chunk, into the OTHER buffer ----
    if k < K:
        c_col = k + tAx
        for i in range(0, BLOCK_M, THREAD_A_LAYOUT_Y):
            r = row0 + i + tAy
            in_bounds = (r < M) and (c_col < K)
            source = "A[%d][%d]" % (r, c_col) if in_bounds else "ZERO_PAD"
            log_row(
                "RELOAD_A",
                k=k, i=i, r=r, c=c_col, bufferId=bufferId ^ 1,
                source=source, destination="sA[%d][%d][%d]" % (bufferId ^ 1, i + tAy, tAx),
                in_bounds=in_bounds,
            )
        for i in range(0, BLOCK_K, THREAD_B_LAYOUT_Y):
            r = k + i + tBy
            for j in range(0, BLOCK_N, THREAD_B_LAYOUT_X):
                c2 = col0 + j + tBx
                n = j + tBx
                kk2 = i + tBy
                mask = (n >> 2) & 7
                in_bounds = (r < K) and (c2 < N)
                source = "B[%d][%d]" % (r, c2) if in_bounds else "ZERO_PAD"
                log_row(
                    "RELOAD_B",
                    k=k, i=i, j=j, kk=kk2, n=n, mask=mask, bufferId=bufferId ^ 1, r=r, c=c2,
                    source=source, destination="sB[%d][%d][%d]" % (bufferId ^ 1, n, kk2 ^ mask),
                    in_bounds=in_bounds,
                )
        log_row("SYNCTHREADS", k=k)

    bufferId ^= 1
    log_row("BUFFER_FLIP", bufferId=bufferId)

# =====================================================================
# EPILOGUE (no "groups of 4" bit-trick -- simpler than matmul_v0.cu):
#   for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
#       int r = row0 + tCy + tm * THREAD_C_LAYOUT_Y;
#       for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
#           int c = col0 + tCx + tn * THREAD_C_LAYOUT_X;
#           if (r < M && c < N) C[r * N + c] = acc[tm][tn];
#       }
#   }
# =====================================================================
for tm in range(THREAD_C_Y_TILE_SIZE):
    r = row0 + tCy + tm * THREAD_C_LAYOUT_Y
    for tn in range(THREAD_C_X_TILE_SIZE):
        c = col0 + tCx + tn * THREAD_C_LAYOUT_X
        in_bounds = (r < M) and (c < N)
        log_row(
            "EPILOGUE_WRITE",
            tm=tm, tn=tn, r=r, c=c,
            source="acc[%d][%d]" % (tm, tn),
            destination="C[%d][%d]" % (r, c) if in_bounds else "SKIPPED (out of bounds)",
            in_bounds=in_bounds,
        )

# =====================================================================
# Print the ENTIRE execution log as ONE unified table.
# =====================================================================
print("=" * 100)
print("UNIFIED EXECUTION LOG -- %d total logged operations, in real kernel order" % len(log))
print("=" * 100)
print_unified_table()

*/

/*
"""
matmul_v2_bfix_simulation_v1.py

A REVERSE-LINEAGE Python simulation of matmul_v2_bfix.cu's
`matrix_multiplication_kernel`: given ONE target element of C (e.g.
C[4][24]), this script discovers and prints the COMPLETE chain of
grid/block/thread identities and index values that produce it --

    A, B  -->  sA, sB  -->  regA, regB  -->  acc  -->  C[target_row][target_col]

-- answering exactly:
  1) which (block, tidx) LOADED each relevant element of A and B into
     shared memory (sA, sB) -- a DIFFERENT set of threads than the one
     that owns the final output, since loading sA/sB is a cooperative,
     whole-block operation using the tAx/tAy/tBx/tBy thread mapping,
     independent of the tCx/tCy mapping used for compute/write.
  2) which (block, tidx) copied the relevant sA/sB values into its own
     private regA/regB registers (un-scrambling B's mask swizzle along
     the way) -- this IS the single "owning" thread.
  3) how that SAME owning thread's acc[tm][tn] builds up, one real
     k-value at a time, from those registers.

This is the matmul_v2_bfix.cu counterpart of simulation_v2.py (which
did the same thing for matmul_v0.cu) -- adapted for THIS kernel's
different structure:
  - A has NO swizzle at all (sA[BLOCK_M][BLOCK_K], direct).
  - B has a MASK swizzle (sB[BLOCK_N][BLOCK_K], mask = (n>>2)&7, XORed
    into the k-index) -- Parts 18-20 of matmul_v0.md.
  - There is no tk-pipelining loop; a thread caches its WHOLE BLOCK_K
    row of A/B at once, then does a full dot-product reduction. The
    direct analog of "one real k-value's contribution" is one `kk`
    term inside that reduction.
  - The epilogue has no "groups of 4" bit-trick -- tm/tn map straight
    to output rows/columns, making the consumer-thread search simpler
    (still done via brute force for consistency and safety).

No real A/B/C data is used anywhere -- only index values.

PRESET VALUES (as given):
    M=256 N=256 K=16
    BLOCK_M=128 BLOCK_N=128 BLOCK_K=8 BLOCK_SIZE=256
    THREAD_A_LAYOUT_X=8   THREAD_A_LAYOUT_Y=32
    THREAD_B_LAYOUT_X=32  THREAD_B_LAYOUT_Y=8
    THREAD_C_LAYOUT_X=16  THREAD_C_LAYOUT_Y=16
    THREAD_C_X_TILE_SIZE=8  THREAD_C_Y_TILE_SIZE=8
    THREAD_C_WARP_X=8  THREAD_C_WARP_Y=4  THREAD_C_WARP_DIM_X=2
"""

# =====================================================================
# PRESET CONSTANTS (as given)
# =====================================================================
M, N, K = 256, 256, 16
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 8
BLOCK_SIZE = 256

THREAD_A_LAYOUT_X = 8
THREAD_A_LAYOUT_Y = 32
THREAD_B_LAYOUT_X = 32
THREAD_B_LAYOUT_Y = 8
THREAD_C_LAYOUT_X = 16
THREAD_C_LAYOUT_Y = 16
THREAD_C_X_TILE_SIZE = 8
THREAD_C_Y_TILE_SIZE = 8
THREAD_C_WARP_X = 8
THREAD_C_WARP_Y = 4
THREAD_C_WARP_DIM_X = 2

# =====================================================================
# THE TARGET: which single element of C are we fully tracing?
# =====================================================================
TARGET_C_ROW = 4
TARGET_C_COL = 24


def thread_identity(tidx):
    """Every fixed per-thread index value -- identical formulas to
    matmul_v0.cu, this part of the kernel is byte-for-byte unchanged."""
    tAx = tidx & (THREAD_A_LAYOUT_X - 1)
    tAy = tidx // THREAD_A_LAYOUT_X
    tBx = tidx & (THREAD_B_LAYOUT_X - 1)
    tBy = tidx // THREAD_B_LAYOUT_X

    warpId = tidx >> 5
    laneId = tidx & 31
    warpx = warpId & (THREAD_C_WARP_DIM_X - 1)
    warpy = warpId // THREAD_C_WARP_DIM_X
    lanex = (laneId & 15) >> 1
    laney = ((laneId >> 4) << 1) + (laneId & 1)
    tCx = warpx * THREAD_C_WARP_X + lanex
    tCy = warpy * THREAD_C_WARP_Y + laney

    return dict(tidx=tidx, tAx=tAx, tAy=tAy, tBx=tBx, tBy=tBy,
                warpId=warpId, laneId=laneId, warpx=warpx, warpy=warpy,
                lanex=lanex, laney=laney, tCx=tCx, tCy=tCy)


def find_consumer(row0, col0):
    """
    Brute-force search for the ONE (tidx, tm, tn) whose real epilogue
    write lands on (TARGET_C_ROW, TARGET_C_COL). matmul_v2_bfix.cu's
    epilogue has NO bit-trick (unlike matmul_v0.cu's), so this could be
    solved directly, but brute force is used anyway for consistency
    and because it is guaranteed correct by construction.
    """
    for tidx in range(BLOCK_SIZE):
        ident = thread_identity(tidx)
        tCx, tCy = ident["tCx"], ident["tCy"]
        for tm in range(THREAD_C_Y_TILE_SIZE):
            r = row0 + tCy + tm * THREAD_C_LAYOUT_Y
            if r != TARGET_C_ROW:
                continue
            for tn in range(THREAD_C_X_TILE_SIZE):
                c = col0 + tCx + tn * THREAD_C_LAYOUT_X
                if c == TARGET_C_COL:
                    return tidx, tm, tn
    return None


def find_A_writer(target_m, target_kk):
    """
    Brute-force search for the ONE thread (and ONE i-iteration) that
    writes sA[*][target_m][target_kk]. A has NO swizzle in this kernel
    (`sA[buf][i+tAy][tAx] = ...`), so tAx must equal target_kk directly,
    and i+tAy must equal target_m directly.
    """
    for tidx in range(BLOCK_SIZE):
        tAx = tidx & (THREAD_A_LAYOUT_X - 1)
        if tAx != target_kk:
            continue
        tAy = tidx // THREAD_A_LAYOUT_X
        for i in range(0, BLOCK_M, THREAD_A_LAYOUT_Y):
            if i + tAy == target_m:
                return tidx, i, tAy
    return None


def find_B_writer(target_n, target_kk):
    """
    Brute-force search for the ONE thread (and ONE j-iteration) that
    writes the slot holding real k-value `target_kk` at N-position
    `target_n`. B's write is `sB[buf][n][kk2^mask] = ...` where
    `mask = (n>>2)&7` depends ONLY on n -- so once n is pinned to
    target_n (forced, since N has no ambiguity of its own), the
    writer's mask is IDENTICAL to the consumer's mask, and the XOR
    cancels out of the equation, leaving kk2 == target_kk directly.
    """
    mask = (target_n >> 2) & 7
    target_physical_slot = target_kk ^ mask
    for tidx in range(BLOCK_SIZE):
        tBy = tidx // THREAD_B_LAYOUT_X
        for i in range(0, BLOCK_K, THREAD_B_LAYOUT_Y):  # i=0 only, BLOCK_K==THREAD_B_LAYOUT_Y here
            kk2 = i + tBy
            tBx = tidx & (THREAD_B_LAYOUT_X - 1)
            for j in range(0, BLOCK_N, THREAD_B_LAYOUT_X):
                n = j + tBx
                if n != target_n:
                    continue
                writer_mask = (n >> 2) & 7  # == mask, since n matches exactly
                if (kk2 ^ writer_mask) == target_physical_slot:
                    return tidx, j
    return None


def print_table(headers, rows):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))


# =====================================================================
# STEP 1: which grid block owns the target C element?
# =====================================================================
block_idx_y = TARGET_C_ROW // BLOCK_M
block_idx_x = TARGET_C_COL // BLOCK_N
row0 = block_idx_y * BLOCK_M
col0 = block_idx_x * BLOCK_N

print("=" * 100)
print("TARGET: C[%d][%d]" % (TARGET_C_ROW, TARGET_C_COL))
print("=" * 100)
print("STEP 1: which grid block owns this element?")
print("    blockIdx.y = %d // %d = %d" % (TARGET_C_ROW, BLOCK_M, block_idx_y))
print("    blockIdx.x = %d // %d = %d" % (TARGET_C_COL, BLOCK_N, block_idx_x))
print("    row0 = %d,  col0 = %d" % (row0, col0))
print()

# =====================================================================
# STEP 2: which thread (tidx), and which (tm, tn) accumulator cell,
# owns this element?
# =====================================================================
consumer_tidx, TM, TN = find_consumer(row0, col0)
consumer = thread_identity(consumer_tidx)

m_for_A = consumer["tCy"] + TM * THREAD_C_LAYOUT_Y
n_for_B = consumer["tCx"] + TN * THREAD_C_LAYOUT_X
mask_for_B = (n_for_B >> 2) & 7

print("STEP 2: which thread, and which acc[tm][tn] cell, owns this element?")
print("    tidx = %d  (found by exhaustively matching the real epilogue formula)" % consumer_tidx)
print("    tm = %d, tn = %d  ->  this thread's acc[%d][%d] IS C[%d][%d]"
      % (TM, TN, TM, TN, TARGET_C_ROW, TARGET_C_COL))
print("    This thread's fixed identity: tAx=%d tAy=%d tBx=%d tBy=%d tCx=%d tCy=%d"
      % (consumer["tAx"], consumer["tAy"], consumer["tBx"], consumer["tBy"],
         consumer["tCx"], consumer["tCy"]))
print("    Derived: m_for_A=%d (the sA row this thread reads), "
      "n_for_B=%d (the sB row this thread reads), mask_for_B=%d"
      % (m_for_A, n_for_B, mask_for_B))
print()
print("    NOTE: A_writer_tidx / B_writer_tidx below will usually be OTHER")
print("    threads -- loading shared memory is a cooperative, whole-block")
print("    job, independent of which thread ends up owning C[%d][%d]."
      % (TARGET_C_ROW, TARGET_C_COL))
print()

# =====================================================================
# STEP 3: walk every REAL k-value (0..K-1), discovering for each one:
# which thread loaded A's element into sA, which thread loaded B's
# element into sB, where the consumer thread re-staged them into
# regA/regB (un-scrambling B's mask along the way), and the resulting
# multiply-add term.
# =====================================================================
rows = []
step = 0

for k_outer in range(BLOCK_K, K + BLOCK_K, BLOCK_K):
    is_first_chunk = (k_outer == BLOCK_K)
    load_k_offset = 0 if is_first_chunk else (k_outer - BLOCK_K)
    load_phase = "PROLOGUE" if is_first_chunk else ("RELOAD@k_outer=%d" % (k_outer - BLOCK_K))
    buf = 0 if is_first_chunk else 1

    for kk in range(BLOCK_K):
        step += 1
        real_k = load_k_offset + kk

        a_writer_tidx, a_i, a_writer_tAy = find_A_writer(m_for_A, kk)
        b_writer_tidx, b_j = find_B_writer(n_for_B, kk)

        if is_first_chunk:
            a_real_row = row0 + a_i + a_writer_tAy
            a_real_col = kk  # prologue reads A[r*K + tAx], tAx == kk here (no swizzle)
            b_real_row = kk  # prologue reads B[r*N + c], r == i+tBy == kk here
            b_real_col = col0 + b_j + (b_writer_tidx & (THREAD_B_LAYOUT_X - 1))
        else:
            a_real_row = row0 + a_i + a_writer_tAy
            a_real_col = load_k_offset + kk  # reload reads A[r*K + c], c = k_outer_prev + tAx
            b_real_row = load_k_offset + kk  # reload reads B[r*N + c2], r = k_outer_prev + i + tBy
            b_real_col = col0 + b_j + (b_writer_tidx & (THREAD_B_LAYOUT_X - 1))

        rows.append([
            step, real_k, load_phase,
            a_writer_tidx, "A[%d][%d]" % (a_real_row, a_real_col),
            "sA[%d][%d][%d]" % (buf, m_for_A, kk),
            b_writer_tidx, "B[%d][%d]" % (b_real_row, b_real_col),
            "sB[%d][%d][%d]" % (buf, n_for_B, kk ^ mask_for_B),
            consumer_tidx,
            "regA[%d][%d]" % (TM, kk),
            "regB[%d][%d]" % (TN, kk),
            "regA[%d][%d]*regB[%d][%d]" % (TM, kk, TN, kk),
        ])

print("STEP 3: the complete lineage, one row per REAL k-value (0..%d),"
      % (K - 1))
print("in the exact order these contributions sum together:")
print()
print_table(
    [
        "step", "real_k", "load_phase",
        "A_writer_tidx", "A_element", "sA_target",
        "B_writer_tidx", "B_element", "sB_target",
        "consumer_tidx", "regA_slot", "regB_slot", "multiply_term",
    ],
    rows,
)

# =====================================================================
# STEP 4: the final accumulation and write-back.
# =====================================================================
print()
print("STEP 4: the final result.")
print("    acc[%d][%d] = sum of all %d multiply_term values shown above"
      % (TM, TN, len(rows)))
print("    (i.e. sum over real_k=0..%d of A[row][real_k] * B[real_k][col])"
      % (K - 1))
print("    EPILOGUE_WRITE: thread tidx=%d writes acc[%d][%d] -> C[%d][%d]"
      % (consumer_tidx, TM, TN, TARGET_C_ROW, TARGET_C_COL))
print()
print("Simulation complete.")

*/

/*
# Matrix Multiplication on the GPU — Full Conversation Write-Up (CUDA + Triton)

This document is a complete, start-to-finish compilation of a conversation about
GPU matrix multiplication (GEMM), covering a hand-tuned CUDA SGEMM kernel and a
from-scratch Triton tiled-GEMM implementation (v0 and v1). It includes every
explanation, worked numeric example, code excerpt, table, and ASCII diagram that
came up, in the order the topics were actually discussed.

No LaTeX is used anywhere in this document — all formulas are plain text /
Python-style expressions, and all diagrams are plain ASCII art inside code
fences. Encoding: UTF-8.

---

## Table of Contents — Questions and Topics Covered

1. **CUDA SGEMM kernel — how does the whole thing work?**
   (shared-memory double buffering, register tiling, float4 vectorized loads,
   warp-aware thread layout, the three-level tiling hierarchy)
2. **CUDA: the constexpr maze** — what do `BLOCK_M`, `THREAD_A_LAYOUT_X`,
   `THREAD_C_WARP_X`, etc. actually mean, and how do they fit together?
3. **CUDA: thread-index decomposition** — how does a flat thread ID become
   `tAx`/`tAy`, `tBx`/`tBy`, and the warp-aware `tCx`/`tCy`?
4. **CUDA: loading A and B into shared memory** — the "prologue" load, the
   bank-conflict-avoidance XOR swizzle, and the boundary masks.
5. **CUDA: the A-tile prologue load, traced line by line** with two fully
   worked numeric examples (row padding, column padding).
6. **CUDA: two-level double buffering** — coarse (global → shared) and fine
   (shared → register) software pipelining, shown as explicit timelines.
7. **CUDA: the accumulation step** — rank-1 (outer-product) updates and why
   the total work balances out.
8. **CUDA: writing the result back to C**, and why the boundary guard here is
   safe (no barrier downstream).
9. **CUDA: bugs found in `solve()`** — swapped M/N/K arguments, a mis-sized
   grid, and a missing `extern "C"` quote.
10. **Triton matmul_v0.py** — the basic tiled GEMM using only native Triton
    primitives (no cuBLAS, no `torch.matmul` inside the kernel).
11. **Triton: why masking makes the boundary-handling safe**, verified against
    a real test suite (non-multiple M, N, K; 1x1; rectangular shapes).
12. **Triton matmul_v1.py** — autotuning, grouped program ordering, software
    pipelining (`num_stages`), and pointer advancing.
13. **The TF32-vs-IEEE precision correction** — catching a misleading "265%
    faster than torch.matmul" benchmark result and fixing it honestly.
14. **Honest performance tables**: v1 vs cuBLAS (TF32-vs-TF32), and v0 vs v1.
15. **Why grouped (L2-cache-friendly) program ordering exists at all** — the
    naive row-major tile ordering problem, and the grouped fix.
16. **Why the group-size clamping logic (`actual_group_size_m`) is load-bearing
    for correctness**, not just a performance nicety — shown with a simulated
    bug where two tiles of C are silently never computed.
17. **The "desk and warehouse" cache analogy**, with measured reuse-gap numbers
    for naive vs. grouped tile ordering.
18. **Full worked table of all 24 tiles in grouped order**, and a step-by-step
    derivation of the reuse-gap sequences (why "1,1,10,1,1" and "3,3,3").
19. **Why flat_id=12 lands at tile (3,0) and not (0,4)** — a grid-picture
    explanation plus a full mechanical trace of the grouping formula.
20. **The `[:, None]` / `[None, :]` broadcasting trick**, explained from first
    principles with a tiny non-GEMM example before being applied to GEMM.
21. **The modulo "wrap" trick** for handling M and N not dividing evenly into
    the tile size — why it's safe, with concrete row numbers.
22. **Turning wrapped indices into real memory addresses** for A and B, fully
    worked with real numbers.
23. **The K-reduction loop** — chunking, masking (`a_k_mask`/`b_k_mask`),
    `tl.dot`, running accumulation, and pointer advancing instead of
    recomputing addresses every iteration.
24. **The final store mask for C** — fully worked example showing a tile that
    overflows both the row and column edges of the real matrix at once.
25. **Why the store mask MUST use the unwrapped indices, not the wrapped
    ones** — what breaks (silent data corruption) if you get this backwards.
26. **Clarifying question: "why does it jump from row 10 to row 13?"** — a
    worked-out proof (plus a runnable Python check) that there was no real
    gap; rows 11 and 12 exist and behave identically to their neighbors.
27. **Follow-up deep dive: where exactly, in the real `matmul_v0.py` and
    `matmul_v1.py` source files, is the "wrapped indices" code for A/B, and
    the "unwrapped indices" code for C?** Includes real file line numbers and
    a side-by-side comparison of v0's per-iteration-masking strategy versus
    v1's wrap-once strategy.
28. **Why K can never be safely wrapped**, even though M and N can.
29. **Follow-up deep dive: the CUDA shared-memory loading loop (`sA` prologue
    load), explained line by line** — where `tAx`/`tAy` come from, why the
    loop runs exactly 4 times, a full numeric trace of which row each thread
    touches, and the transposed `sA[k][m]` storage layout explained with a
    tiny side-by-side 4x2 toy example (natural layout vs. transposed layout).
30. **Follow-up: why transposing matrix A actually helps, proven with a
    complete worked example** — using a concrete 4x2 A times 2x3 B matrix
    multiply, rewriting the direct dot-product formula as a "sum of
    outer products over k," verifying both forms give an identical numeric
    answer, then showing exactly why the outer-product method needs "all
    rows for one fixed k" contiguous in memory, and how the transpose
    delivers that.
31. **Porting the hand-tuned kernel to real, compiled CUDA** (`matmul_v0.cu`)
    — three real bugs found and fixed in the supplied `solve()` wrapper
    (missing `extern "C"` quotes, swapped M/N/K kernel-launch argument
    order, and the grid's column-dimension sized from K instead of N), a
    plain C++ CPU reference used only for verification, a full
    correctness + performance test harness (configurable warm-up/timed
    run counts, CUDA-event latency collection, TFLOPS), and the discovery
    (then correction) of a wrong assumption about CPU-reference timing.
32. **Investigating a suspiciously flat ~2.3ms latency floor** across tiny
    to medium matrix sizes — proven, not just asserted, to be a genuine
    fixed per-call `cudaDeviceSynchronize()` cost on the shared/virtualized
    GPU instance, using a standalone empty-kernel micro-benchmark.
33. **Follow-up: is the opposite transpose choice possible (A untouched, B
    transposed)?** Building `matmul_v1.cu` to test it directly, discovering
    it is NOT a symmetric trade-off, and a reader's sharp follow-up
    challenge (using a concrete 4x2 A times 2x3 B example) that exposes an
    inconsistency in the first explanation.
34. **Resolving the inconsistency: outer-product-style accumulation versus
    dot-product-style accumulation are two different algorithms**, each
    with its own correct transpose choice — traced step by step with the
    reader's own example, plus the real reason (register pressure, not
    contiguity) high-performance GEMM kernels prefer outer-product style.
35. **Building the faithful dot-product-style kernel** (`matmul_v2.cu`) to
    prove the resolution for real — real `nvcc -Xptxas -v` register counts
    compared side by side across all three kernel variants, full
    correctness results, and a genuinely surprising performance finding
    that overturned part of the register-pressure prediction.
36. **Building the FLOAT4 ablation** (`matmul_v0_nof4.cu`) — replacing
    matmul_v0.cu's two FLOAT4 operations with scalar equivalents, an
    identical register count, bit-for-bit identical correctness, and a
    real investigation into confusing, inconsistent latency readings that
    turned out to be shared/virtualized-GPU noise affecting both kernels
    equally rather than a genuine difference between them.
37. **Follow-up deep dive: what do all twelve `constexpr` tiling constants
    actually mean?** (`BLOCK_M/N/K/SIZE`, the three `THREAD_*_LAYOUT`
    pairs, the `THREAD_C_*_TILE_SIZE` pair, and the three `THREAD_C_WARP_*`
    constants) — explained with a full flowchart and a diagram for every
    single constant, showing why the same 256 threads get reshaped three
    different ways for three different jobs.
38. **Follow-up: why does `(tAx << 2)` actually solve the bank conflict?**
    A full first-principles explanation of shared-memory banks, a
    concrete before/after table proving an 8-way bank conflict exists
    without the swizzle and vanishes completely with it, and the bit-level
    reason `tAx << 2` is the specific fix that works.
39. **Follow-up: where does the address formula `tAx*128 + tAy` actually
    come from?** Building up row-major array flattening completely from
    scratch with a tiny 2x3 toy example before applying the same rule to
    the real `sA[2][8][128]` array.
40. **Follow-up: which code reads the swizzled write back, and how do we
    know the round trip is correct?** Identifying the exact matching read
    code, then building and running a standalone CUDA round-trip test (and
    a deliberately broken negative-control version) to prove it for real,
    not just assert it.
41. **Follow-up: do the write-side and read-side addresses actually match
    each other?** A parking-garage analogy, a full coverage proof showing
    both sides independently tile the same 128 real positions via
    different numbering systems, and a complete worked numeric example
    tracing four separate writer threads' values into one reader thread's
    single grouped read.
42. **Follow-up: does every thread in the 256-thread block move data for
    A, move data for B, AND compute C?** Resolved with the SIMT execution
    model (same code, different data per thread, three sequential role
    changes in time rather than three permanent groups of specialists),
    proven with a fully worked trace of one real thread (tidx=42) through
    every phase of the kernel.
43. **Follow-up: what is the per-thread stride for A, B, and C?** A and B
    turn out to have simple, uniform strides (32 each); C does not --
    explained with a from-scratch primer on the `& 3` / `& (~3)` bit
    tricks, showing C's real "4 consecutive, jump 64, 4 more consecutive"
    pattern and tracing why that grouping survives even after FLOAT4 was
    removed in matmul_v0_nof4.cu.
44. **Building matmul_v2_nof4.cu (the FLOAT4 ablation of the dot-product
    kernel)** -- identical register count, bit-for-bit identical
    correctness, and a major, honest correction: a fresh re-check revealed
    the earlier "matmul_v2.cu beats matmul_v0.cu at 4096^3" claim was
    almost certainly an upward noise artifact on this shared GPU, not a
    reliable result -- clean, repeated, back-to-back measurements now show
    matmul_v0.cu consistently winning instead.
45. **Follow-up: why doesn't matmul_v2.cu need `(tAx << 2)` to avoid a
    bank conflict for A?** A full bank-math proof showing the shape flip
    (A stored natural instead of transposed) reverses which index gets
    "erased" by the modulo-32 operation, eliminating the conflict without
    any swizzle -- plus a bonus discovery made while checking: B's write
    pattern in matmul_v2.cu has a real, unaddressed 8-way bank conflict.
46. **Follow-up: investigating and attempting to fix B's bank conflict**
    (`matmul_v2_bfix.cu`) -- a complete from-scratch re-derivation of how
    `sB[0][j+tBx][i+tBy]` becomes a flat address, confirmation that bank
    conflicts affect performance only (never correctness, which is why
    all test cases already passed), an attempted profiler-based
    investigation blocked by this shared GPU's restricted hardware
    counter access, and a derived-and-tested swizzle fix that, measured
    honestly, made performance slightly WORSE rather than better.
47. **Follow-up: why does `mask = (n >> 2) & 7` combined with `k ^ mask`
    actually fix the bank conflict?** A complete worked trace of n=0..7
    showing the collision disappear one step at a time, an explanation of
    why `& 7` isn't just tidiness but prevents real out-of-bounds memory
    corruption, and confirmation that the prologue-load and mid-loop-
    reload snippets are the same fix applied twice, once per load site.
48. **Follow-up: re-deriving "Step 0" itself from raw code and arithmetic**
    -- proves, from scratch, why `tBy` is provably constant across an
    entire warp (integer-division truncation on 32 consecutive numbers),
    why `n` provably sweeps 32 consecutive values (remainder of already-
    small numbers equals themselves), and why the bank collision repeats
    every 4 values of `n` (the `8*4=32` "one full lap of a 32-hour clock"
    modular-arithmetic argument) -- with a flowchart tying all three
    pieces together into the original collision table.
49. **Follow-up: is `THREAD_A_LAYOUT_Y=32` deliberately matched to warp
    size, the same way `THREAD_B_LAYOUT_X=32` is?** No -- traced from the
    real global-memory coalescing requirements for each operand: B's
    THREAD_B_LAYOUT_X=32 is a genuine, deliberate choice (B's 128-wide
    rows have plenty of room for a full 32-wide coalesced burst, and this
    is exactly the choice that causes the bank conflict from Parts 17-20
    as a side effect); A's THREAD_A_LAYOUT_Y=32 is proven to be a pure
    arithmetic byproduct of two unrelated forced choices (BLOCK_SIZE=256,
    BLOCK_K=8) that never even appears in A's own addressing formulas.
50. **Follow-up: why not just set BLOCK_K=32 and every THREAD_*_LAYOUT_X
    to 32, so everything aligns to warp size even when transposed?**
    Two independent reasons it fails: (1) a hard register-budget
    catastrophe -- 1024 threads/block at this kernel's real measured 168
    registers/thread needs 2.6x more registers than an entire SM's
    register file -- and (2) even ignoring that, the same resize helps A
    (removes its conflict) but makes B's conflict dramatically WORSE (an
    8-way conflict becomes a full 32-way conflict), proven with address
    math for both, because the outcome depends on which index (inner-
    added vs outer-multiplied) varies across a warp, not on any constant
    uniformly equaling 32.
51. **Follow-up: why does matmul_v0.cu's swizzle shift LEFT (`tAx << 2`)
    while matmul_v2_bfix.cu's swizzle shifts RIGHT and clamps
    (`(n >> 2) & 7`)?** Both apply the identical recipe (derive a mask
    from the outer index, XOR it into the inner index) but in mirror-
    image shape situations: A's outer index (K) is small and its inner
    index (M) is big, so the mask scales UP with no clamp ever needed;
    B's outer index (N) is big and its inner index (K) is small, so the
    mask must scale DOWN and be clamped with `&7`, or risk genuine
    out-of-bounds memory corruption.
52. **Follow-up: why does matmul_v0.cu's accumulation read
    `tCsA[(tk - 1) & 1]` instead of `tCsA[tk & 1]`?** A complete 9-step
    (`tk=0..8`) trace of the register-level software-pipelining scheme
    proving `(tk-1)&1` always points at the buffer that was filled one
    iteration earlier, with a worked example of what silently breaks if
    `tk` is used instead, and a pipeline diagram showing why staggering
    load and compute by one step hides shared-memory latency behind ALU
    work instead of stalling on it.
53. **Building simulation_v0.py: a full per-thread index table for
    matmul_v0.cu** -- a standalone Python simulation (no GPU needed) that
    computes every index value (tAx, tAy, tCx, tCy, bufferId, the XOR
    swizzles, the (tk-1)&1 pipeline bookkeeping) for one concrete thread
    (tidx=13) across every phase of the kernel, with every single number
    cross-checked against hand derivation before being presented.
54. **Building simulation_v1.py: one continuous, execution-order log**
    -- a line-by-line re-implementation of the real kernel's control flow
    (not split into separate per-phase tables), proving the exact
    compute-then-load ordering within each `tk` step, and making the
    full `A,B -> sA,sB -> tCsA,tCsB -> acc -> C` data chain traceable
    through the log's own source/destination columns.
55. **Follow-up: formatting the source column as `A[][]`/`B[][]`
    instead of a flat `A[]`/`B[]` index** -- a small, targeted fix
    across all four global-memory read sites, re-verified end to end
    with a before/after comparison.
56. **Building simulation_v2.py: reverse-engineering one target C
    element's complete lineage** -- given only `C[4][24]`, a brute-force
    search (never hand-inverted algebra) discovers which grid block,
    which owning thread, and which OTHER cooperating threads loaded the
    relevant A/B data into shared memory -- with the result
    self-verifying, since the 16 discovered A-elements and B-elements
    independently reconstruct the exact mathematical definition
    `C[4][24] = sum(A[4][k] * B[k][24])`.
57. **Building matmul_v2_bfix_simulation_v0.py: the same unified-log
    approach, adapted to a structurally different kernel** -- covers why
    matmul_v2_bfix.cu's trace looks different (no `tk`-pipelining, whole
    BLOCK_K-row register caching, a mask-based swizzle on B instead of A),
    demonstrates the mask descramble concretely (`n=6, mask=1`: a clean
    pairwise swap of adjacent physical slots), and records a real,
    unresolved file-persistence issue encountered while building it
    (the file vanished between turns twice despite verified successful
    creation, with backups put in place as a mitigation).
58. **Building matmul_v2_bfix_simulation_v1.py: reverse lineage for the
    mask-swizzled kernel** -- the matmul_v2_bfix.cu counterpart of
    simulation_v2.py, showing the mirror-image version of the writer-
    search trick (this time B needs the un-scrambling, not A), finding a
    DIFFERENT owning thread (tidx=96, not 13) for the identical target
    `C[4][24]` because this kernel's simpler epilogue assigns output
    cells to threads differently -- yet both kernels' independent traces
    converge on the exact same underlying mathematical sum.
59. **Follow-up: why does matmul_v0.cu's epilogue need BOTH `row0` AND
    `tCy << 2`?** Proven necessary by showing what collides without
    each: removing `row0` makes every block's thread 13 compute the
    identical row numbers (total collision across blocks); removing
    `tCy<<2` makes all 256 threads within one block compute the
    identical row (total collision within a block) -- plus a coverage
    count proving the two pieces together tile exactly BLOCK_M's 128
    rows with zero gaps and zero overlaps (16 tCy-values x 2 i-groups x
    4 i-slots = 128).
60. **Follow-up: is `tCy` really different for almost every thread, and
    why do the `i&~3`/`i&3` bit-tricks exist at all?** Corrected with
    exact numbers: `tCy` only takes 16 distinct values across 256
    threads (NOT 256 distinct values) -- `tidx=13` and `tidx=45` share
    the identical `tCy=1` and differ only in `tCx` (6 vs 14), proving
    it's the full `(tCx,tCy)` PAIR, not `tCy` alone, that is unique per
    thread. The bit-tricks are then shown to be forced, not arbitrary,
    by tracing back to the FLOAT4-driven register-loading scheme from
    Part 24 -- with a concrete demonstration of what silently breaks if
    a naive linear formula (`row0 + tCy*8 + i`) were used instead.
61. **Follow-up: ground-up, fully-visual rebuild for "totally lost" --
    and why not just 8 simple consecutive rows per thread?** A complete
    re-derivation from the simplest possible nested mental model (whole
    matrix -> block -> 16x16 thread grid -> one thread's 8x8 piece),
    revealing the real picture (every thread's 8 rows come in TWO
    separated clusters of 4, one per half of the 128-row block) with a
    full 128-row ownership map, a step-by-step `i=0..7` walkthrough, and
    a zoomed-out 2x2-quadrant diagram of one thread's actual four
    scattered 4x4 squares -- followed by the honest, traceable answer to
    why: the epilogue doesn't choose this scattered pattern, it is
    FORCED to match wherever the compute phase's FLOAT4 register loads
    (2 wide instructions instead of 8 slow scalar ones -- a real,
    concrete 4x instruction-count reduction) already physically placed
    the data.
62. **Follow-up: the outer `for (k=BLOCK_K; k<K+BLOCK_K; k+=BLOCK_K)`
    loop -- what's the logic, and why start at BLOCK_K with step
    BLOCK_K?** Verified with K=16, BLOCK_K=8 that this loop runs exactly
    2 times (k=8, 16) -- one pass per real chunk. `k` always means "the
    chunk to reload NEXT," never "the chunk being computed right now"
    (that's always the previous iteration's reload, already sitting in
    `bufferId`). Explains why the upper bound is `K+BLOCK_K`, not `K` --
    proven that `k<K` alone would silently never compute the final
    chunk, discarding half of K's real contribution.
63. **Follow-up: why must chunk 0 be loaded by a separate PROLOGUE, and
    why must the loop's stride be exactly BLOCK_K?** A restaurant/
    assembly-line analogy for the prologue's "bootstrap" necessity,
    followed by a concrete proof using real tAx values (0..7, one fixed
    value per thread, never looped within a thread) that `k + tAx`
    requires the stride to exactly match `tAx`'s fixed range width --
    with a numeric side-by-side of a stride-1 column collision.
64. **Follow-up: a full real-code, line-by-line proof of the stride-1
    bug** -- quotes matmul_v0.cu's actual reload block (lines 178-198)
    verbatim, traces line 179 (`int c = k + tAx`) and line 183's fixed
    write target under both the correct and broken stride, and backs it
    with an executed Python simulation showing the REAL tally: column 9
    double-counted, climbing to column 15 counted EIGHT separate times
    into the accumulating `acc[][] += ...`, with the exact bufferId/
    reload/compute iteration trace that produces it.
65. **Follow-up: in matmul_v2_bfix.cu's prologue, why does sA's load use
    only row0 (never col0), and sB's load use only col0 (never row0)?**
    Traced to the basic shape of matrix multiplication itself, not a
    CUDA trick: A's column axis IS the K/reduction axis (always taken in
    full), so only row0 (which M-rows) matters; B's row axis IS the
    K/reduction axis, so only col0 (which N-columns) matters. Proven
    concretely by running two DIFFERENT blocks that share one of the two
    offsets: blocks with the same row0 but different col0 load
    byte-for-byte IDENTICAL A data; blocks with the same col0 but
    different row0 load byte-for-byte IDENTICAL B data.
66. **Follow-up: why does matmul_v0.cu's epilogue need the
    `(i&~3)*THREAD_C_LAYOUT_Y + (i&3)` bit-trick while
    matmul_v2_bfix.cu's epilogue (`tm * THREAD_C_LAYOUT_Y` directly)
    does not?** Traced to what FLOAT4 actually groups together in each
    kernel's compute phase: matmul_v0.cu's FLOAT4 groups 4 DIFFERENT
    ROWS into one register load (forcing a 2-cluster scatter that the
    epilogue must undo), while matmul_v2_bfix.cu's FLOAT4 groups 4
    K-VALUES for a single, independent row (never touching the row
    dimension at all) -- confirmed with real numbers showing
    matmul_v0.cu's scattered `tm=0->{4,5,6,7}, tm=1->{68,69,70,71}`
    versus matmul_v2_bfix.cu's clean linear `tm=0->1, tm=1->17, ...,
    tm=7->113`, and traced back to each kernel's underlying shared-
    memory layout (transposed vs natural) as the root cause.
67. **Follow-up: the exact line-by-line smoking gun for the scattered
    vs linear row patterns** -- pinpoints the single structural fork to
    ONE loop bound: matmul_v0.cu line 167's `tm < THREAD_C_Y_TILE_SIZE
    >> 2` (only 2 tm iterations exist, `8>>2=2`) forces line 168's
    `<<2` scaling to bundle 4 real rows per iteration, versus
    matmul_v2_bfix.cu line 141's `tm < THREAD_C_Y_TILE_SIZE` (full 8
    iterations, no shrinking, no `<<2` scaling anywhere on line 142) --
    verified by actually computing both formulas' real row outputs and
    showing they exactly match the previously-established scattered and
    linear patterns.

---

# PART 1 — CUDA SGEMM Kernel (hand-tuned, shared-memory double-buffered)

## 1.1 The big picture: a three-level tiling hierarchy

This is a fully production-grade, highly-tuned CUDA SGEMM (single-precision
`C = A x B`) kernel, using shared-memory double buffering, register-level
tiling, vectorized `float4` loads, and warp-aware thread layout to squeeze
maximum throughput out of the GPU.

GEMM computes `C[M x N] = A[M x K] x B[K x N]`. Doing this naively (one thread
per output element, reading from global memory every time) is extremely
memory-bound. This kernel instead tiles the problem three levels deep:

```
Grid level:    each CUDA block computes a BLOCK_M x BLOCK_N = 128x128 tile of C
Warp level:    each of the 8 warps in the block owns a sub-region of that tile
Thread level:  each of the 256 threads computes an 8x8 = 64-element sub-tile,
               held entirely in registers (the acc[8][8] array)
```

256 threads x 64 elements each = 16,384 = 128 x 128. Every element of the
block's C tile is accounted for exactly once.

## 1.2 Making sense of the constexpr maze

| Constant | Formula | Value | Purpose |
|---|---|---|---|
| `BLOCK_M`, `BLOCK_N` | -- | 128, 128 | size of the C tile one block computes |
| `BLOCK_K` | -- | 8 | how many K-slices get staged into shared memory at once |
| `BLOCK_SIZE` | -- | 256 | threads per block |
| `THREAD_A_LAYOUT_X/Y` | 8, 256/8 | 8, 32 | arranges the 256 threads into an 8x32 grid for cooperatively loading the A tile |
| `THREAD_B_LAYOUT_X/Y` | 32, 256/32 | 32, 8 | arranges threads into a 32x8 grid for loading the B tile |
| `THREAD_C_LAYOUT_X/Y` | 16, 256/16 | 16, 16 | arranges threads into a 16x16 grid for *computing* C |
| `THREAD_C_X/Y_TILE_SIZE` | 128/16, 128/16 | 8, 8 | each thread's own C sub-tile is 8x8 = 64 elements |
| `THREAD_C_WARP_X/Y` | 8, 32/8 | 8, 4 | within one warp (32 threads), arrange them as an 8x4 grid |
| `THREAD_C_WARP_DIM_X` | 16/8 | 2 | 2x4=8 warps tile the 16x16 thread grid |

Sanity check that confirms this all hangs together: shared tiles `sA`/`sB`
each hold `BLOCK_K x BLOCK_M = 8x128 = 1,024` floats; with 256 threads
cooperating, that's exactly 4 elements loaded per thread for each tile --
which is exactly what the loading loops below iterate 4 times for.

## 1.3 The thread-index decompositions, with real numbers

```c
int tAx = tidx & (THREAD_A_LAYOUT_X - 1);   // tidx % 8
int tAy = tidx / THREAD_A_LAYOUT_X;         // tidx / 8
```

These two lines fold a flat thread ID (0..255) into a 2D coordinate for the
purpose of cooperative loading -- nothing more exotic than that for `tA`/`tB`.
A few concrete values:

| tidx | tAx (mod 8) | tAy (div 8) | tBx (mod 32) | tBy (div 32) |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 8 | 0 | 1 | 8 | 0 |
| 31 | 7 | 3 | 31 | 0 |
| 32 | 0 | 4 | 0 | 1 |
| 255 | 7 | 31 | 31 | 7 |

The warp/lane decomposition for *computing* C is deliberately fancier:

```c
int warpId = tidx >> 5;                           // which of the 8 warps
int laneId = tidx & 31;                            // position within the warp (0-31)
int warpx  = warpId & (THREAD_C_WARP_DIM_X - 1);   // warpId % 2
int warpy  = warpId / THREAD_C_WARP_DIM_X;         // warpId / 2
int lanex  = (laneId & 15) >> 1;
int laney  = ((laneId >> 4) << 1) + (laneId & 1);
int tCx = warpx * THREAD_C_WARP_X + lanex;
int tCy = warpy * THREAD_C_WARP_Y + laney;
```

| tidx | warpId | laneId | warpx,warpy | lanex,laney | tCx, tCy |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0, 0 | 0, 0 | 0, 0 |
| 1 | 0 | 1 | 0, 0 | 0, 1 | 0, 1 |
| 33 | 1 | 1 | 1, 0 | 0, 1 | 8, 1 |
| 255 | 7 | 31 | 1, 3 | 7, 3 | 15, 15 |

The key insight: despite all the bit-twiddling, `tCx` always lands in 0..15
and `tCy` always lands in 0..15 -- exactly a 16x16 grid, same total range as a
naive `tCx = tidx % 16; tCy = tidx / 16` would give. The complexity exists
purely to control *which thread gets which (tCx, tCy)*, specifically so that
threads belonging to the same warp end up clustered in a pattern that lines up
with efficient shared-memory broadcast/bank access later -- not to change what
is being computed, just who computes it.

## 1.4 Loading A and B into shared memory (the "prologue")

```c
for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {   // i = 0, 32, 64, 96  (4 iterations)
    int r = row0 + i + tAy;
    sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] =
        (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
}
```

Each thread does 4 loads (i = 0, 32, 64, 96), covering all 128 rows of the
tile cooperatively across 32 distinct `tAy` values x 4 loop iterations.
Notice `sA` is stored transposed relative to A's natural row-major layout --
`sA[buf][k][m]`, with k (0-7) outer and m (0-127) inner. That's deliberate:
during the compute phase we need "all M-values for one fixed k" as a
contiguous, vectorizable chunk -- exactly what this layout gives us.

The `^ ((tAx) << 2)` part is a bank-conflict-avoidance swizzle: shared memory
is split into 32 banks, and if multiple threads in a warp simultaneously
write to different addresses that happen to map to the same bank, those
writes serialize. XOR-ing the target column index with a function of `tAx`
scrambles which thread lands in which bank in a deterministic, reversible way
(the same XOR is applied later when reading it back: `sA[bufferId][tk][r ^
(tk << 2)]`), spreading the 256 threads' accesses evenly across all 32 banks
instead of clustering them.

The B-tile load is simpler (no swizzle, since B's natural row-major layout
already works out favorably here):

```c
for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {    // i = 0 (1 iteration)
    int r = i + tBy;
    for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) { // j = 0, 32, 64, 96 (4 iterations)
        int c = col0 + j + tBx;
        sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
    }
}
```

`__syncthreads()` after both loads ensures the whole tile is visible before
any thread starts computing with it -- same RAW-hazard reasoning that applies
to every barrier in GPU kernel code.

## 1.5 Deep-dive: the A-tile prologue load, traced line by line

The code again, for reference:

```c
#pragma unroll
for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
    int r = row0 + i + tAy;
    sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
}
```

Recall the constants involved: `BLOCK_M=128`, `THREAD_A_LAYOUT_X=8`,
`THREAD_A_LAYOUT_Y=32`, `BLOCK_K=8`.

**Line by line:**

`#pragma unroll`: a compiler hint telling `nvcc` to fully unroll this loop --
replace it with 4 literal, independent copies of the body (since the trip
count, `BLOCK_M / THREAD_A_LAYOUT_Y = 128/32 = 4`, is a compile-time
constant). This eliminates loop-control overhead and, more importantly, gives
the compiler 4 fully independent load instructions it can freely
schedule/interleave for better memory-latency hiding.

`for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y)`: runs for i = 0, 32,
64, 96 -- 4 iterations. Each iteration handles a different 32-row "stripe" of
the block's 128-row tile. Since only 32 distinct `tAy` values exist, it takes
exactly 4 stripes to cover all 128 rows: [0,32), [32,64), [64,96), [96,128).

`int r = row0 + i + tAy;`: the actual global row of matrix A this thread is
responsible for, in this particular iteration. `row0` is the block's starting
row (`blockIdx.y * BLOCK_M`), `i` picks the stripe, `tAy` (0-31) picks this
thread's position within the stripe.

`sA[0][tAx][(i + tAy) ^ ((tAx) << 2)]` -- the write target, in three pieces:
- `sA[0]` -- buffer 0 (the first of the double-buffer pair; this is the
  prologue, loading before the main loop even starts).
- `[tAx]` -- tAx ranges 0..7, and since this is the very first K-chunk
  (columns 0..7 of A), tAx directly is the column-within-chunk index.
- `(i + tAy) ^ (tAx << 2)` -- the row position within the tile (0..127),
  scrambled by XOR-ing with tAx*4. This is the bank-conflict-avoidance
  swizzle: it's a reversible permutation, so different threads writing "the
  same row offset for different tAx" land in different actual shared-memory
  banks instead of colliding.

`(r < M && tAx < K) ? A[r * K + tAx] : 0.0f` -- two independent bounds
checks, each guarding a different kind of "ragged edge":
- `r < M` -- is this row actually inside the real matrix? (Guards the case
  where M isn't an exact multiple of BLOCK_M.)
- `tAx < K` -- is this column actually inside the real matrix? (Guards the
  case where K is smaller than BLOCK_K=8 entirely -- i.e. the whole first
  chunk doesn't even exist.)
- If both pass: `A[r * K + tAx]` is the standard row-major lookup (row r,
  column tAx) into the M x K matrix A.

### Worked example 1: a normal in-bounds load, plus the row-padding case

Let `M=200`, `K=10`, `blockIdx.y=1` (so `row0 = 1*128 = 128`). Take thread
`tidx=3` -> `tAx = 3%8 = 3`, `tAy = 3/8 = 0`.

| i | r = 128+i+0 | r < 200? | tAx<K (3<10)? | value loaded | swizzled write index (i^12) |
|---|---|---|---|---|---|
| 0 | 128 | yes | yes | A[128*10+3] = A[1283] | 0^12 = 12 -> sA[0][3][12] |
| 32 | 160 | yes | yes | A[160*10+3] = A[1603] | 32^12 = 44 -> sA[0][3][44] |
| 64 | 192 | yes | yes | A[192*10+3] = A[1923] | 64^12 = 76 -> sA[0][3][76] |
| 96 | 224 | no! | yes | 0.0f (padding) | 96^12 = 108 -> sA[0][3][108] |

The last row demonstrates exactly why the `r < M` check matters: row 224
doesn't exist (the real matrix only has 200 rows), so that slot in shared
memory gets a clean 0.0f instead of reading garbage or crashing -- padding
that contributes nothing when it's later multiplied into the accumulation.

### Worked example 2: the column-padding case (K smaller than BLOCK_K)

Now suppose `K=5` (narrower than one full chunk, BLOCK_K=8). Take thread
`tidx=6` -> `tAx=6`, `tAy=0`. Since `tAx=6 >= K=5` for every iteration, this
thread never reads real data at all, regardless of row:

| i | r | tAx<K (6<5)? | value loaded |
|---|---|---|---|
| 0 | 128 | no | 0.0f |
| 32 | 160 | no | 0.0f |
| 64 | 192 | no | 0.0f |
| 96 | 224 | no | 0.0f |

This thread writes zeros into `sA[0][6][...]` across the board -- correctly
representing "column 6 of A doesn't exist when K is only 5," so the
shared-memory tile still has a full, well-defined 8x128 shape, just with the
nonexistent columns zeroed out instead of left as garbage.

### Confirming full, non-overlapping coverage

It's worth checking that these 256 threads x 4 iterations really do tile the
whole 8x128 region with no gaps or double-writes:

- `tAx = tidx % 8` takes every value 0..7 exactly 32 times (once per distinct
  tidx in that residue class) -- so all 8 "columns" of the chunk get covered.
- For a fixed tAx, `tAy = tidx / 8` sweeps 0..31 across the 32 threads
  sharing that tAx. Combined with i in {0,32,64,96}, the quantity `i + tAy`
  sweeps every single value in 0..127, exactly once (stripe 0 covers rows
  0-31, stripe 32 covers 32-63, etc. -- no overlap, no gap).

So every one of the 8 x 128 = 1,024 slots in `sA[0]` gets written by exactly
one (thread, iteration) pair -- matching perfectly with 256 threads x 4 loads
each = 1,024 total writes. Clean, complete, non-redundant coverage of the
tile, cooperatively split across the block.

## 1.6 Double buffering at two levels of the memory hierarchy

This is the most sophisticated part of the kernel, and it's genuinely a
two-level software pipeline.

**Level 1 (coarse) -- overlap global->shared loads with compute.** `sA`/`sB`
each have a leading [2] dimension (bufferId and bufferId^1). While the thread
computes using the data already sitting in `sA[bufferId]`/`sB[bufferId]`, it
simultaneously kicks off loading the next K-chunk's data into
`sA[bufferId^1]`/`sB[bufferId^1]` -- so the slow global-memory fetch for chunk
k+1 is hidden behind the compute for chunk k, instead of happening serially
after it.

```c
for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
    // ... compute using sA[bufferId]/sB[bufferId] (current chunk) ...
    if (k < K) {
        // ... load sA[bufferId^1]/sB[bufferId^1] (NEXT chunk) from global memory ...
        __syncthreads();
    }
    bufferId ^= 1;   // swap which buffer is "current" for next iteration
}
```

Worked example: suppose K = 24 (so 3 chunks of BLOCK_K=8). The loop runs for
k = 8, 16, 24:
- k=8: compute using chunk 0 (already loaded in the prologue); since 8 < 24,
  load chunk 1 into the other buffer; swap.
- k=16: compute using chunk 1; load chunk 2 into the other buffer; swap.
- k=24: compute using chunk 2; 24 < 24 is false, so no load (we're out of
  chunks) -- this is the pipeline's "drain" step.

**Timeline table for the coarse pipeline:**

| Outer loop iteration | bufferId (compute FROM here) | chunk being computed | bufferId^1 (load INTO here) | chunk being loaded (if k<K) | after: bufferId ^= 1 |
|---|---|---|---|---|---|
| (prologue, before the loop) | -- | -- | buffer 0 | chunk 0 (k=0..7) | bufferId = 0 |
| k=8 | 0 | chunk 0 | 1 | chunk 1 (k=8..15) | bufferId = 1 |
| k=16 | 1 | chunk 1 | 0 | chunk 2 (k=16..23) | bufferId = 0 |
| k=24 | 0 | chunk 2 | -- | 24 < 24 is false -> no load (drain) | bufferId = 1 (irrelevant, loop ends) |

The key thing this table makes obvious: at every iteration except the very
first and very last, the GPU is simultaneously computing with one chunk's
data and fetching the next chunk's data from global memory -- that overlap is
the entire point of having [2] buffers instead of [1]. Without double
buffering, each iteration would have to be strictly sequential: load chunk ->
wait -> compute chunk -> load next chunk -> wait -> compute next chunk, with
the (slow) global memory latency fully exposed every single time instead of
hidden behind useful work.

**Level 2 (fine) -- overlap shared->register loads with compute**, nested
inside each chunk's compute, using the exact same ping-pong trick at a
smaller scale (`tCsA[2]`, `tCsB[2]`, indexed by `tk & 1`):

```c
for (int tk = 0; tk < BLOCK_K + 1; ++tk) {       // runs 9 times, not 8 -- one extra "drain" iteration
    if (tk > 0) {
        for (tm...) for (tn...)
            acc[tm][tn] += tCsA[(tk-1) & 1][tm] * tCsB[(tk-1) & 1][tn];  // compute with PREVIOUS slice's data
    }
    if (tk < BLOCK_K) {
        // vectorized float4 loads from shared memory into tCsA[tk&1] / tCsB[tk&1] -- the NEXT slice
    }
}
```

So for tk=0: only loads happen (no tk>0 compute yet -- this is the "fill"
step). For tk=1..7: compute using slice tk-1 while simultaneously loading
slice tk. For tk=8: only the final compute happens (8 < BLOCK_K=8 is false,
so no load -- the "drain" step, consuming the last slice loaded at tk=7).
This BLOCK_K+1-iteration loop shape (compute-while-prefetch, with one extra
iteration to drain) is the classic prologue/steady-state/epilogue
software-pipeline pattern, just compressed into a single loop with two guard
conditions.

**Timeline table for the fine pipeline:**

| tk | tk > 0? compute using... | tk < BLOCK_K? load into... |
|---|---|---|
| 0 | (no, skip compute) | yes -- load k-slice 0 into tCsA[0]/tCsB[0] |
| 1 | yes -- compute with slice 0 (tCsA[0]) | yes -- load slice 1 into tCsA[1]/tCsB[1] |
| 2 | yes -- compute with slice 1 (tCsA[1]) | yes -- load slice 2 into tCsA[0]/tCsB[0] (slot reused) |
| 3 | compute with slice 2 (tCsA[0]) | load slice 3 into tCsA[1] |
| ... | ... | ... |
| 7 | compute with slice 6 | load slice 7 into the alternating slot |
| 8 | compute with slice 7 | (no, 8<8 is false) -- drain, no more loads |

Same shape as the outer table, just one level further down the memory
hierarchy, and with only 2 register slots (& 1) instead of 2 shared-memory
buffers -- since registers are far more scarce than shared memory, you can
only afford to keep 2 slices "in flight" at once here, versus the outer
pipeline affording a full 8-slice (BLOCK_K-wide) buffer per stage.

Together, these two tables show the kernel is really running two nested
software pipelines at once: a coarse one hiding global-memory latency behind
chunk-sized compute, and a fine one hiding shared-memory latency behind
individual-slice compute -- both using the exact same "load slot N+1 while
computing with slot N, then flip which slot is current" idiom, just at wildly
different granularities (8 k-values vs. 1 k-value per stage, global memory vs.
shared memory as the thing being hidden).

The vectorized loads themselves:

```c
for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm++) {   // 8>>2 = 2 iterations
    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
}
```

`FLOAT4` reinterprets 4 consecutive floats as one 128-bit float4 load/store --
a single memory instruction instead of four, cutting instruction overhead
substantially. With only 2 iterations grabbing 4 values each, this fills all
8 slots of `tCsA[tk&1][0..7]` using just 2 vectorized instructions. Note
`tm * THREAD_C_LAYOUT_Y` = `tm * 16`: this reveals that each thread's 8-row
Y-tile is not one contiguous chunk -- it's two groups of 4 rows, 64 rows
apart (tCy*4 and tCy*4 + 64), interleaved across threads rather than
contiguous per-thread. That's deliberate, matching how the data was swizzled
into shared memory in the first place.

## 1.7 The accumulation step

```c
acc[tm][tn] += tCsA[(tk-1) & 1][tm] * tCsB[(tk-1) & 1][tn];
```

This is a classic rank-1 (outer-product) update: for one fixed k-value,
every one of a thread's 8 A-values gets multiplied against every one of its 8
B-values, accumulating into all 64 entries of `acc[8][8]`. Over the full
kernel, each thread performs exactly `64 x K` multiply-adds (8 slices/chunk x
K/8 chunks x 64 MACs/slice = 64 x K), and across the whole block,
`256 x 64 x K = 16,384 x K = BLOCK_M x BLOCK_N x K` -- exactly the total work
needed for this tile. Everything balances.

## 1.8 Writing the result back to C

```c
for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
    int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
    for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
        int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
        if (r < M && c < N) C[r * N + c] = acc[i][j];
    }
}
```

`i & 3` extracts "which of the 4 rows within this thread's sub-group" (0-3),
while `i & ~3` (0 or 4) selects "which of the two 4-row groups" --
multiplying that by `THREAD_C_LAYOUT_Y` (16) reproduces the same 64-row jump
we saw in the loading step. The `if (r < M && c < N)` guard is a safe
boundary check here, because -- unlike a mid-kernel early return -- there's
no subsequent `__syncthreads()` for any thread to strand itself away from; it
is purely gating the final global-memory write. The rule: only gate the
memory operation, never gate participation in a barrier.

## 1.9 Bugs found in solve()

Flagged while explaining (not fixed, since the ask was "how does it work"):

1. `solve()` swaps N and K when calling the kernel: the kernel signature is
   `(A, B, C, M, N, K)` but it's invoked as
   `matrix_multiplication_kernel<<<...>>>(A, B, C, M, K, N)` -- the kernel's
   parameter named N actually receives the real K, and vice versa.
   Everywhere inside the kernel that reads N or K is now using the swapped
   value.
2. The grid's x-dimension uses K instead of N:
   `dim3 blocksPerGrid((K + BLOCK_N - 1) / BLOCK_N, ...)` -- since
   `blockIdx.x` is meant to tile across the output's column count (N), this
   should almost certainly be `(N + BLOCK_N - 1) / BLOCK_N`.
3. `extern C void solve(...)` is missing quotes -- should be
   `extern "C" void solve(...)` to actually disable C++ name mangling (as
   currently written, this likely won't even compile).

Items 1 and 2 interact (a swapped dimension feeding a mis-sized grid), so this
kernel would very likely produce wrong results or out-of-bounds access for
non-square matrices as currently wired up.

---

# PART 2 — Triton Tiled GEMM (matmul_v0.py -> matmul_v1.py)

## 2.1 matmul_v0.py -- basic tiled GEMM, native Triton primitives only

Built at `/home/jupyter/jeff_workspace/triton/leetgpu/matrixMultiplication/matmul_v0.py`
-- using only `tl.load`, `tl.store`, `tl.dot`, `tl.arange`, `tl.zeros` (no
cuBLAS/`torch.matmul` inside `solve()` -- that's reserved for the test harness
as a trusted oracle). All 9 test cases pass, including 1x1 and a fully
rectangular 257x63 @ 63x129.

Problem statement (from the file's own docstring):

```
Given a matrix A of dimensions (M x K) and a matrix B of dimensions
(K x N), compute the product matrix C = A @ B, which has dimensions
(M x N).

All three matrices (A, B, C) are stored in ROW-MAJOR format, meaning
that for a matrix with R rows and S columns, element (row, col) lives
at flat memory offset:

    flat_offset = row * S + col
```

Implementation requirements enforced in the file:
1. Only "native" features are used: PyTorch (for GPU memory allocation /
   tensor bookkeeping only) and Triton's own built-in language primitives
   (tl.load, tl.store, tl.dot, etc). No external GEMM library (cuBLAS,
   cuDNN, cutlass, torch.matmul, etc.) performs the actual computation.
2. The `solve(...)` function signature is fixed and must not change.
3. The final result is written into matrix C in place.

Tile size configuration used in v0 (kept simple/symmetric, not aggressively
tuned):

```python
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32
```

**Traced by hand**: `A=[[1,2],[3,4]]`, `B=[[5,6],[7,8]]` ->
`C=[[19,22],[43,50]]`:

```
C[0][0] = 1*5 + 2*7 = 19
C[0][1] = 1*6 + 2*8 = 22
C[1][0] = 3*5 + 4*7 = 43
C[1][1] = 3*6 + 4*8 = 50
C = [[19, 22], [43, 50]]
```

Kernel steps, imagining a tile size exactly matching this tiny example:
1. Tile ownership (STEP 1-2): one program instance is launched for this
   single 2x2 tile of C -- global_row_indices = [0,1], global_col_indices =
   [0,1].
2. Accumulator (STEP 3): accumulator = [[0,0],[0,0]].
3. The K-loop (STEP 4): with K=2 fitting in one chunk, the kernel loads the
   whole A tile [[1,2],[3,4]] and the whole B tile [[5,6],[7,8]], then does
   tl.dot(a_tile, b_tile) -- a single hardware-accelerated tile-matmul --
   producing [[19,22],[43,50]] in one shot, added into the accumulator.
4. Write-back (STEP 5): the accumulator is stored into C, masked against
   m=2, n=2 (trivially all in-bounds here).

In the real code, BLOCK_SIZE_K=32, so for bigger K this loop runs multiple
times, each pass handling 32 columns of A / 32 rows of B and accumulating
(accumulator = accumulator + partial_product) rather than overwriting -- this
is exactly the reduction-over-K concept:

```
C[row, col] = sum over k_index from 0 to K-1 of:
                  A[row, k_index] * B[k_index, col]
```

just computed 32 k-values at a time instead of one at a time.

### The actual kernel code (STEP 1 through STEP 4, A and B loading)

```python
# -----------------------------------------------------------
# STEP 1: figure out which tile of C this program instance owns.
# -----------------------------------------------------------
program_id_along_rows = tl.program_id(axis=0)
program_id_along_cols = tl.program_id(axis=1)

# -----------------------------------------------------------
# STEP 2: compute the GLOBAL row and column indices (into C) that
# this program instance is responsible for.
# -----------------------------------------------------------
tile_start_row = program_id_along_rows * BLOCK_SIZE_M
local_row_offsets = tl.arange(0, BLOCK_SIZE_M)
global_row_indices = tile_start_row + local_row_offsets

tile_start_col = program_id_along_cols * BLOCK_SIZE_N
local_col_offsets = tl.arange(0, BLOCK_SIZE_N)
global_col_indices = tile_start_col + local_col_offsets

# -----------------------------------------------------------
# STEP 3: allocate the accumulator that will hold this tile's
# partial (and eventually final) result, entirely in registers.
# -----------------------------------------------------------
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

# -----------------------------------------------------------
# STEP 4: the reduction loop over the K dimension.
# -----------------------------------------------------------
number_of_k_chunks = tl.cdiv(k, BLOCK_SIZE_K)

for k_chunk_index in range(number_of_k_chunks):

    # The starting k-index of this particular chunk.
    tile_start_k = k_chunk_index * BLOCK_SIZE_K
    local_k_offsets = tl.arange(0, BLOCK_SIZE_K)
    global_k_indices = tile_start_k + local_k_offsets

    # ---------------------------------------------------
    # Load the (BLOCK_SIZE_M x BLOCK_SIZE_K) tile of A that this
    # chunk needs.
    # ---------------------------------------------------
    a_row_term = global_row_indices[:, None] * stride_a_row
    a_col_term = global_k_indices[None, :] * stride_a_col
    a_tile_addresses = a_ptr + a_row_term + a_col_term

    # Boundary mask: M might not be an exact multiple of
    # BLOCK_SIZE_M, and K might not be an exact multiple of
    # BLOCK_SIZE_K. This mask is True only for addresses that
    # correspond to real elements of A.
    a_row_in_bounds = global_row_indices[:, None] < m
    a_col_in_bounds = global_k_indices[None, :] < k
    a_mask = a_row_in_bounds & a_col_in_bounds

    # Masked-out elements are loaded as 0.0, so that they
    # contribute nothing to the dot-product accumulation below.
    a_tile = tl.load(a_tile_addresses, mask=a_mask, other=0.0)

    # ---------------------------------------------------
    # Load the (BLOCK_SIZE_K x BLOCK_SIZE_N) tile of B that this
    # chunk needs. Same reasoning as for A.
    # ---------------------------------------------------
    b_row_term = global_k_indices[:, None] * stride_b_row
    b_col_term = global_col_indices[None, :] * stride_b_col
    b_tile_addresses = b_ptr + b_row_term + b_col_term

    b_row_in_bounds = global_k_indices[:, None] < k
    b_col_in_bounds = global_col_indices[None, :] < n
    b_mask = b_row_in_bounds & b_col_in_bounds

    b_tile = tl.load(b_tile_addresses, mask=b_mask, other=0.0)

    # ---------------------------------------------------
    # Multiply this chunk's A-tile and B-tile together, and add
    # the result into the running accumulator.
    # ---------------------------------------------------
    partial_product = tl.dot(a_tile, b_tile)
    accumulator = accumulator + partial_product

# -----------------------------------------------------------
# STEP 5: write the finished tile back into C, in global memory.
# -----------------------------------------------------------
c_row_term = global_row_indices[:, None] * stride_c_row
c_col_term = global_col_indices[None, :] * stride_c_col
c_tile_addresses = c_ptr + c_row_term + c_col_term

c_row_in_bounds = global_row_indices[:, None] < m
c_col_in_bounds = global_col_indices[None, :] < n
c_mask = c_row_in_bounds & c_col_in_bounds

tl.store(c_tile_addresses, accumulator, mask=c_mask)
```

### Why the masking matters, verified by the test suite

Test case (33, 32, 32) -- M=33 isn't a multiple of BLOCK_SIZE_M=32 --
exercises exactly the boundary logic from the CUDA GEMM discussion: the last
row-tile only has 1 real row (row 32) and 31 padding rows. The mask
`global_row_indices[:, None] < m` correctly zeroes out the loads for those
nonexistent rows (other=0.0), so they contribute nothing to the accumulation,
and the final store's mask (`c_row_in_bounds & c_col_in_bounds`) prevents
writing to any C address beyond the real 33x32 bounds. Same logic,
independently, covers N not dividing evenly (32,33,32) and K not dividing
evenly (32,32,33); the (65,65,65) and (257,129,63) cases confirm all three
boundary conditions working together correctly, matching the "mask
everything, never branch/return early" discipline that is the safe pattern
throughout GPU kernel programming.

One honest note on `tl.dot`: it is a genuine native Triton-language primitive
-- it's the native way Triton exposes tile-level matrix multiplication
(compiling down to tensor-core instructions on hardware that supports them),
not a call into an external GEMM library. It's the same category of
primitive as `tl.load`/`tl.store`/`tl.arange` -- built into the language
itself.

## 2.2 matmul_v1.py -- autotuning + grouping + pipelining, honest benchmarks

Four real techniques layered on v0's foundation:
1. Autotuning (`@triton.autotune` across 10 tile-size/warp-count/
   pipeline-depth configs, benchmarked once per shape and cached).
2. Grouped (L2-cache-friendly) program ordering.
3. Multi-stage software-pipelined loads (`num_stages`) -- Triton's compiler
   automatically generates the same double/multi-buffering hand-traced in the
   CUDA kernel above, from a single integer hint.
4. Pointer advancing instead of recomputing addresses every loop iteration.

All 9 correctness tests pass, including 1x1 and several deliberately ragged,
non-tile-aligned shapes.

### The TF32-vs-IEEE precision correction

The first benchmark run claimed the kernel beat `torch.matmul` by up to 265%.
That number was real but misleading -- caught by being suspicious of an
extraordinary claim and checking it, rather than just reporting it. The
cause: `tl.dot`'s documented default for float32 inputs on tensor-core GPUs
is `input_precision="tf32"` -- the kernel was silently doing reduced-precision
tensor-core math the whole time, while being compared against
`torch.matmul`'s default, which is full-precision IEEE float32
(`torch.backends.cuda.matmul.allow_tf32 == False` by default since PyTorch
1.12). That's comparing two different amounts of work, not two GEMM
implementations. Fix:
1. Made the precision choice explicit in the code
   (`tl.dot(a_tile, b_tile, input_precision="tf32")`, with a comment
   explaining the tradeoff and how to switch to "ieee" if exact
   full-precision float32 is required instead).
2. Rewrote the benchmark to compare TF32-vs-TF32 (the fair comparison) and
   show full-IEEE separately, clearly labeled, for context only.

### Honest performance numbers (TF32 vs TF32, apples-to-apples)

| Size | Our kernel | cuBLAS (torch.matmul, TF32) | Our kernel's share of cuBLAS |
|---|---|---|---|
| 512^3 | 2.14 TFLOPS | 2.18 TFLOPS | 98.3% |
| 1024^3 | 15.58 TFLOPS | 17.06 TFLOPS | 91.3% |
| 2048^3 | 49.59 TFLOPS | 119.02 TFLOPS | 41.7% |
| 4096^3 | 79.33 TFLOPS | 327.51 TFLOPS | 24.2% |

Genuinely competitive with cuBLAS at small-to-medium sizes, but falls
substantially behind at large sizes. That's expected and not a failure of
this implementation -- cuBLAS represents years of NVIDIA's own
hardware-specific hand-tuning (multiple algorithm variants selected
per-shape, warp specialization, Blackwell-specific tensor-core instruction
paths) that a single, reasonably well-optimized Triton kernel realistically
isn't going to match at the largest scales.

### The more meaningful comparison: did v1's techniques actually help v0?

| Size | v0 (fixed 32x32x32 tiles, no autotuning) | v1 (autotuned + grouped + pipelined) | Speedup |
|---|---|---|---|
| 512^3 | 2.15 TFLOPS | 2.17 TFLOPS | 1.01x |
| 1024^3 | 14.72 TFLOPS | 15.57 TFLOPS | 1.06x |
| 2048^3 | 41.37 TFLOPS | 49.67 TFLOPS | 1.20x |
| 4096^3 | 45.08 TFLOPS | 79.18 TFLOPS | 1.76x |

This is the real, well-supported story: the autotuning/grouping/pipelining
techniques genuinely earn their keep, and the benefit grows with matrix size
(small matrices don't have enough work to benefit from better cache reuse or
deeper pipelining; large matrices do). A 76% speedup at 4096x4096 from
switching from one fixed tile size to an autotuned, cache-aware, pipelined
kernel is a real, honest, measured win -- just not a win over cuBLAS
specifically.

## 2.3 Why grouped program ordering exists, and why the clamping logic is load-bearing

### The problem with the naive, row-major approach

The simplest way to map a flat program ID to a 2D tile coordinate is
"row-major order":

```python
pid_m = flat_program_id // num_tiles_along_n
pid_n = flat_program_id % num_tiles_along_n
```

Say num_tiles_along_m = 6, num_tiles_along_n = 4 (6 row-tiles of A/C, 4
column-tiles of B/C). This naive scheme gives:

| flat_id | pid_m | pid_n |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 0 | 3 |
| 4 | 1 | 0 |
| 5 | 1 | 1 |

A modern GPU doesn't run these program instances one at a time -- it runs
many of them concurrently across its many SMs (streaming multiprocessors).
If a GPU can run, say, 40 of these tiles at once, under naive ordering those
40 concurrent instances would be flat_ids 0-39 -- which, with only 4 tiles
per row, spans 10 different row-tiles of A (pid_m = 0 through 9) while
cycling through all 4 column-tiles of B repeatedly. That means 40
concurrently-running SMs all simultaneously need 10 different 128-row slices
of A to be resident -- a large combined footprint competing for space in the
GPU's single, shared L2 cache. If that footprint doesn't fit, tiles of A keep
getting evicted and re-fetched from slow global memory instead of staying
cached.

### The grouped fix

```python
num_tiles_per_group = GROUP_SIZE_M * num_tiles_along_n
```

Instead of letting pid_m drift across many different values within any
window of concurrently-running tiles, grouping clusters GROUP_SIZE_M
row-tiles together and forces the ordering to exhaust all column-tiles for
that small cluster of rows before moving to the next cluster. With
GROUP_SIZE_M=3 and the same 6x4 example:

| flat_id | pid_m | pid_n |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 0 | 1 |
| 4 | 1 | 1 |
| 5 | 2 | 1 |

Now, any window of up to num_tiles_per_group = 12 consecutive flat_ids only
ever touches 3 distinct row-tiles of A ({0,1,2}), no matter how many of them
run concurrently. That's the entire point: bound the number of distinct
A-tiles any batch of concurrently-scheduled instances needs, so that small
set has a real chance of staying resident in L2 across the whole batch,
instead of a large, constantly-changing set of A-tiles that can't all fit.
The tradeoff is that pid_n now changes more rapidly (more distinct B-tiles
touched per unit time) -- you can't minimize both sides' working sets at once
with a single traversal order; GROUP_SIZE_M is the dial that decides how hard
to lean toward "small A footprint" versus "small B footprint."

### Tracing the actual code, line by line, with the ragged-edge case

Let num_tiles_along_m = 7, num_tiles_along_n = 4, GROUP_SIZE_M = 3 --
deliberately picked so 7 is NOT a multiple of 3, to exercise every line
including the clamp.

```python
num_tiles_per_group = GROUP_SIZE_M * num_tiles_along_n   # = 3 * 4 = 12
```

Each "group" conceptually spans 12 flat_ids: 3 row-tiles x all 4
column-tiles.

```python
group_id = flat_program_id // num_tiles_per_group
```

For flat_id = 0..11: group_id = 0. For flat_id = 12..23: group_id = 1. For
flat_id = 24..27: group_id = 2.

```python
first_m_tile_in_group = group_id * GROUP_SIZE_M
```

Group 0 starts at row-tile 0. Group 1 starts at row-tile 3. Group 2 starts at
row-tile 6.

```python
remaining_m_tiles_in_grid = num_tiles_along_m - first_m_tile_in_group
actual_group_size_m = tl.minimum(remaining_m_tiles_in_grid, GROUP_SIZE_M)
```

Group 0: remaining = 7-0 = 7, actual_group_size_m = min(7,3) = 3 (a full
group). Group 1: remaining = 7-3 = 4, actual_group_size_m = min(4,3) = 3
(still full). Group 2: remaining = 7-6 = 1, actual_group_size_m = min(1,3) =
1 -- only one real row-tile (6) is left; there is no row-tile 7 or 8, since
valid row-tiles only go up to 6.

```python
position_within_group = flat_program_id % num_tiles_per_group
row_offset_within_group = position_within_group % actual_group_size_m
pid_m = first_m_tile_in_group + row_offset_within_group
pid_n = position_within_group // actual_group_size_m
```

Running this exact arithmetic for all 28 flat_ids (7x4=28 total tiles)
produces a perfect, non-overlapping cover of every single (pid_m, pid_n)
pair from (0,0) through (6,3). The critical rows are the last 4:

| flat_id | pid_m | pid_n |
|---|---|---|
| 24 | 6 | 0 |
| 25 | 6 | 1 |
| 26 | 6 | 2 |
| 27 | 6 | 3 |

Because actual_group_size_m = 1 for this last, ragged group,
row_offset_within_group = position_within_group % 1 is always 0 -- pid_m
stays pinned at 6 (the only real row-tile left) for the entire group, while
pid_n = position_within_group // 1 correctly walks through all 4
column-tiles. Exactly right.

### The part that matters most: what breaks without the clamp

Re-running the identical arithmetic but using GROUP_SIZE_M (the unclamped
constant, 3) instead of actual_group_size_m everywhere -- i.e. forgetting
that the last group is smaller -- for the same last 4 flat_ids:

| flat_id | pid_m (unclamped) | pid_n | |
|---|---|---|---|
| 24 | 6 | 0 | valid |
| 25 | 7 | 0 | invalid -- row-tile 7 doesn't exist! |
| 26 | 8 | 0 | invalid -- row-tile 8 doesn't exist! |
| 27 | 6 | 1 | valid |

pid_m=7 and pid_m=8 are nonsense -- there are only 7 row-tiles (0..6). Those
two program instances compute a tile of C that's entirely beyond the
matrix's real bounds; the final store's mask (`global_row_indices < m`)
correctly discards their output, so they're not wrong, just completely
wasted launches. But look at what that leaves behind: the simulation
confirms tiles (6,2) and (6,3) are never computed by any flat_id at all --
there are only 28 program instances total, and under the broken (unclamped)
mapping, none of them land on those two coordinates. Those positions of the
output matrix C would simply be left as whatever uninitialized garbage was
already sitting in that memory -- a silent, severe correctness bug, not
merely a slowdown, and it would trigger for any M that isn't an exact
multiple of GROUP_SIZE_M * BLOCK_SIZE_M (which, in practice, is most M
values). That's exactly why those three lines about remaining_m_tiles_in_grid
/ actual_group_size_m exist -- they're not a performance nicety, they're
load-bearing for correctness on any ragged-edged matrix.

## 2.4 The "desk and warehouse" cache analogy

Imagine the GPU's L2 cache as a small desk that can only hold a handful of
papers at once. The "papers" are chunks of matrix data (a row of A, a column
of B). The actual data lives in a big warehouse far away (global memory,
DRAM) -- grabbing a paper from the warehouse is slow; if the paper is already
on your desk, grabbing it again is instant.

Every time one "tile" (one (pid_m, pid_n) pair) is processed, that tile's
A-paper and B-paper go on the desk. The desk is small, so papers you haven't
touched in a while get cleared off to make room for newer ones. The question
that actually matters for speed is: when I need a paper again, how many other
papers have passed through the desk since I last touched it? If that number
is small, it's probably still there. If it's large, it's almost certainly
gone, and the slow warehouse trip is paid again.

### What was actually measured

```
NAIVE order:    (0,0) (0,1) (0,2) (0,3) (1,0) (1,1) (1,2) (1,3) (2,0) ...
GROUPED order:  (0,0) (1,0) (2,0) (0,1) (1,1) (2,1) (0,2) (1,2) (2,2) ...
```

B-column 0 (the second number in each pair -- every flat_id where pid_n ==
0):

| | which steps use B-column 0 | gap between each reuse |
|---|---|---|
| Naive | 0, 4, 8, 12, 16, 20 | 4, 4, 4, 4, 4 -- always exactly 4 |
| Grouped | 0, 1, 2, 12, 13, 14 | 1, 1, 10, 1, 1 -- mostly 1! |

A-row 0 (the first number -- flat_ids where pid_m == 0):

| | which steps use A-row 0 | gap between each reuse |
|---|---|---|
| Naive | 0, 1, 2, 3 | 1, 1, 1 -- perfect, then never again |
| Grouped | 0, 3, 6, 9 | 3, 3, 3 |

### What this actually shows

Under naive order, B-column 0 is needed again every 4 steps -- forever,
consistently, for the entire kernel. That 4 is literally
num_tiles_along_n (the number of column-tiles) -- and in a real matmul,
num_tiles_along_n isn't 4, it's often 32 or more (e.g. a 4096-wide output
with 128-wide tiles = 32 column-tiles). So under naive order, every single
B-tile reuse has to survive a 32-tile-long gap before it's needed again -- by
then, dozens of other tiles' worth of A and B data have flowed across the
desk, and the odds that specific B-tile is still there are low. Meanwhile
A-row 0 gets perfect, immediate reuse... but only for its first 4 uses, and
then it's never touched again for the rest of the entire kernel (once you
move to row 1, you never come back to row 0).

Under grouped order, B-column 0 mostly gets reused with a gap of just 1 --
almost always the very next thing processed needs the exact same B data
again, the best possible case for caching. The tradeoff: A-row 0's reuse gap
grows from "immediate" to "every 3 steps" -- worse than naive's initial
burst, but still small and bounded, not zero.

### Why this trade is worth making

The reason this trade is a net win in practice: num_tiles_along_n (naive's
B-gap, which you can't control, it's just however wide the matrix is) is
typically much bigger than GROUP_SIZE_M (grouped's A-gap, which you do
control and pick deliberately small, like 8). You're trading away a small,
bounded degradation on A's side (gap grows from 1 to GROUP_SIZE_M, which you
chose to keep small) in exchange for a huge improvement on B's side (gap
shrinks from num_tiles_along_n, which could be 32+, down to mostly 1). Small
guaranteed cost, large likely benefit -- that's the whole trade GROUP_SIZE_M
is tuning.

## 2.5 Full worked table of all 24 tiles, and the reuse-gap derivation

Setup: 6 row-tiles (pid_m can be 0-5), 4 column-tiles (pid_n can be 0-3),
GROUP_SIZE_M=3. That's 6 x 4 = 24 total tiles, so flat_id runs from 0 to 23
-- every single one:

| flat_id | pid_m | pid_n | | flat_id | pid_m | pid_n |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | | 12 | 3 | 0 |
| 1 | 1 | 0 | | 13 | 4 | 0 |
| 2 | 2 | 0 | | 14 | 5 | 0 |
| 3 | 0 | 1 | | 15 | 3 | 1 |
| 4 | 1 | 1 | | 16 | 4 | 1 |
| 5 | 2 | 1 | | 17 | 5 | 1 |
| 6 | 0 | 2 | | 18 | 3 | 2 |
| 7 | 1 | 2 | | 19 | 4 | 2 |
| 8 | 2 | 2 | | 20 | 5 | 2 |
| 9 | 0 | 3 | | 21 | 3 | 3 |
| 10 | 1 | 3 | | 22 | 4 | 3 |
| 11 | 2 | 3 | | 23 | 5 | 3 |

### Deriving the B-column-0 gap sequence (1, 1, 10, 1, 1)

Scanning down the pid_n column, writing down the flat_id every time it equals
0:

- flat_id=0: pid_n=0 -> keep (0)
- flat_id=1: pid_n=0 -> keep (1)
- flat_id=2: pid_n=0 -> keep (2)
- flat_id=3..11: pid_n = 1, 2, or 3 -> skip all
- flat_id=12: pid_n=0 -> keep (12)
- flat_id=13: pid_n=0 -> keep (13)
- flat_id=14: pid_n=0 -> keep (14)
- flat_id=15..23: pid_n = 1, 2, or 3 -> skip all

Collected list: 0, 1, 2, 12, 13, 14. Turning this into gaps (subtracting each
number from the one after it):

| from | to | subtraction | gap |
|---|---|---|---|
| 0 | 1 | 1 - 0 | 1 |
| 1 | 2 | 2 - 1 | 1 |
| 2 | 12 | 12 - 2 | 10 |
| 12 | 13 | 13 - 12 | 1 |
| 13 | 14 | 14 - 13 | 1 |

Gap sequence: 1, 1, 10, 1, 1. That's literally "B-column 0 gets reused
immediately three times in a row (steps 0->1->2), then there's one long wait
(10 steps) before it's needed again at step 12, then it's reused immediately
again twice (12->13->14)."

### Deriving the A-row-0 gap sequence (3, 3, 3)

Scanning the pid_m column for 0:

- flat_id=0: pid_m=0 -> keep (0)
- flat_id=1,2: pid_m=1,2 -> skip
- flat_id=3: pid_m=0 -> keep (3)
- flat_id=4,5: skip
- flat_id=6: pid_m=0 -> keep (6)
- flat_id=7,8: skip
- flat_id=9: pid_m=0 -> keep (9)
- flat_id=10,11: skip
- flat_id=12 through 23: pid_m is always 3, 4, or 5 -> skip every single one,
  forever

Collected list: 0, 3, 6, 9. Gaps: 3-0=3, 6-3=3, 9-6=3 -> 3, 3, 3.

Compare to the naive ordering, where B-column 0's gap was a flat, unchanging
4 forever, and A-row 0 got used 4 times total and then never again for the
rest of the whole computation. Grouping is just a different order to visit
the same 24 tiles -- same total work, same final answer, just a schedule that
keeps reuse gaps smaller and more even on both sides instead of great-then-
zero.

## 2.6 Why flat_id=12 lands at tile (3,0), not (0,4)

### Step 1: draw the actual grid of tiles

The output matrix C is divided into a grid of tiles: 6 row-tiles (pid_m = 0
through 5 -- there is no row-tile 6 or beyond, only 6 exist) crossed with 4
column-tiles (pid_n = 0 through 3 -- there is no column-tile 4 or beyond,
only 4 exist). That's the grid that must be fully covered, cell by cell:

```
                pid_n=0   pid_n=1   pid_n=2   pid_n=3
   pid_m=0   [   ?    ] [   ?    ] [   ?    ] [   ?    ]
   pid_m=1   [   ?    ] [   ?    ] [   ?    ] [   ?    ]
   pid_m=2   [   ?    ] [   ?    ] [   ?    ] [   ?    ]
   pid_m=3   [   ?    ] [   ?    ] [   ?    ] [   ?    ]
   pid_m=4   [   ?    ] [   ?    ] [   ?    ] [   ?    ]
   pid_m=5   [   ?    ] [   ?    ] [   ?    ] [   ?    ]
```

24 cells total. Every one must get filled in by exactly one flat_id, because
every one of those 24 cells is a real tile of the output matrix that needs
computing. The only question is: in what order do flat_ids 0, 1, 2, ... 23
get assigned to these 24 cells?

### Step 2: fill it in, and watch what GROUP_SIZE_M=3 actually does

GROUP_SIZE_M=3 means: chop the 6 rows into chunks of 3 first ({0,1,2} and
{3,4,5}), and fully finish one chunk (all 4 columns, for all 3 of its rows)
before moving to the next chunk. Here's the grid, filled in with which
flat_id visits each cell:

```
                pid_n=0   pid_n=1   pid_n=2   pid_n=3
   pid_m=0   [   0    ] [   3    ] [   6    ] [   9    ]   -+
   pid_m=1   [   1    ] [   4    ] [   7    ] [  10    ]    |- CHUNK 1 (rows 0,1,2) -- flat_ids 0 to 11
   pid_m=2   [   2    ] [   5    ] [   8    ] [  11    ]   -+
   pid_m=3   [  12    ] [  15    ] [  18    ] [  21    ]   -+
   pid_m=4   [  13    ] [  16    ] [  19    ] [  22    ]    |- CHUNK 2 (rows 3,4,5) -- flat_ids 12 to 23
   pid_m=5   [  14    ] [  17    ] [  20    ] [  23    ]   -+
```

Look at CHUNK 1 by itself: it's a 3-row x 4-column block. Inside that block,
flat_ids 0,1,2 go straight down the first column (pid_m = 0,1,2 while pid_n
stays 0), then flat_ids 3,4,5 go down the second column (pid_n=1), then
6,7,8 down the third column, then 9,10,11 down the fourth. That's 3 rows x 4
columns = 12 cells, using up flat_ids 0 through 11 -- completely filling
Chunk 1, with nothing left over.

### Step 3: why flat_id=12 can't go back into Chunk 1

A guessed alternative of 12 -> (0,4) would try to put flat_id 12 into row 0,
column 4. But look at the grid above -- there is no column 4. The grid is
only 4 columns wide (pid_n only ever means 0, 1, 2, or 3, because the real
matrix B was only cut into 4 column-tiles in the first place -- that number
comes from how wide the matrix actually is, and has nothing to do with
grouping at all). Asking for pid_n=4 is like asking for the 5th seat in a
4-seat row -- it doesn't exist.

That guess also keeps pid_m cycling through {0,1,2} forever and never
reaching {3,4,5}. But rows 3, 4, and 5 are real rows of the matrix that
still need to be computed -- if pid_m never advances past 2, those three rows
of C would never get a single tile assigned to them, and (exactly like the
clamping bug from section 2.3) a third of the output matrix would simply
never be computed. That's not allowed.

So the only place left for flat_id=12 to go is: the first still-empty cell,
which is the top-left corner of Chunk 2 -- row 3, column 0. That's it. It's
not a special rule, it's just "Chunk 1 is full, so go fill Chunk 2, starting
from its own top-left corner."

### Step 4: mechanically verify this with the actual formula

```python
GROUP_SIZE_M = 3
num_tiles_along_n = 4
num_tiles_per_group = GROUP_SIZE_M * num_tiles_along_n     # 3 * 4 = 12

flat_id = 12

group_id = flat_id // num_tiles_per_group                  # 12 // 12 = 1
first_m_tile_in_group = group_id * GROUP_SIZE_M             # 1 * 3 = 3
position_within_group = flat_id % num_tiles_per_group       # 12 % 12 = 0
row_offset_within_group = position_within_group % GROUP_SIZE_M   # 0 % 3 = 0
pid_m = first_m_tile_in_group + row_offset_within_group      # 3 + 0 = 3
pid_n = position_within_group // GROUP_SIZE_M                # 0 // 3 = 0

# Result: pid_m=3, pid_n=0
```

The crucial line is `group_id = flat_id // num_tiles_per_group`. Since
num_tiles_per_group = 12 (the size of one full chunk), as soon as flat_id
reaches 12, group_id ticks over from 0 to 1 -- meaning "I'm now in the second
chunk of rows." And `first_m_tile_in_group = group_id * GROUP_SIZE_M = 1*3 =
3` is exactly what shifts us from rows {0,1,2} to rows {3,4,5}.

Same mechanical plug-in for flat_id=13 and 14:

```python
flat_id = 13
group_id = 13 // 12 = 1
first_m_tile_in_group = 1 * 3 = 3
position_within_group = 13 % 12 = 1
row_offset_within_group = 1 % 3 = 1
pid_m = 3 + 1 = 4
pid_n = 1 // 3 = 0
# Result: pid_m=4, pid_n=0

flat_id = 14
group_id = 14 // 12 = 1
first_m_tile_in_group = 3
position_within_group = 14 % 12 = 2
row_offset_within_group = 2 % 3 = 2
pid_m = 3 + 2 = 5
pid_n = 2 // 3 = 0
# Result: pid_m=5, pid_n=0
```

And flat_id=15 is where pid_n finally advances to 1 within Chunk 2:

```python
flat_id = 15
group_id = 15 // 12 = 1
first_m_tile_in_group = 3
position_within_group = 15 % 12 = 3
row_offset_within_group = 3 % 3 = 0
pid_m = 3 + 0 = 3
pid_n = 3 // 3 = 1
# Result: pid_m=3, pid_n=1
```

This exactly reproduces (12,3,0), (13,4,0), (14,5,0), (15,3,1) from the
table -- not by coincidence, but because that's literally what the
arithmetic computes. The grid picture in Step 2 is the "why it has to look
like this" (only 6 rows, only 4 columns, fill every cell exactly once); the
formula in Step 4 is the "mechanical recipe that produces that exact
picture."

## 2.7 The [:, None] / [None, :] broadcasting trick, from first principles

This is just a way to turn two separate lists of numbers into a full grid of
sums, without writing a loop. Take a tiny, unrelated example:
`column_vector = [10, 20, 30]` and `row_vector = [1, 2]`.

```python
reshaped_column = column_vector[:, None]   # [10,20,30] becomes a 3x1 column:
#   [[10]
#    [20]
#    [30]]

reshaped_row = row_vector[None, :]         # [1,2] becomes a 1x2 row:
#   [[1 2]]

result = reshaped_column + reshaped_row    # adding them broadcasts to a 3x2 grid:
#   [[11 12]
#    [21 22]
#    [31 32]]
```

The rule: `result[i][j] = column_vector[i] + row_vector[j]`. So `[:, None]`
makes something "tall" (one value per row), `[None, :]` makes something
"wide" (one value per column), and adding them fills in every (row, col)
combination automatically. Every line in the matmul kernel that uses
`[:, None]` or `[None, :]` is doing exactly this trick -- nothing fancier.

## 2.8 The modulo "wrap" trick for M and N

Say the real matrix A only has m = 10 rows (valid row indices: 0 through 9 --
there is no row 10 or beyond). Say BLOCK_SIZE_M = 8, and we're looking at the
second row-tile (pid_m = 1), so:

```python
tile_start_row = pid_m * BLOCK_SIZE_M            # 1 * 8 = 8
global_row_indices = tile_start_row + [0,1,2,3,4,5,6,7]
#                   = [8, 9, 10, 11, 12, 13, 14, 15]
```

Rows 8 and 9 are real. Rows 10 through 15 do not exist -- this tile runs off
the bottom edge of the matrix. Now apply the wrap:

```python
global_row_indices_wrapped = global_row_indices % m     # % 10
#  8 % 10 = 8     (unchanged, already valid)
#  9 % 10 = 9     (unchanged, already valid)
# 10 % 10 = 0     (wrapped!)
# 11 % 10 = 1     (wrapped!)
# 12 % 10 = 2     (wrapped!)
# 13 % 10 = 3     (wrapped!)
# 14 % 10 = 4     (wrapped!)
# 15 % 10 = 5     (wrapped!)
#
# Result: [8, 9, 0, 1, 2, 3, 4, 5]
```

So lanes 2 through 7 (which were originally pointed at the nonexistent rows
10-15) now point at rows 0-5 instead -- real, valid rows, just not the rows
this tile was actually supposed to compute. That's deliberate "borrowing" of
valid-but-wrong data, which is safe only because the final store at the end
of the kernel throws those lanes' results away anyway, using a mask based on
the original, unwrapped global_row_indices.

## 2.9 Turning wrapped indices into actual memory addresses

Say A has K=4 columns, so (row-major) stride_a_row = 4 (jump 4 elements to
move down one row) and stride_a_col = 1 (jump 1 element to move one column
over). And say BLOCK_SIZE_K=4 (so local_k_offsets = [0,1,2,3]).

```python
a_row_term = global_row_indices_wrapped[:, None] * stride_a_row
```

Take the wrapped rows [8, 9, 0, 1, 2, 3, 4, 5], reshape into a column, and
multiply each by 4:

```
[[32]    <- 8*4
 [36]    <- 9*4
 [ 0]    <- 0*4
 [ 4]    <- 1*4
 [ 8]    <- 2*4
 [12]    <- 3*4
 [16]    <- 4*4
 [20]]   <- 5*4
```

This is "which row does each lane start at, measured in flat memory offset
units."

```python
a_col_term = local_k_offsets[None, :] * stride_a_col
```

Take [0,1,2,3], reshape into a row, multiply by 1 (no change):
[[0, 1, 2, 3]]. This is "how far to walk within that row, for each of the 4
k-values we need this chunk."

```python
a_tile_ptrs = a_ptr + a_row_term + a_col_term
```

Adding the column (a_row_term) to the row (a_col_term) broadcasts into a
full 8x4 grid (8 lanes x 4 k-values each):

```
[[32 33 34 35]      <- lane 0 (real row 8):  A[8][0], A[8][1], A[8][2], A[8][3]
 [36 37 38 39]      <- lane 1 (real row 9):  A[9][0], A[9][1], A[9][2], A[9][3]
 [ 0  1  2  3]      <- lane 2 (wrapped to row 0): A[0][0..3]  <- "wrong" row, discarded later
 [ 4  5  6  7]      <- lane 3 (wrapped to row 1): A[1][0..3]  <- discarded later
 [ 8  9 10 11]      <- lane 4 (wrapped to row 2): A[2][0..3]  <- discarded later
 [12 13 14 15]      <- lane 5 (wrapped to row 3): A[3][0..3]  <- discarded later
 [16 17 18 19]      <- lane 6 (wrapped to row 4): A[4][0..3]  <- discarded later
 [20 21 22 23]]     <- lane 7 (wrapped to row 5): A[5][0..3]  <- discarded later
```

Adding a_ptr (the actual base memory address A lives at) to every one of
these 32 numbers gives the 32 real memory addresses to read from -- that's
what `tl.load(a_tile_ptrs, ...)` will fetch a few lines later.

The B lines are the exact same recipe, just transposed:

```python
b_row_term = local_k_offsets[:, None] * stride_b_row
b_col_term = global_col_indices_wrapped[None, :] * stride_b_col
b_tile_ptrs = b_ptr + b_row_term + b_col_term
```

Here it's the k-values that get reshaped into a column ([:, None]) and the
wrapped column-indices that get reshaped into a row ([None, :]) -- because
for B, k plays the role of "row" and pid_n's columns play the role of
"column" (B is K x N, so its row-tiles come from the K dimension, not M).
Same wrap-then-broadcast mechanism, just applied to B's own shape instead of
A's.

Summary of the whole block: Piece 2 (the wrap) fixes up any row/column index
that runs past the matrix edge so it points somewhere real instead of off the
end of memory; Piece 1's broadcasting trick then turns those row-indices and
k-indices into a complete grid of byte-offsets; adding the base pointer turns
that grid of offsets into the actual addresses the very first tl.load will
read from.

## 2.10 The K-reduction loop: chunking, masking, tl.dot, accumulation, pointer advancing

Imagine K = 10, BLOCK_SIZE_K = 4. The real reduction dimension only has 10
values (k = 0..9), but it's processed in chunks of 4 at a time.

```python
number_of_k_chunks = tl.cdiv(k, BLOCK_SIZE_K)     # ceil(10/4) = 3
```

Three chunks are needed: a chunk only holds 4 k-values, and 2 chunks x 4 = 8
isn't enough to cover all 10, so a 3rd is needed, even though that 3rd one
will be mostly empty.

local_k_offsets = [0, 1, 2, 3] is always the same (it's just "the 4 slots
within one chunk"). What changes every iteration is how many of those 4 slots
actually correspond to real data:

| k_chunk_index | k_values_remaining = k - k_chunk_index*BLOCK_SIZE_K | this chunk's k-values would be | mask (local_k_offsets < k_values_remaining) |
|---|---|---|---|
| 0 | 10 - 0*4 = 10 | 0,1,2,3 | [T,T,T,T] -- all 4 real |
| 1 | 10 - 1*4 = 6 | 4,5,6,7 | [T,T,T,T] -- all 4 still real (7 < 10) |
| 2 | 10 - 2*4 = 2 | 8,9,10,11 | [T,T,F,F] -- only slots 0,1 real (k=8,9 exist; k=10,11 don't) |

Only the last chunk (iteration 2, where k_values_remaining=2 < 4) actually
masks anything off. `tl.load(..., mask=a_k_mask, other=0.0)` then loads real
data for slots 0-1 and silently substitutes 0.0 for slots 2-3 -- those
nonexistent k=10,11 values contribute nothing when multiplied in.

Why is a_k_mask shaped [None, :] (a row) but b_k_mask is shaped [:, None] (a
column)? Because of the broadcasting rule above: a_tile_ptrs has shape
(BLOCK_SIZE_M, BLOCK_SIZE_K) -- k is the column axis of A's tile, so the mask
needs to be a row to broadcast correctly down every M-row. b_tile_ptrs has
shape (BLOCK_SIZE_K, BLOCK_SIZE_N) -- k is the row axis of B's tile instead,
so its mask needs to be a column. Same k-values, opposite orientation,
because A and B store the K dimension on opposite sides of their tiles.

### tl.dot and accumulation, a fully concrete trace

Pretend BLOCK_SIZE_M=2, BLOCK_SIZE_N=2. Here's chunk 0's actual loaded tiles
and what tl.dot does with them:

```
A_chunk0 = [[1, 2, 3, 4],       B_chunk0 = [[1, 0],
            [5, 6, 7, 8]]                   [0, 1],
                                             [1, 1],
                                             [2, 0]]

tl.dot(A_chunk0, B_chunk0) = A_chunk0 @ B_chunk0 = [[12,  5],
                                                      [28, 13]]
```

(check row 0: 1*1 + 2*0 + 3*1 + 4*2 = 1+0+3+8 = 12 -- matches)

accumulator starts at [[0,0],[0,0]], and after chunk 0: accumulator =
[[12,5],[28,13]].

Chunk 1 loads a different 2x4 slice of A and 4x2 slice of B (the k=4..7
values):

```
A_chunk1 = [[9,10,11,12],      B_chunk1 = [[1,1],
            [13,14,15,16]]                 [0,2],
                                            [1,0],
                                            [0,1]]

tl.dot(A_chunk1, B_chunk1) = [[20, 41],
                               [28, 57]]
```

accumulator = accumulator + partial_product = [[12+20, 5+41], [28+28,
13+57]] = [[32, 46], [56, 70]].

This is the whole point of the loop: each chunk only computes a partial sum
using 4 of the 10 k-values; accumulator += keeps a running total across all 3
chunks, so by the time the loop ends, every one of the 10 real k-values has
contributed exactly once to the final sum -- which is the actual mathematical
definition of matrix multiplication (C[i][j] = sum over all k of
A[i][k]*B[k][j]), just computed in pieces instead of all at once.

### Why "advance the pointer" gives the exact same answer as recomputing it

Using stride_a_row=10 (K=10), stride_a_col=1, rows [0,1]:

```
A offsets for chunk 0 (computed from scratch):
[[ 0  1  2  3]
 [10 11 12 13]]
```

Advance by BLOCK_SIZE_K * stride_a_col = 4 * 1 = 4 (just add 4 to every
number):

```
[[ 4  5  6  7]
 [14 15 16 17]]
```

Compare to recomputing chunk 1's addresses completely from scratch
(k_start=4):

```
[[ 4  5  6  7]
 [14 15 16 17]]
```

Identical. That's the whole justification for
`a_tile_ptrs = a_tile_ptrs + BLOCK_SIZE_K * stride_a_col`: since every
chunk's starting row is exactly the same (only the k-offset shifts forward
by BLOCK_SIZE_K each time), you can get the next chunk's addresses by just
adding a constant to the previous ones, instead of redoing the
row-multiplication (`global_row_indices_wrapped[:, None] * stride_a_row`)
every single iteration -- cheaper, same result.
`b_tile_ptrs += BLOCK_SIZE_K * stride_b_row` is the mirror-image move for B
(advancing its row-axis, since K is B's row dimension).

## 2.11 The final store mask for C, fully worked

Real matrix M=10 rows, N=6 columns. Writing the tile at pid_m=1, pid_n=1 with
BLOCK_SIZE_M=8, BLOCK_SIZE_N=4 -- deliberately chosen so this tile overflows
both the bottom edge (rows) and the right edge (columns) of the real matrix
at once.

```python
global_row_indices = pid_m*BLOCK_SIZE_M + [0..7] = [8, 9, 10, 11, 12, 13, 14, 15]
global_col_indices = pid_n*BLOCK_SIZE_N + [0..3] = [4, 5, 6, 7]
```

Only rows 8,9 are real (M=10 means valid rows are 0-9). Only columns 4,5 are
real (N=6 means valid columns are 0-5).

### Building the mask, piece by piece

```python
c_row_in_bounds = global_row_indices[:, None] < m
```

Reshape the 8 row-indices into a column, compare each to m=10:

```
[[ True]    <- row 8:  8 < 10, real
 [ True]    <- row 9:  9 < 10, real
 [False]    <- row 10: 10 < 10 is False, doesn't exist
 [False]    <- row 11
 [False]    <- row 12
 [False]    <- row 13
 [False]    <- row 14
 [False]]   <- row 15
```

```python
c_col_in_bounds = global_col_indices[None, :] < n
```

Reshape the 4 col-indices into a row, compare each to n=6:

```
[[True, True, False, False]]
     ^col4  ^col5  ^col6(doesn't exist)  ^col7(doesn't exist)
```

```python
c_mask = c_row_in_bounds & c_col_in_bounds
```

This is the exact same broadcasting trick, just using & (AND) instead of +.
An (8,1) column AND'd with a (1,4) row broadcasts into a full (8,4) grid,
where cell[i][j] = row_ok[i] AND col_ok[j]. The actual result:

```
[[ True,  True, False, False],   <- row 8:  cols 4,5 ok; cols 6,7 not
 [ True,  True, False, False],   <- row 9:  cols 4,5 ok; cols 6,7 not
 [False, False, False, False],   <- row 10: entire row is fake, all False regardless of column
 [False, False, False, False],   <- row 11: entire row is fake
 [False, False, False, False],   <- row 12: entire row is fake
 [False, False, False, False],   <- row 13: entire row is fake
 [False, False, False, False],   <- row 14: entire row is fake
 [False, False, False, False]]   <- row 15: entire row is fake
```

Only 4 cells out of 32 are True -- exactly the top-left 2x2 corner, which
corresponds to the only 4 positions that are simultaneously a real row and a
real column: (8,4), (8,5), (9,4), (9,5). Every other cell in this 8x4 tile is
fake in at least one dimension (fake row, fake column, or both), so it gets
masked off. `tl.store(c_tile_ptrs, accumulator, mask=c_mask)` then only
actually writes those 4 real values into C; the other 28 computed (but
meaningless) accumulator values are simply dropped on the floor, never
touching memory.

NOTE on the row labels above: this matrix has exactly 8 rows, corresponding
to global_row_indices = [8,9,10,11,12,13,14,15] -- a run of 8 consecutive
integers with no gaps (see Section 2.14 for a full derivation of why rows 11
and 12 cannot be skipped). If you ever see a transcription of this matrix
that appears to jump from row 10 straight to row 13, that is a copy/paste
artifact (two visually-identical "all False" rows accidentally dropped), not
an actual gap in the computation.

## 2.12 Why the store mask must use the UNWRAPPED indices

Recall from Section 2.8: `global_row_indices_wrapped = global_row_indices %
m` was computed to safely redirect the loads for A away from nonexistent
rows. The STEP 9 store code deliberately does NOT reuse that wrapped version
-- it uses the original, unwrapped global_row_indices. Here's what happens if
you (incorrectly) used the wrapped version for this mask instead:

```python
global_row_indices_wrapped = [8, 9, 0, 1, 2, 3, 4, 5]    # from the wrap step
wrong_row_in_bounds = global_row_indices_wrapped < 10
#                   = [True, True, True, True, True, True, True, True]
```

Every single value comes out True -- because after wrapping, every value
genuinely is less than 10 (that's the entire point of wrapping: to make it a
valid row index!). If this wrong mask were used, the kernel would think rows
10 through 15 are perfectly fine to write -- and it would write the borrowed
data (e.g. lane 2's accumulator, which was actually computed using A's real
row 0, intended to represent the nonexistent row 10) into C's actual row 0
(since the address computation would also have to use the wrapped row index
to match) -- silently overwriting C's correct row-0 answer with garbage.
That's a severe, silent correctness bug, not just wasted work.

So the design relies on a careful split: wrapped indices are only ever used
to decide WHERE IT'S SAFE TO READ FROM (avoiding illegal memory access during
loads for A and B), while unwrapped indices are what decide WHETHER A RESULT
IS REAL ENOUGH TO BE WRITTEN (the store mask for C). Mixing these up in
either direction breaks correctness -- this is exactly the mechanism that
makes the claim "wrapping is safe because the final mask discards the bad
lanes" actually true.

## 2.13 Why K can never be safely wrapped, even though M and N can

Straight from the code comment in matmul_v1.py (around the start of the
K-reduction loop):

```python
# Unlike the M and N axes, the K axis CANNOT be safely wrapped
# with modulo arithmetic -- K is the dimension being SUMMED OVER,
# and there is no later "masked store" to discard a wrong
# contribution. If a wrapped k-index accidentally re-read a REAL
# k-slice a second time, that slice's contribution would be added
# into the accumulator TWICE, silently corrupting the sum. So the
# K axis keeps an explicit boundary mask on every load, specific
# to how many real k-values remain in the current chunk.
```

M and N are "pick one final answer, throw away the rest" dimensions -- C's
mask is the safety net. K is a "sum every contribution exactly once"
dimension -- there is no safety net for K, because every chunk's partial
product gets permanently added into the accumulator with += before the loop
even finishes. So K always keeps its own live mask (a_k_mask / b_k_mask), in
both v0 and v1 -- that part never changes between the two kernel versions.

## 2.14 Clarifying question: "why does it jump from row 10 to row 13?"

A reasonable point of confusion arose when looking at the Section 2.11 mask
table: it appeared (in a transcription) to skip directly from "row 10" to
"row 13," as if rows 11 and 12 had been intentionally omitted. The short
answer: there is no real skip. Rows 11 and 12 absolutely exist; they went
missing between re-typing / copy-pasting six visually-identical
`[False, False, False, False]` lines, not from any actual gap in the
computation.

### Step 1: where do the 8 row-numbers even come from?

```python
BLOCK_SIZE_M = 8
pid_m = 1          # this is the SECOND row-tile (tiles are numbered starting at 0)

local_offsets = tl.arange(0, BLOCK_SIZE_M)
# local_offsets = [0, 1, 2, 3, 4, 5, 6, 7]   <- "my position within this tile", always 0..7

tile_start_row = pid_m * BLOCK_SIZE_M
# tile_start_row = 1 * 8 = 8    <- this tile starts at global row 8

global_row_indices = tile_start_row + local_offsets
# global_row_indices = 8 + [0,1,2,3,4,5,6,7]
#                     = [8, 9, 10, 11, 12, 13, 14, 15]
```

Walk through the addition one element at a time:

| array position (index into local_offsets) | local_offsets[i] | 8 + local_offsets[i] | global_row_indices[i] |
|---|---|---|---|
| 0 | 0 | 8+0 | 8 |
| 1 | 1 | 8+1 | 9 |
| 2 | 2 | 8+2 | 10 |
| 3 | 3 | 8+3 | 11 |
| 4 | 4 | 8+4 | 12 |
| 5 | 5 | 8+5 | 13 |
| 6 | 6 | 8+6 | 14 |
| 7 | 7 | 8+7 | 15 |

That's local_offsets = [0,1,2,3,4,5,6,7] -- 8 consecutive integers, zero
gaps, by construction of tl.arange -- plus a constant 8. Adding the same
constant to a run of consecutive integers can never produce a gap. So
global_row_indices is mathematically guaranteed to be
[8,9,10,11,12,13,14,15] -- eight numbers, every single one in a row, no way
for 11 or 12 to vanish.

### Step 2: the mask, row by row, nothing skipped

```python
m = 10   # the REAL matrix only has 10 rows: valid row numbers are 0 through 9
c_row_in_bounds = global_row_indices[:, None] < m
```

| array position | global_row_indices[i] | is it < 10? | result |
|---|---|---|---|
| 0 | 8 | 8 < 10 -> yes | True |
| 1 | 9 | 9 < 10 -> yes | True |
| 2 | 10 | 10 < 10 -> no | False |
| 3 | 11 | 11 < 10 -> no | False |
| 4 | 12 | 12 < 10 -> no | False |
| 5 | 13 | 13 < 10 -> no | False |
| 6 | 14 | 14 < 10 -> no | False |
| 7 | 15 | 15 < 10 -> no | False |

### Runnable Python check (actually executed to confirm this)

```python
import numpy as np

pid_m, BLOCK_SIZE_M, m = 1, 8, 10
local_offsets = np.arange(BLOCK_SIZE_M)
global_row_indices = pid_m * BLOCK_SIZE_M + local_offsets
print("global_row_indices:", global_row_indices.tolist())

mask = global_row_indices < m
for r, ok in zip(global_row_indices, mask):
    print(f"row {r}: in_bounds={ok}")
```

Actual output when run:

```
global_row_indices: [8, 9, 10, 11, 12, 13, 14, 15]
row 8:  in_bounds=True
row 9:  in_bounds=True
row 10: in_bounds=False
row 11: in_bounds=False
row 12: in_bounds=False
row 13: in_bounds=False
row 14: in_bounds=False
row 15: in_bounds=False
```

Bottom line: rows 11 and 12 are there, consecutive, exactly where expected,
both False for the same boring reason as 10, 13, 14, 15 (row >= m=10, matrix
doesn't have that row). No skip, no special-casing of 11/12, just plain
`pid_m*BLOCK_SIZE_M + [0..7]` giving 8 numbers in a row.

## 2.15 Follow-up deep dive: where, in the real source files, is the "wrapped" code for A/B and the "unwrapped" code for C?

This section grounds the whole wrapped/unwrapped design rule in the actual
files on disk:
`/home/jupyter/jeff_workspace/triton/leetgpu/matrixMultiplication/matmul_v0.py`
and
`/home/jupyter/jeff_workspace/triton/leetgpu/matrixMultiplication/matmul_v1.py`.

The key discovery: ONLY matmul_v1.py uses wrapped indices at all.
matmul_v0.py never wraps anything -- it solves the same ragged-edge problem a
completely different way (masking every load directly, every iteration).
Seeing both side by side is the fastest way to understand what "wrapped" vs
"unwrapped" actually buys you.

### Where's the wrapped-index code for A and B?

matmul_v1.py, lines 298-335 (STEP 4, STEP 5, STEP 6):

```python
# STEP 4 (line 298-299): build the UNWRAPPED indices first
global_row_indices = tile_start_row + local_row_offsets      # e.g. [8,9,10,11,12,13,14,15]
global_col_indices = tile_start_col + local_col_offsets

# STEP 5 (line 321-322): wrap them -- THIS is "the wrapped indices"
global_row_indices_wrapped = global_row_indices % m           # e.g. [8,9,0,1,2,3,4,5]  (if m=10)
global_col_indices_wrapped = global_col_indices % n

# STEP 6 (line 329-335): build A's and B's memory addresses using the WRAPPED version
a_row_term = global_row_indices_wrapped[:, None] * stride_a_row    # <- wrapped!
a_col_term = local_k_offsets[None, :] * stride_a_col
a_tile_ptrs = a_ptr + a_row_term + a_col_term

b_row_term = local_k_offsets[:, None] * stride_b_row
b_col_term = global_col_indices_wrapped[None, :] * stride_b_col    # <- wrapped!
b_tile_ptrs = b_ptr + b_row_term + b_col_term
```

`a_tile_ptrs` and `b_tile_ptrs` -- the addresses that `tl.load` reads A and B
from, every single iteration of the K-loop -- are built from
`global_row_indices_wrapped` / `global_col_indices_wrapped`, never the plain
unwrapped versions.

### Where's the unwrapped-index code for C?

matmul_v1.py, lines 408-416 (STEP 9):

```python
c_row_term = global_row_indices[:, None] * stride_c_row    # <- UNWRAPPED (plain global_row_indices)
c_col_term = global_col_indices[None, :] * stride_c_col    # <- UNWRAPPED
c_tile_ptrs = c_ptr + c_row_term + c_col_term

c_row_in_bounds = global_row_indices[:, None] < m          # <- UNWRAPPED, compared to real m
c_col_in_bounds = global_col_indices[None, :] < n          # <- UNWRAPPED, compared to real n
c_mask = c_row_in_bounds & c_col_in_bounds

tl.store(c_tile_ptrs, accumulator, mask=c_mask)
```

`global_row_indices` and `global_col_indices` here are the exact same
variables from STEP 4 -- never reassigned, never wrapped. The wrapped
versions (the `_wrapped` suffix variables) only ever get used for A and B's
addresses; C's store always uses the originals. That's not an accident of
naming -- it's the entire mechanism that makes wrapping safe.

### What's actually different, implementation-wise, between v0 and v1?

matmul_v0.py solves the exact same "ragged edge" problem a completely
different way -- it never wraps anything at all. Compare the A-tile load in
each file:

v0.py (no wrapping -- masks every load directly), around lines 168-192:

```python
a_row_term = global_row_indices[:, None] * stride_a_row      # plain, unwrapped
a_col_term = global_k_indices[None, :] * stride_a_col
a_tile_addresses = a_ptr + a_row_term + a_col_term

a_row_in_bounds = global_row_indices[:, None] < m             # mask computed EVERY loop iteration
a_col_in_bounds = global_k_indices[None, :] < k
a_mask = a_row_in_bounds & a_col_in_bounds

a_tile = tl.load(a_tile_addresses, mask=a_mask, other=0.0)
```

v1.py (wrapping -- mask computed once, outside the loop), lines 298-335 plus
362-363:

```python
# done ONCE, before the K-loop even starts:
global_row_indices_wrapped = global_row_indices % m
a_row_term = global_row_indices_wrapped[:, None] * stride_a_row
a_tile_ptrs = a_ptr + a_row_term + a_col_term

# inside the K-loop, the ONLY mask needed is for K (can't be wrapped -- see Section 2.13):
a_k_mask = local_k_offsets[None, :] < k_values_remaining
a_tile = tl.load(a_tile_ptrs, mask=a_k_mask, other=0.0)
```

| | v0 (unwrapped, masked) | v1 (wrapped) |
|---|---|---|
| Row/col mask for A/B? | Yes -- a_row_in_bounds & a_col_in_bounds, recomputed every K-loop iteration | None -- wrapping makes every address always point at a real row/col, so there's nothing to mask |
| K-dimension mask? | Yes, every iteration (a_col_in_bounds checks k) | Yes, every iteration (a_k_mask) -- K is never wrapped, in either file |
| Can an invalid address ever be computed for A/B? | Yes (e.g. row 15 when m=10) -- but masked-out before the actual memory read happens | No -- row 15 gets turned into row 5 before any address math happens, so the address is always real |
| Where does "this lane is fake" get decided? | At A/B's own load, every iteration | Only once, at the very end, at C's store |

Concretely, with m=10, BLOCK_SIZE_M=8, pid_m=1 (rows 8-15, only 8,9 real):

```python
# v0's way: compute the real (unwrapped) row address, then mask the LOAD
row = 15                      # a genuinely out-of-range row
address = a_ptr + 15*stride_a_row + ...     # computed, but...
mask = (15 < 10) = False      # ...this lane's load is masked off
tl.load(address, mask=False, other=0.0)     # -> returns 0.0, memory at `address` is never touched

# v1's way: wrap the row FIRST, so the address itself is always valid
row_wrapped = 15 % 10 = 5                   # a REAL row (just the wrong one)
address = a_ptr + 5*stride_a_row + ...      # points at real data: A's actual row 5
tl.load(address, mask=a_k_mask, ...)        # no row/col mask needed at all -- loads A[5][...] for real
```

Both are "safe" (neither one corrupts memory) -- the difference is when and
how often the M/N boundary check happens: v0 pays for it on every single A/B
load, every K-loop iteration; v1 pays for it exactly once (the modulo), then
trades it for free correctness later because row 5's wrong data for this lane
gets quietly discarded at the finish line.

Why can't B do the same trick for K, saving even more masking? See Section
2.13: M and N are "pick one final answer, throw away the rest" dimensions --
C's mask is the safety net. K is a "sum every contribution exactly once"
dimension -- there is no safety net for K, so K always keeps its own live
mask, in both v0 and v1.

---

# Appendix A — File locations referenced in this document

- CUDA SGEMM kernel: discussed conceptually (source pasted into the
  conversation, not a tracked file in this repo).
- Triton matmul v0:
  `/home/jupyter/jeff_workspace/triton/leetgpu/matrixMultiplication/matmul_v0.py`
- Triton matmul v1:
  `/home/jupyter/jeff_workspace/triton/leetgpu/matrixMultiplication/matmul_v1.py`
- This document:
  `/home/jupyter/jeff_workspace/triton/leetgpu/matrixMultiplication/matmul_v0.md`

# Appendix B — Quick-reference glossary

- **RAW hazard (Read-After-Write)**: a read of a location must happen after
  the write that produced the value it needs, or it sees stale data.
- **WAR hazard (Write-After-Read)**: a write to a location must wait until an
  earlier read has finished consuming the old value, or the reader sees
  data that arrived too early.
- **Bank conflict**: when multiple threads in a warp simultaneously access
  different addresses that map to the same shared-memory bank, forcing
  those accesses to serialize instead of running in parallel.
- **Swizzle**: a deterministic, reversible permutation (often via XOR)
  applied to a memory index specifically to avoid bank conflicts.
- **Double buffering / software pipelining**: using two (or more) buffers so
  that loading the "next" chunk of data can overlap with computing on the
  "current" chunk, hiding memory latency behind useful work.
- **Masking (Triton)**: passing a boolean tensor into `tl.load`/`tl.store`
  so that out-of-bounds lanes simply skip the memory operation (returning a
  fallback value on load, or not writing at all on store) instead of
  branching/exiting -- this is what makes Triton kernels immune to the
  classic CUDA "early return before a barrier causes a deadlock" bug.
- **Wrapping (modulo trick)**: replacing an out-of-range index with
  `index % real_size` so that a computed memory address always points at
  real, allocated data, deferring the "is this result actually valid"
  decision to a later masked store.
- **TF32**: a reduced-precision floating point format (about 10 mantissa
  bits instead of float32's 23) used by NVIDIA tensor cores for much higher
  throughput at a small accuracy cost; `tl.dot` defaults to it for float32
  inputs on supported hardware.

---

# PART 3 -- Follow-up: The CUDA Shared-Memory Loading Loop, Fully Unpacked

## 3.1 The code being explained

```c
for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {   // i = 0, 32, 64, 96 (4 iterations)
    int r = row0 + i + tAy;
    sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
}
```

## 3.2 What is this loop even trying to do?

Matrix A is huge and lives in slow global memory. One CTA (block) is about
to do a bunch of math using a small 128-row x 8-column slice of A, over and
over. Repeatedly re-reading that slice from slow global memory would be
wasteful -- so first, all 256 threads in the block cooperate to copy that
slice into `sA`, a small, fast on-chip scratchpad (shared memory). This loop
is that copy operation. Nothing is computed here yet -- it is purely "move
data from far away to nearby."

## 3.3 Where do tAx and tAy come from?

Recall `THREAD_A_LAYOUT_X = 8`, `THREAD_A_LAYOUT_Y = 32`. The 256 threads of
this block get reshaped (purely for the purpose of this copy) into an
8-wide x 32-tall grid:

```c
int tAx = tidx & (THREAD_A_LAYOUT_X - 1);   // tidx % 8   -> ranges 0..7
int tAy = tidx / THREAD_A_LAYOUT_X;         // tidx / 8   -> ranges 0..31
```

| tidx | tAx | tAy |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 7 | 7 | 0 |
| 8 | 0 | 1 |
| 15 | 7 | 1 |
| 255 | 7 | 31 |

Think of tAx as "which of the 8 columns of this K-chunk am I responsible
for" and tAy as "which of 32 row-groups am I in."

## 3.4 Why does the loop run exactly 4 times?

The slice being copied is 128 rows x 8 columns = 1024 elements. There are
256 threads. 1024 / 256 = 4 -- so every thread must personally copy 4
elements, no more, no less. That is the whole reason for
`for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y)`: it runs 128/32 = 4
times (i = 0, 32, 64, 96), each time handling a different 32-row "stripe,"
because tAy only ever covers 32 distinct values on its own.

Picture it as 4 horizontal stripes of the 128-row tile, each 32 rows tall:

```
rows   0.. 31   <- stripe i=0    (tAy = 0..31 directly gives the row within this stripe)
rows  32.. 63   <- stripe i=32
rows  64.. 95   <- stripe i=64
rows  96..127   <- stripe i=96
```

## 3.5 Tracing r -- which real row of A does a thread touch?

```c
int r = row0 + i + tAy;
```

Say this block's tile starts at row0 = 0 (the very first row-tile). Take
thread tidx = 5 -> tAx = 5, tAy = 0:

| loop iter | i | r = row0 + i + tAy | meaning |
|---|---|---|---|
| 1 | 0 | 0+0+0 = 0 | row 0 of A |
| 2 | 32 | 0+32+0 = 32 | row 32 of A |
| 3 | 64 | 0+64+0 = 64 | row 64 of A |
| 4 | 96 | 0+96+0 = 96 | row 96 of A |

Notice tAx (=5, fixed for this thread) never changes -- only r changes
across the 4 iterations. So thread 5 copies column 5 of 4 different rows
(0, 32, 64, 96). That is the whole "4 loads per thread" story, concretely.

## 3.6 The write target, peeled apart

```c
sA[0][tAx][(i + tAy) ^ ((tAx) << 2)]
```

Three things are smashed together here. Peel them apart one at a time,
ignoring the XOR for a moment (pretend it is just `sA[0][tAx][i + tAy]`):

- `sA[0]` -> buffer 0 (double-buffering, not relevant here)
- `[tAx]` -> this is the K-index (which of the 8 columns)
- `[i + tAy]` -> this is the M-index (which of the 128 rows, within this tile)

So without the XOR, thread 5 (tAx=5) would write its 4 loaded values to
`sA[0][5][0]`, `sA[0][5][32]`, `sA[0][5][64]`, `sA[0][5][96]`.

The array is indexed as `sA[buffer][K-index][M-index]` -- K comes first
(outer), M comes second (inner). That is backwards compared to A's own
natural layout (`A[row][col]`, i.e. M outer, K inner). That reversal is
deliberate, and it is what "transposed" means here.

## 3.7 Why store it backwards? A concrete side-by-side comparison

Forget the real 128x8 size for a second -- use a tiny toy 4x2 slice of A so
every number fits on screen:

```
A (4 rows, 2 columns, this K-chunk only):
A[0] = [10, 11]
A[1] = [20, 21]
A[2] = [30, 31]
A[3] = [40, 41]
```

Option A -- "natural" layout, same shape as A itself (`sA_natural[m][k]`, M
outer / K inner -- i.e. what you would write if you did NOT transpose):

```
sA_natural[0] = [10, 11]
sA_natural[1] = [20, 21]
sA_natural[2] = [30, 31]
sA_natural[3] = [40, 41]
```
In memory, this is one flat array: `[10, 11, 20, 21, 30, 31, 40, 41]`.

Option B -- the kernel's actual "transposed" layout (`sA[k][m]`, K outer / M
inner):

```
sA_transposed[0] = [10, 20, 30, 40]   <- fix k=0, sweep ALL rows (m=0..3)
sA_transposed[1] = [11, 21, 31, 41]   <- fix k=1, sweep ALL rows (m=0..3)
```
In memory, this flat array is: `[10, 20, 30, 40, 11, 21, 31, 41]`.

## 3.8 Why Option B wins -- "contiguous, vectorizable chunk"

Jump ahead to the compute phase. For a fixed k (say k=0), the kernel needs
to grab a chunk of many different m-values to multiply against B -- that is
literally what a matrix multiply does: `C[m] += A[m][k] * B[k][n]` for every
m in the tile, one k at a time.

- Using Option B (what the real code does): "give me all m-values for k=0"
  = `sA_transposed[0] = [10, 20, 30, 40]`. These 4 numbers are sitting right
  next to each other in memory -- 16 contiguous bytes (4 floats). That is
  exactly the shape a single FLOAT4 vectorized load wants: one instruction,
  4 values, done.

- Using Option A (the natural, non-transposed layout): "give me all
  m-values for k=0" means reading `sA_natural[0][0]`, `sA_natural[1][0]`,
  `sA_natural[2][0]`, `sA_natural[3][0]` -- in the flat array
  `[10, 11, 20, 21, 30, 31, 40, 41]`, these are at positions 0, 2, 4, 6 --
  every other element, not contiguous. A GPU cannot vectorize a strided
  access like that into one instruction; it would need 4 separate scalar
  loads instead of 1 vectorized one.

So the transpose is a trade: pay a slightly "weird-looking" write pattern
once, during this loading loop, in exchange for a fast, contiguous,
vectorizable read pattern that gets used repeatedly afterward (once per
k-value, for every one of the many compute iterations that follow). This
payoff shows up directly in the kernel's later compute-phase code:

```c
FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
```

Fix tk (the k-index, outer), sweep r (the m-index, inner, via the swizzle)
-- reading 4 contiguous m-values in one FLOAT4 instruction. That line only
works efficiently because sA was stored transposed back in the loading
loop.

## 3.9 The XOR -- a quick, honest mention

The `^ (tAx << 2)` part is a separate concern layered on top of the
transpose: it is purely about avoiding shared-memory bank conflicts
(scrambling which of 32 memory banks each thread lands in, so threads do
not collide). It does NOT affect which row/column gets loaded -- only
exactly which shared-memory slot it is parked in. For example, with
i+tAy = 12:

```
tAx=0: 12 ^ (0<<2) = 12 ^ 0  = 12
tAx=1: 12 ^ (1<<2) = 12 ^ 4  = 8
tAx=2: 12 ^ (2<<2) = 12 ^ 8  = 4
tAx=3: 12 ^ (3<<2) = 12 ^ 12 = 0
```

Notice the write locations (12, 8, 4, 0) come out reversed/jumbled, not in
simple ascending order -- that scrambling is the whole point, spreading
nearby threads' writes across different memory banks instead of clustering
them. It is a real, separate optimization from the transpose.

---

# PART 4 -- Follow-up: Why Transposing Matrix A Actually Helps, Proven With a Complete Worked Example

## 4.1 The example matrices

```
A (4 rows x 2 columns):          B (2 rows x 3 columns):
a00  a01                          b00  b01  b02
a10  a11                          b10  b11  b12
a20  a21
a30  a31
```

C = A @ B (4 rows x 3 columns), given directly as:

```
C[0][0] = a00*b00 + a01*b10      C[0][1] = a00*b01 + a01*b11      C[0][2] = a00*b02 + a01*b12
C[1][0] = a10*b00 + a11*b10      C[1][1] = a10*b01 + a11*b11      C[1][2] = a10*b02 + a11*b12
C[2][0] = a20*b00 + a21*b10      C[2][1] = a20*b01 + a21*b11      C[2][2] = a20*b02 + a21*b12
C[3][0] = a30*b00 + a31*b10      C[3][1] = a30*b01 + a31*b11      C[3][2] = a30*b02 + a31*b12
```

## 4.2 Your formula is already secretly "sum over k of outer products"

Look closely: every single entry above is a sum of exactly 2 terms -- one
term involving `a_*0` and `b0*` (column 0 of A times row 0 of B), and one
term involving `a_*1` and `b1*` (column 1 of A times row 1 of B). That is
because K=2 here (A has 2 columns, B has 2 rows). So the whole C can be
split into two separate pieces and added:

```
"k=0 piece" = everything with a_*0 and b0*
"k=1 piece" = everything with a_*1 and b1*
C = (k=0 piece) + (k=1 piece)
```

This regrouping is not an approximation -- it is the exact same arithmetic,
just grouped differently (addition does not care what order you do it in).
This "one k at a time, do the whole tile, add it to a running total" is
literally how the real CUDA kernel's K-loop works.

## 4.3 Verify it with real numbers, both ways

Plug in actual numbers:

```
A =  1  2        B =  1  0  1
     3  4             0  1  1
     5  6
     7  8
```
(a00=1, a01=2, a10=3, a11=4, a20=5, a21=6, a30=7, a31=8,
 b00=1, b01=0, b02=1, b10=0, b11=1, b12=1)

Direct formula, plugging in numbers:

```
C[0][0] = 1*1 + 2*0 = 1      C[0][1] = 1*0 + 2*1 = 2      C[0][2] = 1*1 + 2*1 = 3
C[1][0] = 3*1 + 4*0 = 3      C[1][1] = 3*0 + 4*1 = 4      C[1][2] = 3*1 + 4*1 = 7
C[2][0] = 5*1 + 6*0 = 5      C[2][1] = 5*0 + 6*1 = 6      C[2][2] = 5*1 + 6*1 = 11
C[3][0] = 7*1 + 8*0 = 7      C[3][1] = 7*0 + 8*1 = 8      C[3][2] = 7*1 + 8*1 = 15
```
```
C =  1   2   3
     3   4   7
     5   6  11
     7   8  15
```

Now the "outer product per k" way. For k=0, take column 0 of A (`[1,3,5,7]`)
and row 0 of B (`[1,0,1]`), and multiply every pair:

```
k=0 contribution:
[1*1, 1*0, 1*1]     [1, 0, 1]
[3*1, 3*0, 3*1]  =  [3, 0, 3]
[5*1, 5*0, 5*1]     [5, 0, 5]
[7*1, 7*0, 7*1]     [7, 0, 7]
```
For k=1, take column 1 of A (`[2,4,6,8]`) and row 1 of B (`[0,1,1]`):

```
k=1 contribution:
[2*0, 2*1, 2*1]     [0, 2, 2]
[4*0, 4*1, 4*1]  =  [0, 4, 4]
[6*0, 6*1, 6*1]     [0, 6, 6]
[8*0, 8*1, 8*1]     [0, 8, 8]
```
Add them together:

```
[1+0, 0+2, 1+2]     [1,  2,  3]
[3+0, 0+4, 3+4]  =  [3,  4,  7]
[5+0, 0+6, 5+6]     [5,  6, 11]
[7+0, 0+8, 7+8]     [7,  8, 15]
```

Identical to the direct computation. Now that both methods are confirmed to
agree, look at what each method needs to fetch from memory.

## 4.4 The outer-product method needs "all of column k" in one shot

Notice what the k=0 step needed: `[1, 3, 5, 7]` -- that is a00, a10, a20,
a30 -- i.e. every row's value in column 0, all at once. That is exactly the
"all M-values for one fixed k" phrase from Part 3. The GPU thread doing
this step wants to grab those 4 numbers together, ideally in a single
memory instruction instead of 4 separate ones.

## 4.5 Where do those 4 numbers actually live in memory?

A is stored row-major (that is just how the problem defines it) -- meaning
its real, physical, flat layout in memory is:

```
flat A in memory:  [ a00, a01, a10, a11, a20, a21, a30, a31 ]
                  = [  1,   2,   3,   4,   5,   6,   7,   8 ]
                     pos0 pos1 pos2 pos3 pos4 pos5 pos6 pos7
```

Now find where a00, a10, a20, a30 (the column-0 values needed for k=0)
actually sit:

```
a00 = position 0
a10 = position 2
a20 = position 4
a30 = position 6
```

They are at positions 0, 2, 4, 6 -- every OTHER slot, not next to each
other. This is called a "strided" access (stride = 2, since K=2 here). A
GPU load instruction that fetches a contiguous chunk of memory is fast and
cheap (one instruction); grabbing 4 values that are scattered 2-apart is
NOT something a single vectorized instruction can do -- it needs 4 separate
individual reads instead.

## 4.6 The transpose trick -- rearrange A once, so column-0 becomes contiguous

A = 

a00, a01

a10, a11

a20, a21

a30, a31

B = 

b00, b01, b02

b10, b11, b12

C = 

a00*b00 + a01*b10,    a00*b01 + a01*b11,    a00*b02 + a01*b12,

a10*b00 + a11*b10,    a10*b01 + a11*b11,    a10*b02 + a11*b12,

a20*b00 + a21*b10,    a20*b01 + a21*b11,    a20*b02 + a21*b12,

a30*b00 + a31*b10,    a30*b01 + a31*b11,    a30*b02 + a31*b12,


Here is the fix the kernel applies: when copying A into shared memory,
instead of keeping the same [row][col] order, flip it to [col][row] order
-- write all of column 0 together, then all of column 1 together:

```
sA_transposed (what actually gets written to shared memory):

  "k=0 block" (all 4 rows' column-0 values, grouped together):
      [ a00, a10, a20, a30 ]  =  [ 1, 3, 5, 7 ]    <- positions 0,1,2,3

  "k=1 block" (all 4 rows' column-1 values, grouped together):
      [ a01, a11, a21, a31 ]  =  [ 2, 4, 6, 8 ]    <- positions 4,5,6,7

flat sA_transposed in memory:  [ 1, 3, 5, 7, 2, 4, 6, 8 ]
```

Now look again at where the k=0 values live: positions 0, 1, 2, 3 --
perfectly contiguous, back to back. A single vectorized load (FLOAT4 -- one
instruction that reads 4 floats at once) can now grab `[1, 3, 5, 7]` in one
shot. Same for k=1's values at positions 4,5,6,7.

## 4.7 Side-by-side, so the difference is undeniable

| | Natural layout (A's own row-major order) | Transposed layout (what the kernel does) |
|---|---|---|
| Flat memory | `[1, 2, 3, 4, 5, 6, 7, 8]` | `[1, 3, 5, 7, 2, 4, 6, 8]` |
| Where is "column 0" (needed for k=0)? | positions 0, 2, 4, 6 -- scattered | positions 0, 1, 2, 3 -- contiguous |
| How many instructions to fetch it? | 4 separate scalar reads | 1 vectorized FLOAT4 read |
| Where is "column 1" (needed for k=1)? | positions 1, 3, 5, 7 -- scattered | positions 4, 5, 6, 7 -- contiguous |

Nothing about the actual numbers changes -- it is the same 8 values either
way, and the final answer C comes out identical regardless (proven in
Section 4.3). The only thing that changes is how cheaply the GPU can fetch
"all rows for one fixed k" when it is time to do that k's outer-product
step -- 1 instruction instead of 4.

## 4.8 Why this matters far more at real-world scale

In the 4-row toy example, "scattered reads" only costs 4 instructions
instead of 1 -- a 4x difference, once. In the real kernel, BLOCK_M = 128,
not 4. Every single k-value's outer-product step needs "all 128 rows' worth
of data for that k" -- natural layout would need 128 scattered individual
reads; the transpose turns that into roughly 32 vectorized FLOAT4 reads
(128 divided by 4 values-per-instruction). And this is not a one-time cost
-- it happens once per k-value, for every one of the 8 k-values in a chunk,
for every chunk of K, for every tile of the output matrix, across the
entire matmul. Paying the "weird write order" cost exactly once (during the
loading loop from Part 3) buys a cheap, vectorized read on every one of
those many, many repeated accesses afterward -- that multiplication of
savings across the whole kernel run is the entire economic argument for
transposing.

---

# PART 5 -- Porting the Kernel to Real, Compiled CUDA (matmul_v0.cu)

## 5.1 The file and the problem statement

Everything up to this point in the document was conceptual / Python / Triton.
This part switches to an actual compiled CUDA program: `matmul_v0.cu`, built
and run for real on an NVIDIA GB200 (compute capability sm_100), computing
C = A @ B for row-major A (M x K), B (K x N), C (M x N).

The file is self-contained: the GPU kernel itself (supplied, unchanged),
a plain C++ CPU reference used ONLY to verify GPU results, a set of test
cases from 1x1x1 up to 4096x4096x4096, and a performance harness with
configurable warm-up/timed run counts.

## 5.2 Three real bugs found and fixed in the supplied solve() wrapper

The kernel's own compute logic was used exactly as supplied. The host-side
`solve()` wrapper that launches it had three separate, real bugs:

1. `extern C void solve(...)` is not valid C++ -- the linkage specifier
   must be a string literal. Fixed to `extern "C" void solve(...)`.

2. The kernel launch passed arguments in the order (A, B, C, M, K, N), but
   the kernel's own parameter list is (A, B, C, M, N, K). That swap means
   the kernel's "N" parameter would silently receive the real K, and vice
   versa. Fixed to launch with (M, N, K) in the kernel's actual declared
   order.

3. The grid's x-dimension (which tiles across C's COLUMNS, i.e. the N
   dimension, since `col0 = blockIdx.x * BLOCK_N`) was sized using
   `(K + BLOCK_N - 1) / BLOCK_N` instead of `(N + BLOCK_N - 1) / BLOCK_N`.
   For any non-square matrix where N != K, this either leaves part of C's
   columns never computed (if N > K) or launches wasted blocks (if N < K).
   Fixed to use N.

Bugs 2 and 3 compound: fixing only one without the other still produces
wrong results for non-square, non-K==N matrices. Both were fixed together,
and every non-square test case (e.g. 257x129x63) specifically exercises
this fix.

```c
// Fixed solve():
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                       int K) {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

## 5.3 The CPU reference baseline, and why its loop order matters

Used only to verify the GPU's answer, never to do the "real" computation:

```c++
void cpu_matmul_reference(const std::vector<float> &A,
                           const std::vector<float> &B, std::vector<float> &C,
                           int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < M; ++i) {
        float *c_row = &C[static_cast<size_t>(i) * N];
        for (int k = 0; k < K; ++k) {
            float a_val = A[static_cast<size_t>(i) * K + k];
            const float *b_row = &B[static_cast<size_t>(k) * N];
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}
```

Loop order is deliberately i-k-j, not the more obvious i-j-k: with row-major
storage, this keeps the innermost loop walking B and C one full contiguous
ROW at a time, which is dramatically more cache-friendly on a CPU than the
strided access an i-j-k ordering would cause. This is the exact same
contiguity idea from Parts 3 and 4 of this document, just applied to a CPU
cache line instead of a GPU shared-memory bank.

## 5.4 Test cases and the perf-testing harness design

13 test cases: tiny (1x1x1), smaller-than-one-tile (16/31), exactly one
tile (128x128x128), ragged edges (+1 row, +1 column, +1 reduction step),
rectangular (257x129x63), and large square sizes (512, 1000x777x333, 1024,
2048, 4096).

```c++
constexpr int NUM_WARMUP_RUNS = 5;
constexpr int NUM_TIMED_RUNS = 20;
```

For each test case: `NUM_WARMUP_RUNS` untimed calls to `solve()` first (to
warm up clocks/caches), then `NUM_TIMED_RUNS` timed calls, each bracketed
by `cudaEvent_t` start/stop events (the standard, accurate way to time a
CUDA kernel, since it measures real GPU execution time rather than
host-side wall-clock noise):

```c++
for (int i = 0; i < NUM_TIMED_RUNS; ++i) {
    cudaEventRecord(start);
    solve(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    latencies_ms.push_back(elapsed_ms);
}
```

From those latencies: min, max, mean, median, and TFLOPS (`2*M*N*K` total
floating point ops, divided by mean latency).

## 5.5 First run: 10/10 correctness pass, plus three "SKIP" entries

The first version of the test-case list set `verify_correctness = false`
for the three largest sizes (1024/2048/4096 cubed), based on an assumption
(not yet measured) that a naive CPU reference would be "impractically
slow" at that scale:

```
M          N        K        Correct?   Mean(ms)     TFLOPS     MaxErr
1          1        1        PASS         2.3535       0.00   0.000000
...
512        512      512      PASS         2.4415       0.11   0.000000
1000       777      333      PASS         2.4146       0.21   0.000000
1024       1024     1024     SKIP         2.5243       0.85        n/a
2048       2048     2048     SKIP         2.8124       6.11        n/a
4096       4096     4096     SKIP         8.0507      17.07        n/a
----------------------------------------------------------------------
Correctness: 10 / 10 checked test cases passed
```

## 5.6 A suspicious finding: latency looks flat (~2.3ms) at every small size

Looking at the full table, test cases from 1x1x1 up through roughly 1024
cubed all show nearly identical ~2.3-2.5ms latency, no matter how tiny the
matrix. That is NOT what you'd expect if the number were dominated by real
compute time -- a 1x1x1 matmul should take microseconds, not milliseconds.

Rather than assume this was noise or a bug, it was isolated and measured
directly with a standalone, minimal CUDA program containing an EMPTY
kernel (does nothing at all):

```c++
__global__ void empty_kernel() {}

int main() {
    for (int i = 0; i < 5; ++i) empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 10; ++i) {
        cudaEventRecord(start);
        empty_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("empty kernel + sync call %d: %.4f ms\n", i, ms);
    }
}
```

Result, run for real on this box:

```
empty kernel + sync call 0: 2.3541 ms
empty kernel + sync call 1: 2.3459 ms
empty kernel + sync call 2: 2.3552 ms
...
empty kernel + sync call 9: 2.3422 ms
```

An EMPTY kernel costs the identical ~2.35ms. This confirms the flat
latency across small/medium test cases is a genuine, fixed per-call
overhead from `cudaDeviceSynchronize()` on this particular shared or
virtualized GPU instance -- not a bug in the test harness, and not a
property of the matmul kernel itself. Since `solve()` is deliberately
synchronous (it calls `cudaDeviceSynchronize()` internally, matching the
expected "call it, result is ready when it returns" API contract), this
floor is real and unavoidable when benchmarking `solve()` as given. Actual
compute time only clearly exceeds this floor once the matrix is large
enough (around 2048/4096).

## 5.7 Correcting a wrong assumption: the three "SKIP" sizes didn't need to be skipped

The assumption behind marking 1024/2048/4096 as `verify_correctness = false`
("the CPU reference would be impractically slow") was never actually
measured -- it was asserted. When directly questioned about it, it was
checked for real: the exact `cpu_matmul_reference()` function was compiled
standalone with `-O3` and timed:

```
M=N=K=1000: CPU reference took 0.022 seconds (23.22 GFLOPS)   <- this size DID run in the real test
M=N=K=1024: CPU reference took 0.092 seconds (23.40 GFLOPS)   <- was skipped for "being too slow"
M=N=K=2048: CPU reference took 0.705 seconds (24.36 GFLOPS)
M=N=K=4096: CPU reference took 9.422 seconds (14.59 GFLOPS)
```

GCC's `-O3` auto-vectorizes the cache-friendly i-k-j inner loop well enough
that even the largest case (4096 cubed, ~137 GFLOP of CPU work) finishes in
under 10 seconds. There was no real reason to skip CPU verification at any
of these sizes. The fix: flip `verify_correctness` to `true` for all three,
recompile, rerun:

```
M          N        K        Correct?   Mean(ms)     TFLOPS     MaxErr
1024       1024     1024     PASS         2.5261       0.85   0.000000
2048       2048     2048     PASS         2.8179       6.10   0.000000
4096       4096     4096     PASS         7.9228      17.35   0.000000
----------------------------------------------------------------------
Correctness: 13 / 13 checked test cases passed
Average throughput across all 13 test cases: 1.89 TFLOPS
ALL CORRECTNESS TESTS PASSED!
```

All 13/13 test cases now pass, with exactly `0.000000` max absolute error
at every single size. The lesson: "this would probably be too slow" is a
claim that needs to be measured, not just stated, before it's allowed to
change what gets tested -- the same discipline used earlier in this
document for the TF32-vs-IEEE benchmarking correction.

---

# PART 6 -- Follow-up: Can You Transpose B Instead of A?

A = 

a00, a01

a10, a11

a20, a21

a30, a31


B = 

b00, b01, b02

b10, b11, b12


B' = 

b00, b10,

b01, b11, 

b02, b12


C = 

a00*b00 + a01*b10,    a00*b01 + a01*b11,    a00*b02 + a01*b12,

a10*b00 + a11*b10,    a10*b01 + a11*b11,    a10*b02 + a11*b12,

a20*b00 + a21*b10,    a20*b01 + a21*b11,    a20*b02 + a21*b12,

a30*b00 + a31*b10,    a30*b01 + a31*b11,    a30*b02 + a31*b12,


## 6.1 Recap: why matmul_v0.cu transposes A and not B

From Parts 3-4: the compute loop needs, for one fixed k at a time, a
contiguous COLUMN of A (all m values) and a contiguous ROW of B (all n
values). A's own row-major storage makes a ROW contiguous (wrong shape) --
so A gets transposed in shared memory to fix that. B's own row-major
storage ALREADY makes a ROW contiguous (right shape already) -- so B is
left alone.

## 6.2 Building matmul_v1.cu: A untouched, B transposed

A direct follow-up question: is the opposite possible? `matmul_v1.cu` was
built to test this directly -- shared-memory shapes flipped:

```c
// matmul_v0.cu:
__shared__ float sA[2][BLOCK_K][BLOCK_M];  // A TRANSPOSED: k outer, m inner
__shared__ float sB[2][BLOCK_K][BLOCK_N];  // B natural:    k outer, n inner

// matmul_v1.cu:
__shared__ float sA[2][BLOCK_M][BLOCK_K];  // A natural:    m outer, k inner
__shared__ float sB[2][BLOCK_N][BLOCK_K];  // B TRANSPOSED: n outer, k inner
```

Crucially, matmul_v1.cu kept the exact same outer-product compute loop as
v0 -- only the shared-memory LAYOUT changed, not how it gets consumed.
Since neither array's layout now matched what that compute loop needed,
both A's and B's per-k register-staging reads had to become scalar
(4 separate reads) instead of a single vectorized FLOAT4 read:

```c
// matmul_v1.cu's register-staging step, fixed tk, 4 scalar reads for A:
int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
tCsA[tk & 1][(tm << 2) + 0] = sA[bufferId][r + 0][tk];
tCsA[tk & 1][(tm << 2) + 1] = sA[bufferId][r + 1][tk];
tCsA[tk & 1][(tm << 2) + 2] = sA[bufferId][r + 2][tk];
tCsA[tk & 1][(tm << 2) + 3] = sA[bufferId][r + 3][tk];
// ...and the same 4-scalar-read pattern for B, which used to be one FLOAT4 in v0.
```

## 6.3 Results: correctness perfect, performance notably worse

All 13/13 correctness tests passed, `0.000000` max abs error -- confirming
the math is identical regardless of which operand is transposed (a
transpose is just a storage reshuffle; correctness survives as long as
read-side indices match). Performance, measured on the same GB200, same
test harness:

```
           v0 (A transposed)   v1 (A natural, B transposed)
  1024^3   0.85 TFLOPS         0.82 TFLOPS   (~4% slower, dominated by the ~2.3ms fixed floor)
  2048^3   6.10 TFLOPS         5.13 TFLOPS   (~16% slower)
  4096^3   17.35 TFLOPS        8.19 TFLOPS   (~53% slower -- roughly half the throughput)
```

## 6.4 A reader's sharp challenge, using their own worked example

A natural objection: "If B is transposed, B's data IS contiguous to read.
And A, left alone, is ALSO contiguous to read (its own natural row-major
layout). So why would performance get worse for both?" This was
illustrated with a concrete example:

```
A = a00, a01        B = b00, b01, b02        B' = b00, b10
    a10, a11             b10, b11, b12            b01, b11
    a20, a21                                       b02, b12
    a30, a31

C[0][0] = a00*b00 + a01*b10   (a dot product of A's row 0 and B''s row 0)
C[0][1] = a00*b01 + a01*b11   (a dot product of A's row 0 and B''s row 1)
...
```

The proposed algorithm: for each output cell, read a whole ROW of A
(contiguous, natural layout) and a whole ROW of B-transposed (contiguous,
since B' stores each of B's original columns as a row), and compute a dot
product. This reasoning turns out to be completely correct -- but it
describes a DIFFERENT algorithm than the one matmul_v0.cu/v1.cu use.

## 6.5 Resolving it: two different algorithms, each with its own correct transpose

**Style 1 -- outer-product / rank-1-update (what v0 and v1 actually use):**
walk k one value at a time. For each k, grab ALL of A's column k and ALL of
B's row k, multiply every pair, and add the result into EVERY cell of C at
once.

```
k=0: A's column 0 = [a00, a10, a20, a30]     B's row 0 = [b00, b01, b02]
     outer product -> added to ALL of C:
     C[0][0]+=a00*b00  C[0][1]+=a00*b01  C[0][2]+=a00*b02
     C[1][0]+=a10*b00  C[1][1]+=a10*b01  C[1][2]+=a10*b02
     ...
k=1: A's column 1 = [a01, a11, a21, a31]     B's row 1 = [b10, b11, b12]
     outer product -> added to ALL of C (accumulating on top of k=0):
     ...
```

What this style needs, memory-wise: A's natural row-major storage makes a
ROW contiguous (fixed m, varying k) -- but this style needs a COLUMN
(fixed k, varying m) -- scattered, needs transposing. B's natural storage
makes a ROW contiguous (fixed k, varying n) -- exactly what this style
needs already -- no transpose needed.
**Conclusion for this style: A transposed, B natural -- exactly v0.**

**Style 2 -- dot-product / inner-product (the reader's proposal):** walk
each output cell (m, n) one at a time. For each, grab ALL of A's row m and
ALL of B-transposed's row n, multiply elementwise, and reduce (sum) to one
number.

```
C[0][0] = dot(A row 0, B' row 0) = dot([a00,a01], [b00,b10]) = a00*b00 + a01*b10
C[0][1] = dot(A row 0, B' row 1) = dot([a00,a01], [b01,b11]) = a00*b01 + a01*b11
...
```

What this style needs, memory-wise: A's row-major storage ALREADY makes a
row contiguous -- no transpose needed. B-transposed's row IS literally what
transposing B produces -- contiguous. **Conclusion for this style: A
natural, B transposed -- exactly the reader's proposal, and the reasoning
is entirely correct for THIS style.**

**So where did matmul_v1.cu actually go wrong?** It used Style 2's storage
layout (A natural, B transposed) but kept Style 1's (outer-product)
compute loop, completely unchanged from v0. That mismatch -- not
"transposing B" by itself -- is what forced both operands' register-
staging reads to become scalar instead of vectorized. v1 built
dot-product-shaped storage and fed it into outer-product-shaped
consumption: the worst of both worlds, not a fair test of the dot-product
algorithm itself.

## 6.6 So why do real GEMM kernels use outer-product style at all, if dot-product style can also be made contiguous?

If both styles CAN achieve full contiguity (with their own correctly
matched transpose choice), why does essentially every high-performance GEMM
kernel (cuBLAS, CUTLASS, and this hand-tuned kernel) use outer-product
style? The predicted reason, before it was measured: REGISTER PRESSURE, not
contiguity.

For this kernel's 8x8 per-thread output tile:

- Outer-product style: at any one moment, a thread only needs the CURRENT
  k's one A-value-per-m and one B-value-per-n (8+8=16 "scratch" registers,
  double-buffered = 32) plus the 64 running accumulators (`acc[8][8]`) that
  persist across the whole K reduction. Estimated ~96 registers per thread.
- Dot-product style, done properly with the same 8x8 tile: to get the same
  reuse, a thread needs to cache a WHOLE chunk of A's row for all 8 of its
  m's, AND a whole chunk of B-transposed's row for all 8 of its n's,
  simultaneously -- `8 x BLOCK_K` (64) registers for A, another 64 for B,
  plus the same 64 persistent accumulators. Estimated ~192 registers per
  thread -- roughly double.

More registers per thread means fewer threads resident on the GPU at once
(lower occupancy), which can hurt the GPU's ability to hide memory latency
by switching between warps. This is a reasoned, sound prediction -- but
this whole document's standard has been "measure it, don't just assert
it," so the next part builds the real, faithful dot-product kernel and
checks.

---

# PART 7 -- Building the Faithful Dot-Product Kernel (matmul_v2.cu)

## 7.1 Design: cache whole rows via FLOAT4, then do real dot products

Unlike matmul_v1.cu, matmul_v2.cu's compute loop actually matches its
storage layout. Shared memory is identical to v1's (A natural, B
transposed), but now, because k is the INNERMOST axis of both shared
arrays, reading "all BLOCK_K values for one fixed m" (a full row of A) or
"all BLOCK_K values for one fixed n" (a full row of B-transposed) is
genuinely contiguous -- BLOCK_K=8 floats = exactly two back-to-back FLOAT4
reads:

```c
// Stage this thread's 8 rows of A and 8 rows of B' for the current chunk,
// fully vectorized:
for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
    int m = tCy + tm * THREAD_C_LAYOUT_Y;
    FLOAT4(regA[tm][0]) = FLOAT4(sA[bufferId][m][0]);
    FLOAT4(regA[tm][4]) = FLOAT4(sA[bufferId][m][4]);
}
for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
    int n = tCx + tn * THREAD_C_LAYOUT_X;
    FLOAT4(regB[tn][0]) = FLOAT4(sB[bufferId][n][0]);
    FLOAT4(regB[tn][4]) = FLOAT4(sB[bufferId][n][4]);
}

// The actual dot-product accumulation: for every (m, n) this thread owns,
// reduce across this chunk's BLOCK_K values and add into the running total.
for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
        float partial = 0.0f;
        for (int kk = 0; kk < BLOCK_K; kk++) {
            partial += regA[tm][kk] * regB[tn][kk];
        }
        acc[tm][tn] += partial;
    }
}
```

Both operands now get real vectorized loads -- exactly the contiguity the
reader's own reasoning in Part 6.4 predicted, implemented faithfully this
time instead of mismatched.

## 7.2 The real register count, measured with nvcc -Xptxas -v

All three kernel variants were compiled with `-Xptxas -v`, which reports
the actual register allocation the compiler produced (not a hand count):

```
=== v0 (outer-product, A transposed) ===
ptxas info    : Used 126 registers, used 1 barriers, 16384 bytes smem

=== v1 (mismatched: dot-product layout + outer-product compute) ===
ptxas info    : Used 168 registers, used 1 barriers, 16384 bytes smem

=== v2 (faithful dot-product style) ===
ptxas info    : Used 168 registers, used 1 barriers, 16384 bytes smem
```

The directional register-pressure prediction from Part 6.6 held: dot-
product style does cost more registers (126 -> 168, about 33% more). The
naive hand estimate (96 vs 192, roughly double) was too aggressive --
the real compiler reuses temporaries more cleverly than a raw sum of array
sizes suggests. Also notable: v1 and v2 have IDENTICAL register counts.
That makes sense in hindsight -- both hold the exact same SET of values (8
cached rows x 8 elements for A and for B); only the INSTRUCTIONS used to
load them differ (v1: 4 scalar loads per group; v2: 1 FLOAT4 load per
group covering the same 4 values).

## 7.3 Correctness: 13/13 pass, with a small, explainable wrinkle

```
M          N        K        Correct?   Mean(ms)     TFLOPS     MaxErr
1          1        1        PASS         2.3534       0.00   0.000000
31         31       31       PASS         2.3574       0.00   0.000001
128        128      128      PASS         2.3778       0.00   0.000009
512        512      512      PASS         2.4515       0.11   0.000034
1024       1024     1024     PASS         2.5513       0.84   0.000072
2048       2048     2048     PASS         3.0975       5.55   0.000168
4096       4096     4096     PASS         6.8721      20.00   0.000359
----------------------------------------------------------------------
Correctness: 13 / 13 checked test cases passed
```

Unlike v0/v1 (exactly `0.000000` error at every size), v2 shows small
nonzero errors that grow with size -- still far under the test's tolerance
(`1e-2 + 1e-3*K`), so every case still correctly PASSes. This is expected,
normal floating-point behavior, not a bug: v2 sums each output cell's
BLOCK_K products in a genuinely different order (a private per-cell
reduction) than v0/v1's incremental cross-chunk accumulation, and
floating-point addition is not associative -- different valid summation
orders produce slightly different rounding.

## 7.4 Performance: the prediction from Part 6.6 does NOT fully hold up

```
           v0 (outer-product)   v1 (mismatched)   v2 (faithful dot-product)
  1024^3   0.85 TFLOPS          0.82 TFLOPS       0.84 TFLOPS
  2048^3   6.10 TFLOPS          5.13 TFLOPS        5.55 TFLOPS
  4096^3   17.35 TFLOPS         8.19 TFLOPS       20.00 TFLOPS
```

**v2 actually beats v0 at the largest size (20.00 vs 17.35 TFLOPS),
despite using more registers.** The register-pressure theory from Part 6.6
was directionally correct about the REGISTER COUNT (confirmed: 168 > 126),
but it turned out NOT to be the deciding factor for real measured
throughput once the kernel was implemented faithfully with proper
vectorization on both sides. What v1's earlier bad numbers were actually
showing all along was the cost of its load/compute MISMATCH, not anything
inherent to dot-product-style accumulation being slower. Done properly,
with real FLOAT4 loads on both operands exactly as the reader's reasoning
predicted in Part 6.4, dot-product style is competitive with -- and at the
largest tested size, faster than -- the textbook outer-product style this
particular hand-tuned kernel uses.

One additional honest observation, not fully explained: v2's 4096 cubed
latencies are noticeably noisier than v0/v1's (min 5.4ms, max 12.6ms,
versus v0's tight 7.9-8.3ms band). This is plausibly related to the higher
register count affecting occupancy/scheduling consistency, but that
connection was not independently verified -- it is reported as an
observation, not a proven causal claim.

---

# PART 8 -- matmul_v0_nof4.cu: The FLOAT4 Ablation

## 8.1 What this file changes

matmul_v0.cu uses FLOAT4 in exactly two places, both in the register-
staging step inside the kernel's hottest loop:

```c
FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
```

Each is replaced with 4 separate scalar reads of the exact same values, in
the exact same order -- computing the swizzled base address once, then
reading `base+0, base+1, base+2, base+3` individually:

```c
int base = r ^ (tk << 2);
tCsA[tk & 1][(tm << 2) + 0] = sA[bufferId][tk][base + 0];
tCsA[tk & 1][(tm << 2) + 1] = sA[bufferId][tk][base + 1];
tCsA[tk & 1][(tm << 2) + 2] = sA[bufferId][tk][base + 2];
tCsA[tk & 1][(tm << 2) + 3] = sA[bufferId][tk][base + 3];
```

Nothing else changes: same transpose, same swizzle, same accumulation
order, same bug fixes in `solve()`, same 13 test cases, same perf harness.

## 8.2 Register count and correctness

`nvcc -Xptxas -v` register count: **126, identical to matmul_v0.cu.** This
makes sense -- the same set of values needs to be held in registers
either way; only the load/store INSTRUCTION WIDTH changes, not how many
distinct values exist.

Correctness: 13/13 PASS, exactly `0.000000` max abs error at every size --
bit-for-bit identical to matmul_v0.cu, exactly as expected, since no
arithmetic or accumulation order changed, only instruction width.

## 8.3 A real investigation into confusing performance numbers

The first run of matmul_v0_nof4.cu showed some test cases finishing in
under a millisecond -- MUCH faster than the ~2.3ms floor established
earlier in this document -- while other sizes stayed at the expected
floor. Rather than report this as "scalar loads are sometimes faster,"
which would make no physical sense, it was checked directly: the exact
same binary was run a second time, and the SAME test cases that had been
fast were now back to the expected ~2.3-2.5ms floor, while a DIFFERENT,
previously-normal test case now showed an anomaly instead.

To isolate whether this was specific to the no-FLOAT4 kernel, both
matmul_v0.cu and matmul_v0_nof4.cu were run twice each, back to back, in
the same shell session:

```
=== v0 (FLOAT4) run A ===     all 13 sizes: consistent, expected timings
=== v0_nof4 run A ===         all 13 sizes: consistent, expected timings
=== v0 (FLOAT4) run B ===     "One extra column" test: min=0.0332ms (anomaly!)
                               1024^3: ALL 20 timed runs land near 0.15ms (anomaly!)
=== v0_nof4 run B ===         512^3 and 1000x777x333: anomalies
                               1024^3: ALL 20 timed runs land near 0.154ms (anomaly!)
```

The anomalies appeared in BOTH binaries, at different, inconsistent test
cases, in different runs. This confirms it is environmental noise on this
shared/virtualized GPU instance (likely a transient clock-boost or
scheduling effect), not a property of either kernel -- the comparison
methodology caught its own confound instead of reporting misleading
numbers.

## 8.4 The one size that stayed clean across all 4 runs: 4096 cubed

```
           v0 (FLOAT4)        v0_nof4 (scalar)
  run A:   17.00 TFLOPS       16.74 TFLOPS
  run B:   17.03 TFLOPS       16.71 TFLOPS
```

A small, consistent **~1.7% slowdown** without FLOAT4 -- real, but far
smaller than the naive "4x more load instructions" framing might suggest.
Likely explanation: the register-staging loads this ablation touches are
a small fraction of each k-step's total work compared to the 64 FMA
operations that immediately follow them -- this part of the kernel is
FMA-bound, not load-bound, so even a 4x jump in load instruction count
barely moves the needle.

---

# PART 9 -- Follow-up: What Do All Twelve constexpr Tiling Constants Mean?

## 9.1 The flowchart: one thread block's entire job

```
  Grid launch: many thread BLOCKS, each with 256 THREADS, each block
  owns one 128x128 chunk of the output matrix C.
                        |
                        v
  Step 1: reshape the SAME 256 threads into an 8x32 grid
          ("THREAD_A_LAYOUT"), cooperatively copy a slice of A
          into shared memory
                        |
                        v
  Step 2: reshape the SAME 256 threads into a 32x8 grid
          ("THREAD_B_LAYOUT"), cooperatively copy a slice of B
          into shared memory
                        |
                        v
  Step 3: reshape the SAME 256 threads into a 16x16 grid
          ("THREAD_C_LAYOUT"), each of those 256 cells now
          permanently owns an 8x8 patch of the OUTPUT tile
          ("THREAD_C_TILE_SIZE") -- the real multiply-and-add
          math happens here
                        |
                        v
            Write this block's 128x128 results into C
```

**The one idea that unlocks everything**: `tidx` (0 to 255) never changes
-- it's the same 256 physical threads the whole time. The code just
reinterprets that same flat number as a different (x, y) position
depending on which job it's doing right now. Three jobs, three
reshapings, three different sets of constants.

## 9.2 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_SIZE -- the shape of one block's work

```c
constexpr int BLOCK_M = 128;      // rows of C this block computes
constexpr int BLOCK_N = 128;      // columns of C this block computes
constexpr int BLOCK_K = 8;        // K-values processed per chunk
constexpr int BLOCK_SIZE = 256;   // threads cooperating on all of the above
```

```
                N (columns) ------------------->
  M     +----------+----------+----------+-----
  (rows)| Block    | Block    | Block    |
  |     | (0,0)    | (1,0)    | (2,0)    | ...
  v     | 128x128  | 128x128  | 128x128  |
        +----------+----------+----------+-----
        | Block    | Block    | Block    |
        | (0,1)    | (1,1)    | (2,1)    | ...
        +----------+----------+----------+-----
```

`BLOCK_K = 8` is separate from this picture -- it's about K (the summed-
over dimension, not shown in C at all): instead of loading all of K at
once, the kernel walks K in bite-sized chunks of 8, because the whole K
dimension usually cannot fit in shared memory at once.

## 9.3 Why three different thread layouts?

Because the three jobs move differently-shaped chunks of data:

| Job | Data shape moved | Thread layout needed |
|---|---|---|
| Load A | BLOCK_M(128) x BLOCK_K(8) = 1024 elements | 8 wide x 32 tall |
| Load B | BLOCK_K(8) x BLOCK_N(128) = 1024 elements | 32 wide x 8 tall |
| Compute C | BLOCK_M(128) x BLOCK_N(128) = 16384 elements | 16 wide x 16 tall |

`8x32 = 256`, `32x8 = 256`, `16x16 = 256` -- all three reshapings use all
256 threads, just arranged differently.

## 9.4 THREAD_A_LAYOUT_X/Y -- reshaping for loading A

```c
constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X;  // 32
```
```
         tAx: 0   1   2   3   4   5   6   7
tAy  0  [ t0  t1  t2  t3  t4  t5  t6  t7 ]
     1  [ t8  t9  t10 t11 t12 t13 t14 t15]
    ...
     31 [ ...                       t255 ]
```
```c
int tAx = tidx & (THREAD_A_LAYOUT_X - 1);   // tidx % 8  -> which K-offset (0..7)
int tAy = tidx / THREAD_A_LAYOUT_X;         // tidx / 8  -> which M-stripe (0..31)
```
`tAx` is 8 wide because `BLOCK_K = 8` -- every K-value in this chunk gets
its own column of threads. `tAy` is 32 tall, so covering all 128 rows
needs `BLOCK_M / THREAD_A_LAYOUT_Y = 4` loop iterations.

## 9.5 THREAD_B_LAYOUT_X/Y -- reshaping for loading B

```c
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X; // 8
```
The mirror image of A's layout (32 wide / 8 tall instead of 8 wide / 32
tall), because B's chunk is shaped the opposite way. `tBy` is 8 tall
because `BLOCK_K = 8` again -- K gets its own ROW of threads this time.
`tBx` is 32 wide, so covering 128 columns needs `BLOCK_N / 32 = 4` loop
iterations.

## 9.6 THREAD_C_LAYOUT_X/Y and the tile sizes -- reshaping for computing C

```c
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X;  // 16
constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_Y;  // 8
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y;  // 8
```
A square: 16 wide x 16 tall = 256, matching the block's one 128x128
output tile exactly. Each of those 256 grid cells is one thread, owning
an 8x8 patch of real output:
```
    N (within this 128-wide tile) ------------------------------->
      0..7      8..15     16..23            ...            120..127
    +--------+--------+--------+--      --+--------+
 M  |Thread  |Thread  |Thread  |          |Thread  |   <- row of 16
 0..7(0,0)   |(1,0)   |(2,0)   |   ...    |(15,0)  |      threads
    |8x8=64  |8x8=64  |        |          |        |
    |elements|elements|        |          |        |
    +--------+--------+--------+--      --+--------+
       ...                                             (16 rows total)

16 threads x 8 cols/thread = 128 cols (matches BLOCK_N)
16 threads x 8 rows/thread = 128 rows (matches BLOCK_M)
256 threads x 64 elements/thread = 16384 = 128x128 (matches the whole tile)
```

## 9.7 THREAD_C_WARP_X/Y and THREAD_C_WARP_DIM_X -- grouping threads into warps

A GPU schedules threads in groups of 32 called warps. 256/32 = 8 warps.
These constants describe one warp's shape inside the 16x16 grid:
```c
constexpr int THREAD_C_WARP_X = 8;                                   // warp spans 8 cells wide
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X;                // 4 cells tall (8*4=32)
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X; // 2 warp-rectangles wide
```
```
        X: 0..7          X: 8..15
      +---------------+---------------+
Y:0-3 |    Warp 0     |    Warp 1     |
      +---------------+---------------+
Y:4-7 |    Warp 2     |    Warp 3     |
      +---------------+---------------+
Y:8-11|    Warp 4     |    Warp 5     |
      +---------------+---------------+
Y:12-15|   Warp 6     |    Warp 7     |
      +---------------+---------------+

2 warps wide x 4 warps tall = 8 warps total (matches 256/32)
```

## 9.8 Quick-reference table

| Constant | Value | Means |
|---|---|---|
| BLOCK_M | 128 | rows of C per block |
| BLOCK_N | 128 | columns of C per block |
| BLOCK_K | 8 | K-values processed per chunk |
| BLOCK_SIZE | 256 | threads per block |
| THREAD_A_LAYOUT_X/Y | 8 / 32 | thread shape for loading A |
| THREAD_B_LAYOUT_X/Y | 32 / 8 | thread shape for loading B |
| THREAD_C_LAYOUT_X/Y | 16 / 16 | thread shape for computing C |
| THREAD_C_X/Y_TILE_SIZE | 8 / 8 | real output elements owned per thread |
| THREAD_C_WARP_X/Y | 8 / 4 | shape of one warp inside the C-grid |
| THREAD_C_WARP_DIM_X | 2 | how many warps fit side by side |

---

# PART 10 -- Follow-up: Why Does (tAx << 2) Actually Solve the Bank Conflict?

## 10.1 What is a bank, physically?

Shared memory has 32 banks, each holding one 4-byte float. Consecutive
floats cycle through the banks, then wrap:
```
element index:   0    1    2    3    4   ...  31   32   33   34  ...  63
bank number:     0    1    2    3    4   ...  31    0    1    2  ...  31
```
`bank = element_index % 32`.

## 10.2 What is a bank conflict?

A warp is 32 threads executing in lockstep. If two or more threads in
that warp write to different addresses landing on the SAME bank during
the same instruction, the hardware serializes them instead of doing them
in parallel -- an 8-way collision takes roughly 8x as long as it should.

## 10.3 The conflict, found with real numbers

`sA[0][tAx][m]` sits at flat position `tAx*128 + m`. Take warp 0
(`tidx=0..31`, `tAx=tidx%8`, `tAy=tidx/8`). WITHOUT the swizzle, for the
first 8 threads (`tAy=0`, `tAx=0..7`):

| tidx | tAx | address = tAx*128 + 0 | bank = address % 32 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 128 | 0 |
| 2 | 2 | 256 | 0 |
| 3 | 3 | 384 | 0 |
| 4 | 4 | 512 | 0 |
| 5 | 5 | 640 | 0 |
| 6 | 6 | 768 | 0 |
| 7 | 7 | 896 | 0 |

All 8 threads land on bank 0 -- an 8-way conflict. Why? `128 = 4*32`, an
exact multiple of 32, so adding any multiple of 128 never changes
`address % 32`. `tAx` contributes ZERO to the bank; only `tAy` matters,
and since these 8 threads share `tAy=0`, they collide. This repeats for
`tAy=1,2,3` -- 4 separate 8-way pileups per warp.

## 10.4 With the swizzle, the same 8 threads

With `(i+tAy) ^ (tAx<<2)`, for `tAy=0`, `tAx=0..7`:

| tAx | tAx<<2 | 0 ^ (tAx<<2) | address = tAx*128 + that | bank |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 4 | 4 | 132 | 4 |
| 2 | 8 | 8 | 264 | 8 |
| 3 | 12 | 12 | 396 | 12 |
| 4 | 16 | 16 | 528 | 16 |
| 5 | 20 | 20 | 660 | 20 |
| 6 | 24 | 24 | 792 | 24 |
| 7 | 28 | 28 | 924 | 28 |

8 different banks, zero conflict. Doing the same for `tAy=1,2,3` gives:
```
tAy=0: banks 0, 4,  8, 12, 16, 20, 24, 28
tAy=1: banks 1, 5,  9, 13, 17, 21, 25, 29
tAy=2: banks 2, 6, 10, 14, 18, 22, 26, 30
tAy=3: banks 3, 7, 11, 15, 19, 23, 27, 31
```
All 32 threads of this warp land on all 32 banks, each exactly once --
a perfect, conflict-free spread.

## 10.5 Why `tAx << 2` specifically?

`bank = address % 32` only cares about the lowest 5 bits. Without the
swizzle, `tAx * 128` never touches those 5 bits (it's always a multiple
of 32). The fix must inject a tAx-dependent value into exactly those
bottom 5 bits. `tAx` is 3 bits (0..7); `tAx << 2` shifts those 3 bits up
to occupy bit positions 2, 3, 4 -- squarely inside the 5-bit range that
determines the bank, while leaving the bottom 2 bits (which come from
`tAy`) untouched.

## 10.6 Before vs after

```
WITHOUT swizzle: bank = f(tAy) only, tAx ignored
  tAy=0 group (8 threads) ---> ALL pile onto bank 0   (8-way conflict)
  tAy=1 group (8 threads) ---> ALL pile onto bank 1   (8-way conflict)
  ... (repeats for tAy=2,3)

WITH swizzle: bank = f(tAy, tAx), spread out
  tAy=0 group ---> banks  0, 4, 8,12,16,20,24,28   (8 DIFFERENT banks)
  ... (all 32 banks used exactly once across the whole warp)
```

Correctness note: for any single fixed `tAx`, XOR with a constant mask is
a reversible, one-to-one shuffle of all 128 `m` slots within that same
`tAx`'s own row -- it never touches a different `tAx`'s data, and every
real value still lands in a unique, valid slot, just reordered.

---

# PART 11 -- Follow-up: Where Does "tAx*128 + tAy" Actually Come From?

## 11.1 Computer memory is never actually 2D or 3D

No matter how many `[...]` brackets you write, physical memory is one
flat line of slots. A "2D array" is a convenience the compiler provides;
underneath, it still does arithmetic to turn `[row][col]` into one flat
slot number.

## 11.2 The tiniest example: a 2x3 array

```c
float tiny[2][3];
```
```
Logical view:                                  Physical memory:
  tiny[0][0]  tiny[0][1]  tiny[0][2]            flat slot:   0     1     2     3     4     5
  tiny[1][0]  tiny[1][1]  tiny[1][2]            contains: [0][0][0][1][0][2][1][0][1][1][1][2]
```
Row 0 occupies slots 0,1,2. Row 1 starts right after, at slots 3,4,5 --
no gap, no real 2D grid in memory, just row 0's elements then row 1's,
stuck end to end.

## 11.3 The general rule

```
flat_slot = row * (number of columns) + col
```
Check: `tiny[1][2]` should be the last slot, #5: `1*3+2=5`. Correct.
`tiny[0][1]` should be slot #1: `0*3+1=1`. Correct. "Number of columns"
is the STRIDE of the row dimension -- how far you jump in flat memory
for every +1 step in the row index.

## 11.4 Applying it to the real sA array

```c
__shared__ float sA[2][BLOCK_K][BLOCK_M];   // sA[2][8][128]
```
Fixing the first index to 0 (buffer 0), `sA[0]` behaves exactly like
`tiny[row][col]`, just bigger: 8 rows (indexed by `tAx`) x 128 columns
(indexed by the m-position). Using the same rule:
```
flat_slot = tAx * 128 + (m-position)
```
That is the entire origin of the "128" -- it is `BLOCK_M`, the number of
columns in `sA[0]`'s 8x128 shape, i.e. how many flat slots to skip to
move from one `tAx` row to the next.

## 11.5 Verified against real numbers (tAy=0, tAx=0..7)

| tAx | rows skipped | flat_slot = tAx*128 + 0 |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1 (128 slots) | 128 |
| 2 | 2 (256 slots) | 256 |
| 3 | 3 (384 slots) | 384 |
| 4 | 4 (512 slots) | 512 |
| 5 | 5 (640 slots) | 640 |
| 6 | 6 (768 slots) | 768 |
| 7 | 7 (896 slots) | 896 |

## 11.6 The full 3D version, for completeness

```
flat_slot = buffer * (BLOCK_K * BLOCK_M) + tAx * BLOCK_M + m
          = buffer * 1024 + tAx * 128 + m
```
With `buffer=0` this first term vanishes, leaving the 2-term formula used
throughout. Buffer 1 would just add a flat 1024 to every address --
since 1024 = 32*32 is also an exact multiple of 32, the bank pattern
comes out identically regardless of which buffer is active.

---

# PART 12 -- Follow-up: Which Code Reads the Swizzled Write Back, and How Do We Know It's Correct?

## 12.1 The matching read code

Write (prologue, `tAx` = which k-offset):
```c
sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
```
Read (compute loop, `tk` = which k-offset, same role as tAx):
```c
int base = r ^ (tk << 2);
tCsA[tk & 1][(tm << 2) + 0] = sA[bufferId][tk][base + 0];
```
When reading k-chunk `tk`, you are reading data originally written with
`tAx = tk` (same physical k-slice). Since both sides use the SAME mask
for the SAME k, XOR cancels: `(value ^ mask) ^ mask = value`.

## 12.2 Flowchart of the round trip

```
  WRITE (once, per k-chunk):
    scrambled_slot = m ^ (tAx << 2)
    sA[0][tAx][scrambled_slot] = A's real value
          |
          v
     __syncthreads()
          |
          v
  READ (later, once per tk, tk == tAx for this k-slice):
    scrambled_slot = m ^ (tk << 2)     <- SAME formula, SAME mask
    value = sA[bufferId][tk][scrambled_slot]
    ---> lands on the exact slot the write used, recovers the exact value
```

## 12.3 Not just asserted -- built and run for real

A standalone CUDA program was written that does nothing but test this
round trip. Every one of the 8x128=1024 `(tAx, m)` cells gets a UNIQUE
marker value (`tAx*1000 + m`) written through the real swizzle
expression, then read back through the matching un-swizzle, checked for
an exact match:

```c
__global__ void swizzle_round_trip_test(int *mismatches, int *cells_checked) {
    __shared__ float sA[BLOCK_K][BLOCK_M];
    int tAx = tidx & (THREAD_A_LAYOUT_X - 1);
    int tAy = tidx / THREAD_A_LAYOUT_X;

    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int m = i + tAy;
        float marker = tAx * 1000.0f + m;
        sA[tAx][m ^ (tAx << 2)] = marker;          // same swizzle as matmul_v0.cu
    }
    __syncthreads();

    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int m = i + tAy;
        float expected = tAx * 1000.0f + m;
        float got = sA[tAx][m ^ (tAx << 2)];
        if (got != expected) atomicAdd(mismatches, 1);
        atomicAdd(cells_checked, 1);
    }
}
```

Real output, compiled and run on this box's GB200:
```
Expected total (tAx, m) slots: 1024
Cells actually checked:        1024
Mismatches found:              0
PASS: every swizzled write round-tripped to the exact correct value
```

## 12.4 Proving the test itself is meaningful -- a negative control

To rule out the test trivially passing no matter what, a deliberately
BROKEN version was built, reading with a wrong mask (`(tAx+1)<<2` instead
of `tAx<<2`):
```
Expected total (tAx, m) slots: 1024
Cells actually checked:        1024
Mismatches found:              1024
FAIL (expected -- proves the test can actually detect a broken swizzle)
```
Every single cell was correctly flagged as wrong. This confirms the
original test's "0 mismatches" result is a real, meaningful pass, not a
false positive from a test that never fails.

## 12.5 The same evidence, at a bigger scale

The full matmul_v0.cu/matmul_v0_nof4.cu correctness suite is the same
kind of proof, indirectly: 13 test cases compare the GPU's final output
(which depends on this exact swizzle round trip happening correctly, over
and over, for every k-chunk of every tile) against a plain CPU reference
that uses no swizzle at all. Getting `0.000000` max error across ragged,
non-aligned sizes (K=63, K=129, K=333, K=1) is the same proof, viewed
through "does the whole kernel produce the right matrix" instead of
"does this one array round-trip right." Both angles agree.

---

# PART 13 -- Follow-up: Do the Write-Side and Read-Side Addresses Actually Match?

## 13.1 The parking garage analogy

Imagine a 128-space parking garage. Valet A parks cars in the morning,
walking the garage one way (row 1, then row 2, then row 3, then row 4).
Valet B retrieves cars in the afternoon, walking it in a completely
different pattern (color-coded zones). As long as (1) every car got
parked somewhere with no two cars sharing a spot, (2) every car that was
parked is one that gets looked for later, and (3) both valets agree on
one shared rule for translating "ticket number" into "physical spot" --
it does not matter that they walked the garage in different orders. This
is exactly what happens between the write code (grouped by tAx/tAy) and
the read code (grouped by tCx/tCy, a totally unrelated grouping of the
same 256 threads).

## 13.2 The write side's coverage pattern

```c
for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {   // i = 0,32,64,96
    int m = i + tAy;   // tAy fixed per-thread, 0..31
}
```
Every `m` from 0 to 127 splits uniquely as `m = i + tAy` (long division by
32: `i = 32*(m/32)`, `tAy = m%32`). As `i` sweeps 4 values and `tAy`
sweeps 32 values across different threads, every one of the 128 `m`
values gets hit exactly once, collectively, by the whole block.

## 13.3 The read side's coverage pattern

```c
int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;   // THREAD_C_LAYOUT_Y=16, tm=0 or 1
```
`tCy + tm*16` is long division by 16, giving a unique value `v` in 0..31.
`<<2` turns those 32 unique values into the 32 multiples of 4 from 0 to
124 -- each the start of a 4-wide read group. 32 groups of 4 = 128
values, tiled with zero gaps and zero overlaps.

## 13.4 Same 128 slots, two different numbering systems

```
   Write side (grouped into 4 bands of 32, by "i"):     Read side (grouped into 32 bands of 4, by "base"):
   i=0:  tAy=0..31  -> m = 0..31                         base=0:   m = 0..3
   i=32: tAy=0..31  -> m = 32..63                         base=4:   m = 4..7
   i=64: tAy=0..31  -> m = 64..95                         ...
   i=96: tAy=0..31  -> m = 96..127                        base=124: m = 124..127
```

## 13.5 Complete worked numeric trace: k=3, m=8,9,10,11

Write side -- 4 DIFFERENT threads, each writing ONE value:

| real m | i | tAy | tidx of the writer | scrambled slot = m^(3<<2) = m^12 |
|---|---|---|---|---|
| 8 | 0 | 8 | 67 | 4 |
| 9 | 0 | 9 | 75 | 5 |
| 10 | 0 | 10 | 83 | 6 |
| 11 | 0 | 11 | 91 | 7 |

Four different threads (67, 75, 83, 91) land on physical slots 4, 5, 6, 7
-- four consecutive slots, purely a side effect of the swizzle's
"last-2-bits-preserved" property.

Read side -- ONE thread, reading all 4 of those slots at once: working
backwards from `base=4`: `r = base ^ (tk<<2) = 4^12 = 8`; `r = (tCy +
tm*16)<<2 = 8` gives `tm=0, tCy=2`. The thread with `tCy=2` (tm=0, tk=3)
reads physical slots 4,5,6,7 in one instruction.

```
           WRITE                                    READ
  (4 separate threads,                      (1 thread, 1 instruction,
   4 separate instructions)                  reading all 4 at once)

  tidx=67 writes m=8  --> physical slot 4  <---+
  tidx=75 writes m=9  --> physical slot 5  <---+---  tCy=2, tm=0, tk=3
  tidx=83 writes m=10 --> physical slot 6  <---+     reads slots 4,5,6,7
  tidx=91 writes m=11 --> physical slot 7  <---+     in ONE grouped read
```

XOR-ing each of those 4 slots back with the same mask recovers exactly
`m = 8, 9, 10, 11`: `4^12=8, 5^12=9, 6^12=10, 7^12=11` -- the same four
real row numbers that were written. Nothing lost, duplicated, or swapped.

## 13.6 The resolution, in one sentence

The write and read sides do not need to match thread-for-thread or
formula-for-formula -- they only need to (1) both fully and uniquely
cover the same 128 slots, and (2) agree on the same `m ^ (k << 2)`
translation rule for turning "real position" into "physical slot." Given
both of those, any thread can safely retrieve data any other thread
wrote, no matter how differently the two phases divided up the work.

---

# PART 14 -- Follow-up: Does Every Thread Move A, Move B, AND Compute C?

## 14.1 Short answer: yes, all three, but not the way it might sound

It is tempting to read `THREAD_A_LAYOUT`, `THREAD_B_LAYOUT`, and
`THREAD_C_LAYOUT` as describing three DIFFERENT GROUPS of threads (some
threads permanently specialize in loading A, others in loading B, others
in computing C). That is not what happens. Every one of the 256 threads
runs ALL of the kernel's code, in sequence, over time -- it just
recomputes its own "identity" differently for each job.

## 14.2 Why every thread runs every line: the SIMT model

GPUs execute a model called SIMT (Single Instruction, Multiple Threads):
all 256 threads in a block run the exact same kernel function, line by
line. There is no `if (threadIdx.x < 85) { load A } else { ... }`
branching that permanently splits threads into specialists. Every thread
runs every line, substituting its own `tidx` into the formulas, so each
thread touches different DATA while running identical INSTRUCTIONS.

```
          ALL 256 threads run this SAME function body:

  compute tAx, tAy, tBx, tBy, tCx, tCy   <- every thread computes ALL SIX
          |
          v
  Phase 1: everyone helps load A   (using tAx, tAy)
          |
          v
  Phase 2: everyone helps load B   (using tBx, tBy)
          |
          v
  __syncthreads()
          |
          v
  Phase 3 (repeated many times): everyone computes their own
           permanent 8x8 patch of C (using tCx, tCy), AND
           everyone ALSO helps reload the next chunk of A/B
           (using tAx/tAy/tBx/tBy again)
          |
          v
  Epilogue: everyone writes their own 8x8 patch into C
```

## 14.3 Tracing one real thread (tidx=42) through all three identities

```c
int tAx = tidx & 7;         // 42 & 7  = 2
int tAy = tidx / 8;         // 42 / 8  = 5

int tBx = tidx & 31;        // 42 & 31 = 10
int tBy = tidx / 32;        // 42 / 32 = 1

// tCx, tCy (warp-aware derivation):
// warpId=1, laneId=10, warpx=1, warpy=0, lanex=5, laney=0
tCx = 1*8 + 5 = 13
tCy = 0*4 + 0 = 0
```
Thread 42 is `tAx=2, tAy=5` for loading A; `tBx=10, tBy=1` for loading B;
`tCx=13, tCy=0` for computing C. Three different-looking numbers, one
physical thread.

## 14.4 What thread 42 actually does in Phase 1 (loading A)

```c
for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {   // i = 0, 32, 64, 96
    int r = row0 + i + tAy;
    sA[0][tAx][(i + tAy) ^ (tAx << 2)] = ...
}
```
With `tAx=2, tAy=5`:

| i | m = i+tAy | writes A's row (row0+m), column k=2 | scrambled slot m^8 |
|---|---|---|---|
| 0 | 5 | row0+5 | 13 |
| 32 | 37 | row0+37 | 45 |
| 64 | 69 | row0+69 | 77 |
| 96 | 101 | row0+101 | 109 |

Thread 42 personally moves exactly 4 individual float values of A (one
column, k=2, at 4 specific rows).

## 14.5 What thread 42 actually does in Phase 2 (loading B)

```c
for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {   // i = 0 only
    int r = i + tBy;
    for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {  // j = 0,32,64,96
        sB[0][i + tBy][j + tBx] = ...
    }
}
```
With `tBx=10, tBy=1`: `r = 0+1 = 1` (this thread always handles B's row
k=1). For `j=0,32,64,96`, it writes B's row 1 at columns `10, 42, 74,
106`. Thread 42 personally moves 4 more values -- this time from B --
into a completely different part of shared memory.

## 14.6 What thread 42 actually does in Phase 3 (computing C)

With `tCx=13, tCy=0`, this thread's permanent output patch covers
`m = tCy + tm*16` for `tm=0..7` and `n = tCx + tn*16` for `tn=0..7` --
64 output cells total, updated a little on every pass through the
K-loop, not computed once.

## 14.7 And Phase 1/2 keep happening, not just once

Inside the main loop, thread 42 does not stop being an A/B-loader after
the prologue -- the same `tAx/tAy` and `tBx/tBy` get reused every time
the kernel needs the next chunk of K:
```c
if (k < K) {
    sA[bufferId ^ 1][tAx][...] = ...   // identical code to the prologue
    sB[bufferId ^ 1][...][tBx] = ...
}
```

## 14.8 Full timeline for thread 42

```
  [Phase 1] Move 4 values of A      (role: tAx=2, tAy=5)
  [Phase 2] Move 4 values of B      (role: tBx=10, tBy=1)
  [sync]
  [Phase 3, iteration 1] Accumulate into its 64 C-cells  (role: tCx=13, tCy=0)
  [Phase 3, iteration 1] ALSO move 4 more values of A + 4 more of B for next chunk
  [Phase 3, iteration 2] Accumulate again
  [Phase 3, iteration 2] ALSO move next chunk's A/B again
  ... (repeats for every K-chunk) ...
  [Epilogue] Write its final 64 accumulated values into real C
```

Every one of the 256 threads moves some of A, moves some of B, AND
permanently owns a chunk of C's final answer. `THREAD_A_LAYOUT`,
`THREAD_B_LAYOUT`, and `THREAD_C_LAYOUT` describe the same 256 threads,
reinterpreted three different ways at three different points in time --
like one person wearing three different hats one after another, not
three different people each wearing one hat.

---

# PART 15 -- Follow-up: What Is the Per-Thread Stride for A, B, and C?

## 15.1 Quick-answer table

| Matrix | Stride pattern for one thread |
|---|---|
| A | Simple, uniform: 32 |
| B | Simple, uniform: 32 |
| C | NOT uniform: 4 consecutive, jump 64, 4 more consecutive |

## 15.2 A's stride -- simple and uniform

```c
for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {   // i = 0, 32, 64, 96
    int r = row0 + i + tAy;
}
```
One thread's `tAy` is fixed; only `i` changes, jumping by
`THREAD_A_LAYOUT_Y = 32` every time. Real rows touched:
`tAy, tAy+32, tAy+64, tAy+96` -- evenly spaced, 32 apart.

```
Thread's 4 real rows of A:    tAy  tAy+32  tAy+64  tAy+96
                               |------32------|------32------|------32------|
```

## 15.3 B's stride -- also simple and uniform

```c
for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {   // j = 0, 32, 64, 96
    int c = col0 + j + tBx;
}
```
Same structure, mirrored: real columns touched are `tBx, tBx+32, tBx+64,
tBx+96` -- again 32 apart.

## 15.4 C's stride -- not uniform at all

The actual write-back code (identical between matmul_v0.cu and
matmul_v0_nof4.cu):

```c
for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {        // i = 0..7
    int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
}
```

Two bit tricks need unpacking. `i & 3` keeps only the lowest 2 bits of
`i` -- "where am I within a group of 4?" For `i=0..7`: gives `0,1,2,3,
0,1,2,3`. `i & (~3)` clears the lowest 2 bits, keeping the rest -- "which
group of 4 am I in?" For the same `i` values: gives `0,0,0,0,4,4,4,4`.

```
i:         0   1   2   3   4   5   6   7
i & 3:     0   1   2   3   0   1   2   3     <- position within group
i & ~3:    0   0   0   0   4   4   4   4     <- which group (0 or 4)
```

## 15.5 Trace thread 42's actual 8 rows of C

With `tCy = 0`: `r = row0 + 0 + (i & ~3)*16 + (i & 3)`:

| i | i&3 | i&~3 | (i&~3)*16 | r (relative to row0) |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 | 1 |
| 2 | 2 | 0 | 0 | 2 |
| 3 | 3 | 0 | 0 | 3 |
| 4 | 0 | 4 | 64 | 64 |
| 5 | 1 | 4 | 64 | 65 |
| 6 | 2 | 4 | 64 | 66 |
| 7 | 3 | 4 | 64 | 67 |

```
Thread 42's 8 real rows of C:   0, 1, 2, 3,  [big jump of 64],  64, 65, 66, 67
                                 |--4 in a row--|                |--4 in a row--|
```

Not a uniform stride -- 4 consecutive rows, a jump of 64, then 4 more
consecutive rows. The identical pattern happens for the columns (`n`),
using `THREAD_C_LAYOUT_X` instead of `THREAD_C_LAYOUT_Y`. For thread 42
(`tCx=13`): columns `52,53,54,55, [jump 64], 116,117,118,119`.

## 15.6 Why is C different? The historical reason, still baked in

This "4 consecutive, then jump" shape comes from the register-staging
step, which always processes 4 real output positions as one group,
because matmul_v0.cu wanted to load those 4 with a single FLOAT4
instruction:

```c
// register-staging, still groups by 4 even without FLOAT4 in matmul_v0_nof4.cu:
for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm++) {   // tm = 0, 1
    tCsA[tk & 1][(tm << 2) + 0] = ...
    tCsA[tk & 1][(tm << 2) + 1] = ...
    tCsA[tk & 1][(tm << 2) + 2] = ...
    tCsA[tk & 1][(tm << 2) + 3] = ...
}
```

Even in the no-FLOAT4 ablation, this still loads 4 values per group, just
with 4 scalar instructions instead of 1 vectorized one -- the grouping of
4 is a property of the index math (`acc[][]`'s layout), not of FLOAT4
itself. The final C write-back has to use the matching "4 consecutive,
jump 64" pattern to put each accumulated value back where it belongs --
removing FLOAT4 only changed HOW data moved into those groups, not WHICH
positions make up a group.

## 15.7 Side-by-side summary

```
A: one thread's 4 real rows:     tAy, tAy+32, tAy+64, tAy+96        (uniform stride 32)

B: one thread's 4 real columns:  tBx, tBx+32, tBx+64, tBx+96        (uniform stride 32)

C: one thread's 8 real rows:     r, r+1, r+2, r+3, [jump 64], r+64, r+65, r+66, r+67
                                  (groups of 4 back-to-back, groups spaced 64 apart)
   (and the identical pattern, independently, for its 8 real columns)
```

---

# PART 16 -- matmul_v2_nof4.cu, and a Major Honest Correction

## 16.1 The ablation itself

matmul_v2.cu uses FLOAT4 in exactly four places, all in the row-caching
step (fetching a thread's BLOCK_K-length row of A, and its BLOCK_K-length
row of B-transposed):
```c
FLOAT4(regA[tm][0]) = FLOAT4(sA[bufferId][m][0]);
FLOAT4(regA[tm][4]) = FLOAT4(sA[bufferId][m][4]);
FLOAT4(regB[tn][0]) = FLOAT4(sB[bufferId][n][0]);
FLOAT4(regB[tn][4]) = FLOAT4(sB[bufferId][n][4]);
```
Each was replaced with 8 scalar reads per row. Unlike matmul_v0.cu's
ablation, there is no swizzle/XOR involved here at all -- v2's layout
already has k as the innermost axis for both operands, so these reads
were always plain, unscrambled, contiguous lookups.

## 16.2 Register count and correctness

`nvcc -Xptxas -v`: 168 registers, identical to matmul_v2.cu. Correctness:
13/13 PASS, with the exact same small nonzero errors as matmul_v2.cu
(`0.000001` to `0.000359`, from the dot-product's own summation order) --
confirming bit-identical floating-point behavior.

## 16.3 FLOAT4 vs scalar for v2: negligible, unlike v0's ~1.7% edge

Two clean back-to-back runs each:
```
           v2 (FLOAT4)        v2_nof4 (scalar)
  run A:   10.90 TFLOPS       10.89 TFLOPS
  run B:   10.89 TFLOPS       10.88 TFLOPS
```
Essentially no measurable difference -- v2's row-caching reads were
already simple, well-behaved memory accesses either way, unlike v0's
swizzled, scrambled register-staging reads.

## 16.4 The important correction: the earlier "v2 beats v0" claim was likely wrong

While running this comparison, matmul_v2.cu's 4096^3 number came out as
10.89-10.90 TFLOPS, consistently -- nowhere near the 20.00 TFLOPS claimed
earlier in this document (Part 7.4). GPU state was checked (full P0
clock, no throttling, no contention) and a fresh, back-to-back, twice-
repeated comparison of matmul_v0.cu and matmul_v2.cu was run, right in
the same session:
```
           Run A              Run B
v0:        16.93 TFLOPS       17.07 TFLOPS
v2:        10.90 TFLOPS       10.89 TFLOPS
```
Clean and reproducible: v0 consistently beats v2 by a wide margin. Given
this whole project has already demonstrated this shared/virtualized GPU
produces occasional noise spikes in BOTH directions (not just slow
anomalies, but suspiciously fast ones too), the most honest conclusion is
that the earlier "20.00 TFLOPS, v2 wins" measurement was very likely an
upward noise artifact, not a reliable result. The original register-
pressure prediction (outer-product should win, fewer registers: 126 vs
168) probably holds up better than previously claimed. This correction is
recorded here rather than silently edited into Part 7, following the same
"show the mistake, don't hide it" discipline used for the TF32 and
"SKIP sizes" corrections earlier in this document.

---

# PART 17 -- Follow-up: Why Doesn't matmul_v2.cu Need (tAx << 2) for A?

## 17.1 The two shapes, side by side

```c
// matmul_v0.cu:  A TRANSPOSED
__shared__ float sA[2][BLOCK_K][BLOCK_M];   // sA[2][8][128] -- k outer, m inner

// matmul_v2.cu:  A NATURAL
__shared__ float sA[2][BLOCK_M][BLOCK_K];   // sA[2][128][8] -- m outer, k inner
```
This single shape flip is the entire reason the swizzle disappears.

## 17.2 v2's A-loading code (no swizzle at all)

```c
for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {   // i = 0, 32, 64, 96
    int r = row0 + i + tAy;
    sA[0][i + tAy][tAx] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
}
```
Since `sA[0]` is shaped `[128][8]` (8 = row stride), flat address of
`sA[0][m][k]` is `m*8 + k`. Tracing one real warp (i=0, so m=tAy):

| tidx | tAy (=m) | tAx (=k) | address = m*8 + k | bank = address % 32 |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 | 1 |
| ... | | | | |
| 7 | 0 | 7 | 7 | 7 |
| 8 | 1 | 0 | 8 | 8 |
| ... | | | | |
| 31 | 3 | 7 | 31 | 31 |

All 32 threads land on all 32 different banks, zero overlaps -- with no
swizzle at all.

## 17.3 Why? The "erasure by 32" mechanism, in reverse

In matmul_v0.cu, `tAx` was the OUTER index, multiplied by a row-stride of
128 (`tAx * 128`), and `128 = 4*32` is an exact multiple of 32 -- so
`tAx`'s variation got completely erased by `% 32`, leaving only `tAy` to
determine the bank. In matmul_v2.cu, `tAx` is the INNER index, added
directly with no multiplication (`m*8 + tAx`) -- it's never multiplied by
anything, so it can never get erased.

```
v0's address:  tAx * 128  +  (i+tAy)         <- tAx multiplied by a multiple of 32: ERASED
v2's address:  (i+tAy) * 8  +  tAx           <- tAx added directly, never erased
```

The deeper reason: `BLOCK_K = 8` and `THREAD_A_LAYOUT_X = 8` are the same
number, and `8` divides evenly into the warp size `32` exactly 4 times.
One warp always spans exactly 4 distinct `tAy` values, and since the
row-stride (8) cleanly tiles 32 with no remainder, those 4 `tAy`-groups
land on 4 completely separate, non-overlapping 8-bank chunks.

## 17.4 Bonus discovery: checking B the same way turned up something unaddressed

B got transposed in v2, taking on the same `[128][8]` shape class as A:
```c
sB[0][j + tBx][i + tBy] = ...   // sB[2][BLOCK_N][BLOCK_K] = sB[2][128][8]
```
But `tBx` ranges the FULL `0..31` within one warp (`THREAD_B_LAYOUT_X =
32`), unlike `tAx` which only ranges `0..7` per group. Address = `(j+tBx)*8
+ tBy`. For warp 0 (`tBy=0` warp-wide, `j=0`): address = `tBx*8`:

| tBx | address = tBx*8 | bank = address % 32 |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 8 | 8 |
| 2 | 16 | 16 |
| 3 | 24 | 24 |
| 4 | 32 | 0 (collides with tBx=0!) |
| 5 | 40 | 8 (collides with tBx=1!) |

An 8-way bank conflict on B's write, every chunk reload, never fixed in
matmul_v2.cu.

---

# PART 18 -- Follow-up: Investigating and Attempting to Fix B's Bank Conflict

## 18.1 Re-deriving the address formula for B from scratch

Same row-major flattening rule as always. `sB[2][BLOCK_N][BLOCK_K]` =
`sB[2][128][8]`. Fixing buffer 0, `sB[0]` is 128 rows x 8 columns.
`flat_slot = row * COLS + col`, COLS = BLOCK_K = 8.
```c
sB[0][ j + tBx ][ i + tBy ]
       ^^^^^^^^    ^^^^^^^^
       SECOND       THIRD bracket = COLUMN (0..7)
       bracket = ROW (0..127)
```
So `flat_slot = (j+tBx)*8 + (i+tBy)`. A tiny toy example: `sB_toy[4][2]`
(4 rows, 2 columns) lays out as slots 0,1 (row0), 2,3 (row1), 4,5 (row2),
6,7 (row3). Check: `sB_toy[3][1]` should be the last slot (7):
`3*2+1=7`. Correct -- same rule, scaled up to real B's 128x8 shape.

## 18.2 Does B need a swizzle, and what would it look like?

Yes, by the math. Within one warp, `tBy` is constant and `tBx` sweeps
the full 0..31; address = `tBx*8 + constant` repeats every 4 values of
`tBx` (since 8*4=32), collapsing 32 threads onto 4 banks (8-way conflict
each). A derived fix: `mask = (n >> 2) & 7`, then write/read at `k ^
mask` instead of plain `k`. Verified by hand for all 32 threads of warp
0: spreads them across all 32 banks with zero overlap. Caveat: unlike
A's `tAx << 2` (always a multiple of 4, preserving "4 stays together"),
this mask can be any value 0-7, breaking FLOAT4's "4 consecutive"
requirement -- forcing 8 scalar reads instead of 2 vectorized ones.

## 18.3 Why did all 13 test cases pass despite this unaddressed conflict?

Because a bank conflict is a PERFORMANCE problem, not a CORRECTNESS
problem. When several threads in a warp hit the same bank, the hardware
doesn't compute the wrong answer -- it serializes those specific
accesses (one at a time instead of all 32 at once), costing extra clock
cycles, but every thread still reads/writes its own correct, distinct
address. Like a single-lane toll booth: 8 cars needing that lane queue up
and go through one at a time -- slower, but every car reaches its own
correct destination. Nothing is corrupted, lost, or swapped.

## 18.4 Investigating with a real profiler -- blocked by environment restrictions

`ncu` (NVIDIA Nsight Compute) is installed, but attempting to profile
hung/failed with "No kernels were profiled," even when targeting the
kernel by name directly. This is the standard NVIDIA driver restriction
on hardware performance-counter access for shared/cloud GPU instances
(`NVreg_RestrictProfilingToAdminUsers`), confirmed by checking for `sudo`
access (none available). This is a real, honest environment limitation,
not a failure to find the right flag.

## 18.5 Building and measuring the real fix instead

matmul_v2_bfix.cu applies the derived swizzle to B's write and read
(keeping A's FLOAT4 untouched, since A has no conflict):
```c
// write:
int mask = (n >> 2) & 7;
sB[0][n][kk ^ mask] = <value for real (n, k)>;

// read:
int mask = (n >> 2) & 7;
regB[tn][0] = sB[bufferId][n][0 ^ mask];
... (8 scalar reads total, same mask)
```
Correctness: 13/13 PASS, with the EXACT SAME small errors as matmul_v2.cu
(`0.000001` to `0.000359`) -- confirms the swizzle math recovers the
identical real values.

## 18.6 The honest result: the fix made things slightly WORSE

Measured back to back, repeatedly, against matmul_v2.cu and
matmul_v2_nof4.cu:
```
           v2 (has conflict)   v2_nof4 (has conflict)   v2_bfix (fixed)
  1024^3:  0.84 TFLOPS         0.84 TFLOPS              0.84 TFLOPS
  2048^3:  5.55 TFLOPS         5.54 TFLOPS              5.41 TFLOPS
  4096^3:  10.86 TFLOPS        10.89 TFLOPS              10.26 TFLOPS
```
Fixing the bank conflict made performance 2-6% WORSE, reproducibly. The
likely reason: the mask computation (`(n>>2)&7`) adds real extra
arithmetic to every read, and losing FLOAT4 here costs a little too --
stacking those costs apparently outweighs whatever the bank conflict
itself was costing. The conflict is real and provable by the address
math, but it is not actually the bottleneck for this kernel's overall
performance -- the 64 FMA operations per chunk likely dwarf the cost of
one serialized shared-memory write that happens far less often. A real,
verified, previously-unaddressed bank conflict exists, but fixing it,
measured honestly rather than assumed, is not worth doing here.

---

# PART 19 -- Follow-up: Why Does `mask = (n >> 2) & 7` With `k ^ mask` Actually Fix It?

## 19.1 The code behind n and k in the fix

```c
int mask = (n >> 2) & 7;
sB[0][n][kk ^ mask] = (r < K && c < N) ? B[r * N + c] : 0.0f;
```

## 19.2 What `mask = (n >> 2) & 7` computes

`n >> 2` divides `n` by 4 (shifting right by 2 bits = integer division by
4). `& 7` keeps only the lowest 3 bits, clamping the result to `0..7`.

```
n (binary)        n >> 2 (= n / 4)      & 7
0  = 0000000      0                     0
1  = 0000001      0                     0
2  = 0000010      0                     0
3  = 0000011      0                     0
4  = 0000100      1                     1
5  = 0000101      1                     1
6  = 0000110      1                     1
7  = 0000111      1                     1
8  = 0001000      2                     2
```
Notice: `n=0,1,2,3` (the first colliding "wave") all produce `mask=0`.
`n=4,5,6,7` (the second colliding wave, the one that crashed into the
first) all produce `mask=1`. This is exactly a "which 4-wide group am I
in" counter -- a different number for each colliding group.

## 19.3 Why XOR-ing k with this mask breaks the collision -- full trace

Tracing `n=0..7`, with `k = tBy = 0` (constant across this warp):

| n | mask = (n>>2)&7 | k^mask (k=0) | address = n*8 + (k^mask) | bank |
|---|---|---|---|---|
| 0 | 0 | 0^0 = 0 | 0*8+0 = 0 | 0 |
| 1 | 0 | 0^0 = 0 | 1*8+0 = 8 | 8 |
| 2 | 0 | 0^0 = 0 | 2*8+0 = 16 | 16 |
| 3 | 0 | 0^0 = 0 | 3*8+0 = 24 | 24 |
| 4 | 1 | 0^1 = 1 | 4*8+1 = 33 | 1 (used to be 0!) |
| 5 | 1 | 0^1 = 1 | 5*8+1 = 41 | 9 (used to be 8!) |
| 6 | 1 | 0^1 = 1 | 6*8+1 = 49 | 17 (used to be 16!) |
| 7 | 1 | 0^1 = 1 | 7*8+1 = 57 | 25 (used to be 24!) |

`n=4` no longer lands on bank 0 -- it lands on bank 1, completely clear
of `n=0`. The first 4 `n`'s (0-3) were already conflict-free among
themselves, and `mask=0` leaves them untouched. The next group of 4
(`n=4-7`) gets `mask=1` stamped into their `k` value, shoving their whole
address over by exactly 1 -- just enough to dodge the previous group,
without creating any new collision.

## 19.4 The intuition in plain words

The collision happened because the part of `n` that distinguishes `n=0`
from `n=4` from `n=8` ("which group of 4" -- `n >> 2`) was getting thrown
away by `n*8 mod 32`. The fix rescues exactly that discarded information
and smuggles it back in through `k`: since `k` feeds directly into the
low bits of the address, XOR-ing a bit of "which group" information into
`k` means two threads that used to look identical to the bank
calculation (same `n mod 4`, same `k`) now look different, because their
effective `k` values differ.

## 19.5 Why `& 7` isn't just tidiness -- it prevents real memory corruption

`sB[0][n]` only has 8 valid slots (`k = 0..7`). Without `& 7`, for a
bigger `n` (say `n=100`), `n >> 2 = 25` -- and `k ^ 25` for `k=0` gives
`25`, way outside the valid 0-7 range for that row. That is not a
performance problem anymore, it is a genuine out-of-bounds write,
silently corrupting a different `n`'s data (since `sB[0]` is one
contiguous block of memory, writing to "slot 25" of row `n` actually
lands inside the data belonging to `n+3`). The `& 7` caps the mask to the
same `0..7` range `k` itself lives in, guaranteeing `k ^ mask` always
stays a valid index.

## 19.6 The two code snippets are the same fix, applied twice

```c
// (1) prologue -- the FIRST time this chunk's B data gets loaded:
int mask = (n >> 2) & 7;
sB[0][n][kk ^ mask] = (r < K && c < N) ? B[r * N + c] : 0.0f;

// (2) mid-loop refill -- loading the NEXT chunk's B data, into the other buffer:
int mask = (n >> 2) & 7;
sB[bufferId ^ 1][n][kk2 ^ mask] = (r < K && c2 < N) ? B[r * N + c2] : 0.0f;
```
Identical in spirit -- the only difference is which buffer (`sB[0]` vs
`sB[bufferId^1]`) and which chunk of K is being loaded (`kk` vs `kk2`,
just different variable names for the same role). Exactly like A's
prologue-load and mid-loop-reload code being near-duplicates of each
other, B needs the fix applied every single time new data gets written
into shared memory -- not just once -- because every fresh load is a
fresh opportunity for the same 32 threads to collide the same way.

---

# PART 20 -- Follow-up: Re-deriving "Step 0" Itself From Raw Code and Arithmetic

## 20.1 The real code behind "n" and "k" in the recap

```c
int tBx = tidx & (THREAD_B_LAYOUT_X - 1);   // THREAD_B_LAYOUT_X = 32, so tBx = tidx % 32
int tBy = tidx / THREAD_B_LAYOUT_X;         // tBy = tidx / 32

for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {   // THREAD_B_LAYOUT_Y = 8 = BLOCK_K, so i=0 only
    for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {   // j = 0, 32, 64, 96
        int n = j + tBx;
        int kk = i + tBy;   // since i=0, this is just tBy
        sB[0][n][kk] = ...
    }
}
```
"n" in the recap is `j + tBx`; "k" is `tBy`.

## 20.2 Claim 1: tBy is the same for all 32 threads in a warp -- verified by hand

`tBy = tidx / 32`, using integer division (any remainder is thrown away,
not rounded). A warp is 32 consecutive thread numbers. Take warp 0
(`tidx = 0` through `31`) and compute `tBy` for every one:
```
tidx:    0   1   2   3  ...  29  30  31
tBy:     0   0   0   0  ...   0   0   0     <- every single one gives 0!
```
Integer division by 32 only ticks up once the number actually reaches 32.
Since every `tidx` in warp 0 is strictly less than 32, dividing any of
them by 32 truncates down to exactly 0. Only at `tidx = 32` (the first
thread of the next warp) does `tBy` become 1:
```
tidx:    32  33  34  ...  63
tBy:      1   1   1  ...   1     <- warp 1: constant at a DIFFERENT value (1)
```
General rule: dividing any 32 consecutive numbers that start at a
multiple of 32 (0-31, 32-63, 64-95, ...) by 32 always gives the exact
same answer for all 32 of them. A warp is exactly 32 consecutive thread
numbers, so `tBy` is guaranteed constant for the whole warp (a different
constant per warp).

## 20.3 Claim 2: n sweeps 32 consecutive values -- verified by hand

`tBx = tidx & 31`, same as `tidx % 32`. For warp 0 (`tidx = 0..31`),
since every number is already less than 32, the remainder is the number
itself:
```
tidx:    0   1   2   3  ...  29  30  31
tBx:     0   1   2   3  ...  29  30  31     <- tBx = tidx exactly, for this warp
```
Every one of the 32 threads gets a different `tBx`, covering 0 through
31 with no repeats. `j` is a loop variable, the same value for every
thread at any given moment (every thread runs identical code in
lockstep -- SIMT -- sharing the same program counter). At `j=0`:
```
n = j + tBx = 0 + tBx = tBx = 0, 1, 2, 3, ..., 31
```
32 consecutive values, one per thread.

## 20.4 Claim 3: the collision repeats every 4 values of n -- the modular arithmetic, from scratch

`bank = address % 32`. Think of `% 32` like a 32-hour clock: adding
exactly 32 to any time lands back on the same hour (`5 o'clock + 32
hours = 5 o'clock again`, just a different day). Watch what happens to
the address when `n` increases by exactly 4 (holding `k=0` fixed, since
`tBy` is constant for the whole warp):
```
address(n+4) = (n+4)*8 + 0
             = n*8 + 4*8 + 0
             = n*8 + 32
             = address(n) + 32
```
Adding 4 to `n` always adds exactly 32 to the address -- and 32 is
precisely the "size of the clock" for `% 32`. Adding one full lap around
a 32-hour clock always brings you back to the same hour:
```
bank(n+4) = (address(n) + 32) % 32 = address(n) % 32 = bank(n)
```
`n` and `n+4` are mathematically guaranteed to land on the same bank, no
matter what `n` is -- forced by `8 * 4 = 32` exactly, not a coincidence
of the specific numbers in the table.

## 20.5 Putting the three pieces together: the flowchart

```
  tBy = tidx/32           tBx = tidx%32              8*4 = 32
  -------------           -------------              ---------
  CONSTANT across          Covers 0..31              Adding 4 to n
  one whole warp           across one warp            adds exactly
  (verified: division      (verified: remainder       one full "lap"
  truncates identically    of already-small           around the
  for 32 consecutive       numbers = themselves)       32-bank clock
  small numbers)                 |                          |
       |                          |                          |
       v                          v                          v
  k = tBy stays THE          n sweeps 32              bank(n) == bank(n+4)
  SAME for all 32            DIFFERENT values          for EVERY n
  threads simultaneously     simultaneously
       |                          |                          |
       +--------------------------+--------------------------+
                                   |
                                   v
         One warp, 32 threads, SAME k, 32 DIFFERENT n's,
         but n and n+4 always collide on the same bank
         -----> the 8-way conflict from the Part 18 recap table
```
Every piece of the original "what's broken" recap table -- `tBy` being
constant, `n` sweeping 0-31, and the collision repeating every 4 -- has
now been derived directly from the actual kernel code and basic
arithmetic, with nothing assumed.

---

# PART 21 -- Follow-up: Is THREAD_A_LAYOUT_Y=32 Really Matched to Warp Size?

## 21.1 The six constants, actual values

```c
constexpr int BLOCK_SIZE = 256;
constexpr int THREAD_A_LAYOUT_X = 8;                                 // chosen directly
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X;    // = 256/8  = 32  <- DERIVED, not chosen
constexpr int THREAD_B_LAYOUT_X = 32;                                // chosen directly  <-- picked to BE 32
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X;    // = 256/32 = 8
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X;    // = 256/16 = 16
```
`THREAD_A_LAYOUT_X` and `THREAD_B_LAYOUT_X` are the two directly-chosen
numbers (8 and 32). `THREAD_A_LAYOUT_Y` and `THREAD_B_LAYOUT_Y` are just
256 divided by whichever X was picked -- leftovers, not independent
decisions.

## 21.2 Background: what is coalescing, and why does it want 32?

This is a different performance concern than the shared-memory bank
conflict from Parts 17-20 -- this one is about global memory (the big,
slow, off-chip DRAM holding A and B). The rule: if all 32 threads of a
warp read 32 consecutive addresses in global memory at the same instant,
the hardware bundles them into one efficient transaction (32 threads x 4
bytes = 128 bytes, a clean cache-line-sized burst). If those 32 threads
scatter across far-apart addresses instead, the hardware needs many
separate, wasteful transactions. The warp size, 32, is the natural
"unit" for this optimization.

## 21.3 Tracing B's global read: THREAD_B_LAYOUT_X=32 is deliberate

```c
for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {   // j = 0, 32, 64, 96
    int c = col0 + j + tBx;
    B[r * N + c]        // the GLOBAL memory read
}
```
B is row-major, so consecutive `c` values are consecutive global
addresses. For warp 0 (`tBx = 0..31`) at `j=0`:
```
tBx:     0      1      2      3    ...    31
c:       col0   col0+1 col0+2 col0+3 ...  col0+31
address: B[r*N+col0], B[r*N+col0+1], ..., B[r*N+col0+31]   <- 32 CONSECUTIVE addresses!
```
All 32 threads of the warp read 32 back-to-back floats in one shot -- a
textbook-perfect coalesced transaction. This only works because
`THREAD_B_LAYOUT_X` was set to exactly 32: if it were smaller, fewer
threads would contribute to each contiguous burst. `THREAD_B_LAYOUT_X=32`
is a deliberate choice, made specifically because it equals the warp
size -- and it is also exactly the choice that, as an unavoidable side
effect, caused the shared-memory bank conflict from Parts 17-20.

## 21.4 Tracing A's global read: THREAD_A_LAYOUT_X=8 is forced, not chosen for warp-size reasons

```c
sA[0][i + tAy][tAx] = ... A[r * K + tAx] ...
```
`tAx` ranges only `0..7`, not `0..31`, because A's chunk only has 8 valid
columns to offer in the first place -- `BLOCK_K = 8` is the width of one
K-chunk, and `THREAD_A_LAYOUT_X` must equal `BLOCK_K` so every thread's
`tAx` lands on a real column (`tAx < K` enforces this). Bumping it to 32
would leave 24 of 32 threads with no real column to read -- there is no
9th through 32nd column in an 8-wide chunk. `THREAD_A_LAYOUT_X=8` is
forced by the algorithm's own tile shape, not a coalescing choice.

## 21.5 Where THREAD_A_LAYOUT_Y=32 actually comes from

It is the leftover: `256 total threads / 8 threads-per-row = 32 rows
covered per wave`. Critically, `THREAD_A_LAYOUT_Y` never appears anywhere
in the `tAx`/`tAy` formulas that drive A's addressing:
```c
int tAx = tidx & (THREAD_A_LAYOUT_X - 1);   // uses THREAD_A_LAYOUT_X only
int tAy = tidx / THREAD_A_LAYOUT_X;         // uses THREAD_A_LAYOUT_X only
```
It only shows up in the outer loop stride (`i += THREAD_A_LAYOUT_Y`),
controlling how many passes it takes to cover `BLOCK_M=128`. It has zero
effect on which global address or shared-memory bank any thread touches.
If `BLOCK_SIZE` had instead been 128, `THREAD_A_LAYOUT_Y` would come out
to 16 instead of 32 -- and A's bank-conflict-free property would hold
exactly as before, completely unaffected, because that property only
depends on `THREAD_A_LAYOUT_X (8)` dividing evenly into the hardware's
fixed warp size `(32)`, a relationship `THREAD_A_LAYOUT_Y` plays no part in.

## 21.6 The verdict, side by side

```
THREAD_B_LAYOUT_X = 32            THREAD_A_LAYOUT_Y = 32
-----------------------           -----------------------
DIRECTLY chosen, on purpose       DERIVED: 256 / 8, leftover
Picked BECAUSE it equals the      Picked because BLOCK_SIZE=256
warp size -- maximizes global     and BLOCK_K=8 were each chosen
coalescing for B's wide           for their OWN separate reasons;
(128-column) rows                 32 falls out automatically
       |                                  |
       v                                  v
Causes the shared-memory          Has NO EFFECT on any bank or
bank conflict on B's WRITE        coalescing calculation -- doesn't
(Parts 17-20) as a side           even appear in the tAx/tAy formulas
effect of this same choice
       |                                  |
       v                                  v
   genuinely "by design,              coincidentally equal to 32,
   for warp-size reasons"             but NOT "by design" for
                                       warp-size reasons
```
Short answer: `THREAD_B_LAYOUT_X=32` -- yes, genuinely by design, chosen
specifically to be the warp size for global-memory coalescing.
`THREAD_A_LAYOUT_Y=32` -- no, it only looks like the same kind of choice;
it is an arithmetic byproduct of two unrelated, forced decisions
(`BLOCK_SIZE=256`, `BLOCK_K=8`), and it plays no role in either A's
coalescing or its bank-conflict-free behavior.

---

# PART 22 -- Follow-up: Why Not Just Set BLOCK_K=32 and Every THREAD_*_LAYOUT_X to 32?

## 22.1 Problem 1 (the showstopper): 1024 threads/block collides head-on with register limits

CUDA's hard ceiling is exactly 1024 threads per block. Every SM has a
fixed, small register file -- a figure that has been a common constant
across many NVIDIA architectures: 65,536 (64K) 32-bit registers per SM.
matmul_v2.cu's real measured register usage (`nvcc -Xptxas -v`) is 168
registers per thread. The arithmetic for the proposed `BLOCK_SIZE=1024`:
```
registers needed for ONE block = threads_per_block x registers_per_thread
                                = 1024 x 168
                                = 172,032 registers

registers available on ONE SM  = 65,536 registers

172,032 / 65,536 = 2.6x MORE than the entire SM's register file
```
A single block alone would need 2.6x more registers than physically
exist on one SM. The kernel cannot launch like this -- CUDA would reject
it outright ("too many resources requested for launch"), or the compiler
would be forced to violently slash register usage (spilling the overflow
into local memory, which is slow global memory in disguise), destroying
the entire point of a register-tuned kernel.

## 22.2 Problem 2 (the subtler trap): bumping to 32 everywhere doesn't fix B -- it makes B worse

Setting the register catastrophe aside, trace both A and B with the new
numbers.

### A, with the new numbers: genuinely gets BETTER
```c
// sA[BLOCK_M][BLOCK_K] = sA[128][32]  (K-dimension now 32, matches THREAD_A_LAYOUT_X)
sA[0][i + tAy][tAx] = A[r * K + tAx];
```
For warp 0: `tAy = tidx/32 = 0` (constant). `tAx = tidx` (full `0..31`).
Address = `m*32 + tAx` where `m = 0` (constant): `address = tAx`.
`bank = tAx % 32 = tAx` -- every thread gets a different bank, zero
conflict. This part genuinely works.

### B, with the SAME new numbers: dramatically WORSE
```c
// sB[BLOCK_N][BLOCK_K] = sB[128][32]  (K-dimension now 32)
sB[0][n][kk] = B[r * N + c];     // n = j+tBx, kk = i+tBy
```
For warp 0: `tBy = tidx/32 = 0` (constant). `tBx = tidx` (full `0..31`),
so `n = tBx` (`0..31`). Address = `n*32 + kk = n*32 + 0 = n*32`:
```
n=0: address=0*32=0,   bank = 0 % 32 = 0
n=1: address=1*32=32,  bank = 32 % 32 = 0   <- SAME bank as n=0!
n=2: address=2*32=64,  bank = 64 % 32 = 0   <- SAME bank!
...
n=31: address=31*32=992, bank = 992 % 32 = 0  <- STILL bank 0!
```
All 32 threads in the warp land on bank 0 at the same time -- a full
32-way conflict, the worst possible case. Before, with `BLOCK_K=8`, it
was only an 8-way conflict. Setting `BLOCK_K=32` to match the warp size
doesn't fix B -- it turns every single row-stride step into an exact,
complete lap around the 32-bank "clock," collapsing all 32 threads onto
one bank instead of spreading them across four.

## 22.3 Why the SAME change helps A but hurts B -- the real structural reason

It comes down to which index is varying across the warp, and whether
that index gets added directly to the address or multiplied by the
row-stride first:
```
A's situation:                         B's situation:
----------------                       ----------------
Within a warp, tAy (-> m, the          Within a warp, tBy (-> k, the
OUTER/row index) is CONSTANT.          INNER/column index) is CONSTANT.
tAx (-> k, the INNER/column            tBx (-> n, the OUTER/row index)
index) is what VARIES.                 is what VARIES.

k is ADDED directly into the           n is MULTIPLIED by the row-stride
address (never multiplied) --          before being added -- so n's
so its full range (0..31) shows        variation gets scaled up, and
up directly as 32 different            whenever the stride is itself a
banks, no matter the stride.           multiple of 32, ALL of that
                                        variation gets wiped out by % 32.
```

## 22.4 The flowchart

```
          FLOWCHART: why "just use 32 everywhere" isn't a free lunch
          ------------------------------------------------------------
                    Which index varies across one warp?
                              |
            +-----------------+-----------------+
            |                                    |
     The INNER/column index               The OUTER/row index
     (added directly, e.g. A's k)         (multiplied by stride, e.g. B's n)
            |                                    |
            v                                    v
    Bigger stride (match it to 32)      Bigger stride that's an exact
    spreads threads across MORE         multiple of 32 makes EVERY row
    distinct banks -> HELPS             change a full "lap" -> collapses
                                         ALL threads onto ONE bank -> HURTS
```
The bank-conflict outcome isn't about whether some constant equals 32 --
it's a structural property of which matrix layout (natural vs.
transposed) puts which index in the "varies-across-a-warp" seat. A
happens to have the index-that-varies in the safe (inner, added)
position; B has it in the dangerous (outer, multiplied) position -- an
assignment that comes from B being stored transposed while A is stored
natural. No single magic constant chosen uniformly for `BLOCK_K` and
every `THREAD_X_LAYOUT_X` can fix both at once, because they need
opposite things: A wants a bigger stride, B wants a stride that does NOT
evenly divide 32. Combined with the register-budget catastrophe from
Section 22.1, this "make everything 32" redesign is a non-starter twice
over.

---

# PART 23 -- Follow-up: Why Does v0's Swizzle Shift LEFT While v2_bfix's Shifts RIGHT and Clamps?

## 23.1 The two swizzles, side by side

```c
// matmul_v0.cu -- A is stored sA[2][BLOCK_K][BLOCK_M] = sA[2][8][128]
//                 K is OUTER (8 values), M is INNER (128 values)
sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] = ...;
//                ^^^   ^^^^^^^^^^^^^^^^^^^^^^
//                K-pos    M-pos, XORed with (K-pos << 2)

// matmul_v2_bfix.cu -- B is stored sB[2][BLOCK_N][BLOCK_K] = sB[2][128][8]
//                      N is OUTER (128 values), K is INNER (8 values)
int mask = (n >> 2) & 7;
sB[0][n][kk ^ mask] = ...;
//    ^   ^^^^^^^^^
//   N-pos   K-pos, XORed with ((N-pos >> 2) & 7)
```
Both follow the exact same recipe: take the outer index, derive a mask
from it, and XOR that mask into the inner index. The difference is
entirely in how the mask gets derived -- and that comes down to one
fact: in A, the outer index (K) is the small one (0-7) and the inner
index (M) is the big one (0-127). In B, it is flipped -- the outer index
(N) is big (0-127) and the inner index (K) is small (0-7).

## 23.2 Why A shifts LEFT (`<< 2`, no clamp needed)
```
A's situation:   outer = K (small, 0..7)      inner = M (big, 0..127)
```
The mask needs to land somewhere inside M's big 0-127 range. Taking the
small K value (0-7) and shifting it LEFT by 2 (multiplying by 4)
stretches it to `0, 4, 8, ..., 28` -- still comfortably inside `0..127`,
so no extra clamping is ever needed:
```
tAx (K-pos):   0   1   2   3   4   5   6   7
tAx << 2:      0   4   8  12  16  20  24  28     <- all safely within M's 0..127 range
```

## 23.3 Why B shifts RIGHT and clamps (`>> 2`, then `& 7`)
```
B's situation:   outer = N (big, 0..127)      inner = K (small, 0..7)
```
Now it is reversed: the mask needs to land inside K's tiny 0-7 range, but
the outer value (N) can be as large as 127. Shifting N RIGHT by 2
(dividing by 4) brings it down to `0..31` -- still too big for a 0-7
slot -- so `& 7` clamps it the rest of the way:
```
n (N-pos):      0   1   2   3   4  ...  28  29  30  31  ... 124 125 126 127
n >> 2:         0   0   0   0   1  ...   7   7   7   7  ...  31  31  31  31
(n >> 2) & 7:   0   0   0   0   1  ...   7   7   7   7  ...   7   7   6   7
                                                           (wraps back into 0..7)
```
Without `& 7`, a large `n` like `100` would give `mask = 25` -- XOR-ing
`25` into `kk` (only 8 valid slots, `0..7`) produces something way
outside the array, corrupting a neighboring row's data. This is the same
correctness point from Part 19, now explained structurally: A's mask
never needs clamping because its small outer value, even scaled UP,
still fits the big inner range. B's mask always needs clamping because
its big outer value, even scaled DOWN, is still too big for the small
inner range.

## 23.4 The general rule, as a flowchart
```
            Look at the array's shape: [OUTER][INNER]
                           |
           Which one is numerically SMALLER?
                 |                        |
          OUTER is smaller          OUTER is bigger
          (A's case: K=8            (B's case: N=128
           outer, M=128 inner)       outer, K=8 inner)
                 |                        |
                 v                        v
       Shift outer LEFT to          Shift outer RIGHT to
       stretch it UP into the       shrink it DOWN toward
       big inner range                   the small inner range
       (mask = outer << 2)          ...then CLAMP it the rest
                 |                   of the way with & (inner_size - 1)
                 v                   (mask = (outer >> 2) & 7)
       No clamp needed --                   |
       result always fits                   v
       inside inner's range          Clamp is MANDATORY --
                                      without it: out-of-bounds
                                      write, real memory corruption
```
Bottom line: the two formulas are not different techniques -- they are
the same single idea (XOR a mask derived from the outer index into the
inner index) applied correctly to two mirror-image situations. A has a
small outer / big inner shape, so it scales its mask up and never needs a
safety clamp. B (deliberately transposed for global-memory coalescing,
Part 21) has a big outer / small inner shape, so it must scale its mask
down and clamp it, or risk real memory corruption.

---

# PART 24 -- Follow-up: Why `tCsA[(tk - 1) & 1]`, Not `tCsA[tk & 1]`?

## 24.1 The full loop, annotated
```c
for (int tk = 0; tk < BLOCK_K + 1; ++tk) {        // tk runs 0,1,2,...,8 (9 iterations, BLOCK_K=8)
    if (tk > 0) {
        // COMPUTE: uses data loaded in a PREVIOUS iteration
        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
    }
    if (tk < BLOCK_K) {
        // LOAD: stages THIS iteration's fresh k-data into registers
        FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][...]);
    }
}
```
Two independent conditions control two independent jobs each iteration:
compute (skipped only on the very first iteration, `tk=0`) and load
(skipped only on the very last iteration, `tk=8`). The load always writes
to buffer `tk & 1`. The compute always reads from buffer `(tk-1) & 1`.

## 24.2 Why `(tk-1)&1`? Because that's literally where the data was written

The load line writes k's data into buffer `tk & 1` during iteration `tk`.
To read that data back later, the compute step must use the exact same
buffer index that was used when it was written -- which was computed
using that k's own `tk` value at that time, not the current iteration's
`tk`. Since the compute for "k's data" always happens one full iteration
after the load for "k's data," the compute step must look back and
compute `tk-1` to re-derive which buffer that was.

## 24.3 The full trace, k by k -- nothing skipped

| tk | Load action (writes `tk & 1`) | Compute action (reads `(tk-1) & 1`) |
|---|---|---|
| 0 | buffer 0 <- k=0's data | (skipped -- nothing loaded yet) |
| 1 | buffer 1 <- k=1's data | buffer `(1-1)&1`=0 -> correctly gets k=0's data |
| 2 | buffer 0 <- k=2's data (safe, k=0 already used) | buffer `(2-1)&1`=1 -> correctly gets k=1's data |
| 3 | buffer 1 <- k=3's data | buffer `(3-1)&1`=0 -> correctly gets k=2's data |
| 4 | buffer 0 <- k=4's data | buffer `(4-1)&1`=1 -> correctly gets k=3's data |
| 5 | buffer 1 <- k=5's data | buffer `(5-1)&1`=0 -> correctly gets k=4's data |
| 6 | buffer 0 <- k=6's data | buffer `(6-1)&1`=1 -> correctly gets k=5's data |
| 7 | buffer 1 <- k=7's data | buffer `(7-1)&1`=0 -> correctly gets k=6's data |
| 8 | (skipped -- all 8 already loaded) | buffer `(8-1)&1`=1 -> correctly gets k=7's data (the last one!) |

Every single compute step reads exactly the k-value that was loaded one
iteration earlier -- proven by tracing all 9 iterations, no gaps, no
repeats.

## 24.4 What would break if you used `tk` instead of `tk-1`

At `tk=1`: compute would read buffer `1&1 = 1`. But buffer 1 is being
written by the load that happens later in this same iteration (the load
line comes after the compute line in the code). At the moment the
compute executes, buffer 1 either still holds garbage (never written) or
stale leftover data from two iterations ago -- reading it now would
multiply the wrong numbers together, silently corrupting the result.
Using `tk-1` guarantees "only ever read a buffer that has already been
fully written in a prior iteration."

## 24.5 The pipeline diagram -- why bother with this complexity at all
```
iteration:    tk=0      tk=1      tk=2      tk=3      tk=4     ...
LOAD k:        0    ->   1    ->   2    ->   3    ->   4    -> ...
COMPUTE k:          (none)   0    ->   1    ->   2    ->   3   -> ...
                       ^
              staggered by exactly one step: load k+1 while
              computing with k, so the ALU is never left
              waiting on a fresh load
```
This is the same idea as classic CPU instruction pipelining ("fetch the
next instruction while executing the current one"). Shared-memory reads
are not instant -- they take real cycles. By issuing the load for k's
data a full iteration before it is actually needed, the hardware has time
to service that read in the background while the ALU stays busy
multiplying the previous k's values -- by the time the next iteration
needs that data, it has already arrived. `tk=0` only loads (the pipeline
is still "filling"), and `tk=8` only computes (no more data left to load
-- the pipeline is "draining" its last value). The `-1` is bookkeeping to
always point at "the data staged one step ago," which is what makes the
whole overlap scheme correct instead of reading uninitialized or stale
registers.

---

# PART 25 -- simulation_v0.py: a Full Per-Thread Index Table for matmul_v0.cu

## 25.1 The request and the preset values

A standalone, pure-Python, no-GPU-needed simulation of matmul_v0.cu's
entire per-thread index arithmetic, for one concrete thread, with every
value verified by actually running the script (not hand-typed separately):
```
M=256 N=256 K=16
BLOCK_M=128 BLOCK_N=128 BLOCK_K=8 BLOCK_SIZE=256
THREAD_A_LAYOUT_X=8   THREAD_A_LAYOUT_Y=32
THREAD_B_LAYOUT_X=32  THREAD_B_LAYOUT_Y=8
THREAD_C_LAYOUT_X=16  THREAD_C_LAYOUT_Y=16
THREAD_C_X_TILE_SIZE=8  THREAD_C_Y_TILE_SIZE=8
THREAD_C_WARP_X=8  THREAD_C_WARP_Y=4  THREAD_C_WARP_DIM_X=2
tidx = 13
```
blockIdx was not given, so block (0,0) is assumed (row0=0, col0=0).

## 25.2 Fixed per-thread values, computed once

```c
int tAx = tidx & (THREAD_A_LAYOUT_X - 1);   // 13 & 7  = 5
int tAy = tidx / THREAD_A_LAYOUT_X;         // 13 / 8  = 1
int tBx = tidx & (THREAD_B_LAYOUT_X - 1);   // 13 & 31 = 13
int tBy = tidx / THREAD_B_LAYOUT_X;         // 13 / 32 = 0
```

| variable | row0 | col0 | tAx | tAy | tBx | tBy | warpId | laneId | warpx | warpy | lanex | laney | tCx | tCy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 0 | 0 | 5 | 1 | 13 | 0 | 0 | 13 | 0 | 0 | 6 | 1 | 6 | 1 |

## 25.3 PROLOGUE: loading chunk 0 of A and B

```c
sA[0][tAx][(i + tAy) ^ (tAx << 2)] = A[r * K + tAx];
```

| i | r=row0+i+tAy | tAx | (i+tAy)^(tAx<<2) | sA write target |
|---|---|---|---|---|
| 0 | 1 | 5 | 21 | sA[0][5][21] |
| 32 | 33 | 5 | 53 | sA[0][5][53] |
| 64 | 65 | 5 | 85 | sA[0][5][85] |
| 96 | 97 | 5 | 117 | sA[0][5][117] |

| i | j | r=i+tBy | c=col0+j+tBx | sB write target |
|---|---|---|---|---|
| 0 | 0 | 0 | 13 | sB[0][0][13] |
| 0 | 32 | 0 | 45 | sB[0][0][45] |
| 0 | 64 | 0 | 77 | sB[0][0][77] |
| 0 | 96 | 0 | 109 | sB[0][0][109] |

## 25.4 MAIN LOOP SCAFFOLDING for k=8 -- the pipeline bookkeeping, real values

| k | tk | compute?(tk>0) | load?(tk<8) | tk-1 | (tk-1)&1 | tk&1 | bufferId |
|---|---|---|---|---|---|---|---|
| 8 | 0 | False | True | - | - | 0 | 0 |
| 8 | 1 | True | True | 0 | 0 | 1 | 0 |
| 8 | 2 | True | True | 1 | 1 | 0 | 0 |
| 8 | 3 | True | True | 2 | 0 | 1 | 0 |
| 8 | 4 | True | True | 3 | 1 | 0 | 0 |
| 8 | 5 | True | True | 4 | 0 | 1 | 0 |
| 8 | 6 | True | True | 5 | 1 | 0 | 0 |
| 8 | 7 | True | True | 6 | 0 | 1 | 0 |
| 8 | 8 | True | False | 7 | 1 | - | 0 |

`bufferId` stays 0 the entire time -- it only flips once, after this whole
`tk` loop finishes.

## 25.5 A register-load for k=8 (the swizzle in action, real tk values)

```c
int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
```

| tk | tm_reg | r | tk<<2 | r^(tk<<2) | sA read source |
|---|---|---|---|---|---|
| 0 | 0 | 4 | 0 | 4 | sA[0][0][4] |
| 0 | 1 | 68 | 0 | 68 | sA[0][0][68] |
| 1 | 0 | 4 | 4 | 0 | sA[0][1][0] |
| 1 | 1 | 68 | 4 | 64 | sA[0][1][64] |
| 2 | 0 | 4 | 8 | 12 | sA[0][2][12] |
| 2 | 1 | 68 | 8 | 76 | sA[0][2][76] |

(continues the same pattern through tk=7 -- full 16-row table in the
script's real output)

## 25.6 The ACCUMULATE grid -- an X/Y table, tk=1

```c
acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
```
At tk=1, buffer = (1-1)&1 = 0:

| tm \ tn | 0 | 1 | ... | 7 |
|---|---|---|---|---|
| 0 | tCsA[0][0]*tCsB[0][0] | tCsA[0][0]*tCsB[0][1] | ... | tCsA[0][0]*tCsB[0][7] |
| 1 | tCsA[0][1]*tCsB[0][0] | ... | ... | ... |
| ... | | | | |
| 7 | tCsA[0][7]*tCsB[0][0] | ... | ... | tCsA[0][7]*tCsB[0][7] |

## 25.7 EPILOGUE: the final 8x8 write-back grid, fully verified

```c
int r = row0 + (tCy << 2) + (i & ~3) * THREAD_C_LAYOUT_Y + (i & 3);
int c = col0 + (tCx << 2) + (j & ~3) * THREAD_C_LAYOUT_X + (j & 3);
```

| i \ j | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | C[4,24] | C[4,25] | C[4,26] | C[4,27] | C[4,88] | C[4,89] | C[4,90] | C[4,91] |
| 1 | C[5,24] | C[5,25] | C[5,26] | C[5,27] | C[5,88] | C[5,89] | C[5,90] | C[5,91] |
| 2 | C[6,24] | C[6,25] | C[6,26] | C[6,27] | C[6,88] | C[6,89] | C[6,90] | C[6,91] |
| 3 | C[7,24] | C[7,25] | C[7,26] | C[7,27] | C[7,88] | C[7,89] | C[7,90] | C[7,91] |
| 4 | C[68,24] | ... | ... | ... | C[68,88] | ... | ... | C[68,91] |

Confirms, with real numbers, the exact "4 consecutive, jump 64, 4 more
consecutive" pattern established back in Parts 13-15 -- rows land on
{4,5,6,7,68,69,70,71}, columns on {24,25,26,27,88,89,90,91}.

## 25.8 The flowchart of every phase

```
PROLOGUE (once)          MAIN LOOP (k=8, then k=16)            EPILOGUE (once)
----------------         ---------------------------           ----------------
Load chunk 0 of A/B       For EACH k:                           Write this thread's
into buffer 0                |                                  8x8 acc[][] grid
     |                       v                                  into real C[r][c]
     v                  tk-loop (9 steps):                      positions, using
  (feeds into the         - tk=0: load only                     the i&3/i&~3
   first k=8                - tk=1..7: load AND compute          bit-trick grouping
   iteration below)            (compute uses (tk-1)&1,
                                load writes tk&1)
                              - tk=8: compute only
                                |
                                v
                          if k<K: reload NEXT chunk into
                          the OTHER buffer (bufferId^1)
                                |
                                v
                          bufferId ^= 1 (flip for next k)
```

---

# PART 26 -- simulation_v1.py: One Continuous, Execution-Order Log

## 26.1 Why this is a different kind of simulation than Part 25's

simulation_v0.py split the trace into separate tables per phase.
simulation_v1.py instead builds ONE continuous, unified table, logged in
the EXACT sequence the real CUDA source executes, so the chain
`global A/B -> sA/sB -> tCsA/tCsB -> acc -> C` can be watched unfold
without any reordering or grouping for presentation.

## 26.2 The real, unified output (excerpt)

```
step | phase           | k | tk | tm | tn | tk-1 | (tk-1)&1 | tk&1 | bufferId | source                | destination   | in_bounds
1    | PROLOGUE_LOAD_A | - | -  | -  | -  | -    | -        | -    | -        | A[1][5]               | sA[0][5][21]  | True
...
10   | LOAD_A_REGISTER | 8 | 0  | 0  | -  | -    | -        | 0    | 0        | sA[0][0][4]           | tCsA[0][0:3]  | -
11   | LOAD_A_REGISTER | 8 | 0  | 1  | -  | -    | -        | 0    | 0        | sA[0][0][68]          | tCsA[0][4:7]  | -
12   | LOAD_B_REGISTER | 8 | 0  | -  | 0  | -    | -        | 0    | 0        | sB[0][0][24]          | tCsB[0][0:3]  | -
13   | LOAD_B_REGISTER | 8 | 0  | -  | 1  | -    | -        | 0    | 0        | sB[0][0][88]          | tCsB[0][4:7]  | -
14   | COMPUTE         | 8 | 1  | 0  | 0  | 0    | 0        | -    | 0        | tCsA[0][0]*tCsB[0][0] | acc[0][0]     | -
...  (64 rows of COMPUTE for tk=1, using buffer 0)
78   | LOAD_A_REGISTER | 8 | 1  | 0  | -  | -    | -        | 1    | 0        | sA[0][1][0]           | tCsA[1][0:3]  | -
...  (LOAD for tk=1 happens AFTER its own compute, feeding buffer 1)
82   | COMPUTE         | 8 | 2  | 0  | 0  | 1    | 1        | -    | 0        | tCsA[1][0]*tCsB[1][0] | acc[0][0]     | -
```

This proves the single most important fact about the real control flow:
at tk=1, COMPUTE (steps 14-77) happens entirely BEFORE LOAD_A_REGISTER/
LOAD_B_REGISTER (steps 78-81) -- matching the real code's statement order
(`if (tk>0) {compute}` written before `if (tk<BLOCK_K) {load}`), and
matching the (tk-1)&1 pipelining logic from Part 24. Then tk=2's compute
correctly consumes buffer 1 (the data tk=1 just loaded), proving the
pipeline stagger is real, not just theoretical.

## 26.3 The full A->B->C data chain, visible in the log's own columns

```
PROLOGUE_LOAD_A:  source="A[1][5]"           destination="sA[0][5][21]"
                           |
LOAD_A_REGISTER:  source="sA[0][0][4]"       destination="tCsA[0][0:3]"
                           |
COMPUTE:          source="tCsA[0][0]*tCsB[0][0]"   destination="acc[0][0]"
                           |
EPILOGUE_WRITE:   source="acc[0][0]"         destination="C[1][5]"
```
Every row's source/destination columns literally spell out one hop of
the chain -- any specific sA[...] or tCsA[...] location can be traced
forward or backward through the entire 1,172-row log to see exactly
which global A/B element it came from and which C element it lands in.

## 26.4 The flowchart of one tk iteration's logged rows

```
tk-loop iteration, in REAL code order:
  -----------------------------------
  if tk > 0:
      log 64 COMPUTE rows  (8x8 tm,tn grid, reading buffer (tk-1)&1)
  if tk < BLOCK_K:
      log 2 LOAD_A_REGISTER rows  (tm=0,1 -- writes buffer tk&1)
      log 2 LOAD_B_REGISTER rows  (tn=0,1 -- writes buffer tk&1)

  tk=0: only LOAD runs (4 rows)       -- pipeline "filling"
  tk=1..7: COMPUTE (64) then LOAD (4) -- 68 rows each, in that order
  tk=8: only COMPUTE runs (64 rows)   -- pipeline "draining"
```
Across both k chunks, plus prologue (8 rows), reload (8 rows), two
SYNCTHREADS/BUFFER_FLIP markers, and the epilogue (64 rows): 1,172 total
rows -- the complete, line-for-line trace of everything one thread does,
in the exact sequence the real GPU executes it.

---

# PART 27 -- Follow-up: Formatting the source Column as A[][]/B[][]

## 27.1 The change

All four global-memory source constructions in simulation_v1.py
(PROLOGUE_LOAD_A, PROLOGUE_LOAD_B, RELOAD_A, RELOAD_B) switched from a
flat single-index format to a 2D-looking format, and EPILOGUE_WRITE's
destination was updated to match for consistency:
```python
# Before:
source = "A[%d]" % (r * K + tAx) if in_bounds else "ZERO_PAD"

# After:
source = "A[%d][%d]" % (r, tAx) if in_bounds else "ZERO_PAD"
```

## 27.2 Before vs after, verified from a real re-run

| step | phase | old source | new source |
|---|---|---|---|
| 1 | PROLOGUE_LOAD_A | A[21] | A[1][5] |
| 2 | PROLOGUE_LOAD_A | A[533] | A[33][5] |
| 5 | PROLOGUE_LOAD_B | B[13] | B[0][13] |
| 6 | PROLOGUE_LOAD_B | B[45] | B[0][45] |

```
     Why this is a better read at a glance:
     ----------------------------------------
     A[21]        <- is this row 21? column 21? you'd have to
                     mentally divide by K=16 to find out (21/16=1 r1, rem 5)
     A[1][5]      <- immediately readable: row 1, column 5 -- no
                     mental division needed, matches how A is
                     conceptually a 2D matrix even though it's
                     physically a flat 1D buffer underneath
```
Re-ran the full script end to end (1,172 rows, exit code 0, identical
row count and execution order) -- only the textual formatting of the
source/destination strings changed; every underlying index value, loop
order, and buffer/pipeline logic is untouched.

---

# PART 28 -- simulation_v2.py: Reverse-Engineering One Target C Element's Lineage

## 28.1 The question this answers

Given only a target output, e.g. C[4][24], which grid block, which
owning thread, and which OTHER cooperating threads loaded the relevant
A/B data into shared memory? This requires working BACKWARD through
four layers of memory, discovering (not assuming) which thread did what.

## 28.2 Step 1 and 2: which block, which thread, by brute force

```python
def find_consumer(row0, col0):
    for tidx in range(BLOCK_SIZE):               # try all 256 threads
        tCx, tCy = ...
        for i in range(THREAD_C_Y_TILE_SIZE):     # try all 8x8 = 64 epilogue cells
            for j in range(THREAD_C_X_TILE_SIZE):
                r, c = <exact epilogue formula>    # the REAL formula, unmodified
                if (r, c) == (TARGET_C_ROW, TARGET_C_COL):
                    return tidx, i, j              # found it -- guaranteed correct
```
Result: tidx=13, tm=0, tn=0. Rather than inverting the epilogue's
bit-twiddling formula algebraically (risky), the script tries every
possibility and checks against the real formula directly.

## 28.3 Step 3: the complete lineage, one row per real k-value

```
step | real_k | load_phase       | A_writer_tidx | A_element | sA_target    | B_writer_tidx | B_element | sB_target    | consumer_tidx | multiply_term
1    | 0      | PROLOGUE         | 32            | A[4][0]   | sA[0][0][4]  | 24            | B[0][24]  | sB[0][0][24] | 13            | tCsA[0][0]*tCsB[0][0]
...
8    | 7      | PROLOGUE         | 39            | A[4][7]   | sA[0][7][24] | 248           | B[7][24]  | sB[0][7][24] | 13            | ...
9    | 8      | RELOAD@k_outer=8 | 32            | A[4][8]   | sA[1][0][4]  | 24            | B[8][24]  | sB[1][0][24] | 13            | ...
...
16   | 15     | RELOAD@k_outer=8 | 39            | A[4][15]  | sA[1][7][24] | 248           | B[15][24] | sB[1][7][24] | 13            | ...
```
A_writer_tidx and B_writer_tidx are threads 32-39 and 24-248 -- NOT
thread 13. This is the key revelation: thread 13 (which owns C[4][24])
never loads its own ingredients from global memory -- a completely
different set of threads (assigned via the independent tAx/tAy, tBx/tBy
mapping) cooperatively fill shared memory for the whole block, and
thread 13 just reads whatever ended up in the right spot, using its own
tCx/tCy identity.

## 28.4 The self-verifying result

The 16 A_element entries are exactly A[4][0] through A[4][15], and the
16 B_element entries are exactly B[0][24] through B[15][24] -- precisely
the mathematical definition `C[4][24] = sum(A[4][k] * B[k][24] for k in
0..15)`, discovered purely by brute-force search, not hardcoded.

## 28.5 The flowchart

```
   TARGET: C[4][24]
        |
        v
   STEP 1: brute-force -> blockIdx(0,0), row0=0, col0=0
        |
        v
   STEP 2: brute-force over 256 threads x 64 epilogue cells
        -> tidx=13 owns acc[0][0] = C[4][24]
        |
        v
   STEP 3: for each real_k = 0..15:
        +--> brute-force: WHO wrote sA[buf][tk][swizzled_idx]? -> A_writer_tidx
        +--> brute-force: WHO wrote sB[buf][tk][idx]? -> B_writer_tidx
        +--> thread 13 (the CONSUMER) stages both into tCsA/tCsB, multiplies
        v
   STEP 4: sum all 16 multiply_terms -> acc[0][0] -> C[4][24]
```

---

# PART 29 -- matmul_v2_bfix_simulation_v0.py: the Unified Log for a Structurally Different Kernel

## 29.1 Why this kernel's trace looks different from matmul_v0.cu's

```
matmul_v0.cu                          matmul_v2_bfix.cu
-------------                         -------------------
A stored TRANSPOSED, swizzled          A stored NATURAL, NO swizzle at all
  (tAx<<2 XOR trick)                     (sA[0][i+tAy][tAx] = A[...] directly)
B stored NATURAL, no swizzle           B stored TRANSPOSED, MASK-swizzled
                                          (mask = (n>>2)&7, XORed into k-index)
Register-level PIPELINING:             NO pipelining: caches a thread's WHOLE
  tk-loop, (tk-1)&1 buffer trick         BLOCK_K-row of A/B into regA/regB
  (one real k-value per compute step)    all at once, then a full dot-product
Epilogue uses "groups of 4" bit-trick  Epilogue is a simple, direct
  (i&3 / i&(~3))                         tm*LAYOUT_Y / tn*LAYOUT_X mapping
```

## 29.2 The real, unified output (excerpt) -- the mask descramble, concretely

```
step | phase            | k | i  | j  | n   | mask | source                | destination    | in_bounds
5    | PROLOGUE_LOAD_B  | - | 0  | 0  | 13  | 3    | B[0][13]              | sB[0][13][3]   | True
...
74   | STAGE_B_REGISTER | 8 | -  | -  | 6   | 1    | sB[0][6][1]           | regB[0][0]     | -
75   | STAGE_B_REGISTER | 8 | -  | -  | 6   | 1    | sB[0][6][0]           | regB[0][1]     | -
76   | STAGE_B_REGISTER | 8 | -  | -  | 6   | 1    | sB[0][6][3]           | regB[0][2]     | -
77   | STAGE_B_REGISTER | 8 | -  | -  | 6   | 1    | sB[0][6][2]           | regB[0][3]     | -
...
138  | COMPUTE          | 8 | -  | -  | -   | -    | regA[0][0]*regB[0][0] | acc[0][0]      | -
...
145  | COMPUTE          | 8 | -  | -  | -   | -    | regA[0][7]*regB[0][7] | acc[0][0]      | -
```
Rows 74-77 are the clearest possible demonstration of the mask
descramble: for n=6, mask=1, requesting regB[0][0] reads from physical
slot 0^1=1, requesting regB[0][1] reads from 1^1=0 -- a clean pairwise
swap of adjacent slots, recovering the true logical order from the
scrambled physical layout, exactly as the hand-derived math predicted.

## 29.3 The flowchart of one k-chunk in this kernel

```
STAGE_A_REGISTER (64 rows: 8 tm x 8 kk, straight copy, no swizzle)
        |
        v
STAGE_B_REGISTER (64 rows: 8 tn x 8 kk, each individually un-masked)
        |
        v
COMPUTE (512 rows: 8 tm x 8 tn x 8 kk -- a FULL dot-product per
         (tm,tn) pair, all 8 real k-contributions summed at once,
         unlike v0.cu's one-k-value-per-tk-iteration pipelining)
        |
        v
RELOAD (if more chunks remain) -> BUFFER_FLIP -> next k
```
Total: 1,374 logged rows (slightly more than simulation_v1.py's 1,172,
since STAGE_B_REGISTER is expanded per-kk to make the mask swizzle fully
visible), covering the complete chain for this thread, in exact real-
kernel execution order.

## 29.4 An honest note: a real file-persistence issue encountered while building this

While building this specific file, it was created and verified running
successfully (exit code 0, correct row count, correct content) -- then
found completely missing from disk on the very next check, with no
trace, no error, and no active process touching the directory. This
happened TWICE for this exact file, while three other simulation files
in the same directory (simulation_v0.py, simulation_v1.py,
simulation_v2.py) stayed intact throughout. No root cause was found
from inside the sandbox (checked for background jobs, running
processes, naming collisions with the compiled matmul_v2_bfix binary --
none found). The file was recreated and, as a mitigation, backed up to
two additional locations outside the working directory with matching
MD5 checksums. This is recorded here plainly, per this document's
established practice of not hiding unresolved issues.

---

# PART 30 -- matmul_v2_bfix_simulation_v1.py: Reverse Lineage for the Mask-Swizzled Kernel

## 30.1 Why the writer-search logic had to change for this kernel

```
matmul_v0.cu's find_A_writer:          matmul_v2_bfix.cu's find_A_writer:
  A IS swizzled -- had to search          A has NO swizzle -- tAx equals
  (i+tAy)^(tAx<<2) == target               target_kk DIRECTLY, i+tAy equals
                                            target_m DIRECTLY (simpler!)

matmul_v0.cu's find_B_writer:          matmul_v2_bfix.cu's find_B_writer:
  B NOT swizzled -- simple n/k            B IS swizzled -- mask=(n>>2)&7
  matching                                 depends only on n, so once n is
                                            pinned, writer's mask == consumer's
                                            mask, and the XOR cancels out,
                                            leaving kk2 == target_kk directly
```
A mirror-image swap: matmul_v0.cu needed the cancellation trick for A
(swizzled there); matmul_v2_bfix.cu needs the exact same kind of trick
for B instead (swizzled here) -- confirmed by brute-force search, not
trusted blindly.

## 30.2 Finding the block and owning thread -- a DIFFERENT thread than before

```
STEP 1: blockIdx.y = 4//128 = 0, blockIdx.x = 24//128 = 0, row0=0, col0=0
STEP 2: tidx=96 owns acc[0][1] = C[4][24]
        (NOT tidx=13 -- a completely different thread than matmul_v0.cu's
         trace found for this SAME target, because this kernel's simpler,
         bit-trick-free epilogue formula maps output cells to threads
         differently)
```

## 30.3 The complete lineage, with the mask un-scramble visible

```
step | real_k | load_phase | A_writer | A_element | sA_target   | B_writer | B_element | sB_target    | regB_slot
1    | 0      | PROLOGUE   | 32       | A[4][0]   | sA[0][4][0] | 24       | B[0][24]  | sB[0][24][6] | regB[1][0]
7    | 6      | PROLOGUE   | 38       | A[4][6]   | sA[0][4][6] | 216      | B[6][24]  | sB[0][24][0] | regB[1][6]
```
At real_k=0: logical slot 0, mask=6, physical slot = 0^6=6 ->
sB[0][24][6]. At real_k=6: 6^6=0 -> sB[0][24][0]. The consumer thread
correctly un-scrambles both back to their true logical positions
(regB[1][0] and regB[1][6] respectively).

## 30.4 The verified final result

```
acc[0][1] = sum(A[4][k] * B[k][24] for k in range(16))
EPILOGUE_WRITE: tidx=96 writes acc[0][1] -> C[4][24]
```

## 30.5 The flowchart, side by side with the matmul_v0.cu version

```
   matmul_v0.cu's trace (tidx=13)        matmul_v2_bfix.cu's trace (tidx=96)
   -----------------------------         ------------------------------------
   find_consumer: bit-trick epilogue     find_consumer: simple epilogue
        |                                      |
        v                                      v
   find_A_writer: UN-swizzle A            find_A_writer: direct match (no swizzle)
   find_B_writer: direct match            find_B_writer: UN-swizzle B (mask cancels)
        |                                      |
        v                                      v
   tCsA/tCsB, per-tk pipelined            regA/regB, whole-row cached,
   compute (one real-k per tk)            full dot-product per chunk
        |                                      |
        +------------------+-------------------+
                            |
                            v
              BOTH converge on the same proof:
              C[4][24] = sum(A[4][k]*B[k][24])
              -- discovered, not assumed, by brute-force
              search over each kernel's own real formulas
```
Same mathematical truth, two structurally different kernels, two
different owning threads -- and the simulation discovered both
independently, purely by exhaustively matching each kernel's actual
index formulas.

---

# PART 31 -- Why matmul_v0.cu's Epilogue Needs BOTH row0 AND tCy<<2

## 31.1 The core idea: two completely different jobs, at two completely different scales

```c
int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
         ^^^^   ^^^^^^^^^^
     "WHICH BLOCK?"    "WHICH THREAD within that block?"
```
`row0` is the SAME value for all 256 threads inside one thread block --
it only changes when moving to a different block. `tCy << 2` is
DIFFERENT for (groups of) threads inside one block -- but the exact same
`tCy` value shows up again for "thread #13" in every other block too.
Like a postal address: `row0` is "which city," `tCy<<2` is "which street
address within that city." Both are required, because knowing only one
leaves the location completely ambiguous.

## 31.2 What breaks if you remove row0

With M=256, BLOCK_M=128, there are 2 thread blocks stacked vertically.
Computing `r` for the SAME thread (tidx=13, tCy=1) in both blocks:
```
Block (blockIdx.y=0): row0 = 0*128 = 0
    r = 0 + (1<<2) + ... = 4, 5, 6, 7, 68, 69, 70, 71

Block (blockIdx.y=1): row0 = 1*128 = 128
    r = 128 + (1<<2) + ... = 132, 133, 134, 135, 196, 197, 198, 199
```
Without `row0`, both blocks' thread 13 would compute the exact same `r`
values and both would try to write `C[4]`, `C[5]`, etc. -- total
collision. Block 1 is responsible for rows 128-255 and has no other way
to know that.

## 31.3 What breaks if you remove tCy<<2 (and the i-based term)

If every thread in a block used `r = row0` directly, with no
thread-specific component, every one of the 256 threads in that block
would compute the exact same `r` (since `row0` doesn't depend on
`tidx`). All 256 threads would race to write to the very same row of C
-- collisions within a single block, and 127 of the block's 128 real
output rows would never get written at all.

## 31.4 The flowchart: both pieces are mandatory, for different reasons

```
                Full matrix C has M rows
                         |
          +--------------+--------------+
          |                             |
   Block 0 owns rows 0-127      Block 1 owns rows 128-255
   (needs row0=0 to know this)  (needs row0=128 to know this)
          |                             |
   256 threads inside Block 0    256 threads inside Block 1
   must each cover a DIFFERENT   must each cover a DIFFERENT
   slice of those 128 rows       slice of those 128 rows
   (needs tCy<<2 + i-bits to     (needs tCy<<2 + i-bits to
    spread them out)              spread them out -- SAME
                                   formula, reused identically)
```
`row0` solves "which block." `tCy << 2` (with the `i&~3`/`i&3` bit-trick)
solves "which thread within the block." Removing either collapses a
different dimension of the parallelism into collisions.

## 31.5 Proof there are no gaps either -- the two pieces exactly tile the block together

With THREAD_C_LAYOUT_Y=16, `tCy` ranges over 16 distinct values (0..15),
so `tCy<<2` produces `0,4,8,...,60`. Adding the i-bit-trick's two
possible outcomes (`i&~3` contributes 0 or 64, `i&3` contributes 0..3):
```
tCy<<2 range:        0 .. 60   (16 values, step 4)
+ (i&3):             0 .. 3    (fills in every value between the steps of 4)
= combined, group 0: 0 .. 63   (every single row covered, no gaps)
+ (i&~3)=64, group 1: 64 .. 127 (the second half of the block)
---------------------------------------------------------------
TOTAL: 0 .. 127 -- exactly BLOCK_M's 128 rows, covered once each,
       by the combination of tCy (16 values) x i's 2 groups x 4 slots
       = 16 * 2 * 4 = 128
```
Neither piece alone could do both jobs -- the coarse selector (`row0`)
and the fine selector (`tCy<<2` + bit-trick) together give a complete,
collision-free, gap-free address.

---

# PART 32 -- Correcting "tCy Is Different For Almost Every Thread" And Why the Bit-Tricks Are Forced

## 32.1 The precise correction, with real numbers

```python
tCy = warpy * THREAD_C_WARP_Y + laney    # THREAD_C_WARP_Y = 4
```
Running this for all 256 threads shows `tCy` only ever takes 16 distinct
values (0..15) -- not 256. Each of those 16 values is shared by exactly
16 different threads:
```
threads sharing tCy=1:
  (tidx=1, tCx=0)  (tidx=3, tCx=1)   (tidx=5, tCx=2)   (tidx=7, tCx=3)
  (tidx=9, tCx=4)  (tidx=11, tCx=5)  (tidx=13, tCx=6)  (tidx=15, tCx=7)
  (tidx=33,tCx=8)  (tidx=35,tCx=9)   (tidx=37,tCx=10)  (tidx=39,tCx=11)
  (tidx=41,tCx=12) (tidx=43,tCx=13)  (tidx=45,tCx=14)  (tidx=47,tCx=15)
```
The established example thread (tidx=13, tCy=1) is NOT alone -- tidx=45
has the exact same tCy=1:
```
tidx=13: tCx=6,  tCy=1
tidx=45: tCx=14, tCy=1    <- same tCy!  different tCx!
```
Since both share tCy=1, they compute the exact same set of output ROWS
(4,5,6,7,68,69,70,71). What makes them write to genuinely different
places in C is that their tCx differs (6 vs 14), producing different
COLUMNS. It's the full (tCx, tCy) PAIR together -- not tCy alone -- that
is unique per thread.

## 32.2 The corrected mental model

```
256 threads total
        |
   16 distinct tCy values  x  16 distinct tCx values  =  256 unique (tCx,tCy) pairs
   ("which row-band?")        ("which column-band?")

tCy alone:  shared by 16 threads (they differ only in tCx)
tCx alone:  shared by 16 threads (they differ only in tCy)
(tCx,tCy):  unique to exactly ONE thread
```
```
   tCy=1 "row band" (shared by 16 threads, tCx=0..15):
   -----------------------------------------------------
   tidx=1 (tCx=0)  | tidx=3 (tCx=1) | ... | tidx=13(tCx=6) | ... | tidx=47(tCx=15)
   writes rows          writes rows           writes rows            writes rows
   {4,5,6,7,            {4,5,6,7,              {4,5,6,7,               {4,5,6,7,
    68,69,70,71}         68,69,70,71}           68,69,70,71}            68,69,70,71}
   at columns determined by ITS OWN tCx (0, 1, ..., 6, ..., 15 respectively)
```

## 32.3 Why the i&~3 / i&3 split exists at all

Why not something simpler, like `r = row0 + tCy*8 + i`? Tracing back to
how acc[][] actually got filled, in the register-staging step (Part 24):
```c
int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
```
For tm=0, this FLOAT4 pulls data tied to row-group tCy directly. For
tm=1, it pulls data tied to row-group `tCy + 1*THREAD_C_LAYOUT_Y` -- a
row-band 16 positions away, not the row immediately next to tm=0's. The
`tm` index in acc[tm][tn] was never meant to represent "the next row in
sequence" -- it represents "the next THREAD_C_LAYOUT_Y-separated
row-band."

## 32.4 What a naive linear formula would get wrong, concretely

```
Naive (WRONG):  r = row0 + tCy*8 + i
  tidx=13, tCy=1:  r = 0 + 8 + {0..7} = 8, 9, 10, 11, 12, 13, 14, 15

Real formula (CORRECT):  r = row0 + (tCy<<2) + (i&~3)*16 + (i&3)
  tidx=13, tCy=1:  r = 4, 5, 6, 7, 68, 69, 70, 71
```
The naive formula writes acc[4]'s value (which actually holds data
sourced from row-band `tCy + 1*16 = 17`'s territory) into row 12 --
completely wrong. The bit-trick is the exact mathematical inverse of how
FLOAT4 physically grouped the data during loading, and nothing simpler
works.

## 32.5 Why specifically & ~3 and & 3, and why jump by exactly 16

```
16 distinct tCy values x 4 (one FLOAT4 group) = 64 rows covered by i=0..3 alone
   tCy=0: rows 0-3      tCy=1: rows 4-7      tCy=2: rows 8-11   ...   tCy=15: rows 60-63
   (tCy<<2 tiles rows 0-63 EXACTLY, no gaps, no overlaps)

But BLOCK_M=128 -- there are 64 MORE rows (64-127) left uncovered!
That's exactly what i=4..7 (the i&~3=4 branch) is for: REUSE the same
16 tCy-bands a second time, offset by 64, to cover the second half:
   tCy=0: rows 64-67    tCy=1: rows 68-71    ...   tCy=15: rows 124-127
```
```
          i value:        0   1   2   3   4   5   6   7
          i & ~3:         0   0   0   0   4   4   4   4    <- which half of the 128-row block?
          i & 3:          0   1   2   3   0   1   2   3    <- which of the 4 rows within that half's band?
          real row (tCy=1): 4   5   6   7  68  69  70  71
```

## 32.6 The flowchart tying both parts together

```
256 threads, 16x16 grid of (tCx, tCy) identities
        |
        v
tCy picks a "row band" (SHARED by 16 threads differing in tCx)
        |
        v
tCy<<2 places that band's FIRST 4 rows somewhere in 0..63
        |
        v
i's low 2 bits (i&3) walk through those 4 rows one at a time
        |
        v
i's remaining bit (i&~3) decides: first half of the block (rows
0-63, use the band as-is) OR second half (rows 64-127, reuse the
SAME band, jumped forward by exactly THREAD_C_LAYOUT_Y*4=64)
        |
        v
Together: EVERY one of the block's 128 rows is covered exactly
once, by exactly the right thread, in exactly the physical order
FLOAT4 originally produced the data -- no naive linear formula
could reproduce this without breaking correctness.
```

---

# PART 33 -- Ground-Up Visual Rebuild, and Why Not Simple Consecutive Rows

## 33.1 The nesting, confirmed

```
Whole matrix C
    |
    +-- One thread BLOCK covers a 128x128 tile (BLOCK_M=128, BLOCK_N=128)
            |
            +-- That tile is covered by 256 threads, arranged as a 16x16 grid
                    |
                    +-- Each ONE thread covers an 8x8 = 64-element piece
                        (128 rows / 16 threads-per-column = 8 rows each
                         128 cols / 16 threads-per-row    = 8 cols each)
```

## 33.2 The wrong (but natural) guess, and the real picture

The natural guess: "thread tCy=1 probably owns 8 consecutive rows, like
rows 8-15." This is wrong. The complete, real map of all 128 rows,
grouped by which tCy owns them:
```
tCy= 0 -> rows: [0, 1, 2, 3,   64, 65, 66, 67]
tCy= 1 -> rows: [4, 5, 6, 7,   68, 69, 70, 71]
tCy= 2 -> rows: [8, 9, 10, 11, 72, 73, 74, 75]
...
tCy=15 -> rows: [60, 61, 62, 63,  124, 125, 126, 127]
```
Every thread's 8 rows come in TWO separate clusters of 4, far apart --
one cluster in the top half (rows 0-63), one in the bottom half (rows
64-127).
```
 Row:  0  1  2  3  4  5  6  7  8  9 10 11 ... 60 61 62 63 | 64 65 66 67 68 69 70 71 ...124 125 126 127
owner: 0  0  0  0  1  1  1  1  2  2  2  2 ... 15 15 15 15 |  0  0  0  0  1  1  1  1 ... 15  15  15  15
       |________________________________|                  |________________________________|
              TOP HALF (rows 0-63)                                BOTTOM HALF (rows 64-127)
```

## 33.3 Walking the formula step by step for tCy=1

```c
int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
```
With row0=0, tCy=1, THREAD_C_LAYOUT_Y=16, walking i=0..7 one at a time:
```
i=0:  i&~3=0 (TOP)   i&3=0   r = 0 + 4 + 0*16 + 0 = 4
i=1:  i&~3=0 (TOP)   i&3=1   r = 4 + 0 + 1 = 5
i=2:  i&~3=0 (TOP)   i&3=2   r = 4 + 0 + 2 = 6
i=3:  i&~3=0 (TOP)   i&3=3   r = 4 + 0 + 3 = 7
      -------------------------------------------------
      <- TOP cluster done: rows 4,5,6,7
i=4:  i&~3=4 (BOTTOM!) i&3=0  r = 0 + 4 + 4*16 + 0 = 4 + 64 + 0 = 68
i=5:  i&~3=4 (BOTTOM)  i&3=1  r = 4 + 64 + 1 = 69
i=6:  r = 4 + 64 + 2 = 70
i=7:  r = 4 + 64 + 3 = 71
      -------------------------------------------------
      <- BOTTOM cluster done: rows 68,69,70,71
```
Final result for thread tCy=1, i=0..7 in order: rows 4, 5, 6, 7, 68, 69,
70, 71.

## 33.4 What each identifier does, in plain English

```
(A) row0        = "which 128-row BLOCK am I even in?"           (same for all 256 threads)
(B) tCy << 2    = "within this block, where does MY cluster
                    of 4 rows start, in the TOP half?"           (e.g. tCy=1 -> starts at row 4)
(C) (i&~3)*16   = "am I currently in the TOP half (contributes
                    0) or the BOTTOM half (contributes 64)?"     (a simple on/off switch)
(D) (i&3)       = "within whichever half I'm in right now,
                    which of the 4 rows am I on? (0,1,2, or 3)"  (walks 0->1->2->3)
```

## 33.5 Why the split into TOP and BOTTOM exists at all

16 distinct tCy values x 4 rows each = only 64 rows -- that only covers
HALF of the 128-row block (0-63). The other 64 rows (64-127) have to be
covered too, so the design simply reuses the same 16 tCy clusters a
second time, shifted down by exactly 64 rows, instead of inventing a
whole separate set of thread identities.

## 33.6 Zooming out: the full 2D picture -- four scattered 4x4 squares, not one 8x8 block

The exact same TOP/BOTTOM splitting trick happens independently for
columns too (using j, tCx, THREAD_C_LAYOUT_X). So one thread's real
8x8=64 elements aren't one solid square -- they're four separate 4x4
squares, scattered into the four quadrants of the 128x128 block:
```
        col-cluster A          col-cluster B
        (cols 24-27)           (cols 88-91)
      +----------------+     +----------------+
row   |  4x4 square 1  |     |  4x4 square 2  |      <- row-cluster A (rows 4-7)
      +----------------+     +----------------+
                  ... (rest of the 128x128 block, owned by other threads) ...
      +----------------+     +----------------+
      |  4x4 square 3  |     |  4x4 square 4  |      <- row-cluster B (rows 68-71)
      +----------------+     +----------------+
```
Thread tidx=13 (tCx=6, tCy=1) owns exactly these four scattered 4x4
squares -- not one contiguous 8x8 block.

## 33.7 Why not simple consecutive rows? The honest, traceable answer

The epilogue doesn't get to choose this pattern -- it's just reporting
where data already landed, placed there by an earlier step (the compute
phase), which the epilogue must simply match.

The real cause is FLOAT4 -- a hardware instruction that loads 4 numbers
at once:
```c
int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
```
The concrete instruction-count comparison:
```
WITHOUT FLOAT4 (8 separate scalar loads, one per row):
    8 TOTAL LOAD INSTRUCTIONS

WITH FLOAT4 (2 wide loads, 4 rows each):
    2 TOTAL LOAD INSTRUCTIONS  <- 4x fewer instructions!
```
FLOAT4 only works on 4 values that are physically adjacent in memory.
So the natural design is: "group 1" = the first physically-adjacent
4-float chunk (tm=0), "group 2" = a second, separate physically-adjacent
4-float chunk (tm=1), pulled from a region THREAD_C_LAYOUT_Y=16
positions away in the thread grid's own numbering -- not immediately
next-door.

```
 If you wanted SIMPLE CONSECUTIVE rows instead:
 -----------------------------------------------
 You'd need shared memory organized so that ALL 8 of a thread's
 logical rows sit physically back-to-back -- a DIFFERENT, equally-
 valid design choice, not disallowed by the hardware. It's simply
 not what THIS kernel's author built. The author instead chose to
 let EACH of the 256 threads' two FLOAT4 chunks land in TWO
 DIFFERENT HALVES of the 128-row block, so every thread contributes
 a small piece to BOTH halves, rather than having 128 threads handle
 the top half exclusively and a different 128 threads handle the
 bottom half exclusively.
```

## 33.8 The full causal chain, as a flowchart

```
DESIGN CHOICE: use FLOAT4 for register loading (performance: 4x
fewer load instructions than scalar loads)
        |
        v
CONSEQUENCE: a thread's 8 rows must be filled via exactly 2
separate FLOAT4 chunks of 4 (8 rows / 4 = 2 chunks)
        |
        v
CONSEQUENCE: those 2 chunks come from 2 DIFFERENT physical
locations in shared memory (tm=0's location, and tm=1's location,
offset by THREAD_C_LAYOUT_Y=16 in the thread-grid's own numbering,
which becomes a 64-row jump in REAL row terms)
        |
        v
CONSEQUENCE: acc[tm][tn]'s "tm" index is forever tied to "which
FLOAT4 chunk", not "which row in simple increasing order"
        |
        v
FORCED RESULT: the epilogue, writing acc[][] back to C, MUST use
a formula that correctly maps "which FLOAT4 chunk" back to "which
real row" -- exactly the (i&~3)/(i&3) bit-trick. It has no other
choice, because it must match wherever the data actually physically is.
```

So: it's not that "8 consecutive rows" is impossible -- it's that this
specific kernel's design starts from "use FLOAT4 for fast register
loading," and that one upstream choice mechanically forces everything
downstream (how acc[][] is organized, and therefore how the epilogue
must write it back) into this scattered, two-cluster-of-4 shape. The
epilogue's job is never to decide the pattern -- only to correctly undo it.

---

# PART 34 -- The Outer k-Loop: for (k=BLOCK_K; k<K+BLOCK_K; k+=BLOCK_K)

## 34.1 What is this loop's logic, at the highest level?

This loop processes the K dimension in chunks of size BLOCK_K. With
K=16, BLOCK_K=8, there are exactly 2 chunks:
```
Chunk 0 = columns [0, 8)    Chunk 1 = columns [8, 16)
```
The loop variable `k` does NOT mean "which chunk am I computing right
now." It means "which chunk should I reload NEXT, to get ready for the
FOLLOWING iteration." This is a software-pipelined prefetch loop -- the
same "load-ahead" pattern as the inner tk loop (Parts 21-24), just
operating one level up, at whole-chunk granularity.

## 34.2 Why does k start at BLOCK_K instead of 0?

Because chunk 0 was already loaded before this loop even begins, by the
PROLOGUE:
```c
// PROLOGUE (runs ONCE, before the k-loop):
sA[0][...] = A[row][tAx];   // loads chunk 0 (columns 0..7) into buffer 0
sB[0][...] = B[tBy][col];   // loads chunk 0 (columns 0..7) into buffer 0
__syncthreads();

int bufferId = 0;
for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) { ... }
```
Since chunk 0 is already sitting in buffer 0 before the loop starts, the
loop's first pass computes using chunk 0 (already there) while
simultaneously reloading chunk 1 (the next one). The starting column of
"the next chunk to load" is exactly BLOCK_K=8 -- so k has to start there.
```
       PROLOGUE (before loop)             k-LOOP iteration 1 (k=8)
       --------------------------         ---------------------------
       loads chunk 0 (cols 0-7)           COMPUTE using chunk 0 (already
       into buffer 0                      loaded) WHILE reloading chunk 1
                                           (cols 8-15, because k=8) into
                                           buffer 1
```

## 34.3 Why does k use BLOCK_K as the step size?

Because each loop iteration is responsible for exactly one whole
chunk's worth of columns, and chunks are BLOCK_K columns wide. Stepping
by BLOCK_K moves k from one chunk's starting column directly to the
next chunk's starting column. The reload code uses k directly as a
whole-chunk offset (`A[row][k + tAx]`, `B[k + tBy][col]`), so k must
jump in BLOCK_K-sized strides to land exactly on chunk boundaries.

## 34.4 Why does the loop go all the way to K + BLOCK_K, not just K?

Traced with real numbers (K=16, BLOCK_K=8):
```
for (k = 8; k < 16 + 8; k += 8)   // i.e. k < 24
    k=8:  iteration 1
    k=16: iteration 2
    (k=24 would be next, but 24 is NOT < 24, so loop stops)
```
If the bound were just `k < K` (`k < 16`) instead, k=16 would never be
reached -- the loop would run only ONE time (k=8), and chunk 1 (columns
8-15), reloaded into buffer 1 during that single iteration, would NEVER
get computed. Its data would sit loaded in shared memory and just be
thrown away -- a correctness bug, silently dropping half of K's real
contribution.
```
   WRONG (k < K):                       CORRECT (k < K + BLOCK_K):
   --------------------                 ---------------------------
   k=8: compute chunk0, reload chunk1   k=8:  compute chunk0, reload chunk1
   (loop stops here -- BUG!)            k=16: compute chunk1, NO reload
                                               (drains the last chunk)
   Chunk 1 loaded but NEVER computed    Both chunks fully computed
```
The extra iteration (k=16) exists purely to "drain" the pipeline -- to
consume the very last chunk that got reloaded one step earlier, with no
new reload needed since there's nothing left beyond K.

## 34.5 The guard that makes this safe

```c
for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
    // ... compute using the CURRENTLY loaded buffer (bufferId) ...
    if (k < K) {
        // reload the NEXT chunk (starting at column k) into bufferId^1
    }
    bufferId ^= 1;
}
```
At k=8: `8 < 16` is True -> reload happens (fetching chunk 1). At k=16:
`16 < 16` is False -> reload is skipped (there's no chunk 2).

## 34.6 The full real timeline

```
TIME -->

PROLOGUE          k=8 (iter 1)              k=16 (iter 2, "drain")
--------          -------------------       -------------------------
load chunk0  -->  COMPUTE chunk0       -->  COMPUTE chunk1
into buffer0      (from buffer0)             (from buffer1, loaded
                      +                       during k=8's iteration)
                  RELOAD chunk1                   +
                  into buffer1               (k<K is FALSE,
                  (k=8 < K=16, so this        so NO reload here --
                   reload DOES happen)        nothing left to fetch)
```

## 34.7 The flowchart

```
            PROLOGUE: load chunk 0 into buffer 0 (runs once)
                              |
                              v
            bufferId = 0
                              |
                              v
        +---------------------------------------------+
        |  for k = BLOCK_K; k < K+BLOCK_K; k += BLOCK_K |
        +---------------------------------------------+
                              |
                              v
             COMPUTE using the chunk already sitting
             in buffer `bufferId` (loaded last time)
                              |
                              v
                  is k < K?  (is there a NEXT
                   chunk that actually exists?)
                    /                    \
                 YES                     NO
                  |                       |
                  v                       v
      RELOAD chunk starting       skip reloading --
      at column k into            nothing left to fetch
      buffer `bufferId ^ 1`       (this is the final,
                  |                "drain" iteration)
                  v                       |
          bufferId ^= 1  <----------------+
                  |
                  v
           back to top of loop (or exit, if k+BLOCK_K
           is no longer < K+BLOCK_K)
```
The loop variable k always means "the chunk I'm about to reload for
NEXT time," not "the chunk I'm computing right now." Starting at
BLOCK_K skips re-doing the prologue's job; running one extra time past
K ensures the very last reloaded chunk actually gets computed instead
of being silently discarded.

---

# PART 35 -- Why a Separate PROLOGUE, and Why the Stride Must Be Exactly BLOCK_K

## 35.1 Why does chunk 0 need a separate PROLOGUE? The bootstrapping problem

A restaurant analogy: Worker A preps ingredients (loading data), Worker
B cooks dishes (computing). To keep BOTH busy simultaneously, Worker B
should be cooking dish N while Worker A is already prepping dish N+1's
ingredients. But for the very first dish, Worker B has nothing to cook
yet -- nobody prepped it in advance. Someone has to prep dish 1's
ingredients alone, before the overlapping pattern can even start. That
one-time, solo step is exactly the PROLOGUE.

## 35.2 What goes wrong without a separate prologue

```c
// naive, no overlap:
for (k = 0; k < K; k += BLOCK_K) {
    load chunk starting at k into buffer bufferId;
    __syncthreads();              // must WAIT here for the load to finish
    compute using buffer bufferId;  // only AFTER the load is fully done
    bufferId ^= 1;
}
```
This works correctly, but has zero overlap -- strictly load, wait,
compute, load, wait, compute. During each load, the ALUs sit idle.
During each compute, the memory system sits idle. The entire
performance benefit of double-buffering is lost.

## 35.3 The real kernel's approach -- true overlap

```c
// PROLOGUE (runs once, synchronously, before anything else):
load chunk 0 into buffer 0;
__syncthreads();

int bufferId = 0;
for (k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
    compute using buffer bufferId;         // uses chunk ALREADY loaded
    if (k < K) {
        reload NEXT chunk into buffer (bufferId^1);   // issued concurrently
    }
    bufferId ^= 1;
}
```
Because chunk 0 is already sitting in shared memory, fully synced,
before the loop's first iteration even starts, that first iteration can
begin computing immediately while the hardware simultaneously fetches
chunk 1 in the background. The load's latency gets hidden behind useful
ALU work.
```
  WITHOUT prologue (naive):        WITH prologue (real kernel):
  --------------------------       -----------------------------
  load0 -> wait -> compute0        load0 (once, upfront)
  load1 -> wait -> compute1             |
  (ALU idle during every load,          v
   memory idle during every        compute0  +  load1 (SIMULTANEOUSLY)
   compute -- never overlapped)         |
                                        v
                                   compute1  (load1 already finished
                                   by the time compute0 was done)
```

## 35.4 Why k + tAx works, and why the stride must be exactly BLOCK_K

Important clarification: `tAx` is NOT something one thread loops over --
it's a fixed identity, baked into each thread once, forever, based on
its tidx. The "coverage of a whole chunk" comes from many different
threads, each with a different fixed tAx, all running in parallel.
```
THREAD_A_LAYOUT_X = 8   BLOCK_K = 8   (equal, by deliberate design!)

Across all 256 threads, tAx (= tidx & 7) takes EXACTLY these values:
[0, 1, 2, 3, 4, 5, 6, 7]  -- covers 0..BLOCK_K-1, nothing more, nothing less
```
For a fixed k (say k=8), the collective set of 256 threads -- each
contributing its own fixed tAx -- computes k+tAx for every value 0..7
simultaneously:
```
thread with tAx=0 loads real column  8+0 = 8
thread with tAx=1 loads real column  8+1 = 9
...
thread with tAx=7 loads real column  8+7 = 15
```
Together, in ONE iteration, all 256 threads collectively load every
column of chunk 1 (columns 8-15) exactly once. This is the exact same
"coarse + fine" addressing idiom from Parts 31-33's `row0 + tCy<<2` --
here it's k (coarse: which chunk) + tAx (fine: which column within
that chunk).

## 35.5 What breaks if the stride were 1 instead of BLOCK_K

```
k=8: thread with tAx=3 loads real column  8+3 = 11
k=9: thread with tAx=3 loads real column  9+3 = 12   <- only ONE step later!
```
If k incremented by 1, the very next iteration would re-load almost the
same columns again, while making almost no forward progress. It would
eventually push k+tAx past K-1=15 into invalid territory, and the loop
would run 16 times instead of 2 -- doing far more work than necessary,
re-computing overlapping chunks and corrupting the final sum by
counting some real k-values' contributions multiple times.
```
stride = BLOCK_K (8):  k=8 covers cols [8,16), k=16 is past the end -- PERFECT, no gaps, no overlap
stride < BLOCK_K (e.g. 1):  massive overlap between consecutive iterations -- wasteful, double-counts data
stride > BLOCK_K (e.g. 16): would SKIP chunk coverage entirely if K had more chunks -- leaves gaps, missing data
```

## 35.6 The flowchart, combining both answers

```
PROLOGUE (bootstrapping -- the "first dish" problem):
   No previous iteration exists to have prefetched chunk 0,
   so it must be loaded once, synchronously, up front.
        |
        v
MAIN LOOP (k = BLOCK_K, BLOCK_K*2, ...):
   Each thread's FIXED tAx (0..BLOCK_K-1) + the loop's k
   together address exactly one real column: k + tAx
        |
        v
   256 threads, each with a different fixed tAx, COLLECTIVELY
   cover all BLOCK_K columns of chunk `k` in ONE iteration
        |
        v
   k MUST jump by exactly BLOCK_K each time, matching the
   width tAx already covers -- otherwise columns get skipped
   (gap, missing data) or redundantly reloaded (waste, double-counting)
```

---

# PART 36 -- Line-by-Line Real-Code Proof of the Stride-1 Bug

## 36.1 The exact code from matmul_v0.cu (lines 178-198)

```c
178:    if (k < K) {
179:        int c = k + tAx;
180:  #pragma unroll
181:        for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
182:            int r = row0 + i + tAy;
183:            sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] =
184:                (r < M && c < K) ? A[r * K + c] : 0.0f;
185:        }
186:  #pragma unroll
187:        for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
188:            int r = k + i + tBy;
189:  #pragma unroll
190:            for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
191:                int c2 = col0 + j + tBx;
192:                sB[bufferId ^ 1][i + tBy][j + tBx] =
193:                    (r < K && c2 < N) ? B[r * N + c2] : 0.0f;
194:            }
195:        }
196:        __syncthreads();
197:    }
198:    bufferId ^= 1;
```

## 36.2 Line 179 is the single most important line: `int c = k + tAx;`

This is the only place in the whole reload block where k determines
which real column of A gets fetched. Traced for thread tAx=3:
```
CORRECT stride:
  Iteration k=8:   c = 8 + 3  = 11
  Iteration k=16:  (line 178's guard k<K is False -- block never executes,
                    zero redundant work)

BROKEN stride=1:
  Iteration k=8:   c = 8 + 3  = 11
  Iteration k=9:   c = 9 + 3  = 12    <- only shifted by 1!
  Iteration k=10:  c = 10 + 3 = 13
  Iteration k=11:  c = 11 + 3 = 14
```
With broken stride, line 178's guard `k < K` (`k < 16`) stays True for
iterations k=8 through k=15 -- eight separate times the reload block
fires, each time shifting the fetched column by only 1.

## 36.3 Line 183: WHERE data gets written never changes, only WHAT does

```c
183:        sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] = ...
```
The destination index depends only on tAx, i, tAy -- none of which
depend on k. Every reload call for a given thread writes to the exact
same shared-memory slot, overwriting it with a different source column
each time:
```
Iteration k=8:   sA[1][3][...] = A[row][11]
Iteration k=9:   sA[1][3][...] = A[row][12]    <- SAME slot, overwritten
Iteration k=10:  sA[1][3][...] = A[row][13]    <- SAME slot, overwritten
```

## 36.4 Why overwriting still causes double-counting, not just wasted writes

COMPUTE reads from the buffer in between these overwrites, once per
outer-loop iteration, using whichever buffer the previous iteration's
reload just filled:
```
k=8:  COMPUTE reads buffer0 (cols 0-7, from PROLOGUE)         | RELOAD writes buffer1 with cols [8..15]
k=9:  COMPUTE reads buffer1 (cols [8..15], just written)      | RELOAD OVERWRITES buffer0 with cols [9..16]
k=10: COMPUTE reads buffer0 (cols [9..16]!! overlaps k=8's)   | RELOAD OVERWRITES buffer1 with cols [10..17]
```
At k=9, COMPUTE correctly processes columns 8-15. But at k=10, COMPUTE
processes columns 9 through 16 -- columns 9-15 were already added into
acc[][] one iteration ago, and now get added AGAIN.

## 36.5 The real, executed Python simulation -- the actual tally

```
k= 8: COMPUTE buffer0(cols 0-7)      | RELOAD cols 8-15  into buffer1
k= 9: COMPUTE buffer1(cols 8-15)     | RELOAD cols 9-16  into buffer0
k=10: COMPUTE buffer0(cols 9-16)     | RELOAD cols 10-17 into buffer1
k=11: COMPUTE buffer1(cols 10-17)    | RELOAD cols 11-18 into buffer0
k=12: COMPUTE buffer0(cols 11-18)    | RELOAD cols 12-19 into buffer1
k=13: COMPUTE buffer1(cols 12-19)    | RELOAD cols 13-20 into buffer0
k=14: COMPUTE buffer0(cols 13-20)    | RELOAD cols 14-21 into buffer1
k=15: COMPUTE buffer1(cols 14-21)    | RELOAD cols 15-22 into buffer0
k=16: COMPUTE buffer0(cols 15-22)    | ... keeps going forever
```
Column-by-column tally (verified by actually running the simulation):
```
column 0:  counted 1 time   (correct)
...
column 8:  counted 1 time   (correct)
column 9:  counted 2 times  <- BUG: added into acc[][] TWICE
column 10: counted 3 times  <- BUG: added THREE times
column 11: counted 4 times  <- BUG
column 12: counted 5 times  <- BUG
column 13: counted 6 times  <- BUG
column 14: counted 7 times  <- BUG
column 15: counted 8 times  <- BUG: added EIGHT times!
```
The damage snowballs: column 15 (the last real column) gets caught in
the overlap of nearly every subsequent iteration, because the 8-wide
window keeps sliding forward by just 1 step while staying 8 wide.

## 36.6 Visualizing the sliding window

```
k=8:  [8  9 10 11 12 13 14 15]
k=9:     [9 10 11 12 13 14 15 16]
k=10:       [10 11 12 13 14 15 16 17]
k=11:          [11 12 13 14 15 16 17 18]
                              ^^^^^^^^^^^ column 15 stays INSIDE the
                                          window for many iterations
                                          in a row -- counted every time
```
With the correct stride (BLOCK_K=8), the window jumps its full width
each time -- the next window starts exactly where the previous one
ended, zero overlap:
```
k=8:  [8  9 10 11 12 13 14 15]
k=16:                          [16 17 18 ...]   <- starts right after, no overlap
```

## 36.7 Why acc[][] += ... makes overlap catastrophic, not just wasteful

```c
acc[tm][tn] += tCsA[...][tm] * tCsB[...][tn];
```
This is always an accumulation -- every computed column's contribution
gets ADDED to the running total, never replaces a previous value. If
column 15's data gets computed 8 separate times, its true contribution
gets multiplied by roughly 8x, corrupting every output element that
depends on it -- which, since column 15 is a real part of the K-
dimension sum, is every single output element in the result matrix.

## 36.8 The out-of-bounds problem, handled safely by the ternary check

```c
184:    (r < M && c < K) ? A[r * K + c] : 0.0f;
193:    (r < K && c2 < N) ? B[r * N + c2] : 0.0f;
```
As the broken window marches past K=16 (e.g. columns 16-22 at k=15),
these ternary bound checks catch it and substitute 0.0f instead of
reading invalid memory -- so the bug does NOT crash or corrupt memory,
it silently computes using zero for the invalid positions while still
double/triple/octuple-counting the valid ones.

## 36.9 The flowchart, anchored to the exact real line numbers

```
line 178: if (k < K)?
    |
    +-- CORRECT (k jumps by BLOCK_K): only fires for k=8 (then k=16 fails the
    |   check, reload block skipped entirely -- zero redundant work)
    |
    +-- BROKEN (k jumps by 1): fires for EVERY k=8,9,10,...,15 (8 separate
        times instead of 1) -- line 179's `c = k + tAx` only advances by 1
        each time, so line 183's write target (fixed per-thread) gets
        overwritten 8 times with heavily overlapping column data, and
        line 198's bufferId flip ensures COMPUTE re-reads nearly the same
        columns on almost every outer iteration -- producing the exact
        8x-overcounting on column 15 and the cascading 2x..7x overcounting
        on columns 9-14 shown in the real executed tally above
```
The entire bug traces back to just two numbers interacting: line 179's
`+tAx` offset (fixed width BLOCK_K) versus however much k itself
advances between calls. They are only compatible -- zero overlap, zero
gaps -- when k's step exactly equals that fixed width, which is
precisely BLOCK_K.

---

# PART 37 -- matmul_v2_bfix.cu's Prologue: Why sA Uses Only row0, and sB Uses Only col0

## 37.1 The reason lives in basic matrix-multiplication shapes, not CUDA tricks

```
A is M x K  (256 x 16)        B is K x N  (16 x 256)        C is M x N (256 x 256)
   K is A's COLUMN axis           K is B's ROW axis
```
The math definition: `C[r][c] = sum over k of ( A[r][k] * B[k][c] )`. To
compute one tile of C spanning rows [row0, row0+BLOCK_M) and columns
[col0, col0+BLOCK_N), the sum requires:
- For A: only the specific rows [row0, row0+BLOCK_M) -- but ALL K
  columns, every single one, because the sum runs over every k.
- For B: only the specific columns [col0, col0+BLOCK_N) -- but ALL K
  rows, every single one, for the same reason.
```
   A's relevant slice:  A[row0 : row0+BLOCK_M,  0 : K]       <- uses row0, spans FULL K width
   B's relevant slice:  B[0 : K,  col0 : col0+BLOCK_N]       <- uses col0, spans FULL K height
```
col0 is about which columns of C (and B) we want -- irrelevant to A,
since A's column axis IS the K/reduction axis, always taken in full.
Same logic in reverse for B and row0.

## 37.2 Confirming this directly against the real code

```c
// A's prologue:
int r = row0 + i + tAy;                          // <- row0 used HERE
sA[0][i + tAy][tAx] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
//                                              ^^^ tAx indexes K DIRECTLY -- no col0 anywhere!
```
```c
// B's prologue:
int r = i + tBy;                                  // <- NO row0 here! r indexes K directly
int c = col0 + j + tBx;                           // <- col0 used HERE
sB[0][n][kk ^ mask] = (r < K && c < N) ? B[r * N + c] : 0.0f;
```
A's loading formula never mentions col0 anywhere in its math. B's
loading formula never mentions row0 anywhere in its math. This isn't an
omission -- neither operand needs that information.

## 37.3 The concrete, verified proof -- two different blocks loading identical data

Running this for real, for the same thread (tidx=13) in two different
blocks sharing the same row0 but different col0:
```
A loads for block(row0=0, col0=0):    [(1,5), (33,5), (65,5), (97,5)]
A loads for block(row0=0, col0=128):  [(1,5), (33,5), (65,5), (97,5)]
--> IDENTICAL!
```
Two completely different blocks -- one responsible for C's left half,
one for C's right half -- load the exact same data from A, because A
doesn't care which N-slice of C you're computing. Flip it for B -- same
col0, different row0:
```
B loads for block(row0=0, col0=0):    [(0,13),(0,45),(0,77),(0,109),(8,13),(8,45),(8,77),(8,109)]
B loads for block(row0=128, col0=0):  [(0,13),(0,45),(0,77),(0,109),(8,13),(8,45),(8,77),(8,109)]
--> IDENTICAL!
```
Two blocks -- one for C's top half, one for C's bottom half -- load the
exact same data from B.

## 37.4 The diagram that makes it click

```
                    N (columns of B / C)
              col0=0          col0=128
           +-----------+-----------+
  row0=0   | Block(0,0)| Block(1,0)|   <- BOTH these blocks load the
           +-----------+-----------+      SAME rows of A (rows 0-127),
  row0=128 | Block(0,1)| Block(1,1)|      because A doesn't know about N
           +-----------+-----------+
    M (rows of A / C)

  Block(0,0) and Block(1,0): SAME row0=0  -> load IDENTICAL A data
                              DIFFERENT col0 -> load DIFFERENT B data

  Block(0,0) and Block(0,1): SAME col0=0  -> load IDENTICAL B data
                              DIFFERENT row0 -> load DIFFERENT A data
```

## 37.5 The flowchart

```
Computing C's tile at (row0, col0)
        |
        +--> Need A's data: WHICH rows of C am I responsible for?
        |         -> row0 picks the ROWS of A (matches C's rows)
        |         -> col0 is IRRELEVANT -- always take ALL of K
        |              (A[row0 : row0+BLOCK_M, every column 0..K-1])
        |
        +--> Need B's data: WHICH columns of C am I responsible for?
                  -> col0 picks the COLUMNS of B (matches C's columns)
                  -> row0 is IRRELEVANT -- always take ALL of K
                       (B[every row 0..K-1, col0 : col0+BLOCK_N])
```
`row0` answers "which rows of the OUTPUT (and therefore A) do I own?"
-- a question B has no stake in. `col0` answers "which columns of the
OUTPUT (and therefore B) do I own?" -- a question A has no stake in.
The K dimension itself is always taken in full by both operands,
because that's the axis being summed over, not sliced.

---

# PART 38 -- Why matmul_v0.cu's Epilogue Needs the Bit-Trick and matmul_v2_bfix.cu's Doesn't

## 38.1 The one-sentence answer

matmul_v0.cu uses FLOAT4 to group ROWS together (4 different tm
row-slots share one load), which scatters rows and forces the bit-trick
to undo it. matmul_v2_bfix.cu uses FLOAT4 to group K-VALUES together
instead (one tm row gets its own private load of 4 K-values), which
never touches the row dimension at all -- so there's nothing to undo.

## 38.2 matmul_v0.cu's compute phase -- FLOAT4 groups 4 DIFFERENT ROWS into one instruction

```c
int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
```
`tm` only takes 2 values (0 and 1) -- each single FLOAT4 call fills 4
register slots at once:
```
tm=0: ONE FLOAT4 call fills tCsA slots [0,1,2,3] -- all 4 pulled from ONE
      physically-adjacent shared-memory chunk, tied to row-band tCy
tm=1: ONE FLOAT4 call fills tCsA slots [4,5,6,7] -- all 4 pulled from a
      DIFFERENT, physically-adjacent chunk, tied to row-band tCy+16
```
Slots 0-3 and slots 4-7 come from two separate, far-apart physical
sources -- exactly why the epilogue needs `(i&~3)*THREAD_C_LAYOUT_Y +
(i&3)` to correctly map "which scattered group" and "which position
within the group" back to the real row.

## 38.3 matmul_v2_bfix.cu's compute phase -- FLOAT4 groups 4 K-VALUES for ONE row

```c
for (tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
    int m = tCy + tm * THREAD_C_LAYOUT_Y;
    FLOAT4(regA[tm][0]) = FLOAT4(sA[bufferId][m][0]);   // 4 K-values for row m
    FLOAT4(regA[tm][4]) = FLOAT4(sA[bufferId][m][4]);   // 4 MORE K-values, same row m
}
```
`tm` ranges over all 8 values, and every single tm gets its own,
completely independent, dedicated pair of FLOAT4 loads -- the
vectorization groups 4 K-values (columns within one chunk) for a
single row, never combining multiple rows into one load:
```
tm=0: m = 1 + 0*16  = 1     (own dedicated loads, nobody else touches row 1)
tm=1: m = 1 + 1*16  = 17    (own dedicated loads, nobody else touches row 17)
tm=2: m = 1 + 2*16  = 33
...
tm=7: m = 1 + 7*16  = 113
```
Every tm maps to its real row via one single, simple, linear formula --
`tCy + tm*THREAD_C_LAYOUT_Y`, no branching, no bit-trick, nothing to
undo. The epilogue just repeats this exact same formula (`r = row0 +
tCy + tm*THREAD_C_LAYOUT_Y`) because the compute phase never scrambled
anything in the first place.

## 38.4 Why the two kernels made opposite choices -- traced to shared-memory layout

```
matmul_v0.cu:        sA is stored TRANSPOSED: sA[buf][k][m]
                     -- M is the INNER (fastest-varying) index
                     -- 4 physically-adjacent values = 4 DIFFERENT rows (m)
                     -- so FLOAT4 naturally groups ROWS -> scatters them

matmul_v2_bfix.cu:   sA is stored NATURAL: sA[buf][m][k]
                     -- K is the INNER (fastest-varying) index
                     -- 4 physically-adjacent values = 4 DIFFERENT k-values
                     -- so FLOAT4 naturally groups K-VALUES, rows stay untouched
```
Each kernel's FLOAT4 usage simply follows whichever axis happens to be
physically adjacent in its own shared-memory layout -- and that layout
choice is what ultimately decides whether the epilogue needs a
scatter-undoing bit-trick or not.

## 38.5 The flowchart, side by side

```
matmul_v0.cu                              matmul_v2_bfix.cu
-------------                             -------------------
sA stored TRANSPOSED (k outer, m inner)   sA stored NATURAL (m outer, k inner)
        |                                          |
        v                                          v
FLOAT4 groups 4 ADJACENT "m" values        FLOAT4 groups 4 ADJACENT "k" values
into ONE register load                     into ONE register load
        |                                          |
        v                                          v
tm only needs 2 values (0,1) to cover      tm needs all 8 values (0..7), each
all 8 rows -- rows get SPLIT into          with its OWN independent row --
2 scattered clusters of 4                  rows stay in SIMPLE linear order
        |                                          |
        v                                          v
EPILOGUE must use (i&~3)*LAYOUT_Y+(i&3)    EPILOGUE just repeats the SAME
to correctly un-scatter the two            simple linear formula the compute
clusters back to real rows                 phase already used -- nothing to undo
```

## 38.6 Direct proof: comparing the two kernels' real row-to-tm mappings

```
matmul_v0.cu (scattered):     tm=0 -> rows {4,5,6,7}   tm=1 -> rows {68,69,70,71}
                               (a "4-then-jump-64" pattern, needs the bit-trick)

matmul_v2_bfix.cu (linear):   tm=0->row 1, tm=1->row 17, tm=2->row 33, ..., tm=7->row 113
                               (a clean, even, 16-apart sequence -- no bit-trick needed)
```
The bit-trick isn't a "feature" matmul_v2_bfix.cu is missing -- it's a
repair for a scattering that only happens if your register-loading
scheme groups rows together via FLOAT4. Since matmul_v2_bfix.cu's
FLOAT4 usage groups K-values instead, there's simply nothing broken
that needs fixing in its epilogue.

---

# PART 39 -- The Exact Line-by-Line Smoking Gun: One Loop Bound Causes Everything

## 39.1 matmul_v0.cu -- the exact real code (lines 164-171)

```c
164:            }
165:            if (tk < BLOCK_K) {
166:  #pragma unroll
167:                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm++) {
168:                    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
169:                    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
170:                }
```
Line 167 is the single most important line: `tm < THREAD_C_Y_TILE_SIZE
>> 2`. With THREAD_C_Y_TILE_SIZE=8, this bound is `8 >> 2 = 2` -- tm
only ever takes the values 0 and 1, two total iterations, even though
the thread needs 8 rows total.

## 39.2 Walking through line 168 and 169, for both values of tm

```
tm=0:  line 168: r = (tCy + 0*16) << 2 = (1) << 2 = 4
       line 169: FLOAT4(tCsA[tk&1][0]) = ...   <- fills slots 0,1,2,3
                 the "<<2" scaling means this ONE source position (4)
                 actually represents 4 CONSECUTIVE real rows: 4,5,6,7

tm=1:  line 168: r = (tCy + 1*16) << 2 = (17) << 2 = 68
       line 169: FLOAT4(tCsA[tk&1][4]) = ...   <- fills slots 4,5,6,7
                 this source position (68) represents rows 68,69,70,71
```
This is exactly why only 2 tm values produce 4-row clusters each: the
loop only runs twice (line 167's `>>2` bound), and each single run's
FLOAT4 write (line 169) physically fills 4 register slots at once,
with the `<<2` scaling on line 168 meaning "one logical tm position
actually represents a whole group of 4 real rows."
```
                    line 167 loop bound: 8 >> 2 = 2
                              |
                 only 2 iterations possible: tm=0, tm=1
                              |
              line 168's "<<2" scales each tm into a GROUP of 4 rows
                              |
              line 169's FLOAT4 fills 4 register slots per iteration
                              |
                 RESULT: 8 total rows, but delivered as
                 2 GROUPS of 4 -- exactly the scattered pattern
```

## 39.3 matmul_v2_bfix.cu -- the exact real code (lines 138-145)

```c
138:    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
139:        // ---- Stage A: unchanged, still FLOAT4. ----
140:  #pragma unroll
141:        for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
142:            int m = tCy + tm * THREAD_C_LAYOUT_Y;
143:            FLOAT4(regA[tm][0]) = FLOAT4(sA[bufferId][m][0]);
144:            FLOAT4(regA[tm][4]) = FLOAT4(sA[bufferId][m][4]);
145:        }
```
Line 141 is the key contrast: `tm < THREAD_C_Y_TILE_SIZE` -- no `>>2`
shrinking at all. tm runs the FULL range 0..7, all 8 values, every
single one its own separate loop iteration.

## 39.4 Walking through line 142, 143, 144, for every tm

```
tm=0: line 142: m = tCy + 0*16 = 1 + 0  = 1      -> this tm owns ONLY row 1
tm=1: line 142: m = tCy + 1*16 = 1 + 16 = 17     -> this tm owns ONLY row 17
tm=2: line 142: m = tCy + 2*16 = 1 + 32 = 33     -> this tm owns ONLY row 33
tm=3: line 142: m = tCy + 3*16 = 1 + 48 = 49     -> this tm owns ONLY row 49
tm=4: line 142: m = tCy + 4*16 = 1 + 64 = 65     -> this tm owns ONLY row 65
tm=5: line 142: m = tCy + 5*16 = 1 + 80 = 81     -> this tm owns ONLY row 81
tm=6: line 142: m = tCy + 6*16 = 1 + 96 = 97     -> this tm owns ONLY row 97
tm=7: line 142: m = tCy + 7*16 = 1+112 = 113     -> this tm owns ONLY row 113
```
Line 142 has NO `<<2` scaling anywhere -- a plain, direct, unscaled
formula. Lines 143-144's two FLOAT4 calls both use this exact same m --
they're loading 8 K-values (columns 0-3, then 4-7) for that single row,
never touching any other row.
```
                 line 141 loop bound: FULL 8 (no shrinking)
                              |
                  8 separate iterations: tm=0..7
                              |
              line 142 has NO scaling -- each tm gets exactly ONE row
                              |
       lines 143-144's two FLOAT4 calls both target THAT SAME row m,
       grouping 4 K-VALUES together, never grouping multiple rows
                              |
                 RESULT: 8 total rows, delivered as
                 8 INDIVIDUAL, evenly-spaced rows -- no scattering
```

## 39.5 Side-by-side flowchart -- the ONE structural fork that causes everything downstream

```
                    Does the tm-loop bound include ">> 2"?
                                    |
                    YES (matmul_v0.cu, line 167)          NO (matmul_v2_bfix.cu, line 141)
                           |                                          |
                           v                                          v
             Only 2 tm iterations exist               All 8 tm iterations exist
             (8 rows must be squeezed                 (each tm gets its own,
              into just 2 "slots")                     dedicated row)
                           |                                          |
                           v                                          v
             Each slot's FLOAT4 write (line 169)       Each row's FLOAT4 writes
             must cover 4 REAL ROWS AT ONCE             (lines 143-144) cover 4
             (line 168's "<<2" scaling makes             K-VALUES of ONE row
              this explicit)                             (no scaling of m needed)
                           |                                          |
                           v                                          v
             tm=0->{4,5,6,7}  tm=1->{68,69,70,71}       tm=0->1, tm=1->17, ..., tm=7->113
             SCATTERED, needs epilogue bit-trick         LINEAR, no bit-trick needed
```

## 39.6 The one-sentence summary

Everything traces back to a single difference in the for loop's upper
bound: matmul_v0.cu's line 167 divides the row count by 4 (`>> 2`)
because it's using FLOAT4 to bundle 4 rows into each loop pass;
matmul_v2_bfix.cu's line 141 uses the full row count unchanged because
its FLOAT4 (lines 143-144) bundles 4 K-values within a single row
instead. That one loop-bound decision is the entire root cause of the
scattered-vs-linear row pattern, and therefore of needing (or not
needing) the epilogue's bit-trick.

*/