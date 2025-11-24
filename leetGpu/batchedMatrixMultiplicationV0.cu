#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>



// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define get_offset(T, ptr, row, col, ld) ((T*)((ptr) + (row) * (ld) + (col)))
#define local_tile(ptr, tileM, tileN, row, col, ld) ((ptr) + (row)*(tileM)*(ld) + (col)*(tileN))

template<int x>
__host__ __device__ static constexpr int get_log2x() {
    static_assert(x>0 and (x & (x-1)) == 0);
    int v = x;
    int res = 0;
    while (v>1) {
        v>>= 1;
        ++res;
    }
    return res;
}

__global__ void sgemm_Kernel_Naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int batch_id = blockIdx.z;
    A += batch_id * M * K;
    B += batch_id * N * K;
    C += batch_id * M * N;

    if (row<M and col<N) {
        float rC = 0.0f;
        for (int k=0; k<K; ++k) {
            rC += A[row*K + k] * B[k * N + col];
        }
        C[row * N + col] = rC;
    }
}

template <int TileM, int TileN, int cta_size, bool trans, typename vec_t, typename T>
__device__ __forceinline__ void load_Global_To_Shared(T* dst, const T* src, int ld_dst, int ld_src, int M, int N, int tid) {
    constexpr int vec_size = sizeof(vec_t) / sizeof(T);
    static_assert(TileN % vec_size == 0);
    static_assert(TileM * TileN / vec_size >= cta_size);
    constexpr int num_threads_per_N = TileN / vec_size;
    //constexpr int num_threads_per_M = cta_size / num_threads_per_N;
    constexpr int num_vec_elements = TileM * TileN  / vec_size;
    constexpr int num_loop = num_vec_elements / cta_size;
    static_assert(num_vec_elements % cta_size == 0);

    #pragma unroll
    for (int loopid=0, idx=tid; loopid<num_loop; ++loopid, idx+=cta_size) {
        int n = (idx & (num_threads_per_N-1)) * vec_size;
        int m = idx>>get_log2x<num_threads_per_N>();

        if (m>=M || n>=N) continue;

        if constexpr (!trans) {
            get_offset(vec_t, dst, m, n, ld_dst)[0] = get_offset(vec_t, src, m, n, ld_src)[0];
        } else {
            auto vec_v = get_offset(vec_t, src, m, n, ld_src)[0];

            #pragma unroll
            for (int vid=0; vid<vec_size; ++vid) {
                get_offset(T, dst, n+vid, m, ld_dst)[0] = reinterpret_cast<T*>(&vec_v)[vid];
            }
        }
    }
}



template <int BM, int BN, int BK, int WM, int WN, int WK, int TM, int TN, 
            typename vec_t, typename T>
__device__ __forceinline__ void load_Shared_To_Reg(T* tArA, T* tBrB, const T* sA, const T* sB, int wm_id, int wn_id, int wk_id, int tm_vec_id, int tn_vec_id) {
    // tArA: WK * TM, sA: BK * BM
    // tBrB: WK * TN, sB: BK * BN
    constexpr int vec_size = sizeof(vec_t) / sizeof(T);
    constexpr int TM_vec_num = TM / vec_size;
    constexpr int TN_vec_num = TN / vec_size;
    constexpr int WM_per_loop = WM / TM_vec_num;
    constexpr int WN_per_loop = WN / TN_vec_num;

    const auto* tAsA = local_tile(sA, WK, WM, wk_id, wm_id, BM);
    const auto* tBsB = local_tile(sB, WK, WN, wk_id, wn_id, BN);

    #pragma unroll
    for (int kid=0; kid<WK; ++kid) {
        //load A
        #pragma unroll
        for (int tm_loop=0; tm_loop<TM_vec_num; ++tm_loop) {
            int m = tm_loop * WM_per_loop + tm_vec_id * vec_size;
            int _m = tm_loop * vec_size;
            get_offset(vec_t, tArA, kid, _m, TM) [0] = get_offset(vec_t, tAsA, kid, m, BM) [0];
        }

        //load B
        #pragma unroll
        for (int tn_loop=0; tn_loop<TN_vec_num; ++tn_loop) {
            int n = tn_loop * WN_per_loop + tn_vec_id * vec_size;
            int _n = tn_loop * vec_size;
            get_offset(vec_t, tBrB, kid, _n, TN) [0] = get_offset(vec_t, tBsB, kid, n, BN) [0];
        }
    }
}


template <int WK, int TM, int TN, int TK, typename T>
__device__ __forceinline__ void mma(T* tCrC, const T* tArA, const T* tBrB)  {
    // static_assert(WK==4 and TM==8 and TN==8 and TK==1, "This MMA implementation is designed for WK=4, TM=8, TN=8, TK=1");
    // rA: WK * TM, rB: WK * TN, rC: TM * TN
    #pragma unroll
    for (int tk=0; tk<WK; tk+=TK) {
        #pragma unroll
        for (int k=0; k<TK; ++k) {
            int _k = tk + k;
            #pragma unroll
            for (int m=0; m<TM; ++m) {
                #pragma unroll
                for (int n=0; n<TN; ++n) {
                    tCrC[m * TN + n] += tArA[_k * TM + m] * tBrB[_k * TN + n];
                }
            }
        }
    }
}


template <int BM, int BN, int WM, int WN, int TM, int TN, typename VecT, typename T>
__device__ __forceinline__ void store_Reg_To_Global(T* tCgC, const T* tCrC, int ldc, int M, int N, int wm_id, int wn_id, int tm_vec_id, int tn_vec_id) {
    // tCgC: BM * BN, tCrC: TM * TN
    constexpr int VecSz = sizeof(VecT) / sizeof(T);
    constexpr int TM_vec_num = TM / VecSz;   // 8/4=2
    constexpr int TN_vec_num = TN / VecSz;
    constexpr int WM_per_loop = WM / TM_vec_num;  // 64/2=32
    constexpr int WN_per_loop = WN / TN_vec_num;  // 32/2=16

    auto* tCtCgC = local_tile(tCgC, WM, WN, wm_id, wn_id, ldc);

    int validM = M - wm_id * WM_per_loop;
    int validN = N - wn_id * WN_per_loop;
    //int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // if (blockIdx.x == 1 && blockIdx.y == 0 && wm_id == 0 && wn_id == 0) {
    //     printf("store_reg_to_global: WM_per_loop=%d, WN_per_loop=%d, TM_vec_num=%d, TN_vec_num=%d\n", WM_per_loop, WN_per_loop, TM_vec_num, TN_vec_num);
    //     printf("wm_id=%d, wn_id=%d, tm_vec_id=%d, tn_vec_id=%d, M=%d, N=%d, validM=%d, validN=%d\n", wm_id, wn_id, tm_vec_id, tn_vec_id, M, N, validM, validN);
    // }

    #pragma unroll
    for (int tm_loop = 0; tm_loop < TM_vec_num; ++tm_loop) {
        #pragma unroll
        for (int vid = 0; vid < VecSz; ++vid) {
            int m = tm_loop * WM_per_loop + tm_vec_id * VecSz + vid;
            int _m = tm_loop * VecSz + vid;
            #pragma unroll
            for (int tn_loop = 0; tn_loop < TN_vec_num; ++tn_loop) {
                int n = tn_loop * WN_per_loop + tn_vec_id * VecSz;
                int _n = tn_loop * VecSz;
                if (m < validM && n < validN) {
                    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
                    //     printf("store: tid=%d, validM=%d, validN=%d, wm_id=%d, wn_id=%d, m=%d, n=%d, _m=%d, _n=%d, tm_loop=%d, tn_loop=%d\n", tid, validM, validN, wm_id, wn_id, m, n, _m, _n, tm_loop, tn_loop);
                    // }
                    get_offset(VecT, tCtCgC, m, n, ldc)[0] = get_offset(VecT, tCrC, _m, _n, TN)[0];
                }
            }
        }
    }
}


template<typename T>
__device__ __forceinline__ void printTensor(const T* tensor, int rows, int cols, int ld) {
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            printf("%.2f ,", tensor[i*ld+j]);        
        }
        printf("\n");
    }
}

template <int BM, int BN, int BK, int WM, int WN, int WK, int TM, int TN, int TK, 
            int cta_size, typename vec_t>
__global__ __launch_bounds__(cta_size)
void sgemm_Kernel_Universal_Pipeline_TT(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    //int tidx = threadIdx.x;
    //int tidy = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.z;

    A += batch_id * M * K;
    B += batch_id * N * K;
    C += batch_id * M * N;

    extern __shared__ float smem[];
    float* sA[2] = {smem, smem+BK*BM};
    float* sB[2] = {smem+2*BK*BM, smem+2*BK*BM+BK*BN};

    int bmid = blockIdx.y;
    int bnid = blockIdx.x;

    int curr_buffer_id = 0;

    const int rest_m = M - bmid*BM;
    const int rest_n = N - bnid*BN;

    constexpr int vec_size = sizeof(vec_t) / sizeof(float);
    //constexpr int num_load_per_thread = (BM*BK/vec_size) / cta_size;
    //constexpr int num_elem_ld_per_row_A = BK / vec_size;
    //constexpr int num_elem_ld_per_row_B = BN / vec_size;

    auto* gA = A;
    auto* gB = B;
    auto* gC = C;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    auto* tCgC = local_tile(gC, BM, BN, bmid, bnid, ldc);

    int bkid=0;
    auto* tAgA = local_tile(gA, BM, BK, bmid, bkid, lda);
    auto* tBgB = local_tile(gB, BK, BN, bkid, bnid, ldb);

    //warp level
    constexpr int NWarps_dim_N = (BN/WN);
    constexpr int NWarps_dim_M = (BM/WM);
    static_assert((NWarps_dim_N & (NWarps_dim_N-1)) == 0);
    static_assert((NWarps_dim_M & (NWarps_dim_M-1)) == 0);
    const int warp_id = tid>>5;
    const int lane_id = tid & 0x1F;
    static_assert(get_log2x<NWarps_dim_N>() == 2);
    const int wm_id = warp_id >> get_log2x<NWarps_dim_N>();
    const int wn_id = warp_id & (NWarps_dim_N-1);

    //thread level
    //constexpr int TM_vec_num = TM / vec_size;
    constexpr int TN_vec_num = TN / vec_size;
    //constexpr int WM_per_loop = WM /TM_vec_num;
    constexpr int WN_per_loop = WN /TN_vec_num;
    const int tm_vec_id = lane_id >> get_log2x<WN_per_loop/vec_size>();
    const int tn_vec_id = lane_id & (WN_per_loop/vec_size-1);

    //thread register
    float tArA[WK*TM];
    float tBrB[WK*TN];
    float tCrC[TM*TN] = {0.0f};

    //load
    load_Global_To_Shared<BM, BK, cta_size, true, vec_t>(sA[curr_buffer_id], tAgA, BM, lda, rest_m, (K - bkid*BK), tid);
    load_Global_To_Shared<BK, BN, cta_size, false, vec_t>(sB[curr_buffer_id], tBgB, BN, ldb, (K - bkid*BK), rest_n, tid);
    __syncthreads();

    //no unrolling
    for(; bkid<K/BK-1; ++bkid) {
        auto next_buffer_id = 1^curr_buffer_id;
        //load
        auto* tAgA = local_tile(gA, BM, BK, bmid, bkid+1, lda);
        auto* tBgB = local_tile(gB, BK, BN, bkid+1, bnid, ldb);
        load_Global_To_Shared<BM, BK, cta_size, true, vec_t>(sA[next_buffer_id], tAgA, BM, lda, rest_m, (K-bkid*BK), tid);
        load_Global_To_Shared<BK, BN, cta_size, false, vec_t>(sB[next_buffer_id], tBgB, BN, ldb, (K-bkid*BK), rest_n, tid);

        #pragma unroll
        for(int wk_id=0; wk_id<BK/WK; ++wk_id) {
            //load reg
            load_Shared_To_Reg<BM, BN, BK, WM, WN, WK, TM, TN, vec_t>(tArA, tBrB, sA[curr_buffer_id], sB[curr_buffer_id], wm_id, wn_id, wk_id, tm_vec_id, tn_vec_id);
            //mma
            mma<WK, TM, TN, TK>(tCrC, tArA, tBrB);
        }

        //barrier
        __syncthreads();

        //switch buffer
        curr_buffer_id ^= 1;
    }

    #pragma unroll
    for(int wk_id=0; wk_id<BK/WK; ++wk_id) {
        //load reg
        load_Shared_To_Reg<BM, BN, BK, WM, WN, WK, TM, TN, vec_t>(tArA, tBrB, sA[curr_buffer_id], sB[curr_buffer_id], wm_id, wn_id, wk_id, tm_vec_id, tn_vec_id);
        //mma
        mma<WK, TM, TN, TK>(tCrC, tArA, tBrB);
    }

    //store
    store_Reg_To_Global<BM, BN, WM, WN, TM, TN, vec_t>(tCgC, tCrC, ldc, rest_m, rest_n, wm_id, wn_id, tm_vec_id, tn_vec_id);
}


template <int BM, int BN, int BK, int WM, int WN, int WK, int TM, int TN, int TK, 
            int cta_size, int M, int N, int K, typename vec_t>
__global__ __launch_bounds__(cta_size)
void sgemm_Kernel_Universal_Pipeline_TT_Specialized(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    //int tidx = threadIdx.x;
    //int tidy = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.z;

    A += batch_id * M * K;
    B += batch_id * N * K;
    C += batch_id * M * N;

    extern __shared__ float smem[];
    float* sA[2] = {smem, smem+BK*BM};
    float* sB[2] = {smem+2*BK*BM, smem+2*BK*BM+BK*BN};

    int bmid = blockIdx.y;
    int bnid = blockIdx.x;

    int curr_buffer_id = 0;

    const int rest_m = M - bmid*BM;
    const int rest_n = N - bnid*BN;

    constexpr int vec_size = sizeof(vec_t) / sizeof(float);
    //constexpr int num_load_per_thread = (BM*BK/vec_size) / cta_size;
    //constexpr int num_elem_ld_per_row_A = BK / vec_size;
    //constexpr int num_elem_ld_per_row_B = BN / vec_size;

    auto* gA = A;
    auto* gB = B;
    auto* gC = C;
    constexpr int lda = K;
    constexpr int ldb = N;
    constexpr int ldc = N;
    auto* tCgC = local_tile(gC, BM, BN, bmid, bnid, ldc);

    int bkid=0;
    auto* tAgA = local_tile(gA, BM, BK, bmid, bkid, lda);
    auto* tBgB = local_tile(gB, BK, BN, bkid, bnid, ldb);

    //warp level
    constexpr int NWarps_dim_N = (BN/WN);
    constexpr int NWarps_dim_M = (BM/WM);
    static_assert((NWarps_dim_N & (NWarps_dim_N-1)) == 0);
    static_assert((NWarps_dim_M & (NWarps_dim_M-1)) == 0);
    const int warp_id = tid>>5;
    const int lane_id = tid & 0x1F;
    static_assert(get_log2x<NWarps_dim_N>() == 2);
    const int wm_id = warp_id >> get_log2x<NWarps_dim_N>();
    const int wn_id = warp_id & (NWarps_dim_N-1);

    //thread level
    constexpr int TM_vec_num = TM / vec_size;
    constexpr int TN_vec_num = TN / vec_size;
    constexpr int WM_per_loop = WM /TM_vec_num;
    constexpr int WN_per_loop = WN /TN_vec_num;
    const int tm_vec_id = lane_id >> get_log2x<WM_per_loop/vec_size>();
    const int tn_vec_id = lane_id & (WN_per_loop/vec_size-1);

    //thread register
    float tArA[WK*TM];
    float tBrB[WK*TN];
    float tCrC[TM*TN] = {0.0f};

    //load
    load_Global_To_Shared<BM, BK, cta_size, true, vec_t>(sA[curr_buffer_id], tAgA, BM, lda, rest_m, (K - bkid*BK), tid);
    load_Global_To_Shared<BK, BN, cta_size, false, vec_t>(sB[curr_buffer_id], tBgB, BN, ldb, (K - bkid*BK), rest_n, tid);
    __syncthreads();

    //no unrolling
    for(; bkid<K/BK-1; ++bkid) {
        auto next_buffer_id = 1^curr_buffer_id;
        //load
        auto* tAgA = local_tile(gA, BM, BK, bmid, bkid+1, lda);
        auto* tBgB = local_tile(gB, BK, BN, bkid+1, bnid, ldb);
        load_Global_To_Shared<BM, BK, cta_size, true, vec_t>(sA[next_buffer_id], tAgA, BM, lda, rest_m, (K-bkid*BK), tid);
        load_Global_To_Shared<BK, BN, cta_size, false, vec_t>(sB[next_buffer_id], tBgB, BN, ldb, (K-bkid*BK), rest_n, tid);

        #pragma unroll
        for(int wk_id=0; wk_id<BK/WK; ++wk_id) {
            //load reg
            load_Shared_To_Reg<BM, BN, BK, WM, WN, WK, TM, TN, vec_t>(tArA, tBrB, sA[curr_buffer_id], sB[curr_buffer_id], wm_id, wn_id, wk_id, tm_vec_id, tn_vec_id);
            //mma
            mma<WK, TM, TN, TK>(tCrC, tArA, tBrB);
        }

        //barrier
        __syncthreads();

        //switch buffer
        curr_buffer_id ^= 1;
    }

    #pragma unroll
    for(int wk_id=0; wk_id<BK/WK; ++wk_id) {
        //load reg
        load_Shared_To_Reg<BM, BN, BK, WM, WN, WK, TM, TN, vec_t>(tArA, tBrB, sA[curr_buffer_id], sB[curr_buffer_id], wm_id, wn_id, wk_id, tm_vec_id, tn_vec_id);
        //mma
        mma<WK, TM, TN, TK>(tCrC, tArA, tBrB);
    }

    //store
    store_Reg_To_Global<BM, BN, WM, WN, TM, TN, vec_t>(tCgC, tCrC, ldc, rest_m, rest_n, wm_id, wn_id, tm_vec_id, tn_vec_id);
}


// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    auto launch_Naive = [&] () {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N+threadsPerBlock.x-1) / threadsPerBlock.x,
                           (M+threadsPerBlock.x-1) / threadsPerBlock.x,
                           BATCH);

        sgemm_Kernel_Naive<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K); 
    };
    constexpr int BM = 256;
    constexpr int BN = 128;
    constexpr int BK = 16;

    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int WK = 8;

    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int TK = 1;

    auto launch_Pipeline = [&] () {
        using vec_t = uint4;
        constexpr dim3 block_size(32, 16);
        constexpr int num_warps = (block_size.x * block_size.y) / 32;
        static_assert(num_warps == (BM/WM) * (BN/WN));

        const dim3 grid_size((N+BN-1)/BN, (M+BM-1)/BM, BATCH);
        constexpr int smem_size = 2*BK*(BM+BN) * sizeof(float);

        auto func = sgemm_Kernel_Universal_Pipeline_TT<BM, BN, BK, WM, WN, WK, TM, TN, TK, block_size.x * block_size.y, vec_t>;
        auto stream = cudaStream_t(0);
        auto func_attr = cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        CUDA_CHECK(func_attr);
        // printf("running sgemm_kernel_universal_pipeline_TT %d %d %d %d\n", gridSz.x, gridSz.y, blockSz.x, blockSz.y);
        func<<<grid_size, block_size, smem_size, stream>>>(A, B, C, M, N, K);
        //CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        printf("end sgemm_Kernel_Universal_Pipeline_TT_Specialized\n");
    };

    auto launch_Pipeline_Specialized = [&] () {
        using vec_t = uint4;
        constexpr dim3 block_size(32, 16);
        constexpr int num_warps = (block_size.x * block_size.y) / 32;
        static_assert(num_warps == (BM/WM) * (BN/WN));

        const dim3 grid_size((N+BN-1)/BN, (M+BM-1)/BM, BATCH);
        constexpr int smem_size = 2*BK*(BM+BN) * sizeof(float);

        auto func = sgemm_Kernel_Universal_Pipeline_TT_Specialized<BM, BN, BK, WM, WN, WK, TM, TN, TK, block_size.x * block_size.y, 8192, 6144, 4096, vec_t>;
        auto stream = cudaStream_t(0);
        auto func_attr = cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        CUDA_CHECK(func_attr);
        // printf("running sgemm_kernel_universal_pipeline_TT %d %d %d %d\n", gridSz.x, gridSz.y, blockSz.x, blockSz.y);
        func<<<grid_size, block_size, smem_size, stream>>>(A, B, C);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        printf("end sgemm_Kernel_Universal_Pipeline_TT_Specialized\n");
    };

    auto is_aligned = [] (const float* ptr, int size) {
        return (uintptr_t(ptr) % size) == 0;
    };
    if (M==8192 and N==6144 and K==4096) {
        // use the specialized kernel for perf test
        launch_Pipeline_Specialized();
    } else if (is_aligned(A, 16) and is_aligned(B, 16) and is_aligned(C, 16) and 
                M%8==0 and N%8==0 and K%8==0 and 
                M>=BM and N>=BN and K%BK==0) {
        // use the optimized kernel
        //printf()"using the optimized kernel with M=%d, N=%d, K=%d\n", M, N, K);
        launch_Pipeline();
    } else {
        // if M, N, K % 8 !=0, can not use the optimized kernel
        //printf("using naive kernel due to M=%d, N=%d, K=%d not being nultiple of 8 or too small.\n", M, N, K);
        launch_Naive();
    }

} 