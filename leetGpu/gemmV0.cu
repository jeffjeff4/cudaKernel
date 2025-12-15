//method0
/*
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#define DIV_UP(n, x) (((n) + (x) - 1) / (x))

const int BLOCK_SIZE_M = 64;
const int BLOCK_SIZE_N = 64;
const int BLOCK_SIZE_K = 64;
const int THREAD_SIZE_M = 8;
const int THREAD_SIZE_N = 8;

typedef struct __align__(8) {
    half x, y, z, w;
} half4_t;

__device__ __inline__ half4_t load_half4(const half* ptr) {
    return *reinterpret_cast<const half4_t*>(ptr);
}

__device__ __inline__ void store_half4(half* ptr, const half4_t &v) {
    *reinterpret_cast<half4_t*>(ptr) = v;
}


template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void halfKernelMM(const half* matrix_a, const half* matrix_b, half* matrix_c, int M, int N, int K, float alpha, float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int thread_blocks_m = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int thread_blocks_n = BLOCK_SIZE_N / THREAD_SIZE_N;

    const int thread_nums = thread_blocks_m * thread_blocks_n;

    const int ldg_a_num = (BLOCK_SIZE_M * BLOCK_SIZE_K / thread_nums / 4);
    const int ldg_b_num = (BLOCK_SIZE_N * BLOCK_SIZE_K / thread_nums / 4);

    int A_TILE_COL = (tid % (BLOCK_SIZE_K / 4)) * 4;
    int A_TILE_ROW = tid / (BLOCK_SIZE_K / 4);
    int A_TILE_ROW_STRIDE = BLOCK_SIZE_M;
    if (ldg_a_num > 0) {
        A_TILE_ROW_STRIDE /= ldg_a_num;
    }

    int B_TILE_COL = (tid % (BLOCK_SIZE_N / 4)) * 4;
    int B_TILE_ROW = tid / (BLOCK_SIZE_N / 4);
    int B_TILE_ROW_STRIDE = BLOCK_SIZE_K;
    //this is the bug
    if (ldg_a_num > 0) {
        A_TILE_ROW_STRIDE /= ldg_a_num;
    }

    extern __shared__ half sm_mem[];
    half* sm_A_matrix = sm_mem;
    half* sm_B_matrix = sm_mem + (size_t)BLOCK_SIZE_K * BLOCK_SIZE_M;

    auto SM_A = [&](int row_k, int col_m)->half& {
        return sm_A_matrix[(size_t)row_k * BLOCK_SIZE_M + col_m];
    };

    auto SM_B = [&](int row_k, int col_n)->half& {
        return sm_B_matrix[(size_t)row_k * BLOCK_SIZE_N + col_n];
    };

    const int LRG = 16;
    half ldg_a_reg[LRG];

    float reg_a[THREAD_SIZE_M];
    float reg_b[THREAD_SIZE_N];

    const half* base_A = matrix_a + (size_t)by * BLOCK_SIZE_M * (size_t)K;
    const half* base_B = matrix_b + (size_t)bx * BLOCK_SIZE_N;
    half* base_C = matrix_c + (size_t)by * BLOCK_SIZE_M * (size_t)N + (size_t)bx * BLOCK_SIZE_N;

    float sum[THREAD_SIZE_M][THREAD_SIZE_N];
    #pragma unroll
    for (int i=0; i<THREAD_SIZE_M; ++i) {
        //for (int j=0; j<THREAD_SIZE_M; ++j) {
        for (int j=0; j<THREAD_SIZE_N; ++j) {
            sum[i][j] = 0.0f;
        }
    }

    for (int bk=0; bk<K; bk += BLOCK_SIZE_K) {
        #pragma unroll
        for (int i=0; i<BLOCK_SIZE_M; i+= A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE;
                int a_row = i + A_TILE_ROW;
                //int a_col = A_TILE_ROW;
                int a_col = A_TILE_COL;
                int global_row = by * BLOCK_SIZE_M + a_row;
                int global_col = bk + a_col;

            if (global_row < M) {
                if (a_col+3<BLOCK_SIZE_K && (global_col+3) < K) {
                    const half* gptr = base_A + (size_t)a_row * (size_t)K + (size_t)a_col;
                    half4_t v = {__float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f) };

                    if ((global_col + 3) < K) {
                                v = load_half4(gptr);
                    }

                    SM_A(a_col+0, a_row) = v.x;
                    SM_A(a_col+1, a_row) = v.y;
                    SM_A(a_col+2, a_row) = v.z;
                    SM_A(a_col+3, a_row) = v.w;
                } else {
                    for (int vv=0; vv<4; ++vv) {
                        int gc = bk + a_col + vv;
                        half val = __float2half(0.0f);
                        if (gc<K) {
                            val = base_A[(size_t)a_row * (size_t)K + (size_t)(a_col + vv)];
                        }
                        SM_A(a_col+vv, a_row) = val;
                    }
                }
            } else {
                for (int vv=0; vv<4; ++vv) {
                    SM_A(a_col+vv, a_row) = __float2half(0.0f);
                }
            }
        }
        

        #pragma unroll
        for (int i=0; i<BLOCK_SIZE_K; i+= B_TILE_ROW_STRIDE) {
            int b_row = i + B_TILE_ROW;
            int b_col = B_TILE_COL;
            int global_row = bk +b_row;
            int global_col_base = bx * BLOCK_SIZE_N + b_col;

            if (global_row < K) {
                if (global_col_base + 3 < N) {
                    const half* gptr = base_B + (size_t)b_row * (size_t)N + (size_t)b_col;
                    half4_t v = load_half4(gptr);

                    SM_B(b_row, b_col+0) = v.x;
                    SM_B(b_row, b_col+1) = v.y;
                    SM_B(b_row, b_col+2) = v.z;
                    SM_B(b_row, b_col+3) = v.w;
                } else {
                    for (int vv=0; vv<4; ++vv) {
                        int gc = bx * BLOCK_SIZE_N + b_col + vv;
                        half val = __float2half(0.0f);
                        if (gc<N) {
                            val = base_B[(size_t)b_row * (size_t)N + (size_t)(b_col+vv)];
                        }
                        SM_B(b_row, b_col+vv) = val;
                    }
                }
            } else {
                for (int vv=0; vv<4; ++vv) SM_B(b_row, b_col+vv) = __float2half(0.0f);
                
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k_inner=0; k_inner<BLOCK_SIZE_K; ++k_inner) {
            #pragma unroll
            for (int rm=0; rm<THREAD_SIZE_M; ++rm) {
                    int row_in_block = ty * THREAD_SIZE_M + rm;
                    half aval = SM_A(k_inner, row_in_block);
                    reg_a[rm] = __half2float(aval);
            }
            #pragma unroll
            for (int rn=0; rn<THREAD_SIZE_N; ++rn) {
                    int col_in_block = tx * THREAD_SIZE_N + rn;
                    half bval = SM_B(k_inner, col_in_block);
                    reg_b[rn] = __half2float(bval);
            }
            #pragma unroll
            for (int rm=0; rm<THREAD_SIZE_M; ++rm) {
                #pragma unroll
                for (int rn=0; rn<THREAD_SIZE_N; ++rn) {
                    sum[rm][rn] += reg_a[rm] * reg_b[rn];
                }
            }
        }

        __syncthreads();
        base_A += BLOCK_SIZE_K;
        base_B += (size_t)BLOCK_SIZE_K * (size_t)N;
    }

    #pragma unroll
    for (int rm=0; rm<THREAD_SIZE_M; ++rm) {
        int global_row = by * BLOCK_SIZE_M + ty * THREAD_SIZE_M + rm;
        if (global_row >= M) continue;

        #pragma unroll
        for (int rn=0; rn<THREAD_SIZE_N; ++rn) {
            int global_col = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_N + rn;
            if (global_col >= N) continue;

            half oldh = base_C[(size_t)rm * (size_t)N + (size_t)rn];
            half existing = matrix_c[(size_t)global_row * (size_t)N + (size_t)global_col];
            float existing_f = __half2float(existing);
            float newval = alpha * sum[rm][rn] + beta * existing_f;
            matrix_c[(size_t)global_row * (size_t)N + (size_t)global_col] = __float2half(newval);
        }
    }
}



// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int BS_M = BLOCK_SIZE_M;
    const int BS_N = BLOCK_SIZE_N;
    const int BS_K = BLOCK_SIZE_K;
    const int TS_M = THREAD_SIZE_M;
    const int TS_N = THREAD_SIZE_N;

    dim3 threadsPerBlock(BS_N / TS_N, BS_M / TS_M);
    dim3 blockPerGrid(DIV_UP(N, BS_N), DIV_UP(M, BS_M));

    size_t shared_bytes = (size_t)BS_K * ((size_t)BS_M + (size_t)BS_N) * sizeof(half);

    halfKernelMM<BS_M, BS_N, BS_K, TS_M, TS_N><<<blockPerGrid, threadsPerBlock, shared_bytes>>>(A, B, C, M, N, K, alpha, beta);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel function launch failuer: %s\n", cudaGetErrorString(err));
    }
}
*/

//----------------------------------------------------------------------------------
//method1
/*
  sgemm_half_alpha_beta.cu

  实现内容：
  - 将提供的float4_Kernel_MM转换为基于half的核函数，在float中进行累加
  - 核函数计算分块GEMM并写入结果：
        C = alpha * (A * B) + beta * C
    其中A、B、C都是半精度（设备指针）
  - 保持原始代码中的分块/线程布局逻辑，同时选择一组安全的
    模板分块/线程大小，以及匹配原始tid映射的blockDim
  - 尽可能实现向量化的half4加载/存储（使用half4_t结构体），
    并在边界情况下使用标量回退
  - 提供solve(...)函数（extern "C"），使用请求的确切签名来
    启动核函数。未更改签名

  注意事项：
  - 核函数使用模板BLOCK大小确定共享内存大小。确保选择的模板
    参数适合GPU的共享内存；我选择了通常可用的保守默认值，但可以调整
  - 当索引完全在边界内时，假设地址对齐以进行简单的向量化加载/存储
  - 累加器使用float以保证数值稳定性，并在float中实现alpha/beta
  - 这是将原始算法直接转换为half操作数和alpha/beta语义
    进一步的性能调优（协作加载、填充、cp.async、张量核心）超出此转换范围
*/

/*
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#define DIV_UP(n,x) (((n)+(x)-1)/(x))

// 选择模板参数（可调优）
constexpr int BLOCK_SIZE_M_DEF = 64;  // 每个线程块处理的M维度大小
constexpr int BLOCK_SIZE_N_DEF = 64;  // 每个线程块处理的N维度大小
constexpr int BLOCK_SIZE_K_DEF = 16;  // 每个线程块处理的K维度大小
//constexpr int BLOCK_SIZE_K_DEF = 64;  // 每个线程块处理的K维度大小
constexpr int THREAD_SIZE_M_DEF = 8;  // 每个线程处理的M维度大小
constexpr int THREAD_SIZE_N_DEF = 8;  // 每个线程处理的N维度大小

// 定义half4类型，8字节对齐
typedef struct __align__(8) {
    half x, y, z, w;  // 四个half元素组成的向量
} half4_t;

// 辅助函数：从设备指针安全加载half4（如果完全在边界内，不检查对齐）
__device__ __inline__ half4_t load_half4(const half* ptr) {
    return *reinterpret_cast<const half4_t*>(ptr);  // 类型转换并加载
}
__device__ __inline__ void store_half4(half* ptr, const half4_t &v) {
    *reinterpret_cast<half4_t*>(ptr) = v;  // 类型转换并存储
}


//  转换后的核函数：操作半精度矩阵A(M×K)、B(K×N)、C(M×N)
//  在float中进行累加，最终写入时遵循 C = alpha * sum + beta * C
//  模板参数应适当划分矩阵大小以获得良好的分块效果

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void half_Kernel_MM(const half *matrix_a, const half *matrix_b, half *matrix_c, int M, int N, int K, float alpha, float beta)
{
    // 块索引（blockIdx.x遍历N块，blockIdx.y遍历M块）
    int bx = blockIdx.x;  // 列块索引
    int by = blockIdx.y;  // 行块索引

    int tx = threadIdx.x;  // 线程在x维度（0-7）
    int ty = threadIdx.y;  // 线程在y维度（0-7）
    const int tid = ty * blockDim.x + tx;  // 线程ID（0-63）

    // 块内沿M和N的"线程分块"数量
    const int thread_blocks_m = BLOCK_SIZE_M / THREAD_SIZE_M;  // M方向线程分块数 = 8
    const int thread_blocks_n = BLOCK_SIZE_N / THREAD_SIZE_N;  // N方向线程分块数 = 8

    const int thread_nums = thread_blocks_m * thread_blocks_n;  // 总线程数 = 64

    // 每个块每个矩阵需要多少向量加载（4个一组）- 保持与原始逻辑一致
    const int ldg_a_num = (BLOCK_SIZE_M * BLOCK_SIZE_K / thread_nums / 4);  // A的向量加载数
    const int ldg_b_num = (BLOCK_SIZE_N * BLOCK_SIZE_K / thread_nums / 4);  // B的向量加载数

    // 每个线程的分块读取索引（4个一组向量化）
    int A_TILE_COL = (tid % (BLOCK_SIZE_K / 4)) * 4;  // K分块内的列偏移（0..BLOCK_SIZE_K-4）
    int A_TILE_ROW = tid / (BLOCK_SIZE_K / 4);        // BLOCK_SIZE_M分块内的行偏移索引
    int A_TILE_ROW_STRIDE = BLOCK_SIZE_M / ( (ldg_a_num>0)? ldg_a_num : 1 );  // A的行加载步长

    int B_TILE_COL = (tid % (BLOCK_SIZE_N / 4)) * 4;  // BLOCK_SIZE_N内的列偏移
    int B_TILE_ROW = tid / (BLOCK_SIZE_N / 4);        // BLOCK_SIZE_K内的行偏移
    int B_TILE_ROW_STRIDE = BLOCK_SIZE_K / ( (ldg_b_num>0)? ldg_b_num : 1 );  // B的行加载步长

    // 分块的共享内存：注意排序与原始算法一致
    extern __shared__ half sm_mem[];  // 动态分配，以便我们可以在这里放置两个数组
    // 计算偏移量
    // 需要的大小：BLOCK_SIZE_K * BLOCK_SIZE_M + BLOCK_SIZE_K * BLOCK_SIZE_N
    half *sm_A_matrix = sm_mem;  // 大小 BLOCK_SIZE_K * BLOCK_SIZE_M
    half *sm_B_matrix = sm_mem + (size_t)BLOCK_SIZE_K * BLOCK_SIZE_M;  // 大小 BLOCK_SIZE_K * BLOCK_SIZE_N

    // 访问2D共享内存的宏（行主序）
    auto SM_A = [&](int row_k, int col_m)->half& {
        // sm_A_matrix存储为BLOCK_SIZE_K行，每行BLOCK_SIZE_M列
        return sm_A_matrix[(size_t)row_k * BLOCK_SIZE_M + col_m];
    };
    auto SM_B = [&](int row_k, int col_n)->half& {
        // sm_B_matrix存储为BLOCK_SIZE_K行，每行BLOCK_SIZE_N列
        return sm_B_matrix[(size_t)row_k * BLOCK_SIZE_N + col_n];
    };

    // 临时寄存器缓冲区（向量加载存储为half）
    // ldg_a_reg保存此线程读取的组（每组4个half）
    // 限制大小：4 * ldg_a_num
    // 为安全起见分配一些最大值
    const int LRG = 16;  // 大寄存器组大小
    half ldg_a_reg[LRG];  // 确保LRG >= 4 * ldg_a_num；在我们的默认设置下足够

    // 每个线程的小寄存器
    float reg_a[THREAD_SIZE_M];  // 存储A值的寄存器数组
    float reg_b[THREAD_SIZE_N];  // 存储B值的寄存器数组

    // 调整指针到块起始位置
    // matrix_a是M×K行主序：行步长为K
    // 将matrix_a移动到此块M区域的第一个行
    const half *base_A = matrix_a + (size_t)by * BLOCK_SIZE_M * (size_t)K;
    // matrix_b是K×N行主序：我们想要指向列块bx * BLOCK_SIZE_N的指针（在第二个维度偏移）
    const half *base_B = matrix_b + (size_t)bx * BLOCK_SIZE_N;
    // matrix_c是M×N行主序：块起始行和列
    half *base_C = matrix_c + (size_t)by * BLOCK_SIZE_M * (size_t)N + (size_t)bx * BLOCK_SIZE_N;

    // 每个线程的累加缓冲区
    float sum[THREAD_SIZE_M][THREAD_SIZE_N];  // 2D累加器数组
    #pragma unroll  // 循环展开优化
    for (int i=0;i<THREAD_SIZE_M;i++)
        for (int j=0;j<THREAD_SIZE_N;j++)
            sum[i][j] = 0.0f;  // 初始化累加器为0

    // 循环遍历K分块
    for (int bk = 0; bk < K; bk += BLOCK_SIZE_K) {
        // 加载A分块到共享内存（以类似于原始方式转置）
        #pragma unroll  // 循环展开优化
        for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
            int ldg_index = i / A_TILE_ROW_STRIDE;  // 加载索引
            int a_row = i + A_TILE_ROW;  // 在BLOCK_SIZE_M内
            int a_col = A_TILE_COL;      // 在BLOCK_SIZE_K内
            int global_row = by * BLOCK_SIZE_M + a_row;  // 全局行索引
            int global_col = bk + a_col;                 // 全局列索引
            
            if (global_row < M) {  // 检查行边界
                // 如果完全在K和N边界内，则向量化加载
                if (a_col + 3 < BLOCK_SIZE_K && (global_col + 3) < K) {
                    // 从全局A加载half4
                    const half *gptr = base_A + (size_t)a_row * (size_t)K + (size_t)a_col;
                    half4_t v = { __float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f) };
                    // 如果地址在整体边界内，则安全向量加载
                    if ((global_col + 3) < K) v = load_half4(gptr);
                    // 将组件存储到共享内存（注意sm布局）
                    SM_A(a_col + 0, a_row) = v.x;
                    SM_A(a_col + 1, a_row) = v.y;
                    SM_A(a_col + 2, a_row) = v.z;
                    SM_A(a_col + 3, a_row) = v.w;
                } else {
                    // 标量回退：逐个元素处理
                    for (int vv = 0; vv < 4; ++vv) {
                        int gc = bk + a_col + vv;  // 全局列索引
                        half val = __float2half(0.0f);  // 默认值为0
                        if (gc < K) val = base_A[(size_t)a_row * (size_t)K + (size_t)(a_col + vv)];  // 在边界内则加载实际值
                        SM_A(a_col + vv, a_row) = val;  // 存储到共享内存
                    }
                }
            } else {
                // 超出边界行：设置为零
                for (int vv = 0; vv < 4; ++vv) SM_A(a_col + vv, a_row) = __float2half(0.0f);
            }
        }

        // 加载B分块到共享内存
        #pragma unroll  // 循环展开优化
        for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            int b_row = i + B_TILE_ROW;  // 在BLOCK_SIZE_K内
            int b_col = B_TILE_COL;      // 在BLOCK_SIZE_N内
            int global_row = bk + b_row;  // K轴全局行索引
            int global_col_base = bx * BLOCK_SIZE_N + b_col;  // 基础全局列索引
            
            if (global_row < K) {  // 检查行边界
                if (global_col_base + 3 < N) {  // 检查列边界（向量化条件）
                    const half *gptr = base_B + (size_t)b_row * (size_t)N + (size_t)b_col;
                    half4_t v = load_half4(gptr);  // 向量化加载
                    SM_B(b_row, b_col + 0) = v.x;
                    SM_B(b_row, b_col + 1) = v.y;
                    SM_B(b_row, b_col + 2) = v.z;
                    SM_B(b_row, b_col + 3) = v.w;
                } else {
                    // 标量回退：逐个元素处理
                    for (int vv = 0; vv < 4; ++vv) {
                        int gc = bx * BLOCK_SIZE_N + b_col + vv;  // 全局列索引
                        half val = __float2half(0.0f);  // 默认值为0
                        if (gc < N) val = base_B[(size_t)b_row * (size_t)N + (size_t)(b_col + vv)];  // 在边界内则加载实际值
                        SM_B(b_row, b_col + vv) = val;  // 存储到共享内存
                    }
                }
            } else {
                // 超出边界行：填充0
                for (int vv = 0; vv < 4; ++vv) SM_B(b_row, b_col + vv) = __float2half(0.0f);
            }
        }

        __syncthreads();  // 同步所有线程，确保共享内存数据就绪

        // 计算此K分块的部分乘积
        #pragma unroll  // 循环展开优化
        for (int k_inner = 0; k_inner < BLOCK_SIZE_K; ++k_inner) {
            // 为此k_inner加载THREAD_SIZE_M个a值的向量
            #pragma unroll
            for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
                int row_in_block = ty * THREAD_SIZE_M + rm;  // 0..BLOCK_SIZE_M-1
                half aval = SM_A(k_inner, row_in_block);     // 从共享内存加载A值
                reg_a[rm] = __half2float(aval);             // 转换为float存储在寄存器
            }
            // 为此k_inner加载THREAD_SIZE_N个b值的向量
            #pragma unroll
            for (int rn = 0; rn < THREAD_SIZE_N; ++rn) {
                int col_in_block = tx * THREAD_SIZE_N + rn;  // 0..BLOCK_SIZE_N-1
                half bval = SM_B(k_inner, col_in_block);     // 从共享内存加载B值
                reg_b[rn] = __half2float(bval);             // 转换为float存储在寄存器
            }

            // rank-1更新：使用寄存器中的值更新累加器
            #pragma unroll
            for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
                #pragma unroll
                for (int rn = 0; rn < THREAD_SIZE_N; ++rn) {
                    sum[rm][rn] += reg_a[rm] * reg_b[rn];  // 乘积累加
                }
            }
        }

        __syncthreads();  // 同步，确保共享内存可安全覆盖

        // 将基础指针前进到下一个K分块
        // base_A在K维度右移BLOCK_SIZE_K
        base_A += BLOCK_SIZE_K;
        // base_B向前移动BLOCK_SIZE_K行，但由于B存储为K×N，我们添加BLOCK_SIZE_K * N个元素
        base_B += (size_t)BLOCK_SIZE_K * (size_t)N;
    } // 结束每个K分块的循环

    // 将结果写回C，结合alpha/beta参数
    // 每个线程生成THREAD_SIZE_M × THREAD_SIZE_N子块
    #pragma unroll
    for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
        int global_row = by * BLOCK_SIZE_M + ty * THREAD_SIZE_M + rm;  // 计算全局行索引
        if (global_row >= M) continue;  // 跳过超出边界的行
        
        #pragma unroll
        for (int rn = 0; rn < THREAD_SIZE_N; rn++) {
            int global_col = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_N + rn;  // 计算全局列索引
            if (global_col >= N) continue;  // 跳过超出边界的列
            
            // 读取现有的C值并组合
            half oldh = base_C[(size_t)rm * (size_t)N + (size_t)rn];  // 注意：如果base_C未重置则错误
            // 计算C中的正确指针：
            half existing = matrix_c[(size_t)global_row * (size_t)N + (size_t)global_col];  // 从全局内存读取现有值
            float existing_f = __half2float(existing);  // 转换为float
            float newval = alpha * sum[rm][rn] + beta * existing_f;  // 应用alpha和beta参数
            matrix_c[(size_t)global_row * (size_t)N + (size_t)global_col] = __float2half(newval);  // 转换回half并存储
        }
    }
}


//  solve: 主机启动器。我们使用选择的常量实例化模板化核函数
//  保持请求的确切签名

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // 选择实例化参数（可调优）
    constexpr int BS_M = BLOCK_SIZE_M_DEF;
    constexpr int BS_N = BLOCK_SIZE_N_DEF;
    constexpr int BS_K = BLOCK_SIZE_K_DEF;
    constexpr int TS_M = THREAD_SIZE_M_DEF;
    constexpr int TS_N = THREAD_SIZE_N_DEF;

    // 计算匹配tid映射逻辑的网格和块维度
    dim3 threadsPerBlock(BS_N / TS_N, BS_M / TS_M);  // tx: BLOCK_SIZE_N/THREAD_SIZE_N, ty: BLOCK_SIZE_M/THREAD_SIZE_M
    dim3 blocksPerGrid(DIV_UP(N, BS_N), DIV_UP(M, BS_M));  // 计算网格维度

    // 计算所需的共享内存大小：BLOCK_SIZE_K * BLOCK_SIZE_M + BLOCK_SIZE_K * BLOCK_SIZE_N个half
    size_t shared_bytes = (size_t)BS_K * ( (size_t)BS_M + (size_t)BS_N ) * sizeof(half);

    // 实例化并启动
    // 注意：核函数期望const half*，但我们的核函数签名对A、B使用const half*，对C使用half*
    // 在模板函数签名需要的地方，为核函数参数去掉const
    half_Kernel_MM<BS_M, BS_N, BS_K, TS_M, TS_N><<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(A, B, C, M, N, K, alpha, beta);

    cudaError_t err = cudaGetLastError();  // 检查启动错误
    if (err != cudaSuccess) {
        fprintf(stderr, "核函数启动失败: %s\n", cudaGetErrorString(err));
    }
}

*/

//----------------------------------------------------------------------------------
//method2

/*
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#define DIV_UP(n,x) (((n)+(x)-1)/(x))

// 选择模板参数（可调优）
constexpr int BLOCK_SIZE_M_DEF = 64;
constexpr int BLOCK_SIZE_N_DEF = 64;
constexpr int BLOCK_SIZE_K_DEF = 16;
constexpr int THREAD_SIZE_M_DEF = 8;
constexpr int THREAD_SIZE_N_DEF = 8;

// 定义half4类型
typedef struct __align__(8) {
    half x, y, z, w;
} half4_t;

// 辅助函数
__device__ __inline__ half4_t load_half4(const half* ptr) {
    return *reinterpret_cast<const half4_t*>(ptr);
}
__device__ __inline__ void store_half4(half* ptr, const half4_t &v) {
    *reinterpret_cast<half4_t*>(ptr) = v;
}


template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void half_Kernel_MM(const half *matrix_a, const half *matrix_b, half *matrix_c, int M, int N, int K, float alpha, float beta)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int thread_blocks_m = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int thread_blocks_n = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int thread_nums = thread_blocks_m * thread_blocks_n; // 64

    // Shared Memory 加载任务分配
    // 每个线程负责加载的总元素数量，用于计算线性加载步长
    const int A_TILE_ELEMENTS = BLOCK_SIZE_M * BLOCK_SIZE_K; // 64 * 16 = 1024
    const int B_TILE_ELEMENTS = BLOCK_SIZE_N * BLOCK_SIZE_K; // 64 * 16 = 1024
    const int A_LOAD_STEP = (A_TILE_ELEMENTS / thread_nums); // 1024 / 64 = 16 元素/线程
    const int B_LOAD_STEP = (B_TILE_ELEMENTS / thread_nums); // 16 元素/线程

    // 共享内存布局 (A: M x K, B: N x K)
    extern __shared__ half sm_mem[];
    half *sm_A_matrix = sm_mem;  // M * K
    half *sm_B_matrix = sm_mem + (size_t)BLOCK_SIZE_M * BLOCK_SIZE_K;  // N * K 

    // 访问 SM_A: M行，K列 (标准行主序)
    auto SM_A = [&](int row_m, int col_k)->half& {
        return sm_A_matrix[(size_t)row_m * BLOCK_SIZE_K + col_k];
    };
    // 访问 SM_B: N行，K列 (转置)
    auto SM_B = [&](int row_n, int col_k)->half& {
        return sm_B_matrix[(size_t)row_n * BLOCK_SIZE_K + col_k];
    };

    float reg_a[THREAD_SIZE_M];
    float reg_b[THREAD_SIZE_N];

    const half *base_A = matrix_a + (size_t)by * BLOCK_SIZE_M * (size_t)K;
    const half *base_B = matrix_b + (size_t)bx * BLOCK_SIZE_N;
    half *base_C = matrix_c + (size_t)by * BLOCK_SIZE_M * (size_t)N + (size_t)bx * BLOCK_SIZE_N;

    float sum[THREAD_SIZE_M][THREAD_SIZE_N];
    #pragma unroll
    for (int i=0;i<THREAD_SIZE_M;i++)
        for (int j=0;j<THREAD_SIZE_N;j++)
            sum[i][j] = 0.0f;

    // K分块主循环
    for (int bk = 0; bk < K; bk += BLOCK_SIZE_K) {
        
        // --- 1. 加载 A 矩阵 (M x K -> M x K, 无转置) ---
        // 采用线性索引循环，确保小矩阵的所有元素都被访问
        for (int ldg_start = tid * A_LOAD_STEP; ldg_start < (tid + 1) * A_LOAD_STEP; ldg_start += 4) {
            int a_row = ldg_start / BLOCK_SIZE_K; // M 维
            int a_col = ldg_start % BLOCK_SIZE_K; // K 维
            
            int global_row = by * BLOCK_SIZE_M + a_row;
            int global_col = bk + a_col;
            
            half4_t v = { __float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f) };

            if (global_row < M) {  // M 边界检查
                // 向量化加载：从 A[M, K]
                if (a_col + 3 < BLOCK_SIZE_K && (global_col + 3) < K) { // K 边界检查
                    const half *gptr = base_A + (size_t)a_row * (size_t)K + (size_t)a_col;
                    v = load_half4(gptr);
                } else {
                    // 标量回退：逐个元素处理
                    for (int vv = 0; vv < 4; ++vv) {
                        int gc = bk + a_col + vv;
                        if (gc < K) {
                            // 使用正确的索引加载单个元素
                            *((half*)&v + vv) = base_A[(size_t)global_row * (size_t)K + (size_t)(global_col + vv)];
                        }
                    }
                }
            } 
            // 写入 SM_A (M行, K列)
            SM_A(a_row, a_col + 0) = v.x;
            SM_A(a_row, a_col + 1) = v.y;
            SM_A(a_row, a_col + 2) = v.z;
            SM_A(a_row, a_col + 3) = v.w;
        }
        
        // --- 2. 加载 B 矩阵 (K x N -> N x K, 转置) ---
        for (int ldg_start = tid * B_LOAD_STEP; ldg_start < (tid + 1) * B_LOAD_STEP; ldg_start += 4) {
            int b_row = ldg_start / BLOCK_SIZE_N; // K 维
            int b_col = ldg_start % BLOCK_SIZE_N; // N 维
            
            int global_row = bk + b_row;    // K 轴全局行索引
            int global_col = bx * BLOCK_SIZE_N + b_col;
            
            half4_t v = { __float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f) };

            if (global_row < K) {
                if (b_col + 3 < BLOCK_SIZE_N && (global_col + 3) < N) {
                    const half *gptr = matrix_b + (size_t)global_row * (size_t)N + (size_t)global_col;
                    v = load_half4(gptr);
                    
                    // 写入 SM_B (N行, K列) - 转置
                    SM_B(b_col + 0, b_row) = v.x;
                    SM_B(b_col + 1, b_row) = v.y;
                    SM_B(b_col + 2, b_row) = v.z;
                    SM_B(b_col + 3, b_row) = v.w;
                } else {
                    // 标量回退：逐个元素处理
                    for (int vv = 0; vv < 4; ++vv) {
                        int gc = bx * BLOCK_SIZE_N + b_col + vv;
                        if (global_row < K && gc < N) {
                            // 使用正确的索引加载单个元素
                            *((half*)&v + vv) = matrix_b[(size_t)global_row * (size_t)N + (size_t)gc]; 
                        }
                        SM_B(b_col + vv, b_row) = *((half*)&v + vv); 
                    }
                }
            }
        }

        __syncthreads();  

        // --- K 维度限制 (K=3) ---
        int K_limit_in_block = K - bk;
        int k_inner_max = (K_limit_in_block < BLOCK_SIZE_K) ? K_limit_in_block : BLOCK_SIZE_K;

        // --- 3. 计算阶段 (Compute) ---
        #pragma unroll
        for (int k_inner = 0; k_inner < k_inner_max; ++k_inner) { 
            
            #pragma unroll
            for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
                int row_in_block = ty * THREAD_SIZE_M + rm;  
                half aval = SM_A(row_in_block, k_inner);     
                reg_a[rm] = __half2float(aval);
            }
            #pragma unroll
            for (int rn = 0; rn < THREAD_SIZE_N; ++rn) {
                int col_in_block = tx * THREAD_SIZE_N + rn;
                half bval = SM_B(col_in_block, k_inner);     
                reg_b[rn] = __half2float(bval);
            }

            // Rank-1 Update
            #pragma unroll
            for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
                #pragma unroll
                for (int rn = 0; rn < THREAD_SIZE_N; ++rn) {
                    sum[rm][rn] += reg_a[rm] * reg_b[rn];
                }
            }
        }

        __syncthreads();  

        base_A += BLOCK_SIZE_K;
        base_B += (size_t)BLOCK_SIZE_K * (size_t)N;
    } // 结束K分块循环

    // --- 4. 写回 C 矩阵 ---
    #pragma unroll
    for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
        int global_row = by * BLOCK_SIZE_M + ty * THREAD_SIZE_M + rm;
        if (global_row >= M) continue;

        #pragma unroll
        for (int rn = 0; rn < THREAD_SIZE_N; rn++) {
            int global_col = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_N + rn;
            if (global_col >= N) continue;

            half existing = matrix_c[(size_t)global_row * (size_t)N + (size_t)global_col];
            float existing_f = __half2float(existing);
            float newval = alpha * sum[rm][rn] + beta * existing_f;
            matrix_c[(size_t)global_row * (size_t)N + (size_t)global_col] = __float2half(newval);
        }
    }
}


//  solve: 主机启动器

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int BS_M = BLOCK_SIZE_M_DEF;
    constexpr int BS_N = BLOCK_SIZE_N_DEF;
    constexpr int BS_K = BLOCK_SIZE_K_DEF;
    constexpr int TS_M = THREAD_SIZE_M_DEF;
    constexpr int TS_N = THREAD_SIZE_N_DEF;

    dim3 threadsPerBlock(BS_N / TS_N, BS_M / TS_M);  
    dim3 blocksPerGrid(DIV_UP(N, BS_N), DIV_UP(M, BS_M));  

    size_t shared_bytes = (size_t)BS_K * ( (size_t)BS_M + (size_t)BS_N ) * sizeof(half);

    half_Kernel_MM<BS_M, BS_N, BS_K, TS_M, TS_N><<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(A, B, C, M, N, K, alpha, beta);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "核函数启动失败: %s\n", cudaGetErrorString(err));
    }
}
*/

//----------------------------------------------------------------------------------
//method3
//把 method1 改写成 method2 的风格

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#define DIV_UP(n,x) (((n)+(x)-1)/(x))

// 选择模板参数（可调优）
constexpr int BLOCK_SIZE_M_DEF = 64;
constexpr int BLOCK_SIZE_N_DEF = 64;
constexpr int BLOCK_SIZE_K_DEF = 16;
constexpr int THREAD_SIZE_M_DEF = 8;
constexpr int THREAD_SIZE_N_DEF = 8;

// 定义half4类型
typedef struct __align__(8) {
    half x, y, z, w;
} half4_t;

// 辅助函数
__device__ __inline__ half4_t load_half4(const half* ptr) {
    return *reinterpret_cast<const half4_t*>(ptr);
}
__device__ __inline__ void store_half4(half* ptr, const half4_t &v) {
    *reinterpret_cast<half4_t*>(ptr) = v;
}


template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_M, const int THREAD_SIZE_N>
__global__ void half_Kernel_MM(const half *matrix_a, const half *matrix_b, half *matrix_c, int M, int N, int K, float alpha, float beta)
{
    // 块索引和线程索引
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int thread_blocks_m = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int thread_blocks_n = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int thread_nums = thread_blocks_m * thread_blocks_n; // 64

    // Shared Memory 加载任务分配 (使用线性分摊逻辑)
    const int A_TILE_ELEMENTS = BLOCK_SIZE_M * BLOCK_SIZE_K; // M * K
    const int B_TILE_ELEMENTS = BLOCK_SIZE_N * BLOCK_SIZE_K; // N * K
    // 确保加载步长至少为 1
    const int A_LOAD_STEP = (A_TILE_ELEMENTS / thread_nums); 
    const int B_LOAD_STEP = (B_TILE_ELEMENTS / thread_nums);

    // --- 共享内存布局 (Code 1 风格：A是 K x M (转置), B是 K x N (标准)) ---
    extern __shared__ half sm_mem[];
    half *sm_A_matrix = sm_mem;  // K * M (BLOCK_SIZE_K * BLOCK_SIZE_M)
    half *sm_B_matrix = sm_mem + (size_t)BLOCK_SIZE_K * BLOCK_SIZE_M;  // K * N (BLOCK_SIZE_K * BLOCK_SIZE_N)

    // 访问 SM_A: K行，M列 (转置)
    auto SM_A = [&](int row_k, int col_m)->half& {
        return sm_A_matrix[(size_t)row_k * BLOCK_SIZE_M + col_m];
    };
    // 访问 SM_B: K行，N列 (标准)
    auto SM_B = [&](int row_k, int col_n)->half& {
        return sm_B_matrix[(size_t)row_k * BLOCK_SIZE_N + col_n];
    };

    // 寄存器初始化
    float reg_a[THREAD_SIZE_M];
    float reg_b[THREAD_SIZE_N];

    // 调整指针到块起始位置
    const half *base_A = matrix_a + (size_t)by * BLOCK_SIZE_M * (size_t)K;
    const half *base_B = matrix_b + (size_t)bx * BLOCK_SIZE_N;
    half *base_C = matrix_c + (size_t)by * BLOCK_SIZE_M * (size_t)N + (size_t)bx * BLOCK_SIZE_N;

    float sum[THREAD_SIZE_M][THREAD_SIZE_N];
    #pragma unroll
    for (int i=0;i<THREAD_SIZE_M;i++)
        for (int j=0;j<THREAD_SIZE_N;j++) // Code 1 Bug fix: 使用 THREAD_SIZE_N
            sum[i][j] = 0.0f;

    // K分块主循环
    for (int bk = 0; bk < K; bk += BLOCK_SIZE_K) {
        
        // --- 1. 加载 A 矩阵 (M x K -> K x M, 转置) ---
        // 采用 Code 2 的线性加载分摊逻辑
        for (int ldg_start = tid * A_LOAD_STEP; ldg_start < (tid + 1) * A_LOAD_STEP; ldg_start += 4) {
            int a_row = ldg_start / BLOCK_SIZE_K; // GM 的 M 维 (行)
            int a_col = ldg_start % BLOCK_SIZE_K; // GM 的 K 维 (列)
            
            int global_row = by * BLOCK_SIZE_M + a_row;
            int global_col = bk + a_col;
            
            half4_t v = { __float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f) };

            if (global_row < M) {
                if (a_col + 3 < BLOCK_SIZE_K && (global_col + 3) < K) { 
                    const half *gptr = base_A + (size_t)a_row * (size_t)K + (size_t)a_col;
                    v = load_half4(gptr);
                } else { // 标量回退
                    for (int vv = 0; vv < 4; ++vv) {
                        int gc = bk + a_col + vv;
                        if (global_row < M && gc < K) {
                            *((half*)&v + vv) = matrix_a[(size_t)global_row * (size_t)K + (size_t)gc]; 
                        }
                    }
                }
            } 
            // 写入 SM_A (K行, M列) - 转置
            SM_A(a_col + 0, a_row) = v.x;
            SM_A(a_col + 1, a_row) = v.y;
            SM_A(a_col + 2, a_row) = v.z;
            SM_A(a_col + 3, a_row) = v.w;
        }
        
        // --- 2. 加载 B 矩阵 (K x N -> K x N, 无转置) ---
        for (int ldg_start = tid * B_LOAD_STEP; ldg_start < (tid + 1) * B_LOAD_STEP; ldg_start += 4) {
            int b_row = ldg_start / BLOCK_SIZE_N; // K 维
            int b_col = ldg_start % BLOCK_SIZE_N; // N 维
            
            int global_row = bk + b_row;    
            int global_col = bx * BLOCK_SIZE_N + b_col;
            
            half4_t v = { __float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f) };

            if (global_row < K) {
                if (b_col + 3 < BLOCK_SIZE_N && (global_col + 3) < N) {
                    const half *gptr = matrix_b + (size_t)global_row * (size_t)N + (size_t)global_col;
                    v = load_half4(gptr);
                    
                    // 写入 SM_B (K行, N列) - 无转置
                    SM_B(b_row, b_col + 0) = v.x;
                    SM_B(b_row, b_col + 1) = v.y;
                    SM_B(b_row, b_col + 2) = v.z;
                    SM_B(b_row, b_col + 3) = v.w;
                } else {
                    // 标量回退：逐个元素处理
                    for (int vv = 0; vv < 4; ++vv) {
                        int gc = bx * BLOCK_SIZE_N + b_col + vv;
                        if (global_row < K && gc < N) {
                            *((half*)&v + vv) = matrix_b[(size_t)global_row * (size_t)N + (size_t)gc]; 
                        }
                        SM_B(b_row, b_col + vv) = *((half*)&v + vv); 
                    }
                }
            }
        }

        __syncthreads();  

        // --- K 维度限制 (K=3) ---
        int K_limit_in_block = K - bk;
        int k_inner_max = (K_limit_in_block < BLOCK_SIZE_K) ? K_limit_in_block : BLOCK_SIZE_K;

        // --- 3. 计算阶段 (Compute) ---
        #pragma unroll
        for (int k_inner = 0; k_inner < k_inner_max; ++k_inner) { 
            
            // Load A (K x M -> M x K)
            #pragma unroll
            for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
                int row_in_block = ty * THREAD_SIZE_M + rm;  
                half aval = SM_A(k_inner, row_in_block);     // K行, M列
                reg_a[rm] = __half2float(aval);
            }
            // Load B (K x N -> K x N)
            #pragma unroll
            for (int rn = 0; rn < THREAD_SIZE_N; ++rn) {
                int col_in_block = tx * THREAD_SIZE_N + rn;
                half bval = SM_B(k_inner, col_in_block);     // K行, N列
                reg_b[rn] = __half2float(bval);
            }

            // Rank-1 Update
            #pragma unroll
            for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
                #pragma unroll
                for (int rn = 0; rn < THREAD_SIZE_N; ++rn) {
                    sum[rm][rn] += reg_a[rm] * reg_b[rn];  
                }
            }
        }

        __syncthreads();  

        base_A += BLOCK_SIZE_K;
        base_B += (size_t)BLOCK_SIZE_K * (size_t)N;
    } // 结束K分块循环

    // --- 4. 写回 C 矩阵 ---
    #pragma unroll
    for (int rm = 0; rm < THREAD_SIZE_M; ++rm) {
        int global_row = by * BLOCK_SIZE_M + ty * THREAD_SIZE_M + rm;
        if (global_row >= M) continue;

        #pragma unroll
        for (int rn = 0; rn < THREAD_SIZE_N; rn++) {
            int global_col = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_N + rn;
            if (global_col >= N) continue;

            half existing = matrix_c[(size_t)global_row * (size_t)N + (size_t)global_col];
            float existing_f = __half2float(existing);
            float newval = alpha * sum[rm][rn] + beta * existing_f;
            matrix_c[(size_t)global_row * (size_t)N + (size_t)global_col] = __float2half(newval);
        }
    }
}


/*
  solve: 主机启动器
*/
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int BS_M = BLOCK_SIZE_M_DEF;
    constexpr int BS_N = BLOCK_SIZE_N_DEF;
    constexpr int BS_K = BLOCK_SIZE_K_DEF;
    constexpr int TS_M = THREAD_SIZE_M_DEF;
    constexpr int TS_N = THREAD_SIZE_N_DEF;

    dim3 threadsPerBlock(BS_N / TS_N, BS_M / TS_M);  
    dim3 blocksPerGrid(DIV_UP(N, BS_N), DIV_UP(M, BS_M));  

    size_t shared_bytes = (size_t)BS_K * ( (size_t)BS_M + (size_t)BS_N ) * sizeof(half);

    half_Kernel_MM<BS_M, BS_N, BS_K, TS_M, TS_N><<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(A, B, C, M, N, K, alpha, beta);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "核函数启动失败: %s\n", cudaGetErrorString(err));
    }
}
