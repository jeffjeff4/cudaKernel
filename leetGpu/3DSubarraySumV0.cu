
#include <cuda_runtime.h>

#define THREAD_NUM_X		16
#define THREAD_NUM_Y		16
#define THREAD_NUM_Z		4
#define WARP_SIZE           32
#define STRIDE_LENGTH       8
#define DIV_UP(n, x)       ((n+(x)-1)/(x))  // x一定要加上括号!

__device__ __forceinline__ int warp_sum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 分别为深度/行数/列数
__global__ void subarray_sum_3d_equal_kernel(const int* __restrict__ input, int* output, int ndeps, int nrows, int ncols,
int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    /* 0.每个线程每个方向上读取 STRIDE_LENGTH个元素并求和,每个元素在 input中间隔 WARP_SIZE
       以保证每个块读取连续 WARP_SIZE，合并内存访问 */
    int tcol = blockIdx.x * blockDim.x * STRIDE_LENGTH + threadIdx.x;
    int trow = blockIdx.y * blockDim.y * STRIDE_LENGTH + threadIdx.y;
    int tdep = blockIdx.z * blockDim.z * STRIDE_LENGTH + threadIdx.z;

    int sum_val = 0;
    // 三维矩阵是行主序存储的,按照 Z/Y/X方式是可以连续读取的 
    for (int i=0; i<STRIDE_LENGTH; ++i) {
        // 比如,0号线程读取的就是 Z方向上块 0、块 1、...块STRIDE_LENGTH-1的 0号位置元素
        int dep = tdep + i*blockDim.z;
        for (int j=0; j<STRIDE_LENGTH; ++j) {
            int row = trow + j*blockDim.y;
            for (int k=0; k<STRIDE_LENGTH; ++k) {
                int col = tcol + k*blockDim.x;
                if (dep < ndeps && row < nrows && col < ncols) {
                    int pos = (dep+S_DEP)*M*K+(row+S_ROW)*K+(col+S_COL);
                    sum_val += input[pos];
                }
            }
        }
    }

    // 1.每个 warp就是 32个线程,一个块 1024个线程时就是 32个 warp,与维度无关
    // 在每个 warp内规约求和，并将其部分求和结果存储到共享内存中
    __shared__ int shared_partial_sum[WARP_SIZE];
        // 三维块内线性索引（行主序，x变化最快，然后y，最后z）
    int tid = threadIdx.z * (blockDim.x * blockDim.y) +  threadIdx.y * blockDim.x +  threadIdx.x;
    int warp = tid >> 5;    // 当前线程所在的 warp在整个 warp数组中的下标
    int lane = tid & 31;    // 当前线程在当前 warp内的下标

    int wsum = warp_sum(sum_val);
    if (lane == 0) {
        shared_partial_sum[warp] = wsum;
    }
    __syncthreads();

    // 2.将每个块内所有 warp已得到的部分求和结果再进行规约求和
    if (warp == 0) {
        int partial_sum_val = shared_partial_sum[lane];
        shared_partial_sum[0] = warp_sum(partial_sum_val);
    }

    // 3.利用原子加操作,对所有块内的 shared_partial_sum[0]求和
    if (tid == 0) {
        atomicAdd(output, shared_partial_sum[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
// N/M/K分别为深度/行数/列数
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    // 共计 1024个线程
    dim3 threadsPerBlock(THREAD_NUM_X, THREAD_NUM_Y, THREAD_NUM_Z);
	int ncols = E_COL - S_COL + 1;
	int nrows = E_ROW - S_ROW + 1;
	int ndeps = E_DEP - S_DEP + 1;
    dim3 blocksPerGrid(DIV_UP(ncols, THREAD_NUM_X*STRIDE_LENGTH), DIV_UP(nrows, THREAD_NUM_Y*STRIDE_LENGTH), DIV_UP(ndeps, THREAD_NUM_Z*STRIDE_LENGTH));

    // 确保output初始化为0
    cudaMemset(output, 0, sizeof(int));

    subarray_sum_3d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, ndeps, nrows, ncols, N, M, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL);
    cudaDeviceSynchronize();
}



//--------------------------------------------------------------------------------------------------
/*
question0:
不理解，请解释，用例子

这段代码实现了一个高效的 **并行三维子数组求和 (Parallel 3D Subarray Summation)** 算法。它的目标是计算一个大型三维数组（张量）中，用户指定的一个子立方体区域内所有元素的总和。

它采用了 **三级并行归约** 和 **3D Block-Stride 访问** 策略来最大化 GPU 的并行性。

-----

## ⚙️ I. 核心目标与分工机制

### 1\. 目标

计算 \sum input[d, r, c]，其中 d, r, c 位于用户指定的 3D 边界 (S_DEP \dots E_DEP, S_ROW \dots E_ROW, S_COL \dots E_COL) 内。

### 2\. 宏定义和分块

| 宏 | 值 | 含义 |
| :--- | :--- | :--- |
| THREAD_NUM_X, Y, Z | 16, 16, 4 | 线程块的 3D 维度 |
| STRIDE_LENGTH | 8 | 粗粒度因子（每个线程负责的循环次数） |
| **总线程数** | 16 \times 16 \times 4 = 1024 | |

-----

## 🚀 II. Kernel 内部流程

### 1\. 线程到 3D 空间的映射

c
int tcol = blockIdx.x * blockDim.x * STRIDE_LENGTH + threadIdx.x;
int trow = blockIdx.y * blockDim.y * STRIDE_LENGTH + threadIdx.y;
int tdep = blockIdx.z * blockDim.z * STRIDE_LENGTH + threadIdx.z;


  * **目的:** 计算当前线程在子数组（待求和的矩形区域）中的**起始坐标** (tdep, trow, tcol)。
  * **原理:** \text{起始点 = \text{Block 偏移 + \text{Thread 偏移。
      * \text{Block 偏移 = blockIdx \times blockDim \times STRIDE_LENGTH
      * **示例:** 线程 tx=5 位于 bx=1 的 Block。 tcol = 1 \times 16 \times 8 + 5 = 128 + 5 = 133。

### 2\. 局部数据加载和累加 (3D Block-Stride Loop)

c
int sum_val = 0;
for (int i=0; i<STRIDE_LENGTH; ++i) { // 深度 (Z)
    int dep = tdep + i*blockDim.z;
    for (int j=0; j<STRIDE_LENGTH; ++j) { // 行 (Y)
        int row = trow + j*blockDim.y;
        for (int k=0; k<STRIDE_LENGTH; ++k) { // 列 (X)
            int col = tcol + k*blockDim.x;
            if (dep < ndeps && row < nrows && col < ncols) {
                // ... (计算全局索引并累加)
                int pos = (dep+S_DEP)*M*K+(row+S_ROW)*K+(col+S_COL);
                sum_val += input[pos];
            
        
    



  * **目的:** 每个线程计算它负责的 STRIDE_LENGTH^3 = 8^3 = 512 个元素的局部和 sum_val。
  * **步长:** 循环中的步长是 blockDim.z (4), blockDim.y (16), blockDim.x (16)。
  * **交错访问:** 这种 **Block-Stride** 循环确保了所有线程能够协作，以 blockDim 为步长交错访问数据，从而实现内存合并。
  * **全局索引 (`pos`):**
      * **核心:** `pos` 结合了线程的**局部坐标** (dep, row, col) 和子数组的**全局起始偏移** (S_DEP, S_ROW, S_COL)，计算出该点在原始 N \times M \times K 矩阵中的一维索引。

> **示例:** 线程 tx=0 (Block 0) 在 i=0, j=0, k=0 时，计算 pos:
> pos = (0+S_{\text{DEP) \times M \times K + (0+S_{\text{ROW) \times K + (0+S_{\text{COL)
> 线程读取子数组 (S_DEP, S_ROW, S_COL) 处的元素。

### 3\. 第一级归约：Warp 内部求和 (Shuffle)

c
int tid = threadIdx.z * (blockDim.x * blockDim.y) +  threadIdx.y * blockDim.x +  threadIdx.x; // 线性 tid
// ...
int wsum = warp_sum(sum_val);
if (lane == 0) {
    shared_partial_sum[warp] = wsum;

__syncthreads();


  * **线性化 tid:** 3D 的 threadIdx 被线性化为 1D 索引 tid (0 到 1023)。
  * **Warp 归约:** 使用 `warp_sum` (Shuffle 归约) 将 32 个线程的 sum_val 累加起来，结果存储在 lane=0 的线程中。
  * **Shared Memory 存储:** 32 个 Warp Leader 将各自的总和写入 shared_partial_sum 数组。

### 4\. 第二级归约：Block 内部求和 (Shuffle)

c
if (warp == 0) {
    int partial_sum_val = shared_partial_sum[lane];
    shared_partial_sum[0] = warp_sum(partial_sum_val);



  * **目的:** 将 Shared Memory 中的 32 个 Warp 总和，通过 **Warp 0** 进行二次 **Shuffle 归约**。
  * **结果:** 整个 Block 的最终总和集中在 shared_partial_sum[0] 中。

### 5\. 最终原子累加

c
if (tid == 0) {
    atomicAdd(output, shared_partial_sum[0]);



  * 只有 Block 内的第一个线程将总和安全地累加到全局的 output 变量中。

## 💻 III. 主机端求解 (`solve` 函数)

`solve` 函数计算了子数组的维度，并配置 Grid/Block：

c
// ...
dim3 threadsPerBlock(THREAD_NUM_X, THREAD_NUM_Y, THREAD_NUM_Z); // 16x16x4 = 1024 threads
dim3 blocksPerGrid(DIV_UP(ncols, 128), DIV_UP(nrows, 128), DIV_UP(ndeps, 32));
// ...


  * **Grid 尺寸:** Grid 的尺寸是根据子数组的尺寸 (ncols, nrows, ndeps) 和每个 Block 负责的总工作量 (16 \times 8 = 128 或 4 \times 8 = 32) 计算的，确保 Grid 覆盖整个子数组区域。
  * **启动:** 启动 Kernel 完成求和。

//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question1:

不理解，请解释，用例子

1.
Subarray Sum

    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;

    int sum_val = 0;
    for (int i=0; i<STRIDE_LENGTH; ++i) {
        // 比如,0号线程读取的就是块 0、块 1、...块STRIDE_LENGTH-1的 0号位置元素
        int idx = offset*STRIDE_LENGTH + tid + i*blockDim.x;
        if (idx < N) {
            sum_val += input[idx];
        
    

int idx = offset*STRIDE_LENGTH + tid + i*blockDim.x
		= blockIdx.x * blockDim.x *STRIDE_LENGTH + threadIdx.x + i*blockDim.x        

sum_val += input[idx];

//-------------------------------------------------

2.
2D Subarray Sum

    int col = threadIdx.x + CFACTOR * blockDim.x * blockIdx.x;
    int row = threadIdx.y + CFACTOR * blockDim.y * blockIdx.y;
    
    __shared__ int Ms[TILE_X * TILE_Y];

    int v = 0;
    for(int i = 0; i < CFACTOR; i++){
        int row1 = row + i * blockDim.y;
        for(int j = 0; j < CFACTOR; j++){
            int col1 = col + j * blockDim.x;
            if(S_ROW <= row1 && row1 <= E_ROW && S_COL <= col1 && col1 <= E_COL){
                v += input[row1 * M + col1];
            
        
    


int row1 = CFACTOR * blockDim.y * blockIdx.y + threadIdx.y + i * blockDim.y;

int col1 = CFACTOR * blockDim.y * blockIdx.y + threadIdx.y + j * blockDim.x;

v += input[row1 * M + col1];


//--------------------------------------------------

3.
3D Subarray Sum

    // 0.每个线程每个方向上读取 STRIDE_LENGTH个元素并求和,每个元素在 input中间隔 WARP_SIZE
       以保证每个块读取连续 WARP_SIZE，合并内存访问 
       int tcol = blockIdx.x * blockDim.x * STRIDE_LENGTH + threadIdx.x;
       int trow = blockIdx.y * blockDim.y * STRIDE_LENGTH + threadIdx.y;
       int tdep = blockIdx.z * blockDim.z * STRIDE_LENGTH + threadIdx.z;
   
       int sum_val = 0;
       // 三维矩阵是行主序存储的,按照 Z/Y/X方式是可以连续读取的 
       for (int i=0; i<STRIDE_LENGTH; ++i) {
           // 比如,0号线程读取的就是 Z方向上块 0、块 1、...块STRIDE_LENGTH-1的 0号位置元素
           int dep = tdep + i*blockDim.z;
           for (int j=0; j<STRIDE_LENGTH; ++j) {
               int row = trow + j*blockDim.y;
               for (int k=0; k<STRIDE_LENGTH; ++k) {
                   int col = tcol + k*blockDim.x;
                   if (dep < ndeps && row < nrows && col < ncols) {
                       int pos = (dep+S_DEP)*M*K+(row+S_ROW)*K+(col+S_COL);
                       sum_val += input[pos];
                   
               
           
       
   
   
   int row = blockIdx.y * blockDim.y * STRIDE_LENGTH + threadIdx.y + j*blockDim.y;
   
   int col = blockIdx.x * blockDim.x * STRIDE_LENGTH + threadIdx.x + k*blockDim.x;
   
   int dep = blockIdx.z * blockDim.z * STRIDE_LENGTH + threadIdx.z + i*blockDim.z;
   
   int pos = (dep+S_DEP)*M*K+(row+S_ROW)*K+(col+S_COL);
   sum_val += input[pos];
   
   

//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question2:

不理解，请解释，用例子

这里的lane的值不一定是0，如何保证shared_partial_sum[lane]的值包含了这个lane里所有的值？

if (warp == 0) {
    int partial_sum_val = shared_partial_sum[lane];
    shared_partial_sum[0] = warp_sum(partial_sum_val);

//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question3:

不理解，请解释，用例子

这里的lane的值不一定是0，如何保证shared_partial_sum[lane]的值包含了这个lane里所有的值？

if (warp == 0) {
    int partial_sum_val = shared_partial_sum[lane];
    shared_partial_sum[0] = warp_sum(partial_sum_val);


这是一个非常好的问题，它指出了这段代码中 **第二级归约（Warp 间归约）** 的一个 **关键的实现技巧**，同时也揭示了 **CUDA Shuffle 归约的特性**。

您观察到的现象是正确的：`lane` 的值在 0 到 31 之间，但我们只希望 Warp 0 中的前 NUM_WARPS 个线程参与读取和归并。

-----

## ⚙️ I. 核心机制：Warp 归约的输入和输出

### 1\. 第一级归约的结果

在进入这段代码之前，所有 32 个 Warp 的总和都已安全地存储在 shared_partial_sum[0] 到 shared_partial_sum[31] 中（假设 Block 内有 32 个 Warp）。

### 2\. 第二级归约的目标

现在需要将这 32 个值合并成一个最终的总和，并存储在 shared_partial_sum[0] 中。

### 3\. Warp 0 的分工

只有 Warp\ 0 的 32 个线程执行这段代码。

| 线程 ID (lane) | 线程 tx | 任务 |
| :--- | :--- | :--- |
| **0** | 0 | 读取 `shared_partial_sum[0]` (Warp 0 的总和) |
| **1** | 1 | 读取 `shared_partial_sum[1]` (Warp 1 的总和) |
| **31** | 31 | 读取 `shared_partial_sum[31]` (Warp 31 的总和) |

## 🚀 II. 答案：lane 恰好是索引 (Index)

在这里，`lane`（0 到 31）被巧妙地用作读取 shared_partial_sum 数组的**索引**：

c
int partial_sum_val = shared_partial_sum[lane];


  * **线程 0 (lane=0):** 读取 `shared_partial_sum[0]`。
  * **线程 31 (lane=31):** 读取 `shared_partial_sum[31]`。

**问题不在于 lane 的值是否为 0，而在于** **`shared_partial_sum` 数组的索引恰好是 lane。**

### 示例：数据收集

假设线程块有 32 个 Warp (WARPS=32)。

| 线程 tid (Warp 0) | lane | 读取 shared_partial_sum[lane] | 存储内容 |
| :--- | :--- | :--- | :--- |
| **0** | 0 | shared_partial_sum[0] | Warp 0 的总和 |
| **5** | 5 | shared_partial_sum[5] | Warp 5 的总和 |
| **31** | 31 | shared_partial_sum[31] | Warp 31 的总和 |

**结论：** Warp 0 中的 32 个线程，以并行的方式，从 shared_partial_sum 数组中，将所有 32 个 Warp 的总和都加载到了自己的私有变量 partial_sum_val 中。

## 🧭 III. 后续的归并 (`warp_sum`)

一旦每个线程都读取了其分配到的一个 Warp 总和 (partial_sum_val)，整个 Warp 0 随后执行第二次 Shuffle 归约：

c
shared_partial_sum[0] = warp_sum(partial_sum_val);


  * warp_sum 接收这 32 个值，并将它们累加成一个最终总和。
  * 最终总和被写入 shared_partial_sum[0] (覆盖了 Warp 0 自己的原始总和)。

**这个过程是正确且高效的，因为 `lane` 恰好是 Shared Memory 数组的正确索引。**



//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question4:

不理解，请解释，用例子



//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question5:

不理解，请解释，用例子


//--------------------------------------------------------------------------------------------------




//--------------------------------------------------------------------------------------------------
/*
question6:

不理解，请解释，有例子




//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question7:

不理解，请解释，用例子




//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question8:

不理解，请解释，用例子



//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question10:

不理解，请解释，用例子


//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question11:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question12:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question13:


//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question14:


//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question15:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question16:


//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question17:


//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question18:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question19:


//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question20:


//--------------------------------------------------------------------------------------------------
