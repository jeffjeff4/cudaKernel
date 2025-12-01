
#include <cuda_runtime.h>

int div_helper(int a, int b){
    return (a + b - 1) /b;
}

#define TILE 256
#define CFACTOR 2
__global__ void kogge_stone_prefix(const int* input, int* output, int *aux, int N){
    __shared__ int Ms[TILE * CFACTOR];
    int i = threadIdx.x + CFACTOR * blockDim.x * blockIdx.x;
    int tidx = threadIdx.x;
    int end_coord = min(CFACTOR * blockDim.x * (blockIdx.x + 1), N);
    #pragma unroll
    for(int j = 0; j < CFACTOR; j++){
        Ms[tidx + j * blockDim.x] = (i + j * blockDim.x < end_coord) ? input[i + j * blockDim.x] : 0.0f;
    }
    __syncthreads();
    // Managing radius
    for(int j = 1; j < CFACTOR; j++){
        Ms[tidx * CFACTOR + j] += Ms[tidx * CFACTOR + j - 1];
    }
    __syncthreads();

    int pos =  CFACTOR * (threadIdx.x + 1) - 1;
    int f = Ms[pos];
    for(int stride = 1; stride < blockDim.x; stride <<= 1){
        if(threadIdx.x >= stride){
            f += Ms[pos - CFACTOR * stride];
        }
        __syncthreads();
        Ms[pos] = f;
        __syncthreads();
    }

    int prev = (tidx > 0) ? Ms[tidx * CFACTOR - 1] : 0.0f;
    #pragma unroll
    for(int j = 0; j < CFACTOR - 1; j++){
        Ms[tidx * CFACTOR + j] += prev;
    }
    __syncthreads();

    i = threadIdx.x + CFACTOR * blockDim.x * blockIdx.x;
    #pragma unroll
    for(int j = 0; j < CFACTOR; j++){
        if(i < end_coord){
            output[i] = Ms[tidx];
        }
        else{
            break;
        }
        i += blockDim.x;
        tidx += blockDim.x;
    }

    if(threadIdx.x == 0)
        aux[blockIdx.x] = Ms[TILE*CFACTOR - 1];

}

__global__ void add(int* output, int* blocksum, int N){    
    __shared__ int sS;
    if(threadIdx.x == 0){
        sS = blocksum[blockIdx.x];
    }
    __syncthreads();
    int s = sS;
    int i = threadIdx.x + CFACTOR * blockDim.x * (blockIdx.x + 1);
    #pragma unroll
    for(int j = 0; j < CFACTOR; j++){
        if(j * blockDim.x + i < N)
            output[j * blockDim.x + i] = output[j * blockDim.x + i] + s;
    }
}

void rec(const int* input, int* output, int N){
    int *aux_sums;
    int len_aux = div_helper(N, TILE * CFACTOR);
    cudaMalloc(&aux_sums, len_aux * sizeof(int));
    kogge_stone_prefix<<<div_helper(N, TILE*CFACTOR), TILE>>>(input, output, aux_sums, N);
    if(N <= TILE) return;
    rec(aux_sums, aux_sums, len_aux);
    add<<<div_helper(N - 1 - TILE*CFACTOR, TILE*CFACTOR), TILE>>>(output, aux_sums, N);
}

__global__ void max_sum(const int* input, int *output, int N, int window_size){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i + window_size - 1 < N){
        atomicMax(output, input[i + window_size - 1] - ((i) ? input[i-1] : 0));
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int *prefix_sums;
    cudaMalloc(&prefix_sums, N * sizeof(int));
    rec(input, prefix_sums, N);
    int value = INT_MIN;
    cudaMemcpy(output, &value, sizeof(int), cudaMemcpyHostToDevice);
    max_sum<<<(N - window_size + TILE - 1) / TILE, TILE>>>(prefix_sums, output, N, window_size);
}



//--------------------------------------------------------------------------------------------------
/*
question0:
不理解，请解释，用例子

这段代码实现了一个高效的并行算法来解决\*\*最大子数组和问题（Maximum Subarray Sum）\*\*的一个变体：**最大滑动窗口和（Maximum Sliding Window Sum）**。

它利用了 **Kogge-Stone 并行前缀和（Prefix Sum / Scan）** 算法，并结合了**分层递归**来解决大规模数据集上的计算。

-----

## ⚙️ I. 核心数学原理：前缀和 (Prefix Sum)

计算固定大小窗口内的和是昂贵的 O(N \cdot W) 操作。但通过使用前缀和，可以降为 O(1)。

  * **前缀和数组 P:** P[i] = \sum_{j=0^{i A[j]。
  * **滑动窗口和:** 窗口 [i, i + W - 1] 的和 =P[i + W - 1] - P[i - 1]。

这段代码的流程是：

1.  **`rec` (Recursive Scan):** 在 GPU 上计算输入数组的**全局前缀和** P。
2.  **`max_sum`:** 利用 P 数组，在 GPU 上并行计算所有滑动窗口的最大和。

-----

## 🚀 II. 阶段 1: 并行前缀和 Kernel (`kogge_stone_prefix`)

这个 Kernel 负责计算 Block 内的前缀和，并将 Block 的总和输出到 `aux` 数组，为分层递归做准备。

### 1\. 局部数据加载与零填充

c
// ...
#pragma unroll
for(int j = 0; j < CFACTOR; j++){
    Ms[tidx + j * blockDim.x] = (i + j * blockDim.x < end_coord) ? input[i + j * blockDim.x] : 0.0f;

__syncthreads();


  * **CFACTOR=2:** 每个线程 tidx 负责加载 2 个元素。
  * **Block-Interleaved Load:** 线程 t 和 t+1 访问的地址相隔 blockDim.x (256)，确保了内存合并。
  * **零填充:** 如果线程访问的索引超出 N，则用 0 填充 Shared Memory (`Ms`)。

### 2\. 线程内部前缀和

c
for(int j = 1; j < CFACTOR; j++){
    Ms[tidx * CFACTOR + j] += Ms[tidx * CFACTOR + j - 1];

__syncthreads();


  * **目的:** 在每个线程的局部 2 个元素之间，进行串行前缀求和。
  * **示例:** 假设线程 t 加载了 A, B。 Ms[t * 2 + 1] = B + A。

### 3\. Kogge-Stone 块内扫描

c
int pos =  CFACTOR * (threadIdx.x + 1) - 1; // 线程处理区域的末尾索引
int f = Ms[pos];
for(int stride = 1; stride < blockDim.x; stride <<= 1){
    if(threadIdx.x >= stride){
        f += Ms[pos - CFACTOR * stride]; // 从 stride 之外的区域获取累加值
    
    __syncthreads();
    Ms[pos] = f;
    __syncthreads();



  * **机制:** 这是经典的 **Kogge-Stone** 算法，通过 i=1, 2, 4, 8, \dots 的步长进行并行累加。它执行 **Exclusive Scan (独占前缀和)**。
  * **作用:** 将 Ms 数组的前半部分累加到后半部分，最终 Ms[pos] 存储了 **线程 t 之前**所有元素的累积和。

### 4\. 最终块内调整

c
int prev = (tidx > 0) ? Ms[tidx * CFACTOR - 1] : 0.0f; // 获取前一个线程的最终总和
#pragma unroll
for(int j = 0; j < CFACTOR - 1; j++){
    Ms[tidx * CFACTOR + j] += prev; // 独占前缀和的最后调整



  * **目的:** 将 Kogge-Stone 结果调整为 **独占前缀和**，并完成线程内部的最终累加。

### 5\. 输出 Block 总和

c
if(threadIdx.x == 0)
    aux[blockIdx.x] = Ms[TILE*CFACTOR - 1];


  * **目的:** 将整个 Block 的总和（位于 Ms 数组的最后一个有效位置）写入 aux 数组，作为递归下一层的输入。

-----

## 🧭 III. 阶段 2: 层次递归 (`rec` 和 `add`)

### 1\. 递归扫描 (`rec`)

c
void rec(const int* input, int* output, int N){
    // ...
    kogge_stone_prefix<<<...>>>(input, output, aux_sums, N); // 块内扫描
    if(N <= TILE) return;
    rec(aux_sums, aux_sums, len_aux); // 递归调用自身，处理 aux 数组
    add<<<...>>>(output, aux_sums, N); // 将递归结果加回到主数组



  * **机制:** 如果 N 足够大，`rec` 函数会递归地调用自己，对 aux 数组进行扫描。
  * **用途:** 确保所有 Block 的总和在全局范围内也得到正确的累积。

### 2\. 最终累加 (`add` Kernel)

c
__global__ void add(int* output, int* blocksum, int N){    
    // ...
    int s = sS; // sS 是当前 Block 之前的全局累积和
    // ...
    for(int j = 0; j < CFACTOR; j++){
        // ...
        output[j * blockDim.x + i] = output[j * blockDim.x + i] + s;
    



  * **目的:** 将递归计算出的 **全局偏移量** (s) 加到 `output` 数组的相应 Block 区域上。
  * **机制:** 这完成了 **全局前缀和** 的计算，确保 P[i] 包含了所有先前 Block 的总和。

-----

## 🏁 IV. 阶段 3: 最大滑动窗口和 (`max_sum`)

c
__global__ void max_sum(const int* input, int *output, int N, int window_size){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i + window_size - 1 < N){
        atomicMax(output, input[i + window_size - 1] - ((i) ? input[i-1] : 0));
    



  * **输入:** input 是全局前缀和数组 P。
  * **计算:** 每个线程 i 负责检查一个滑动窗口：
    Window Sum = P[i + W - 1] - P[i - 1]
      * P[i + W - 1] 对应 input[i + window_size - 1]。
      * P[i - 1] 对应 input[i - 1]，边界检查用三元运算符处理。
  * **`atomicMax`:** 将计算出的窗口和与全局最大值进行原子比较，并更新 output 中的最大值。

**总结:** 整个 `solve` 函数展示了如何将一个复杂的全局依赖性问题（前缀和）分解为并行可解的子问题，然后利用 O(1) 的滑动窗口公式和原子操作，高效地在 GPU 上解决最大滑动窗口和问题。



//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question1:

不理解，请解释，用例子

不理解，请解释，用例子

int div_helper(int a, int b){
    return (a + b - 1) /b;



void rec(const int* input, int* output, int N){
    int *aux_sums;
    int len_aux = div_helper(N, TILE * CFACTOR);
    cudaMalloc(&aux_sums, len_aux * sizeof(int));
    kogge_stone_prefix<<<div_helper(N, TILE*CFACTOR), TILE>>>(input, output, aux_sums, N);
    if(N <= TILE) return;
    rec(aux_sums, aux_sums, len_aux);
    add<<<div_helper(N - 1 - TILE*CFACTOR, TILE*CFACTOR), TILE>>>(output, aux_sums, N);


这段代码是 **层次并行前缀和（Hierarchical Parallel Prefix Sum / Scan）** 算法的核心实现，旨在高效地计算一个大型数组的全局前缀和。它通过 **递归** 的方式，将大规模的计算分解成 GPU 可以并行处理的小块。

-----

## ⚙️ I. 核心机制：向上取整和递归

### 1\. 向上取整函数 (`div_helper`)

c
int div_helper(int a, int b){
    return (a + b - 1) /b;



  * **用途:** 计算 A 除以 B 的结果，并**向上取整**。
  * **示例:** 假设 A=10，B=4。
      * div_helper(10, 4) = (10 + 4 - 1) / 4 = 13 / 4 = 3。
      * **含义:** 10 个元素需要 3 组，每组 4 个。

### 2\. 递归函数 (`rec`) 的作用

`rec` 函数是整个全局前缀和计算的驱动器。它的目标是：

  * **处理当前层:** 使用 `kogge_stone_prefix` Kernel 计算当前 N 个元素的块内前缀和，并将每个块的**总和**输出到辅助数组 `aux_sums` 中。
  * **处理下一层 (递归):** 将 `aux_sums` 视为一个新的、更小的数组，然后递归地调用 `rec` 来计算它的前缀和。

-----

## 🚀 II. 示例解释：分层递归流程

假设我们有一个非常大的数组，N = 4096 个元素，TILE * CFACTOR = 256 * 2 = 512 (每个 Block 处理 512 个元素)。

### 1\. 初始调用 (`rec(input, prefix_sums, 4096)`)

| 变量 | 计算值 | 含义 |
| :--- | :--- | :--- |
| N | 4096 | 初始数组长度 |
| len_aux | div_helper(4096, 512) = 8 | 需要 8 个 Block，因此辅助数组长度为 8。 |

**执行:**

  * **`kogge_stone_prefix<<<8, 256>>>`:** 启动 8 个 Block。每个 Block 计算 512 个元素的块内前缀和，并将自己的总和写入 `aux_sums` 数组（长度为 8）。
  * **`if(N <= TILE)`:** 4096 \not\le 256。继续递归。
  * **`rec(aux_sums, aux_sums, 8)`:** 递归调用自身，现在数组长度变为 8。

### 2\. 递归调用 (第 2 层)

  * **`N`** (新长度) = 8。
  * len_aux (新长度) = div_helper(8, 512) = 1。

**执行:**

  * **`kogge_stone_prefix<<<1, 256>>>`:** 启动 1 个 Block。计算这 8 个元素的全局前缀和，并将其总和写入 `aux_sums`（长度为 1）。
  * **`if(N <= TILE)`:** 8 \le 256。**递归停止**。

### 3\. 返回和累加 (`add` Kernel)

现在，递归开始返回，执行 `add` Kernel 将全局偏移量累加到主数组。

  * **`add<<<1, 256>>>(output, aux_sums, 4096)`:**
      * `aux_sums` 中存储了第 2 层的全局前缀和。
      * `add` Kernel 将 **Block 0** 的总和（aux_sums[0]）累加到主数组 output 的 0 到 511 区域之后的每一个元素上。

**总结：** `rec` 函数实现了 **多阶段的并行前缀和**。它先用 `kogge_stone_prefix` 解决局部问题，然后用 `rec` 解决 Block 间的依赖问题，最后用 `add` Kernel 将 Block 间的累积总和加回到主数组的相应位置，完成最终的全局前缀和计算。

//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question2:

不理解，请解释，用例子

不理解，请解释，用例子

__global__ void kogge_stone_prefix(const int* input, int* output, int *aux, int N){



answer:
这段代码是 **Kogge-Stone 并行前缀求和 (Prefix Sum / Scan)** 算法的实现。它的核心目的是高效地计算每个线程块内所有元素的累积和，并将每个块的总和输出，以支持更大规模的递归（分层）全局前缀和。

这个 Kernel 的设计非常复杂，因为它混合了 **Block-Stride 加载**、**线程局部串行扫描** 和 **并行树形扫描**。

-----

## ⚙️ I. 核心目标和参数设定

  * **目标:** 计算 input 数组（或其子数组）的**独占前缀和** P[i] = A[0] + \dots + A[i-1]。
  * **方法:** 在共享内存 (Ms) 中完成大部分计算。

### 示例参数 (假设简化)

| 参数 | 值 | 含义 |
| :--- | :--- | :--- |
| TILE | 8 | 线程块线程总数 blockDim.x |
| CFACTOR | 4 | 粗粒度因子（每个线程负责的元素数量） |
| **Block Size** | 8 | Block 线程数 |
| **Ms Size** | 8 * 4 = 32 | 共享内存总长度 |

-----

## 🚀 II. 阶段 1：数据加载与线程局部扫描

此阶段将数据从 Global Memory 协作加载到 Shared Memory，并计算每个线程负责区域的**串行前缀和**。

### 1\. 协作加载 (Block-Stride Load)

c
// ... (i 和 tidx 的计算)
#pragma unroll
for(int j = 0; j < CFACTOR; j++){
    Ms[tidx + j * blockDim.x] = (i + j * blockDim.x < end_coord) ? input[i + j * blockDim.x] : 0.0f;

__syncthreads();


  * **目的:** 所有线程协作加载 8 * 4 = 32 个元素。
  * **索引:** 线程 t 访问 Ms[t], Ms[t+8], Ms[t+16], Ms[t+24]（示例）。**这实现了数据在 Shared Memory 中的交错存储**。

### 2\. 线程内部串行扫描

c
// Managing radius (实际是线程局部前缀和)
i = CFACTOR * (threadIdx.x + blockDim.x * blockIdx.x); // 重新计算全局索引的起始点
for(int j = 1; j < CFACTOR; j++){
    Ms[tidx * CFACTOR + j] += Ms[tidx * CFACTOR + j - 1];

__syncthreads();


  * **目标:** 这一步只在每个线程的 CFACTOR 区域内串行进行。
  * **示例:** 线程 tx 负责 Ms[t * 4] 到 Ms[t * 4 + 3] 的 4 个元素。
      * 假设 Ms[t * 4] = [5, 1, 3, 2]
      * **循环后:** Ms[t * 4] = [5, 6, 9, 11] (串行前缀和)
  * **注意:** 代码先将数据交错加载，然后在这里**重新索引** Ms 数组来计算串行前缀和。

-----

## 🧭 III. 阶段 2：Kogge-Stone 并行扫描 (Block Scan)

此阶段执行树形归约，将所有线程的局部总和累加起来。

c
int pos =  CFACTOR * (threadIdx.x + 1) - 1;
int f = Ms[pos];
for(int stride = 1; stride < blockDim.x; stride <<= 1){
    if(threadIdx.x >= stride){
        f += Ms[pos - CFACTOR * stride];
    
    __syncthreads();
    Ms[pos] = f;
    __syncthreads();



  * **`pos` 的计算:** pos = 4 * (tx + 1) - 1。这是每个线程负责的 **4 元素块的最后一个元素的索引**。
      * **示例:** tx=0 --> pos=3. tx=1 --> pos=7.
  * **机制:** 这是 Kogge-Stone 算法。线程 tx 从其左侧 tx - stride 处的线程获取累积的总和，然后加到自己的 f 上，并写回 Ms[pos]。
  * **结果:** 循环结束后，Ms 数组的 pos 位置存储了**独占前缀和**。

-----

## 🏁 IV. 阶段 3：最终调整和输出

### 1\. 独占前缀和调整

c
int prev = (tidx > 0) ? Ms[tidx * CFACTOR - 1] : 0.0f; // 获取前一个线程的最终总和
#pragma unroll
for(int j = 0; j < CFACTOR - 1; j++){
    Ms[tidx * CFACTOR + j] += prev;

__syncthreads();


  * **目的:** 将前一个线程的累积总和 (prev) 加到当前线程的局部结果上，完成 **全局独占前缀和** 的计算。

### 2\. 写入全局输出 (output)

c
#pragma unroll
for(int j = 0; j < CFACTOR; j++){
    if(i < end_coord){
        output[i] = Ms[tidx]; // 将计算好的前缀和写入 Global Memory
    
    // ... (索引和 tidx 的复杂更新)



  * **目的:** 将 Shared Memory (Ms) 中的最终前缀和结果写回到 Global Memory 的 output 数组中。

### 3\. 输出 Block 总和 (为递归做准备)

c
if(threadIdx.x == 0)
    aux[blockIdx.x] = Ms[TILE*CFACTOR - 1];


  * **目的:** 只有线程 0 将整个 Block 的最终总和（位于 Ms 数组的最后一个位置）写入 aux 数组。aux 数组用于递归的下一层，解决 Block 间的全局偏移问题。

**总结:** 这个 Kernel 巧妙地结合了串行和并行技术，在 Shared Memory 中完成了高效的 Kogge-Stone 扫描，同时解决了数据加载和全局递归所需的输出问题。

//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question3:

不理解，请解释，用例子

不理解，请解释，用例子

__global__ void add(int* output, int* blocksum, int N){    


这段 CUDA Kernel (`add`) 的作用是 **将全局前缀和的偏移量 (Global Offset)** 加回到主输出数组 (output) 的相应分块区域，以完成**层次并行前缀和 (Hierarchical Parallel Prefix Sum)** 的计算。

这个 Kernel 解决了在 `kogge_stone_prefix` 中被分解的 **Block 间的依赖问题**。

-----

## ⚙️ I. 核心目标：累加全局偏移

在层次前缀和算法中：

1.  **第一层 (Block 内部):** 每个 Block 独立地计算自己的局部前缀和 P_local。
2.  **第二层 (Block 之间):** 我们需要加上**所有先前 Block 的总和**，才能得到正确的全局前缀和。

`blocksum` 数组存储的正是这个“所有先前 Block 的总和”（全局偏移量）。`add` Kernel 的任务就是将这个偏移量 s 加到当前 Block 的局部结果上。

## 🔢 II. 示例参数设定

我们假设以下值，与 `solve` 函数中的调用上下文保持一致：

  * **Block 尺寸 (blockDim.x):** T = 256
  * **粗粒度因子 (CFACTOR):** C = 2
  * **总数据量 (N):** 1000
  * **Grid ID (blockIdx.x):** 2 (第三个 Block)

-----

## 🚀 III. 步骤分解与示例

### 1\. 共享内存广播 (`sS`)

c
__shared__ int sS;
if(threadIdx.x == 0){
    sS = blocksum[blockIdx.x]; // Thread 0 读取本 Block 的全局偏移量

__syncthreads();
int s = sS; // 所有线程获取 s


  * **`blocksum[blockIdx.x]` 的值:** 这个数组（由递归计算得出）存储着 **Block 0 到 Block blockIdx.x - 1 的所有元素的总和**。
  * **示例:** blockIdx.x = 2。假设 blocksum[2] = 500。
  * **作用:** 线程 0 将 500 读入 Shared Memory 变量 sS，然后 `__syncthreads()` 确保 Block 内所有线程的私有变量 s 都获取到 500。

> **结论:** s = 500。所有线程都知道：它们需要将 500 加到自己的局部前缀和上。

### 2\. 计算起始索引

c
int i = threadIdx.x + CFACTOR * blockDim.x * (blockIdx.x + 1);


  * **目的:** 这一行计算了当前线程负责的 **Output 数组区域** 的**起始地址**。
      * 这里的 blockIdx.x + 1 是为了在递归返回时，定位到**需要被加上偏移量**的 Block 区域。
  * **示例:**
      * threadIdx.x = 0, blockIdx.x = 2.
      * i = 0 + (2 * 256) * (2 + 1) = 0 + 512 * 3 = 1536.
      * **含义:** 线程 0 的起始索引是 1536 (假设这是需要被修正的第一个 Block 的起始点)。

### 3\. 最终累加和写回 (Block-Stride Loop)

c
#pragma unroll
for(int j = 0; j < CFACTOR; j++){
    if(j * blockDim.x + i < N) // 边界检查
        output[j * blockDim.x + i] = output[j * blockDim.x + i] + s;



  * **CFACTOR=2:** 每个线程执行 2 次循环。
  * **目标:** 线程读取 output 中存储的**局部前缀和**，然后加上全局偏移量 s。
  * **索引:** 循环步长是 blockDim.x (256)。

**示例 (线程 tx=0, s=500):**

| 循环 j | j * 256 + i (全局索引) | 操作 |
| :--- | :--- | :--- |
| **0** | 0 + 1536 = 1536 | output[1536] <-- output[1536] + 500 |
| **1** | 256 + 1536 = 1792 | output[1792] <-- output[1792] + 500 |

> **结论:** 线程 0 将 500 加到了索引 1536 和 1792 处的**局部前缀和**上。这个操作在所有线程上并行执行，从而完成了整个 output 数组的**全局前缀和**计算。

这个 Kernel 的存在，证明了 **层次归并** 的有效性：先分治 (局部计算)，再解决依赖 (全局累加)。


//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question4:

不理解，请解释，用例子

不理解，请解释，用例子

__global__ void max_sum(const int* input, int *output, int N, int window_size){


这段 CUDA Kernel `max_sum` 的目标是利用**前缀和数组 (Prefix Sum array)**，以 O(1) 的时间复杂度并行计算一个大数组中所有 **滑动窗口（Sliding Window）** 的和，并找到其中的 **最大值**。

它通过 **原子操作 (Atomic Operation)** 将所有线程计算的结果安全地归约到全局最大值。

-----

## ⚙️ I. 核心原理：前缀和与滑动窗口

最大滑动窗口和问题，传统上需要 O(N * W) 的时间复杂度。使用前缀和数组 P 可以将其降为 O(N)（每个窗口 O(1)）。

  * **前缀和 P:** P[i] = A[0] + A[1] + \dots + A[i]。
  * **窗口和计算:** 窗口 A[i] 到 A[i + W - 1] 的和等于 P[i + W - 1] - P[i - 1]。

## 🚀 II. Kernel 内部流程与计算

### 1\. 线程分工与边界检查

c
int i = threadIdx.x + blockDim.x * blockIdx.x; // 全局线程索引
if(i + window_size - 1 < N){ // 确保窗口末端在数组 N 范围内
    // ... 计算 ...



  * i：当前滑动窗口的**起始索引**。
  * **边界检查:** 只有当窗口的末端 (i + window_size - 1) 不超过数组总长度 N 时，线程才进行计算。

### 2\. 计算滑动窗口的和 (O(1) 查找)

核心计算是利用前缀和数组 input (即 P 数组) 来计算当前窗口的和：

Window Sum = input[i + window_size - 1] - input[i - 1]

代码实现为：

c
// Window Sum = P[window_end] - P[window_start - 1]
int window_end_index = i + window_size - 1; 
int window_start_minus_1 = i - 1; 

// if (i) ? input[i-1] : 0  => 这是处理数组起始边界 P[-1] 的情况
int window_sum = input[window_end_index] - ((i) ? input[window_start_minus_1] : 0);


### 示例解释

假设 window_size=3，N=10。input 是前缀和数组 P。

| 线程 i (窗口起始) | 窗口末端索引 | P[i-1] 项 | 计算的窗口和 (input) | 含义 |
| :--- | :--- | :--- | :--- | :--- |
| **0** (起始点) | 0 + 3 - 1 = 2 | i=0 \implies 0 | P[2] - 0 | A[0]+A[1]+A[2] |
| **1** | 1 + 3 - 1 = 3 | P[0] | P[3] - P[0] | A[1]+A[2]+A[3] |
| **7** | 7 + 3 - 1 = 9 | P[6] | P[9] - P[6] | A[7]+A[8]+A[9] |

### 3\. 原子最大值累加 (atomicMax)

c
atomicMax(output, window_sum);


  * **机制:** 线程计算出 window_sum 后，使用 atomicMax 指令。
  * **作用:** 它将当前计算出的 window_sum 与 output 变量中存储的**当前最大值**进行比较，并原子性地更新 output 为两者中的较大值。
  * **优势:** 这避免了使用共享内存或全局同步，使得所有线程可以安全、并行地将自己的窗口和贡献给全局最大值。

## 🎯 总结

该 Kernel 实现了 O(N) 的高效最大滑动窗口和查找：

1.  **并行性:** 所有线程同时计算不同的窗口和。
2.  **效率:** 利用 O(1) 的前缀和查找窗口和。
3.  **安全性:** 利用 atomicMax 保证了多线程对全局最大值更新的线程安全。

//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question5:

不理解，请解释，用例子

compare 
1)
int prev = (tidx > 0) ? Ms[tidx * CFACTOR - 1] : 0.0f; // 获取前一个线程的最终总和
#pragma unroll
for(int j = 0; j < CFACTOR - 1; j++){
    Ms[tidx * CFACTOR + j] += prev;

__syncthreads();

目的: 将前一个线程的累积总和 (prev) 加到当前线程的局部结果上，完成 全局独占前缀和 的计算。

and
2)
__global__ void add(int* output, int* blocksum, int N){    
    __shared__ int sS;
    if(threadIdx.x == 0){
        sS = blocksum[blockIdx.x];
    
    __syncthreads();
    int s = sS;
    int i = threadIdx.x + CFACTOR * blockDim.x * (blockIdx.x + 1);
    #pragma unroll
    for(int j = 0; j < CFACTOR; j++){
        if(j * blockDim.x + i < N)
            output[j * blockDim.x + i] = output[j * blockDim.x + i] + s;
    


在层次前缀和算法中：第一层 (Block 内部): 每个 Block 独立地计算自己的局部前缀和 P_local。第二层 (Block 之间): 我们需要加上所有先前 Block 的总和，才能得到正确的全局前缀和。blocksum 数组存储的正是这个“所有先前 Block 的总和”（全局偏移量）。add Kernel 的任务就是将这个偏移量 s 加到当前 Block 的局部结果上。


//--------------------------------------------------------------------------------------------------




//--------------------------------------------------------------------------------------------------
/*
question6:

不理解，请解释，有例子

    rec(aux_sums, aux_sums, len_aux);


这段代码是 **层次并行前缀和（Hierarchical Parallel Prefix Sum / Scan）** 算法中的 **递归调用**，它是解决大规模数组全局前缀和的关键步骤。

---

## ⚙️ I. 核心目标：解决 Block 间的依赖关系

在并行前缀和算法中，当我们将数组分成多个 Block 处理时：

1.  **每个 Block** 独立计算自己的**局部前缀和** P_local。
2.  要得到**全局前缀和** P_global，我们需要知道**前面所有 Block 的总和**（即 **全局偏移量**）。

### `aux_sums` 数组的作用

`aux_sums` 数组存储了第一层所有 Block 的总和。aux_sums[i] 就是 Block i 的总和。

* **问题:** 我们需要计算 aux_sums 数组的**前缀和**，才能知道每个 Block 的正确全局偏移量。

## 🚀 II. 递归调用机制

rec(aux_sums, aux_sums, len_aux)

| 参数 | 传递的值 | 含义 |
| :--- | :--- | :--- |
| **`input`** | `aux_sums` | 将第一层所有 Block 的总和作为 **新数组** 的输入。 |
| **`output`** | `aux_sums` | 将计算结果覆盖回 `aux_sums` 数组。 |
| **`N`** | `len_aux` | 新数组的长度（Block 的总数量）。 |

通过将 **`aux_sums` 作为输入和输出** 传入递归调用，函数计算了：

aux_sums'[i] = sum_(j=0^(i-1)) aux_sums[j]

新的 aux_sums' 数组现在存储了每个 Block 的 **全局偏移量**。

---

## 🔢 III. 示例解释：从 4096 到 1 的归约

假设 N = 4096，每个 Block 处理 512 个元素。

### 初始调用 (第 1 层)

* **输入:** input (4096 个元素)。
* **Block 数 (len_aux):** 4096 / 512 = 8 个 Block。
* **`kogge_stone_prefix` 结果:** aux_sums 数组 (8 个元素) 存储了 8 个 Block 各自的总和。

### 递归调用 (第 2 层)

rec(aux_sums, aux_sums, 8)

* **输入:** 长度为 8 的数组。
* **执行:** 递归调用 `rec`，它会启动 Kernel 对这 8 个元素进行前缀和计算。
    * **结果:** aux_sums[i] 现在存储了 Block 0 到 Block i-1 的总和。
    * aux_sums[0] = 0
    * aux_sums[1] = Block 0 的总和
    * aux_sums[2] = Block 0 + Block 1 的总和

### 递归终止

由于 8 <= TILE=256，递归会在这一层终止。

**总结：** 递归调用将一个 4096 元素的全局依赖问题，有效地分解为：
1.  **4096 --> 8:** 在 GPU 上并行计算 8 个 Block 的局部结果。
2.  **8 --> 8 (Scan):** 递归地计算这 8 个 Block 总和的前缀和。

这样，主函数返回后，`aux_sums` 中就包含了正确的全局偏移量，可以用于最终的 `add` Kernel 累加。


//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question7:

不理解，请解释，用例子

1)
    kogge_stone_prefix<<<div_helper(N, TILE*CFACTOR), TILE>>>(input, output, aux_sums, N);

    写入全局输出 (output)C#pragma unroll
    目的: 将 Shared Memory (Ms) 中的最终前缀和结果写回到 Global Memory 的 output 数组中。

2)
    rec(aux_sums, aux_sums, len_aux);

    ⚙️ I. 核心目标：解决 Block 间的依赖关系在并行前缀和算法中，当我们将数组分成多个 Block 处理时：每个 Block 独立计算自己的局部前缀和 P_local。要得到全局前缀和 P_global，我们需要知道前面所有 Block 的总和（即 全局偏移量）。aux_sums 数组的作用aux_sums 数组存储了第一层所有 Block 的总和。aux_sums[i] 就是 Block i 的总和。问题: 我们需要计算 aux_sums 数组的前缀和，才能知道每个 Block 的正确全局偏移量。

3)
    add<<<div_helper(N - 1 - TILE*CFACTOR, TILE*CFACTOR), TILE>>>(output, aux_sums, N);

返回和累加 (add Kernel)现在，递归开始返回，执行 add Kernel 将全局偏移量累加到主数组。add<<<1, 256>>>(output, aux_sums, 4096):aux_sums 中存储了第 2 层的全局前缀和。add Kernel 将 Block 0 的总和（aux_sums[0]）累加到主数组 output 的 0 到 511 区域之后的每一个元素上。

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
