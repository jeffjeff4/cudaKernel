#include <cuda_runtime.h>

constexpr int NUM_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void monte_carlo_intergration_kernel(const float* y_samples, float* result, float a, float b, int n_samples) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float reduce_smem[NUM_WARPS];
    float sum = (idx < n_samples) ? y_samples[idx] : 0.0f;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0)
      reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
      sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (tid == 0)
    {
      atomicAdd(result, sum);
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    int threadsPerBlock = NUM_THREADS;
    int blocksPerGrid = (n_samples + threadsPerBlock - 1) / threadsPerBlock;
    monte_carlo_intergration_kernel<<<blocksPerGrid, threadsPerBlock>>>(y_samples, result, a, b, n_samples);
    float mem_res;
    cudaMemcpy(&mem_res, result, sizeof(float), cudaMemcpyDeviceToHost);
    mem_res *= (b-a) / n_samples;
    cudaMemcpy(result, &mem_res, sizeof(float), cudaMemcpyHostToDevice);
}


//--------------------------------------------------------------------------------------------------
/*
question0:

不理解，请解释，用例子

这段代码实现了一个基于 **蒙特卡洛方法 (Monte Carlo Method)** 的 **数值积分** CUDA Kernel。它利用了 GPU 的并行归约 (Reduction) 能力，高效地计算大量随机采样的函数值之和。

-----

## ⚙️ I. 核心数学原理：蒙特卡洛积分

蒙特卡洛积分使用随机抽样来近似计算定积分 $\int_a^b f(x) dx$。

基本公式为：
$$\int_a^b f(x) dx \approx (b - a) \cdot \frac{1{N \sum_{i=1^N f(x_i)$$

该 Kernel 的任务是并行计算 $\sum f(x_i)$，即 y_samples 数组中所有元素的总和。最终的乘法 $(b-a)/N$ 在主机端 (solve 函数) 完成。

-----

## 🚀 II. 辅助函数：Warp 归约 (`warp_reduce_sum_f32`)

c
// ...
for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);

return val;


  * **目的:** 在一个 Warp (32 个线程) 内部，使用 **`__shfl_xor_sync`** 指令高效地求和。
  * **机制:** 这是一个树形求和算法。它通过寄存器之间的直接数据交换来累加，避免了慢速的共享内存访问和 `__syncthreads()` 同步，是 GPU 上最快的归约方式。
  * **示例:** 假设 kWarpSize=32$。
    1.  mask=16$：线程 $tx$ 接收来自 $tx \oplus 16$ 的值。
    2.  mask=8$：线程 $tx$ 接收来自 $tx \oplus 8$ 的值。
    3.  ...
    4.  mask=1$：线程 $tx$ 接收来自 $tx \oplus 1$ 的值。
    <!-- end list -->
      * **结果:** Warp 的总和最终集中到 **线程 0**（`lane=0`）的 `val` 变量中。

-----

## 🧠 III. Kernel 执行流程：两级归约

Kernel `monte_carlo_intergration_kernel` 执行一个 **两级归约 (Two-Level Reduction)**：先在 Warp 内求和，再在 Block 内合并 Warp 结果。

### 1\. 线程身份和局部求和

c
int tid = threadIdx.x;
int idx = blockDim.x * blockIdx.x + threadIdx.x;
__shared__ float reduce_smem[NUM_WARPS];
float sum = (idx < n_samples) ? y_samples[idx] : 0.0f; // 边界检查
sum = warp_reduce_sum_f32<WARP_SIZE>(sum);


  * **全局索引 (idx):** 确保每个线程处理 y_samples 数组中的一个独特元素。
  * **局部求和 (sum):** 每个线程加载其数据点，然后 **Warp 归约** 将这 32 个线程的 sum 集中到 lane=0 的线程中。

### 2\. 第一级归约：Warp --> Shared Memory

c
int warp = tid / WARP_SIZE; // 0 到 7
int lane = tid % WARP_SIZE; // 0 到 31
if (lane == 0)
  reduce_smem[warp] = sum; // 只有 lane=0 的线程写入
__syncthreads();


  * **目的:** 将每个 Warp 的总和存储到 Shared Memory 中。
  * **示例:** 假设 NUM_THREADS=256$，NUM_WARPS=8$。
      * 线程 tx=0 (warp=0, \text{lane=0$) 将 Warp 0 的总和写入 reduce_smem[0]。
      * 线程 tx=32 (warp=1, \text{lane=0$) 将 Warp 1 的总和写入 reduce_smem[1]。
  * **`__syncthreads()`:** 确保所有 8 个 Warp 的总和都安全地写入了 reduce_smem 数组。

### 3\. 第二级归约：Shared Memory --> 最终结果

c
sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f; // 线程 0-7 读取数据
if (warp == 0)
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum); // 只有 Warp 0 进行归约
if (tid == 0)
{
  atomicAdd(result, sum); // 线程 0 写入全局结果



  * **数据收集:** 只有 Warp 0 中的前 8 个线程 (lane=0$ 到 $7$) 从 reduce_smem 中读取 8 个 Warp 的总和。
  * **二次归约:** **Warp 0** 对这 8 个值进行第二次 **Warp 归约**，得到整个 Block 的总和。
  * **原子累加:** 最终，只有 tid=0 的线程使用 **`atomicAdd`** 将该 Block 的最终总和累加到全局结果 result 中。

## 🏁 IV. 主机端 (`solve` 函数)

c
// ...
compute_xty<<<blocksPerGrid, threadsPerBlock>>>(...);
float mem_res;
cudaMemcpy(&mem_res, result, sizeof(float), cudaMemcpyDeviceToHost);
mem_res *= (b-a) / n_samples; // 应用蒙特卡洛积分公式的 (b-a)/N 部分
cudaMemcpy(result, &mem_res, sizeof(float), cudaMemcpyHostToDevice);


  * **Grid/Block 启动:** 计算并启动 Kernel，覆盖所有 n_samples。
  * **最终公式:** 在 GPU 完成求和后，主机端将总和 (`mem_res`) 乘以 (b-a) / n_samples，完成最终的积分计算。

//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question1:

不理解，请解释，用例子

这段代码是 **蒙特卡洛数值积分** 的核心 CUDA Kernel，它执行一个\*\*两级并行归约（Two-Level Parallel Reduction）\*\*来计算所有采样值 y_samples 的总和。

这个 Kernel 的目标是：将巨大的求和任务分解到每个线程块（Block）中，然后在块内部进行高效求和。

-----

## ⚙️ I. 示例参数和线程身份

我们假设以下参数：

  * **线程数 (NUM_THREADS)** = 256
  * **Warp 大小 (WARP_SIZE)** = 32
  * **Warp 数量 (NUM_WARPS)** = 256 / 32 = **8**
  * **Block 尺寸 (blockDim.x)** = 256

### 1\. 线程身份分解

c
int tid = threadIdx.x;
int idx = blockDim.x * blockIdx.x + threadIdx.x;
// ...
int warp = tid / WARP_SIZE; // 0 到 7
int lane = tid % WARP_SIZE; // 0 到 31


  * idx：**全局数据索引**。确定当前线程负责处理 `y_samples` 数组中的哪一个元素。
  * warp：线程块内的 **Warp ID** (0 到 7)。
  * lane：Warp 内部的 **线程 ID** (0 到 31)。

-----

## 🚀 II. 第一级归约：Warp 内部求和 (最快速度)

c
float sum = (idx < n_samples) ? y_samples[idx] : 0.0f;
sum = warp_reduce_sum_f32<WARP_SIZE>(sum);


1.  **数据加载:** 每个线程加载其对应的 y_samples[idx] 值到私有变量 sum 中，并进行边界检查。
2.  **Warp 归约:** 调用 `warp_reduce_sum_f32`。这是一个基于 **Shuffle 指令**的函数。
3.  **结果:** 每个 Warp（32 个线程）的总和被累积，并存储到该 Warp 的 **`lane=0` 线程**的 sum 变量中。

> **示例:** 线程 tx=0 到 tx=31（Warp 0）的数据总和，现在存储在线程 tx=0 的 sum 变量中。

-----

## 🧠 III. 第二级归约：Block 内部求和 (Warp 间协作)

这一阶段将所有 8 个 Warp 的结果合并成 Block 的总和。

### 1\. 写入共享内存 (Shared Memory)

c
__shared__ float reduce_smem[NUM_WARPS]; // 8 个浮点数
if (lane == 0)
  reduce_smem[warp] = sum;
__syncthreads();


  * **写入:** 只有每个 Warp 的 lane=0 线程执行写入操作。
      * 线程 tx=0 (warp=0$) 将 Warp 0 的总和写入 reduce_smem[0]。
      * 线程 tx=32 (warp=1$) 将 Warp 1 的总和写入 reduce_smem[1]。
      * ... 共有 8 个 Warp 的总和被写入 `reduce_smem[0]` 到 `reduce_smem[7]`。
  * **`__syncthreads()`:** **关键同步点。** 确保所有 8 个 Warp 的结果都已安全地写入共享内存，才能进入下一步的读取和归并。

### 2\. 读取和最终归约

c
sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f; // NUM_WARPS = 8
if (warp == 0)
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);


  * **数据收集:** 只有 Warp 0 中的前 8 个线程（lane=0$ 到 $7$）会执行读取操作，它们读取 `reduce_smem[0]` 到 `reduce_smem[7]`。其他线程 (lane \ge 8$) 将 sum 设为 0。
  * **二次归约:** 整个 Warp 0 再次执行 **Warp 归约**。这次是将 8 个 Warp 总和（以及 24 个零值）合并成一个总和。
  * **结果:** 整个线程块的总和现在集中在 **线程 tx=0** 的 sum 变量中。

-----

## 🏁 IV. 结果写回

c
if (tid == 0)
{
  atomicAdd(result, sum);



  * 只有 Block 内的第一个线程 (tid=0$) 执行此操作。
  * **`atomicAdd(result, sum)`:** 将当前 Block 计算出的总和安全地累加到全局结果 result 变量中。**原子操作** 确保了来自不同 Block 的总和能够正确、无冲突地合并。

**总结：** 这个 Kernel 利用两级归约（Shuffle --> Shared Memory --> Shuffle）实现了对 n_samples 个数据的超高效求和，并将结果安全地贡献给最终积分结果。


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question2:

不理解，请解释，用例子

不理解，请解释，用例子

sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f; // NUM_WARPS = 8


这段代码是 CUDA 归约 (Reduction) 算法的第二阶段，目的是将 **Shared Memory** 中存储的局部结果（即每个 Warp 的总和）分发给一个 Warp（通常是 Warp 0），以便进行最终的合并计算。

---

## ⚙️ I. 核心目标：数据收集

* **源数据:** reduce_smem 数组，它存储了 Block 内所有 Warp 的总和。
* **目标:** 将 reduce_smem 中的有效数据，安全地导入到 **Warp 0** 的寄存器中。

## 🔢 II. 示例参数设定

我们假设以下参数：

* **NUM_WARPS (Warp 总数)** = 8
* reduce_smem 数组的长度是 8，其中 reduce_smem[0] 到 reduce_smem[7] 存储了 8 个 Warp 的总和。
* **线程身份:** lane 范围是 0 到 31。

### 1. 逻辑分工

代码只在 **Warp 0** 中执行，因为 `if (warp == 0)` 检查紧随其后。所以，我们只看 lane \in [0, 31]$ 的线程。

| 线程 Lane ID (lane) | 检查条件 lane < 8 | 结果 (sum 的新值) | 含义 |
| :--- | :--- | :--- | :--- |
| **0** | True (0 < 8) | reduce_smem[0]$ | 线程 0 读取 Warp 0 的总和。 |
| **1** | True (1 < 8) | reduce_smem[1]$ | 线程 1 读取 Warp 1 的总和。 |
| **7** | True (7 < 8) | reduce_smem[7]$ | 线程 7 读取 Warp 7 的总和。 |
| **8** | **False** (8 不小于 8) | 0.0f | 线程 8 将其 sum 设为 0。 |
| **31** | False (31 不小于 8) | 0.0f | 线程 31 将其 sum 设为 0。 |

### 2. 为什么需要 `if/else`？

* **有效数据定位:** 数组 reduce_smem 只有 8 个有效元素。我们只需要 Warp 0 中的 **前 8 个线程** 来读取这 8 个元素。
* **保持活跃:** Warp 0 中的其余 24 个线程 (lane=8$ 到 $31$) 必须保持活跃（不能退出），以便参与后续的 **Warp 归约** (`warp_reduce_sum_f32`)。将它们的 sum 设为 $0.0f$ 可以确保它们在归约中不影响最终结果，但仍能保持 Warp 的同步。

---

## 🎯 总结

这段代码通过 **数据扇入 (Fan-In)** 机制，将 8 个不同的 Warp 结果，巧妙地分配给了 Warp 0 中 32 个线程中的 **前 8 个线程**。这些线程的 sum 变量现在持有等待最终合并的 8 个数值，为最后的 Warp 归约奠定了基础。
//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question3:

不理解，请解释，用例子


//--------------------------------------------------------------------------------------------------
