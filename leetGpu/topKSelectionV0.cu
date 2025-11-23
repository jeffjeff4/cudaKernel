#include <cuda_runtime.h>

#define BLOCK_THREADS 32
#define MAX_BLOCKS 64

__device__ void insert(float* arr, int idx, float val) {
    if (val<= arr[idx-1]) return;

    int l=0, r=idx-1;
    while (l<r) {
        int mid = l + (r-l)/2;
        if (arr[mid] > val) {
            l = mid+1;
        } else {
            r = mid;
        }
    }

    for (int i=idx-1; i>l; --i) {
        arr[i] = arr[i-1];
    }
    arr[l] = val;
}

__device__ __forceinline__
void mergeTopK(const float* A, const float* B, int k, float* C) {
    int i=0, j=0;
    for (int c=0; c<k; ++c) {
        if (i<k and (j>=k or A[i]>=B[j])) {
            C[c] = A[i++];
        } else {
            C[c] = B[j++];
        }
    }
}

__global__ void topK(const float* input, float* output, int N, int k) {
    extern __shared__ float shared_mem[];
    int tx = threadIdx.x;
    float* buf_a = shared_mem;
    float* buf_b = shared_mem + blockDim.x * k;

    float* my_a = buf_a + tx * k;
    float* my_b = buf_b + tx * k;

    for (int idx=0; idx<k; ++idx) {
        my_a[idx] = -INFINITY;
    }
    __syncthreads();

    for(int idx = tx + blockIdx.x * blockDim.x; idx<N; idx += blockDim.x * gridDim.x) {
        insert(my_a, k, input[idx]);
    }
    __syncthreads();

    bool ping = true;
    for (int stride=blockDim.x/2; stride>=1; stride/=2) {
        if (tx<stride) {
            float* src_a = ping ? my_a : my_b;
            float* src_b = ping ? buf_a + (tx + stride) * k : buf_b + (tx+stride) * k;
            float* dst = ping ? my_b : my_a;
            mergeTopK(src_a, src_b, k, dst);
        }
        __syncthreads();
        ping = !ping;
    }

    if (tx==0) {
        float* final_buf = ping ? buf_a : buf_b;
        for (int idx=0; idx<k; ++idx) {
            output[blockIdx.x * k + idx] = final_buf[idx];
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    int smem_size = 2 * BLOCK_THREADS * k * sizeof(float);
    int blocks_per_grid = min(MAX_BLOCKS, (N+BLOCK_THREADS-1) / BLOCK_THREADS);

    float* d_block_results;
    cudaMalloc(&d_block_results, blocks_per_grid * k * sizeof(float));
    topK<<<blocks_per_grid, BLOCK_THREADS, smem_size>>>(input, d_block_results, N, k);

    int remaining_blocks = blocks_per_grid;
    float* d_current = d_block_results;

    while (remaining_blocks > 1) {
        int new_blocks = (remaining_blocks + BLOCK_THREADS -1) / BLOCK_THREADS;
        float* d_next;
        cudaMalloc(&d_next, new_blocks * k * sizeof(float));

        topK<<<new_blocks, BLOCK_THREADS, smem_size>>>(d_current, d_next, remaining_blocks * k, k);

        //original version, correct
        /*
        if (d_current != d_block_results) {
            cudaFree(d_current);
        }
        */
        //v1, correct
        cudaFree(d_current);
        //v2, correct, no need to cudaFree(d_current)
        d_current= d_next;
        remaining_blocks = new_blocks;
    }
    cudaMemcpy(output, d_current, k*sizeof(float), cudaMemcpyDeviceToDevice);
}



/*
question0:
不理解，请举例说明
why times 2?
    int smem_size = 2 * BLOCK_THREADS * k * sizeof(float);

这个表达式中乘以 **2** 是因为这个 **Top-K 归并算法**使用了 **Ping-Pong (双缓冲)** 技术，需要两倍的空间来存储中间结果。

---

## 💡 为什么要乘以 2？(双缓冲机制)

### 1. 内存需求分析

在 `topK` Kernel 中，共享内存 (`shared_mem`) 被用来存储所有线程的局部 Top-K 列表，并用于块内（Block-wide）的归并操作。

* **线程数:** `BLOCK_THREADS`
* **每个列表长度:** `k`
* **每个元素大小:** `sizeof(float)`

每个线程需要 k 个浮点数的空间来存储它的 Top-K 列表。因此，所有线程的总需求是 BLOCK_THREADS * k * sizeof(float)。

### 2. Ping-Pong 归并 (Merging with Swapping)

在 Kernel 内部，归并循环 (`for (int stride=...`) 实现了树形归并，将 N 个列表合并成 N/2 个列表，然后是 N/4 个，以此类推。

为了执行归并操作而不破坏源数据，代码使用了两个交替的缓冲区：

* **Buffer A (`buf_a`):** 作为源数据。
* **Buffer B (`buf_b`):** 作为目标数据。

归并函数 (`mergeTopK`) 读取 **Buffer A** 和另一个源（例如 **Buffer B** 的一部分），并将结果写入目标缓冲区。在下一次循环迭代中，角色互换，**Buffer B** 成为源，**Buffer A** 成为目标。

总空间 = BLOCK_THREADS * k * sizeof(float) + BLOCK_THREADS * k * sizeof(float)

总空间 = Buffer A 的空间 + Buffer B 的空间

因此，**乘以 2** 确保了有足够的空间容纳这两个独立的缓冲区 (`buf_a` 和 `buf_b`)，从而实现高效且线程安全的 **Ping-Pong 归并**。    

*/


/*
question1:

不理解，请解释，用例子：
归并函数 (mergeTopK) 读取 Buffer A 和另一个源（例如 Buffer B 的一部分），并将结果写入目标缓冲区。在下一次循环迭代中，角色互换，Buffer B 成为源，Buffer A 成为目标。

这是一个关于并行编程中 **Ping-Pong (双缓冲)** 技术如何应用于 **并行归并 (Merge)** 的例子。

在这个 Top-K 算法中，**归并函数 (`mergeTopK`)** 的目标是将两个已排序的 k 长度列表合并成一个新的 k 长度列表（只保留最大的 k 个元素）。

---

## 🔢 示例设置

* k=4 (每个列表长度为 4)
* BLOCK_THREADS=8 (为简化，假设只有 8 个线程)
* **共享内存:** 被分成两个缓冲区，每个线程都有一个 A 区和 B 区。

### 初始数据 (所有线程已找到局部 Top-4)

假设 tx 是线程索引。

| 线程 (tx) | 初始数据 | 存储位置 |
| :--- | :--- | :--- |
| **0** | A_0 = [100, 90, 80, 70] | buf_a |
| **1** | A_1 = [95, 85, 75, 65] | buf_a |
| **2** | A_2 = [110, 105, 98, 92] | buf_a |
| **3** | A_3 = [88, 78, 68, 58] | buf_a |
| ... | ... | ... |

## 1. 第一次归并 (Ping: Buffer A --> Buffer B)

归并循环从最大的步长 stride = BLOCK_THREADS / 2 = 4 开始。

* **条件:** 只有 tx < 4 的线程工作（线程 0, 1, 2, 3）。
* **角色:** `ping` 为 **True**。
    * **源 A:** my_a (线程自己的 buf_a 区域)
    * **源 B:** 位于 buf_a 中，但由 tx + stride 处的线程持有（即线程 4, 5, 6, 7 的初始数据）。
    * **目标:** my_b (线程自己的 buf_b 区域)

| 工作线程 tx | stride=4 | src_a (来自 buf_a) | src_b (来自 buf_a) | dst (写入 buf_b) | 结果 C (Top-4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 0 < 4 | A_0 [100, 90, 80, 70] | A_4 | my_b | C_0 = Merge(A_0, A_4) |
| **1** | 1 < 4 | A_1 [95, 85, 75, 65] | A_5 | my_b | C_1 = Merge(A_1, A_5) |
| **2** | 2 < 4 | A_2 [110, 105, 98, 92] | A_6 | my_b | C_2 = Merge(A_2, A_6) |
| **3** | 3 < 4 | A_3 [88, 78, 68, 58] | A_7 | my_b | C_3 = Merge(A_3, A_7) |

> **关键点:**
> * 线程 0 读取 A_0 和 A_4 (都来自 buf_a)，将结果写入 buf_b 的区域 C_0。
> * buf_a 中的原始数据 A_0, A_1, A_2, A_3 **保持不变**，作为下一轮的潜在输入。

## 2. 第二次归并 (Pong: Buffer B --> Buffer A)

在 `__syncthreads()` 之后，`ping` 变为 **False**。步长 s 变为 2。

* **条件:** 只有 tx < 2 的线程工作（线程 0, 1）。
* **角色:** `ping` 为 **False**。
    * **源 A:** my_b (线程自己的 buf_b 区域)
    * **源 B:** 位于 buf_b 中，但由 tx + stride 处的线程持有（线程 2, 3 的新结果）。
    * **目标:** my_a (线程自己的 buf_a 区域)

| 工作线程 tx | stride=2 | src_a (来自 buf_b) | src_b (来自 buf_b) | dst (写入 buf_a) | 结果 D (Top-4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 0 < 2 | C_0 | C_2 | my_a | D_0 = Merge(C_0, C_2) |
| **1** | 1 < 2 | C_1 | C_3 | my_a | D_1 = Merge(C_1, C_3) |

> **关键点:**
> * 线程 0 读取 C_0 和 C_2 (都来自 buf_b)，将结果写入 buf_a 的区域 D_0。
> * 缓冲区角色互换：buf_b 充当源，buf_a 充当目标。

## 🎯 总结：Ping-Pong 的用途

之所以需要 *** 2 倍的共享内存** 和 **Ping-Pong 机制**，是因为：

1.  **数据依赖性:** 在归并操作中，你必须同时访问两个列表（源 A 和 源 B）。
2.  **原地操作不可行:** 如果你尝试将归并结果直接写回源数据所在的缓冲区，你会破坏下一轮归并所需要的输入数据。
3.  **解决办法:** 两个缓冲区允许数据在 A --> B 和 B --> A 之间交替流动，确保每轮归并都有一个完整的、未被破坏的源数据副本，同时也提供了一个独立的目标空间来存储结果。
*/


/*
question2:
不理解，请解释
buf_a 中的原始数据 A_0, A_1, A_2, A_3 保持不变，作为下一轮的潜在输入。

这是一个关于 **Ping-Pong (双缓冲) 机制** 在并行计算中如何保证数据完整性的核心点。

在并行归并算法中，**数据必须被保留**，因为虽然它可能在当前迭代中不直接作为源数据，但它可能在后续迭代中被另一个线程访问，或者，更重要的是，它会在下一轮迭代中作为**新的目标缓冲区**的一部分被覆写。

---

## 💡 为什么数据需要被保留？

我们以上一轮归并（stride=4）和下一轮归并（stride=2）为例。

### 初始状态 ( stride=4 之前)

* **Buffer A (buf_a):** 包含所有线程的原始局部 Top-K 结果 (A_0, A_1, A_2, A_3, ...)。
* **Buffer B (buf_b):** 闲置/包含旧数据。

### 阶段 1: 第一次归并 (stride=4)

* **工作线程：** 线程 tx=0, 1, 2, 3。
* **操作：** 线程 tx 读取 A_tx 和 A_(tx+4)（都来自 buf_a），将结果 C_tx 写入 buf_b。

| 线程 | 读取源 buf_a | 写入目标 buf_b |
| :--- | :--- | :--- |
| **0** | 读取 A_0, A_4 | 写入 C_0 |
| **1** | 读取 A_1, A_5 | 写入 C_1 |
| **2** | 读取 A_2, A_6 | 写入 C_2 |
| **3** | 读取 A_3, A_7 | 写入 C_3 |

**结果：** 在 `__syncthreads()` 之后，buf_a 中的 A_0, A_1, ... **没有被任何写入操作触碰**，它们仍然保存着原始数据。buf_b 现在包含新的、合并后的结果 C_0, C_1, C_2, C_3。

### 阶段 2: 第二次归并 (stride=2)

* **工作线程：** 线程 tx=0, 1。
* **角色互换：** `ping` 变为 False。现在 buf_b 是源，buf_a 是目标。

| 线程 | 读取源 buf_b | 写入目标 buf_a |
| :--- | :--- | :--- |
| **0** | 读取 C_0, C_2 | 写入 D_0 |
| **1** | 读取 C_1, C_3 | 写入 D_1 |

#### buf_a 中的原始数据有什么用？

在 **阶段 2** 中，线程 0 将归并结果 D_0 写入 buf_a 的 A_0 区域。

* **A_0, A_1, ... 的命运：** buf_a 中的原始数据 (A_0, A_1, ...) 在 **阶段 1** 中作为**源数据**完成了它的使命。在 **阶段 2** 中，它被安全地选作**目标缓冲区**，它的原始内容 (A_0) 被线程 0 写入的新结果 (D_0) **覆盖**了。

> **结论：** buf_a 中的原始数据在第一次归并中保持不变，是为了**防止在被读取之前被覆写**，并等待它在下一轮迭代中被选作**目标缓冲区**，从而被新的归并结果覆盖。整个过程像一个接力赛，数据在 A 和 B 之间交替存储，但始终有一个缓冲区持有当前最新的归并结果。

这就是 * 2 的共享内存空间（Ping-Pong）的价值：它提供了两块区域，允许计算读取一块数据的同时，将结果安全地写入另一块区域，避免了数据损坏和复杂的内存管理。

*/


/*
question3:
不理解，请解释，用例子

        int new_blocks = (remaining_blocks + BLOCK_THREADS -1) / BLOCK_THREADS;
        remaining_blocks = new_blocks;

这是 **层次归并算法 (Hierarchical Merging)** 中，用于计算下一轮归并所需的线程块数量的关键步骤。它发生在主机端（CPU）的 `solve` 函数中。

这两行代码的核心目的是：**确定下一阶段的归并任务需要多少个线程块来执行。**

---

## 🔢 示例设定

我们使用代码中的参数：

* BLOCK_THREADS = 32 (每个线程块的线程总数)
* remaining_blocks：上一轮归并后剩下的 **Top-K 列表数量**。

### 核心原理：向上取整除法

计算下一轮线程块数量时，使用的是标准的**向上取整除法**：

new_blocks = ceil( remaining_blocks / BLOCK_THREADS )

---

## 💡 示例解释

假设经过第一轮和第二轮归并后，我们剩下一些 Top-K 列表，需要进行下一阶段的归并。

### 场景 A: 完美整除

* remaining_blocks = 64 (即有 64 个 Top-K 列表需要归并)。
* BLOCK_THREADS = 32。

#### 计算过程:

1.  **计算 new_blocks：**
    new_blocks = (64 + 32 - 1) / 32 = 95 / 32 = 2
    * **结果:** new_blocks = 2。

2.  **更新 remaining_blocks：**
    remaining_blocks = 2

> **含义:** 64 个 Top-K 列表，每个线程块的 32 个线程可以处理 32 * 2 = 64 个列表（每个线程负责 2 个列表）。所以只需要 **2 个线程块**就能完成下一轮归并。

---

### 场景 B: 需要向上取整（非整除）

* remaining_blocks = 65 (即有 65 个 Top-K 列表需要归并)。
* BLOCK_THREADS = 32。

#### 计算过程:

1.  **计算 new_blocks：**
    new_blocks = (65 + 32 - 1) / 32 = 96 / 32 = 3
    * **结果:** new_blocks = 3。

2.  **更新 remaining_blocks：**
    remaining_blocks = 3

> **含义:**
> * 2 个线程块只能处理 2 * 32 = 64 个列表。
> * 剩下的第 65 个列表必须由 **第 3 个线程块** 来处理。
> * 因此，我们需要启动 **3 个线程块**来完成这 65 个 Top-K 列表的归并。

---

## 🎯 总结

这两行代码和 `while` 循环一起，驱动了 **Top-K 层次归并** 的进程：

1.  **计算:** new_blocks 确定了下一轮归并操作所需的 **Grid 尺寸**。
2.  **更新:** remaining_blocks = new_blocks 将这个新的 Grid 尺寸作为下一轮归并的**输入数据量**。

这个过程会一直重复，直到 remaining_blocks = 1，此时只剩下一个线程块的结果，即为最终的全局 Top-K 列表。        

*/


/*
question4:
why buf_b = shared_mem + blockDim.x * k? 请解释，用例子

这是一个关于 **Shared Memory 寻址**和 **Ping-Pong 缓冲**的关键步骤。

这里的表达式 buf_b = shared_mem + blockDim.x * k 的目的是计算 **Buffer B 的起始地址**，使其紧接在 **Buffer A 之后**，从而将整个共享内存空间平均分成两部分。

---

## 🔢 示例设定

我们沿用之前的示例值：

* blockDim.x (线程块大小) = **8** (即有 8 个线程)
* k (Top-K 列表长度) = **4**
* shared_mem：共享内存的起始指针（地址 0）。
* buf_a：**Buffer A** 的起始指针（地址 0）。

### 核心需求：Buffer A 占用多大空间？

Buffer A (`buf_a`) 必须为线程块内的所有线程存储各自的 Top-K 列表。

* **线程数量:** blockDim.x = 8
* **每个线程的列表长度:** k = 4
* **总元素数 (Buffer A):** 8 * 4 = 32 个浮点数。

**Buffer A 占据了共享内存的前 32 个浮点数位置 (索引 0 到 31)。**

---

## 💡 buf_b 的计算和寻址

### 目标：紧密排列

为了最大限度地利用分配的共享内存空间，Buffer B (`buf_b`) 必须在 Buffer A 结束的位置立即开始。

1.  **Buffer A 的结束地址:** shared_mem 的起始地址 + Buffer A 的总大小。
    End Address of Buffer A = shared_mem + (blockDim.x * k)

2.  **Buffer B 的起始地址:** 由于 buf_b 必须紧随其后，所以：
    buf_b = End Address of Buffer A

### 示例解释:

假设我们使用浮点数（`float`）作为单位（即指针算术基于 `float`）：

* shared_mem 地址 = 0
* **Buffer A 的大小:** 8 * 4 = 32 个浮点数。

buf_b = 0 + 32

* **结果:** buf_b 的地址是 **32**。

> **含义:**
> * `buf_a` 从索引 0 开始。
> * `buf_b` 从索引 32 开始。
> * 这两个指针有效地将整个共享内存分成了两个大小相等的、连续的区域，供 Ping-Pong 归并使用。

### 总结

乘以 blockDim.x * k 是为了计算 **Buffer A 的总尺寸**，并将这个尺寸作为**偏移量**加到共享内存的起始地址上，从而精确地定位第二个缓冲区 (`buf_b`) 的起始位置。

*/


/*
question5:
不理解，请解释
Buffer A (buf_a) 必须为线程块内的所有线程存储各自的 Top-K 列表。
每个线程的列表长度: k = 4

好的，我来解释为什么 **Buffer A** 必须存储所有线程的 Top-K 列表，并用 k=4 的具体例子说明这个布局。

---

## 💡 核心目标：线程块内并行归并

这段代码实现的是一个 **线程块内（Block-wide）** 的归并排序。为了归并，首先必须将所有参与归并的数据集中在一个高速、可共享的地方——即 **Shared Memory**。

### 1. 为什么每个线程都需要 k 个空间？

* **计算分工:** 在 Kernel 的第一阶段，每个线程 (tx) 独立地处理输入数据集的一个子集，并使用 `insert` 函数找到它自己处理的那些数据中的 **Top-k 最大值**。
* **存储要求:** 为了保存这个局部结果，线程 tx 需要一个长度为 k 的浮点数数组。

**示例：k=4**

假设线程 tx=5 处理了一百万个数据点。它找到了这批数据中的最大 **4 个** 值：[95.2, 91.1, 88.0, 85.5]。

* 线程 tx=5 必须将这 4 个值存储在它的私有区域 my_a 中，等待后续归并。

### 2. 为什么 buf_a 必须存储**所有线程**的结果？

在局部 Top-K 查找完成后，所有线程（例如 tx=0 到 7）的局部结果都存储在 **Buffer A** 中：

| 线程 ID (tx) | 局部 Top-K 列表 (k=4) | 存储位置 (在 buf_a 中) |
| :--- | :--- | :--- |
| **0** | [A_0, A_1, A_2, A_3] | shared_mem[0] 到 [3] |
| **1** | [B_0, B_1, B_2, B_3] | shared_mem[4] 到 [7] |
| **2** | [C_0, C_1, C_2, C_3] | shared_mem[8] 到 [11] |
| ... | ... | ... |
| **7** | [...] | shared_mem[28] 到 [31] |

**原因：**

* **归并的输入:** 归并操作（`mergeTopK`）需要同时访问两个列表。例如，在第一轮归并中，线程 0 需要读取它自己的列表 A **和** 线程 4 的列表 D。如果 D 不在共享内存中，就无法进行快速归并。
* **数据集中化:** buf_a 的任务就是将线程块内所有分散的局部结果集中到一个高速、统一的内存区域，作为**并行归并的初始输入**。

### 总结

buf_a 的设计确保了：

1.  **足够的空间：** blockDim.x * k 提供了 32 个浮点数的空间。
2.  **可寻址性：** 任何线程都可以通过计算正确的偏移量（例如 tx * k）快速访问任何其他线程的局部 Top-K 结果。

*/


/*
question6:
不理解，请解释，用例子
为啥idx += blockDim.x * gridDim.x？
    for(int idx = tx + blockIdx.x * blockDim.x; idx<N; idx += blockDim.x * gridDim.x) 
        insert(my_a, k, input[idx]);
    

这是一个在 CUDA 编程中实现**全局数据遍历**的经典且高效的模式，称为 **Grid-Stride Loop**（网格步长循环）。

这种循环模式确保了 GPU 上的所有线程（包括所有 Block 中的所有线程）都能覆盖整个数据集 N，而不会遗漏或重复处理数据。

---

## 🔢 示例设定

我们使用以下简化值：

* blockDim.x (线程块大小) = **8** (每个 Block 8 个线程)
* gridDim.x (Grid 大小) = **4** (共 4 个线程块)
* **N** (总数据量) = 100

### 核心问题：如何分配工作？

我们的 Grid 总共有 4 * 8 = 32 个线程。我们想让这 32 个线程以最高效率处理 input[0] 到 input[99] 的所有数据。

---

## 1. 循环的起始点 (idx 的初始化)

idx = tx + blockIdx.x * blockDim.x

这个初始值计算了当前线程 (blockIdx.x, tx) 在整个数据集中的**专属起始索引**。

### 示例解释 (起始点)

| Block ID (blockIdx.x) | 线程 ID (tx) | blockIdx.x * 8 | idx (起始索引) |
| :--- | :--- | :--- | :--- |
| **0** (第一个 Block) | 0 | 0 | 0 + 0 = 0 |
| **0** (第一个 Block) | 7 | 0 | 7 + 0 = 7 |
| **1** (第二个 Block) | 0 | 8 | 0 + 8 = 8 |
| **3** (第四个 Block) | 7 | 24 | 7 + 24 = 31 |

> **结论：** 每个线程的 idx 都是独一无二的，确保所有 32 个线程从不同的位置开始，均匀地分摊了前 32 个数据点。

---

## 2. 循环的步长 (idx += blockDim.x * gridDim.x)

步长 = blockDim.x * gridDim.x = 8 * 4 = 32

### 为什么要用这个步长？

* **Grid Size:** blockDim.x * gridDim.x 正好等于 GPU 启动的**总线程数** (32)。
* **作用：** 一旦所有 32 个线程完成了它们的第一轮处理（索引 0 到 31），下一步就必须从第 32 个索引开始。

### 示例解释 (数据遍历)

我们跟踪 tx=5 线程（无论它属于哪个 Block）：

| 线程 ID | 初始 idx | 第 1 次迭代 | 第 2 次迭代 | 第 3 次迭代 |
| :--- | :--- | :--- | :--- | :--- |
| **Block 0, tx=5** | 5 | 5 + 32 = 37 | 37 + 32 = 69 | 69 + 32 = 101 (> N, 结束) |
| **Block 1, tx=5** | 13 | 13 + 32 = 45 | 45 + 32 = 77 | 77 + 32 = 109 (> N, 结束) |
| **Block 3, tx=5** | 29 | 29 + 32 = 61 | 61 + 32 = 93 | 93 + 32 = 125 (> N, 结束) |

> **结论：**
> 1.  所有 32 个线程的第一轮迭代处理了索引 0 到 31。
> 2.  所有 32 个线程的第二轮迭代处理了索引 32 到 63。
> 3.  所有 32 个线程的第三轮迭代处理了索引 64 到 95。

通过这种方式，每个线程都会处理相隔 32 个元素的序列，共同确保 N 个数据点的每一个元素都被高效地处理到。    

*/


/*
question7:
不理解，请解释，用例子：

    bool ping = true;
    for (int stride=blockDim.x/2; stride>=1; stride/=2) 
        if (tx<stride) 
            float* src_a = ping ? my_a : my_b;
            float* src_b = ping ? buf_a + (tx + stride) * k : buf_b + (tx+stride) * k;
            float* dst = ping ? my_b : my_a;
            mergeTopK(src_a, src_b, k, dst);
        
        __syncthreads();
        ping = !ping;
    

这段代码是 **Top-K 查找算法**中，将一个线程块（Block）内所有线程的局部 Top-K 列表合并成一个最终 Top-K 列表的核心部分。它采用了**树形归并（Tree Reduction）**和 **Ping-Pong（双缓冲）** 机制。

它负责将 N 个列表归并为 N/2 个，再归并为 N/4 个，直到只剩一个列表。

## 🚀 I. 核心机制：树形归并与 Ping-Pong

### 1. 树形归并（循环结构）

* **`for (int stride = blockDim.x/2; stride >= 1; stride /= 2)`:** 这是一个标准的并行归约循环。它将归并任务分阶段执行，每阶段参与归并的线程数减半，步长也减半。
    * **起始：** 步长最大，stride = 16（假设 blockDim.x=32）。
    * **结束：** stride = 1。
* **`if (tx < stride)`:** 确保只有前半部分的线程参与归并。例如，当 stride=16 时，只有线程 0 到 15 工作。当 stride=1 时，只有线程 0 工作。

### 2. Ping-Pong 切换（双缓冲）

* **`bool ping = true;` 和 `ping = !ping;`:** 这是 Ping-Pong 机制的控制开关。它确保在每次归并时，数据在两个缓冲区（buf_a 和 buf_b）之间交替流动：
    * **Ping (True):** 从 A 读，写入 B。
    * **Pong (False):** 从 B 读，写入 A。
* **`__syncthreads();`:** **至关重要！** 它确保在一个归并阶段（例如 stride=16）完成后，所有线程都已将结果写入目标缓冲区，然后才能进入下一阶段（stride=8），此时缓冲区角色互换。

## 🔢 II. 示例解释：数据流动

假设 blockDim.x=8（为简化），k=4。

| 线程 | 初始数据 | 存储区域 |
| :--- | :--- | :--- |
| **0 - 7** | 局部 Top-4 列表 | **buf_a** (通过 my_a 写入) |

### 阶段 1: stride=4 (Ping)

* **`ping` = True**
* **工作线程:** tx=0, 1, 2, 3
* **指针设置:**
    * `src_a`: my_a (线程 tx 的 buf_a 区域)
    * `src_b`: buf_a + (tx + 4) * k (线程 tx+4 的 buf_a 区域)
    * `dst`: my_b (线程 tx 的 buf_b 区域)
* **操作:**
    * 线程 0 归并 A_0 和 A_4 的结果，写入 B_0。
    * 线程 1 归并 A_1 和 A_5 的结果，写入 B_1。
* **结果：** buf_b 包含了 4 个新的、合并后的 Top-4 列表 (B_0 到 B_3)。buf_a 中的 A_0 到 A_7 保持不变（但下一轮会被覆盖）。

### 阶段 2: stride=2 (Pong)

* **`ping` = False**
* **工作线程:** tx=0, 1
* **指针设置:**
    * `src_a`: my_b (线程 tx 的 buf_b 区域)
    * `src_b`: buf_b + (tx + 2) * k (线程 tx+2 的 buf_b 区域)
    * `dst`: my_a (线程 tx 的 buf_a 区域)
* **操作:**
    * 线程 0 归并 B_0 和 B_2 的结果，写入 A_0。
    * 线程 1 归并 B_1 和 B_3 的结果，写入 A_1。
* **结果：** buf_a 现在包含了 2 个新的、合并后的 Top-4 列表。

这个过程重复进行，直到 stride=1，最终结果集中到 **线程 0** 的区域（my_a 或 my_b）。


*/


/*
question8:


*/


/*
question9:


*/


/*
question10:


*/


/*
question11:


*/


/*
question12:


*/


/*
question13:


*/