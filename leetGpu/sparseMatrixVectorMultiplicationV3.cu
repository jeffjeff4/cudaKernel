/*
question 0:
为啥途中红线表的1），2）的row_index, col_index的顺序换了？

因为 COO 只是把非零元素存成一组三元组 `(row, col, value)`，这些三元组的顺序本身并不影响结果。

### 关键点
- `row_index[i]`、`col_index[i]`、`data[i]` 是一一对应的。
- 只要第 `i` 个元素的行、列、值保持一致，`i` 的顺序可以任意换。
- 也就是说，COO 里不需要像 CSR/ELL 那样按行或按列严格排序。

### 所以 1）和 2）为什么能够换？
- 图中的 1）是按“原始输入顺序”或者“按行分组”的一个表示。
- 图中的 2）是把这些三元组重新排序后的另一种表示。
- 两组数据表示的是同一个稀疏矩阵，只是元素在数组中的排列顺序不同。

### 举个简单类比
如果有三元组：
```
(0,0,3), (0,2,1), (2,1,2)
```
你可以把它写成：
```
(2,1,2), (0,0,3), (0,2,1)
```
只要对应关系不变，矩阵不变。

### 为什么要重排
通常是为了性能，比如：
- 让内存访问更连续
- 让线程读取更规则
- 避免随机访问带来的带宽浪费

所以红线里面的 1）和 2）不是“不同矩阵”，而是“同一个矩阵的两种 COO 排列方式”。

//-----------------------------------------------------------------------

question 1:
能结合图中的例子，具体说明为啥？
通常是为了性能，比如：

让内存访问更连续
让线程读取更规则
避免随机访问带来的带宽浪费

因为图中的 1）和 2）其实是同一个稀疏矩阵的两种 COO 存储顺序，它们只是把相同的三元组 `(row, col, value)` 重新排列了一下。

## 1）和 2）到底怎么换的

图 1）表示按“原始行顺序”存储：
- `row_index = [0, 0, 2, 2, 2, 3, 3]`
- `col_index = [0, 2, 1, 2, 3, 0, 3]`
- `data      = [3, 1, 2, 4, 1, 1, 1]`

这意味着非零元按 row0、row2、row3 顺序排列。

图 2）是把这些三元组打散重排后的结果：
- `row_index = [3, 0, 2, 2, 0, 2, 3]`
- `col_index = [0, 2, 1, 2, 0, 3, 3]`
- `data      = [1, 1, 2, 4, 3, 1, 1]`

注意：
- 它们表示的是同一个矩阵
- 对应位置的 `(row,col,value)` 仍然是一一对应的
- 只是元素在数组里的“排列顺序”换了

---

## 为什么可以换？因为 COO 不要求顺序

COO 只是表达：
- 第 `i` 个元素是 `row_index[i]`
- 第 `i` 个元素是 `col_index[i]`
- 第 `i` 个元素的值是 `data[i]`

只要这一一对应关系保持，顺序就可以任意调整。

所以这两个序列只是“同一组三元组的不同排序”。

---

## 为什么要换？主要是为了 GPU 性能

在 GPU 上，执行 COO SpMV 时通常是：
- 每个非零元由一个线程处理
- 线程读取 `x[col]`
- 线程写入 `y[row]`（通常是 atomicAdd）

因此两个关键成本是：

1. `x[col]` 的访问
   - 如果相邻线程访问的 `col` 很接近，内存访问更连续
   - 这样可以更好地利用 DRAM coalescing / L1/L2 缓存
2. `y[row]` 的写
   - 目标行如果比较集中，写操作更有规律
   - 虽然 atomic 会有竞争，但连续写同一行比完全随机写通常更好缓存

---

## 结合图中例子看差别

假设你有 `x[0..3]`：

### 图 1）读取 `x` 的顺序
```
x[0], x[2], x[1], x[2], x[3], x[0], x[3]
```

### 图 2）读取 `x` 的顺序
```
x[0], x[2], x[1], x[2], x[0], x[3], x[3]
```

你可以看到图 2）中：
- `x[0]` 出现得更集中
- `x[3]` 也更集中

这对于 GPU 来说，意味着：
- 连续线程更可能访问相邻或重复的 `x` 元素
- DRAM/缓存更容易把这些加载到一起
- 访问更规则，性能更好

---

## 结论

图中的 1）和 2）顺序不同，原因不是“矩阵变了”，而是：

- COO 允许你自由重排非零元
- 你可以选择更有利于 GPU 访问模式的排列
- 目标就是：让 `x[col]` 访问更连续，线程读写更规则，减少随机访问浪费

所以这两组 `row_index` / `col_index` 顺序换了，本质上是同一个矩阵的“性能优化型排列”。

//---------------------------------------------------------------
question 2:
if so, why not change to:
- `col_index = [0, 0, 1, 2, 2, 3, 3]? this could introduce better gpu performance

Yes — sorting by `col_index` is a valid COO reordering, but it is not automatically better for the kernel you are likely using.

## Why not just do `col_index = [0,0,1,2,2,3,3]`?

### 1. COO order does not affect correctness
- In COO, any permutation of the triplets `(row, col, value)` is mathematically the same matrix.
- So yes, you can reorder them by column index if you want.

### 2. But GPU performance depends on both reads and writes
In a typical COO kernel:
- each thread reads `x[col]`
- each thread does `atomicAdd(&y[row], value * x[col])`

Sorting by column improves:
- `x` access locality, because nearby threads will likely read nearby `x` entries.

However, it may hurt:
- `y` update locality and atomic behavior, because `row` becomes more scattered.

### 3. Example with your figure
If you reorder just by `col_index`, the read pattern becomes more regular:
- `x[0], x[0], x[1], x[2], x[2], x[3], x[3]`

That is good for `x`.

But the corresponding `row_index` order will be mixed:
- writes to `y[0], y[2], y[0], y[2], y[3], y[?, ?]`

That means:
- the kernel writes to `y` in a less-structured way
- atomic updates may become more irregular
- caching and store efficiency on `y` may worsen

### 4. In COO, the best order depends on your kernel design
- If your kernel is one thread per nonzero with `atomicAdd`, row clustering or row-major order is often better.
- If your kernel first groups entries by row and reduces per row, then row order is essential.
- If you instead build a kernel that streams `x` and accumulates separately, col-sorting can help.

### 5. So the answer is:
- `col_index`-sorted COO is one possible optimization.
- But it is not always the best choice for a GPU SpMV kernel that also has writes to `y[row]`.
- The safest practical choice is usually an order that balances:
  - `x` read locality,
  - `y` write locality,
  - and the reduction method your kernel uses.

> In short: yes, column sorting can improve `x` reads, but it may make the `y[row]` updates worse. That is why we do not simply always reorder COO by `col_index`.

//------------------------------------------------------------------------------
question 3:
why for writing to y,
[0, 2, 1, 2, 0, 3, 3] is better than  [0, 0, 1, 2, 2, 3, 3]?

因为写 `y` 的性能，关键不是“行序列看起来更整齐”，而是“同一个 `y[row]` 的原子更新是否会集中在同一时间段内冲突”。

## 具体原因

### 1. `atomicAdd(&y[row], ...)` 的冲突
如果 `row` 序列是 `[0, 0, 1, 2, 2, 3, 3]`：
- 线程 0、线程 1 都马上写 `y[0]`
- 它们会在几乎相同的时间内竞争同一个地址
- 这会使这两个原子加法产生更强的串行化

而如果序列是 `[0, 2, 1, 2, 0, 3, 3]`：
- 第一个 `y[0]` 更新和第二个 `y[0]` 更新被其他行的写入分开
- 同一地址的冲突被“打散”了一些
- 整体 atomic 竞争可能更低

---

## 2. 这不是“一定要连续写同一行”
直觉上我们会觉得“连续写同一行”好，但对 atomic 更新来说：
- 连续多个线程写同一个 `y[row]`，竞争最严重
- 如果把同一行的更新分散开，GPU 的执行和原子机制反而可能更平滑

---

## 3. 另一层含义：线程调度与 warp 执行
GPU 里一组线程（warp）共享执行节奏：
- 同一个 warp 如果多个线程同时访问同一个 `y[row]`，就会产生热点
- 如果顺序是 `[0,0,...]`，这种热点更容易出现
- 如果顺序是 `[0,2,1,2,0,...]`，热点被“稀释”一些

---

## 4. 不同场景的不同最优
这只是针对“每个非零元一个线程、直接 atomic 加到 y”这种简单 COO kernel：
- 这种情况下，写 `y` 的冲突比严格的行聚集更重要
- 如果采用 “先按行聚合，再写 y” 的方案，反而可能希望行更连续

---

## 结论
所以对于 `y` 的写入来说：
- `[0,0,1,2,2,3,3]` 看起来行更集中，但会让 `y[0]` 的原子更新立即连发，冲突大
- `[0,2,1,2,0,3,3]` 则把同一行的更新分散一些，减少瞬时热点，可能反而更适合简单原子型 COO kernel

> 这就是为什么在 COO 里，单纯“按列排序”或“按行排序”并不是绝对最优，关键是要看你写 `y` 的具体方式。

*/


#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdio>

__global__ void cooSpMVKernel(const int* __restrict__ row_idx,
                              const int* __restrict__ col_idx,
                              const float* __restrict__ values,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) {
        return;
    }

    int row = row_idx[tid];
    int col = col_idx[tid];
    atomicAdd(&y[row], values[tid] * x[col]);
}

namespace {

template <typename T>
void copyToHostBuffer(const T* src, std::vector<T>& dst, size_t count) {
    if (src == nullptr) {
        dst.assign(count, T{});
        return;
    }

    cudaPointerAttributes attr;
    cudaError_t status = cudaPointerGetAttributes(&attr, src);
    if (status == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
        cudaMemcpy(dst.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost);
    } else {
        std::copy(src, src + count, dst.begin());
    }
}

template <typename T>
void copyBuffer(const T* src, T* dst, size_t count) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    cudaPointerAttributes src_attr;
    cudaPointerAttributes dst_attr;

    bool src_is_device = cudaPointerGetAttributes(&src_attr, src) == cudaSuccess &&
                         src_attr.type == cudaMemoryTypeDevice;
    bool dst_is_device = cudaPointerGetAttributes(&dst_attr, dst) == cudaSuccess &&
                         dst_attr.type == cudaMemoryTypeDevice;

    if (src_is_device && dst_is_device) {
        cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice);
    } else if (src_is_device && !dst_is_device) {
        cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
    } else if (!src_is_device && dst_is_device) {
        cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToHost);
    }
}

}  // namespace

extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    if (M <= 0 || N <= 0) {
        return;
    }

    std::vector<float> denseA(static_cast<size_t>(M) * static_cast<size_t>(N));
    std::vector<float> denseX(static_cast<size_t>(N));
    copyToHostBuffer(A, denseA, denseA.size());
    copyToHostBuffer(x, denseX, denseX.size());

    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<float> values;

    row_idx.reserve(std::max(nnz, 0));
    col_idx.reserve(std::max(nnz, 0));
    values.reserve(std::max(nnz, 0));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float v = denseA[static_cast<size_t>(i) * static_cast<size_t>(N) + j];
            if (v != 0.0f) {
                row_idx.push_back(i);
                col_idx.push_back(j);
                values.push_back(v);
            }
        }
    }

    int nz_count = static_cast<int>(row_idx.size());
    if (nz_count == 0) {
        std::vector<float> zero_y(static_cast<size_t>(M), 0.0f);
        copyBuffer(zero_y.data(), const_cast<float*>(y), zero_y.size());
        return;
    }

    int* d_row_idx = nullptr;
    int* d_col_idx = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    cudaMalloc(&d_row_idx, nz_count * sizeof(int));
    cudaMalloc(&d_col_idx, nz_count * sizeof(int));
    cudaMalloc(&d_values, nz_count * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemset(d_y, 0, M * sizeof(float));

    cudaMemcpy(d_row_idx, row_idx.data(), nz_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nz_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nz_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, denseX.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (nz_count + threads - 1) / threads;
    cooSpMVKernel<<<blocks, threads>>>(d_row_idx, d_col_idx, d_values, d_x, d_y, nz_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_row_idx);
        cudaFree(d_col_idx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        return;
    }

    cudaDeviceSynchronize();
    copyBuffer(d_y, const_cast<float*>(y), M);

    cudaFree(d_row_idx);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}
