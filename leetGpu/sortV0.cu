////This is an excellent piece of code. It correctly implements a high-performance, ////iterative, bottom-up merge sort on the GPU.
////
////This design is the standard, fast way to sort on a GPU and directly solves the ////"timeout" problem you had before. The old, host-recursive code was slow because ////it launched thousands of tiny kernels, each with a high overhead cost. This new ////code is fast because it launches only a handful of large, efficient kernels (one ////for each "pass").
////
////Let's break down how it works using a simple example.
////
////Example:
////
////data = [8, 7, 6, 5, 4, 3, 2, 1]
////
////N = 8
////
////This means solve() will run its for loop 3 times (width = 1, 2, 4).
////
////1. solve Function (The "Driver")////
////This function is the "driver" on the CPU. It doesn't do the sort itself; it just ////sets up the passes and tells the GPU what to do.
////
////C++
////
////extern "C" void solve(float* data, int N) {
////    // ...
////    float* d_helper;
////    cudaMalloc(...); // Allocates a temporary buffer
////
////    float* d_src = data;     // Source
////    float* d_dst = d_helper; // Destination
////
////    dim3 threads(256);
////    dim3 blocks(1); // For N=8, cdiv(8, 256) = 1 block
////
////    // The main loop: runs log2(N) times
////    for (int width = 1; width < N; width *= 2) {
////        // ...
////        iterative_merge_kernel<<<blocks, threads>>>(d_src, d_dst, N, width);
////        cudaDeviceSynchronize(); // Wait for the pass to finish
////
////        // --- Ping-Pong Swap ---
////        float* temp = d_src;
////        d_src = d_dst;
////        d_dst = temp;
////    
////    // ... Final Copy (if needed) ...
////
////This for loop is the "bottom-up" part. Here's how it runs for our N=8 example:
////
////Pass 1: width = 1
////
////d_src (data): [8, 7, 6, 5, 4, 3, 2, 1]
////
////d_dst (helper): [?, ?, ?, ?, ?, ?, ?, ?]
////
////Launches iterative_merge_kernel. This kernel merges all adjacent 1-element arrays ////into 2-element sorted arrays.
////
////After kernel: d_dst is [7, 8, | 5, 6, | 3, 4, | 1, 2]
////
////Ping-Pong Swap: d_src now points to d_helper, d_dst points to data.
////
////Pass 2: width = 2
////
////d_src (helper): [7, 8, 5, 6, 3, 4, 1, 2]
////
////d_dst (data): [?, ?, ?, ?, ?, ?, ?, ?]
////
////Launches iterative_merge_kernel. This kernel merges all adjacent 2-element arrays ////into 4-element sorted arrays.
////
////It merges [7, 8] with [5, 6] to get [5, 6, 7, 8].
////
////It merges [3, 4] with [1, 2] to get [1, 2, 3, 4].
////
////After kernel: d_dst is [5, 6, 7, 8, | 1, 2, 3, 4]
////
////Ping-Pong Swap: d_src points to data, d_dst points to d_helper.
////
////Pass 3: width = 4
////
////d_src (data): [5, 6, 7, 8, 1, 2, 3, 4]
////
////d_dst (helper): [?, ?, ?, ?, ?, ?, ?, ?]
////
////Launches iterative_merge_kernel. This kernel merges the two adjacent 4-element ////arrays.
////
////After kernel: d_dst is [1, 2, 3, 4, 5, 6, 7, 8]
////
////Ping-Pong Swap: d_src points to d_helper, d_dst points to data.
////
////Loop Exits: (width becomes 8, 8 < 8 is false).
////
////Final Copy (Lines 131-135):
////
////if (d_src != data): After the loop, d_src points to d_helper. This is not the ////original data array.
////
////This if is TRUE. The code executes cudaMemcpy(data, d_src, ...) to copy the final ////sorted result from d_helper back into the original data array.
////
////2. iterative_merge_kernel (The "Worker")////
////This is the most complex part. This kernel is launched with N threads (in our ////case, 8 threads). Each thread is responsible for moving one element from d_src to ////its correct sorted position in d_dst.
////
////Let's trace one thread during Pass 2 (width = 2).
////
////d_src = [7, 8, | 5, 6, || 3, 4, | 1, 2]
////
////The kernel's job is to merge [7, 8] with [5, 6] AND [3, 4] with [1, 2].
////
////Trace: Thread n = 1
////
////n = 1. value = d_src[1] = 8.
////
////width = 2.
////
////chunk_size = 2 * width = 4.
////
////chunk_start = (n / chunk_size) * chunk_size = (1 / 4) * 4 = 0.
////
////This tells the thread it belongs to the first merge chunk (merging indices 0-3).
////
////left_start = 0.
////
////left_end = min(0 + 2, 8) = 2.
////
////right_start = 2.
////
////right_end = min(2 + 2, 8) = 4.
////
////The thread knows its "left" sub-array is d_src[0...1] ([7, 8]).
////
////It knows its "right" sub-array is d_src[2...3] ([5, 6]).
////
////if (n < right_start): if (1 < 2) is TRUE.
////
////This thread's value (8) is from the LEFT sub-array.
////
////my_index = n - left_start = 1 - 0 = 1. (It's the 2nd element of its sub-array).
////
////rank_in_other = lower_bound_rank(d_src, right_start, right_end, value)
////
////lower_bound_rank(d_src, 2, 4, 8): "Search for 8 in the other array [5, 6]."
////
////lower_bound finds that 8 would be inserted at the end, so it returns index 4.
////
////dest_idx = chunk_start + my_index + (rank_in_other - right_start)
////
////dest_idx = 0 + 1 + (4 - 2) = 3.
////
////d_dst[dest_idx] = value ➡️ d_dst[3] = 8.
////
////This is correct. In the merged array [5, 6, 7, 8], the 8 belongs at index 3.
////
////Trace: Thread n = 2 (to check the else block)
////
////n = 2. value = d_src[2] = 5.
////
////width = 2, chunk_size = 4, chunk_start = 0.
////
////left_start = 0, left_end = 2, right_start = 2, right_end = 4.
////
////Left array: [7, 8]. Right array: [5, 6].
////
////if (n < right_start): if (2 < 2) is FALSE.
////
////This thread's value (5) is from the RIGHT sub-array.
////
////my_index = n - right_start = 2 - 2 = 0. (It's the 1st element of its sub-array).
////
////rank_in_other = upper_bound_rank(d_src, left_start, left_end, value)
////
////upper_bound_rank(d_src, 0, 2, 5): "Search for 5 in the other array [7, 8]."
////
////upper_bound finds that 5 would be inserted at the beginning, so it returns index ////0.
////
////dest_idx = chunk_start + my_index + (rank_in_other - left_start)
////
////dest_idx = 0 + 0 + (0 - 0) = 0.
////
////d_dst[dest_idx] = value ➡️ d_dst[0] = 5.
////
////This is also correct. In the merged array [5, 6, 7, 8], the 5 belongs at index 0.
////
////This logic is performed in parallel by all 8 threads, and the entire d_dst array ////is filled correctly in one go.


//------------------------------------------------------------------------------
//method0

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> // For malloc/free
#include <math.h>   // For fminf

//method0
///*

#define BLOCK_SIZE 256

// @brief Finds the first element//not less than* value. (std::lower_bound)
//
// This function calculates the "rank" of 'value'. It returns the index 'l'
// such that all elements arr[s...l-1] are < value.
//
// @param arr   The array to search in.
// @param s     The starting index (inclusive).
//CH* @param e     The ending index (exclusive).
// @param value The value to search for.
// @return The first index 'l' where 'value' could be inserted.
///

__device__ int lower_bound_rank(const float* arr, int s, int e, float value) {
    int l = s;
    int r = e;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (arr[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

// @brief Finds the first element *greater than* value. (std::upper_bound)
// This is the key to a stable merge for handling equal values.
///
__device__ int upper_bound_rank(const float* arr, int s, int e, float value) {
    int l = s;
    int r = e;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (arr[m] <= value) { // The only change is <=
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

// @brief An iterative, bottom-up parallel merge kernel.
//
// This single kernel is launched for each "pass" of the merge sort.
// Each thread is responsible for placing one element from `d_src` into
// its final sorted position in `d_dst`.
//
// @param d_src The source array (contains sorted sub-arrays of size 'width').
// @param d_dst The destination array (will contain sorted sub-arrays of size '2*width').
// @param N     The total number of elements in the array.
// @param width The size of the sorted sub-arrays in `d_src` (e.g., 1, 2, 4, 8...).
///

__global__ void iterative_merge_kernel(const float* d_src, float* d_dst, int N, int width) {
    // Each thread 'n' is responsible for one element in the source array
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n >= N) return;

    // Get the value this thread is responsible for
    float value = d_src[n];

    // --- Find which merge "chunk" this thread belongs to ---
    // A full merge chunk is two sub-arrays of size 'width'
    int chunk_size = 2 * width;
    int chunk_start = (n / chunk_size) * chunk_size;

    // --- Define the two sub-arrays (Left and Right) for this chunk ---
    int left_start = chunk_start;
    // Handle edge cases where N is not a power of 2
    int left_end = min(left_start + width, N);
    int right_start = left_end;
    int right_end = min(right_start + width, N);

    // --- Find the thread's final destination index ---
    int dest_idx;
    if (n < right_start) {
        // This thread 'n' is in the LEFT sub-array [left_start ... left_end)
        // 1. Find index within its own sub-array:
        int my_index = n - left_start;
        // 2. Find rank in the OTHER sub-array (how many are < me):
        int rank_in_other = lower_bound_rank(d_src, right_start, right_end, value);
        // 3. Final dest = start of chunk + my_index + rank_in_other
        dest_idx = chunk_start + my_index + (rank_in_other - right_start);
    } else {
        // This thread 'n' is in the RIGHT sub-array [right_start ... right_end)
        // 1. Find index within its own sub-array:
        int my_index = n - right_start;
        // 2. Find rank in the OTHER sub-array (how many are <= me for stability):
        int rank_in_other = upper_bound_rank(d_src, left_start, left_end, value);
        // 3. Final dest = start of chunk + my_index + rank_in_other
        dest_idx = chunk_start + my_index + (rank_in_other - left_start);
    }

    // Write the value to its final sorted position in the destination buffer
    d_dst[dest_idx] = value;
}


// @brief High-performance iterative, bottom-up merge sort.
// This is the main "solve" function.
//
extern "C" void solve(float* data, int N) {
    if (N <= 1) return; // Already sorted

    float* d_helper;
    cudaMalloc((void**)&d_helper, N * sizeof(float));

    // --- Ping-Pong Buffer Setup ---
    // We will "ping-pong" between 'data' and 'd_helper'
    float* d_src = data;     // Start with data as the source
    float* d_dst = d_helper; // and helper as the destination

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // --- Iterative Passes ---
    // This loop runs log2(N) times (e.g., N=1M, loop runs 20 times)
    // not N-1 times. This is the fix.
    for (int width = 1; width < N; width *= 2) {
        
        // Launch one kernel for the entire pass
        // This kernel merges all pairs of [width] -> [2*width]
        iterative_merge_kernel<<<blocks, threads>>>(d_src, d_dst, N, width);
        cudaDeviceSynchronize(); // Wait for the *entire pass* to finish

        // --- Ping-Pong Swap ---
        // The destination (d_dst) is now the sorted source for the next pass.
        float* temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }

    // --- Final Copy (if needed) ---
    // After the loop, the final sorted array is in d_src.
    // If the loop ran an ODD number of times, the final data is in d_helper.
    // We must copy it back to the original 'data' pointer.
    if (d_src != data) {
        // The final sorted data is in d_helper, copy it back to data
        cudaMemcpy(data, d_src, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_helper);
    cudaDeviceSynchronize();
}
//*/

//-----------------------------------------------------------------------------
//method1
/*

#define BLOCK_SIZE 256

// @brief Finds the first element//not less than* value. (std::lower_bound)
//
// This function calculates the "rank" of 'value'. It returns the index 'l'
// such that all elements arr[s...l-1] are < value.
//
// @param arr   The array to search in.
// @param s     The starting index (inclusive).
//CH* @param e     The ending index (exclusive).
// @param value The value to search for.
// @return The first index 'l' where 'value' could be inserted.
///

__device__ int lower_bound_rank(const float* arr, int s, int e, float value) {
    int l = s;
    int r = e;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (arr[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

// @brief Finds the first element *greater than* value. (std::upper_bound)
// This is the key to a stable merge for handling equal values.
///
__device__ int upper_bound_rank(const float* arr, int s, int e, float value) {
    int l = s;
    int r = e;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (arr[m] <= value) { // The only change is <=
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

// THIS IS A HYPOTHETICAL, SLOWER VERSION
__global__ void coarsened_merge_kernel(const float* d_src, float* d_dst, int N, int width) {
    
    // 1. Get a "coarse" ID
    // We launch 1/8th the number of threads
    int coarse_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 2. Each thread loops 8 times
    for (int i = 0; i < 8; ++i) {
        
        // 3. Calculate the "real" n for this loop iteration
        int n = (coarse_idx * 8) + i;
        if (n >= N) break; // Stop if we go past the end

        // 4. --- Run the ENTIRE original logic INSIDE the loop ---
        float value = d_src[n];
        int chunk_size = 2 * width;
        int chunk_start = (n / chunk_size) * chunk_size;
        int left_start = chunk_start;
        int left_end = min(left_start + width, N);
        int right_start = left_end;
        int right_end = min(right_start + width, N);

        int dest_idx;
        if (n < right_start) {
            int my_index = n - left_start;
            int rank_in_other = lower_bound_rank(d_src, right_start, right_end, value);
            dest_idx = chunk_start + my_index + (rank_in_other - right_start);
        } else {
            int my_index = n - right_start;
            int rank_in_other = upper_bound_rank(d_src, left_start, left_end, value);
            dest_idx = chunk_start + my_index + (rank_in_other - left_start);
        }
        d_dst[dest_idx] = value;
        // --- End of original logic ---
    }
}

// @brief High-performance iterative, bottom-up merge sort.
// This is the main "solve" function.
//
extern "C" void solve(float* data, int N) {
    if (N <= 1) return; // Already sorted

    float* d_helper;
    cudaMalloc((void**)&d_helper, N * sizeof(float));

    // --- Ping-Pong Buffer Setup ---
    // We will "ping-pong" between 'data' and 'd_helper'
    float* d_src = data;     // Start with data as the source
    float* d_dst = d_helper; // and helper as the destination

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // --- Iterative Passes ---
    // This loop runs log2(N) times (e.g., N=1M, loop runs 20 times)
    // not N-1 times. This is the fix.
    for (int width = 1; width < N; width *= 2) {
        
        // Launch one kernel for the entire pass
        // This kernel merges all pairs of [width] -> [2*width]
        coarsened_merge_kernel<<<blocks, threads>>>(d_src, d_dst, N, width);
        cudaDeviceSynchronize(); // Wait for the *entire pass* to finish

        // --- Ping-Pong Swap ---
        // The destination (d_dst) is now the sorted source for the next pass.
        float* temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }

    // --- Final Copy (if needed) ---
    // After the loop, the final sorted array is in d_src.
    // If the loop ran an ODD number of times, the final data is in d_helper.
    // We must copy it back to the original 'data' pointer.
    if (d_src != data) {
        // The final sorted data is in d_helper, copy it back to data
        cudaMemcpy(data, d_src, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_helper);
    cudaDeviceSynchronize();
}
*/


//--------------------------------------------------------------------------------------------------
/*
question0:
不理解，请解释，用例子

这是一个高度优化的 CUDA 核函数，用于在 GPU 上执行 **并行归并排序 (Parallel Merge Sort)**。

该算法不是传统的指针移动归并，而是使用 **二分查找（Binary Search）** 来计算每个元素在其最终位置上的 **排序等级（Rank）**。

## 1\. 核心思想：比较计数归并 (Comparison Counting Merge)

在并行归并中，每个元素 X 的最终位置，取决于：
Final Index = 本组起始位置 + X 在自身数组中的索引 + X 在另一数组中的排序等级

这里的 **排序等级 (Rank)** 就是指：在另一个已排序的子数组中，有多少个元素小于或等于 X。

-----

## 2\. 辅助函数：并行计算 Rank (O(\log N))

这两个设备函数是并行的关键，它们在 O(\log W) 时间内（W 为子数组宽度）计算排序等级。

### `lower_bound_rank(arr, s, e, value)`

  * **功能：** 查找在范围 [s, e) 中，**第一个大于或等于 `value` 的元素的位置**。
  * **用途：** 用于处理**左子数组**中的元素。当左子数组中的元素 X_{left 查找右子数组中的等级时，使用 `< value` 来确保与 X_{left 相等的元素（如果有）都从右子数组中**排在它后面**，以维持稳定性。

### `upper_bound_rank(arr, s, e, value)`

  * **功能：** 查找在范围 [s, e) 中，**第一个严格大于 `value` 的元素的位置**。
      * **关键差异：** 查找条件是 `arr[m] <= value`。
  * **用途：** 用于处理**右子数组**中的元素。当右子数组中的元素 X_{right 查找左子数组中的等级时，使用 `upper_bound` 意味着它会数出所有 \le X_{right 的元素。这确保了如果 X_{left = X_{right，那么 X_{left 会先被计数，从而在输出中**排在前面**，维持排序稳定性。

-----

## 3\. 核函数：`iterative_merge_kernel`

该 Kernel 是并行归并的核心，每个线程 n 负责将源数组 `d_src` 中的元素 `d_src[n]` 放到 `d_dst` 中的正确位置。

### A. 线程任务确定

```c++
// 确定当前线程所在的归并块的边界
int chunk_size = 2 * width;
int chunk_start = (n / chunk_size) * chunk_size;

// 定义左右子数组的边界
int left_start = chunk_start;
int left_end = min(left_start + width, N);
int right_start = left_end;
int right_end = min(right_start + width, N);
```

### B. 左子数组元素的分派逻辑

如果线程 n 属于左子数组 (`n < right_start`)，它执行以下步骤：

1.  **`my_index`:** 元素 X 在自己的左数组中的相对位置。
2.  **`rank_in_other`:** 调用 `lower_bound_rank` 在右子数组中查找有多少元素严格小于 X。
3.  **`dest_idx`:**
    Final Index = Chunk Start + My Index + (Rank in Right Array)
    其中 `rank_in_other - right_start` 得到在右子数组中排在 X 之前的元素数量。

### C. 右子数组元素的分派逻辑

如果线程 n 属于右子数组 (`n >= right_start`)，它执行以下步骤：

1.  **`my_index`:** 元素 X 在自己的右数组中的相对位置。
2.  **`rank_in_other`:** 调用 **`upper_bound_rank`** 在左子数组中查找有多少元素小于或**等于** X（这是为了保证稳定性）。
3.  **`dest_idx`:** 逻辑与左子数组相同。

### 示例追踪 (N=4, width=2)

假设我们有一个数组 `d_src = [3, 5, 4, 8]`。`width=2`。

  * **左数组 (Left):** `[3, 5]`, `left_start=0`, `left_end=2`
  * **右数组 (Right):** `[4, 8]`, `right_start=2`, `right_end=4`
  * **目标输出:** `[3, 4, 5, 8]`

**追踪线程 n=1 (Value = 5):**

1.  n=1 < right_start=2 (属于 Left 数组).
2.  `my_index` = 1 - 0 = 1. (5 是 Left 数组中的第 1 个元素)
3.  `rank_in_other` = `lower_bound_rank(d_src, 2, 4, 5)`。
      * 在 `[4, 8]` 中查找第一个 >= 5 的元素。是 8。
      * `lower_bound_rank` 返回的全局索引是 3。
      * **右数组中排在 5 前面的元素数量:** 3 - right_start = 3 - 2 = 1 (只有 4)。
4.  `dest_idx` = `chunk_start(0) + my_index(1) + count(1)` = **2**.
5.  **结果:** `d_dst[2] = 5` (正确)。

**追踪线程 n=2 (Value = 4):**

1.  n=2 >= right_start=2 (属于 Right 数组).
2.  `my_index` = 2 - 2 = 0. (4 是 Right 数组中的第 0 个元素)
3.  `rank_in_other` = **`upper_bound_rank`**`(d_src, 0, 2, 4)`。
      * 在 `[3, 5]` 中查找第一个 > 4 的元素。是 5。
      * `upper_bound_rank` 返回的全局索引是 1。
      * **左数组中 \le 4 的元素数量:** 1 - left_start = 1 - 0 = 1 (只有 3)。
4.  `dest_idx` = `chunk_start(0) + my_index(0) + count(1)` = **1**.
5.  **结果:** `d_dst[1] = 4` (正确)。

-----

## 4\. 主机函数：`solve` (迭代归并)

`solve` 函数负责控制整个归并排序的流程：

1.  **Ping-Pong 缓冲 (d_src, d_dst):** 避免在每次归并后都进行一次昂贵的 `cudaMemcpy`。数据在 `data` 和 `d_helper` 之间来回交换，一轮的输出作为下一轮的输入。
2.  **迭代循环:** `for (int width = 1; width < N; width *= 2)`。这是归并排序的特征。它执行 \log_2 N 次循环，每次循环都将子数组大小翻倍。
3.  **并行性:** 在每次循环中，**只启动一次 Kernel**。这个 Kernel 会并行处理数组中所有的归并操作（例如，同时合并所有的 1 \rightarrow 2 归并，所有的 2 \rightarrow 4 归并等）。
4.  **`cudaDeviceSynchronize()`:** 在每次 Kernel 启动后必须同步，以确保当前归并阶段（例如 width=4）的所有线程都已完成，才能进入下一阶段（width=8）。

*/
//--------------------------------------------------------------------------------------------------
