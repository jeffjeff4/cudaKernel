#include <cuda_runtime.h>

#define cdiv(a, b) (a+b+1) / b

#define BLOCK 256
#define SIZE 10000

//---------------------------------------------------------------------------------------------------
//method0
//correct
/*
__global__ void scanBlock(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;
    if (n >= N) return;
    __shared__ float sh[BLOCK];
    sh[t] = input[n];
    __syncthreads();

    int step = 1;
    while (step < BLOCK) {
        float tmp = sh[t];
        if (t >= step) {
            tmp += sh[t-step];
        }
        step *= 2;
        __syncthreads();
        sh[t] = tmp;
        __syncthreads();
    }
    output[n] = sh[t];
    if (t == BLOCK-1) {
        block_sums[blockIdx.x] = sh[t];
    }
}

__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;
    output[n] += block_sums[b];
}

void scan(const float* input, float* output, int N) {
    float* block_sums;
    int block_sums_len = cdiv(N, BLOCK);

    cudaMalloc((void**)&block_sums, block_sums_len * sizeof(float));
    scanBlock<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, block_sums);
    if (N <= BLOCK) return;
    scan(block_sums, block_sums, block_sums_len);
    mergeWithBlockSums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, block_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
} 
*/

//---------------------------------------------------------------------------------------------------
//method1
//correct
/*
__global__ void scanBlock(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;
    if (n >= N) return;
    __shared__ float sh[BLOCK];
    sh[t] = input[n];
    __syncthreads();

    int step = 1;
    while (step < BLOCK) {
    //while (step <= t) {
        float tmp = sh[t];
        if (t >= step) {
            tmp += sh[t-step];
        }
        step *= 2;
        __syncthreads();
        sh[t] = tmp;
        __syncthreads();
    }
    output[n] = sh[t];
    if (t == BLOCK-1) {
        block_sums[blockIdx.x] = sh[t];
    }
}

__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;
    output[n] += block_sums[b];
}

void scan(const float* input, float* output, int N) {
    float* block_sums;
    int block_sums_len = cdiv(N, BLOCK);

    cudaMalloc((void**)&block_sums, block_sums_len * sizeof(float));
    scanBlock<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, block_sums);
    if (N <= BLOCK) return;
    scan(block_sums, block_sums, block_sums_len);
    mergeWithBlockSums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, block_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
} 
*/




//---------------------------------------------------------------------------------------------------
//method2
//correct
/*

#include <cuda_runtime.h>

// A more standard and correct ceiling division macro
#define cdiv(a, b) (a + b - 1) / b 

#define BLOCK 256

//
// * This kernel implements a two-phase (up-sweep and down-sweep)
// * parallel scan.

__global__ void scanBlock(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;

    // Use a temporary variable to hold this thread's original value
    float my_val = 0.0f;
    if (n < N) {
        my_val = input[n];
    }
    
    __shared__ float sh[BLOCK];
    sh[t] = my_val;
    __syncthreads();

    // --- PHASE 1: Up-Sweep (Reduce Phase) ---
    // This builds a "summation tree" in shared memory.
    // We use step *= 2.
    int step = 1;
    while (step < BLOCK) {
        // Threads at the end of each segment add the sum 
        // from the end of the previous segment.
        if (t % (2 * step) == (2 * step) - 1) {
            sh[t] += sh[t - step];
        }
        step *= 2;
        __syncthreads(); // Sync at each level of the tree
    }

    // --- Save Total Sum and Clear Last Element ---
    float block_sum_val = 0.0f;
    if (t == BLOCK - 1) {
        block_sum_val = sh[t]; // Save the block's total sum
        sh[t] = 0.0f;          // Clear the last element for the down-sweep
    }
    __syncthreads();

    // --- PHASE 2: Down-Sweep (Scan Phase) ---
    // This is the step /= 2 part you asked for!
    // It walks back down the tree, propagating partial sums.
    step = BLOCK / 2;
    while (step >= 1) {
        // Check if thread 't' is at the end of the *first*
        // half of a segment (e.g., t=1, 3, 5... or t=1, 5... etc.)
        if (t % (2 * step) == step - 1) {
            // Swap the "left" and "right" sums and
            // propagate the sum to the "right" element.
            float tmp = sh[t];           // Save left (e.g., sh[step-1])
            sh[t] = sh[t + step];        // Set left = right (e.g., sh[2*step-1])
            sh[t + step] = tmp + sh[t + step]; // Set right = left_original + right
        }
        step /= 2;
        __syncthreads(); // Sync at each level
    }

    // --- Final Step: Inclusive Scan ---
    // At this point, sh[t] holds the *exclusive* scan
    // (the sum of all elements *before* t).
    // To get the inclusive scan, we add the original value.
    if (n < N) {
        output[n] = sh[t] + my_val;
    }

    // The last thread writes the total sum it saved earlier.
    if (t == BLOCK - 1) {
        block_sums[blockIdx.x] = block_sum_val;
    }
}

// NO CHANGES to this kernel. It's correct.
__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;
    output[n] += block_sums[b];
}

// NO CHANGES to this host function.
// (Added cudaFree for correctness)
void scan(const float* input, float* output, int N) {
    float* block_sums;
    int block_sums_len = cdiv(N, BLOCK);

    cudaMalloc((void**)&block_sums, block_sums_len * sizeof(float));
    scanBlock<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, block_sums);
    
    if (N <= BLOCK) {
        cudaFree(block_sums);
        return;
    }
    
    scan(block_sums, block_sums, block_sums_len);
    mergeWithBlockSums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, block_sums);
    
    cudaFree(block_sums); // Free the temporary memory
}

// NO CHANGES to this function.
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
}

*/

//---------------------------------------------------------------------------------------------------
//method3
//correct
/*

#include <cuda_runtime.h>

// A more standard and correct ceiling division macro
#define cdiv(a, b) ((a + b - 1) / b)

#define BLOCK 256

//
// * @brief This kernel implements a correct two-phase (up-sweep and down-sweep)
// * parallel Blelloch scan within a block.
// *
// * It correctly calculates two things:
// * 1. The INCLUSIVE scan of its local chunk, which it writes to 'output'.
// * 2. The TOTAL SUM of its local chunk, which it writes to 'block_sums'.
//

__global__ void scanBlock(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;

    // Use a standard name for shared memory
    __shared__ float sh[BLOCK];
    float my_val; // To store the original value

    // 1. Load data safely
    // Each thread loads its original value and also stores it in shared mem
    if (n < N) {
        my_val = input[n];
        sh[t] = my_val;
    } else {
        // Use 0.0 (identity for +) for out-of-bounds threads
        my_val = 0.0f; 
        sh[t] = 0.0f;
    }
    __syncthreads();

    // --- PHASE 1: Up-Sweep (Reduction / Build Sum Tree) ---
    // This builds the tree of partial sums in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // A thread at 't' adds the value from 't - stride'
        // We only need to check if 't' is a "right-child" index
        if ((t + 1) % (2 * stride) == 0) {
            sh[t] += sh[t - stride];
        }
        __syncthreads(); // Wait for all sums at this level
    }

    // --- Save Total Block Sum ---
    // The total sum for the entire block is now in the last element
    if (t == (blockDim.x - 1)) {
        block_sums[blockIdx.x] = sh[t];
        // Set the last element to 0 (identity) for the down-sweep
        sh[t] = 0.0f; 
    }
    __syncthreads();

    // --- PHASE 2: Down-Sweep (Distribute Exclusive Sums) ---
    // Now we "push" the exclusive prefix sums down the tree
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        // This thread is a "parent" (a right-child one)
        if ((t + 1) % (2 * stride) == 0) {
            // Save its left child's partial sum
            float left_child_val = sh[t - stride];
            // Pass its own exclusive sum (from "above") to its left child
            sh[t - stride] = sh[t];
            // Its own value becomes the inclusive sum for its right child
            sh[t] = sh[t] + left_child_val;
        }
        __syncthreads(); // Wait for this level to finish
    }

    // --- Final Write ---
    // sh[t] now holds the EXCLUSIVE scan of the block.
    // We add the thread's original value to make it INCLUSIVE.
    if (n < N) {
        output[n] = sh[t] + my_val;
    }
}

// NO CHANGES to this kernel. It's correct.
__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    // Start at the *second* block (index 1)
    int n = threadIdx.x + blockDim.x * (b + 1);
    if (n >= N) return;
    
    // Add the total sum of the *previous* block
    // (which is at block_sums[b], since the scan of sums is inclusive)
    output[n] += block_sums[b];
}

// Host function to perform the full, multi-block scan
void scan(const float* input, float* output, int N) {
    float* block_sums;
    int block_sums_len = cdiv(N, BLOCK);

    cudaMalloc((void**)&block_sums, block_sums_len * sizeof(float));
    
    // 1. Run scanBlock on all blocks
    // This calculates the local inclusive scan for each block
    // and stores each block's total sum in 'block_sums'
    scanBlock<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, block_sums);
    cudaDeviceSynchronize(); // Wait for it to finish

    // If we only had one block, we're done!
    if (N <= BLOCK) {
        cudaFree(block_sums);
        return;
    }
    
    // 2. Recursively scan the block sums
    // This turns [sumB0, sumB1, sumB2] into
    // [sumB0, sumB0+sumB1, sumB0+sumB1+sumB2]
    scan(block_sums, block_sums, block_sums_len);
    
    // 3. Add the scanned sums to the local block scans
    // We skip the first block (which needs no offset)
    mergeWithBlockSums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, block_sums);
    cudaDeviceSynchronize(); // Wait for merge
    
    cudaFree(block_sums); // Free the temporary memory
}

// NO CHANGES to this function.
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
}
//*/

//---------------------------------------------------------------------------------------------------
//below are wrong methods
//---------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------
//method10
//wrong
/*
__global__ void scanBlock(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;
    if (n >= N) return;
    __shared__ float sh[BLOCK];
    sh[t] = input[n];
    __syncthreads();

    int step = t;
    while (step >= 1) {
    //while (step <= t) {
        float tmp = sh[t];
        if (t >= step) {
            tmp += sh[t-step];
        }
        step /= 2;
        __syncthreads();
        sh[t] = tmp;
        __syncthreads();
    }
    output[n] = sh[t];
    if (t == BLOCK-1) {
        block_sums[blockIdx.x] = sh[t];
    }
}

__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;
    output[n] += block_sums[b];
}

void scan(const float* input, float* output, int N) {
    float* block_sums;
    int block_sums_len = cdiv(N, BLOCK);

    cudaMalloc((void**)&block_sums, block_sums_len * sizeof(float));
    scanBlock<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, block_sums);
    if (N <= BLOCK) return;
    scan(block_sums, block_sums, block_sums_len);
    mergeWithBlockSums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, block_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
} 
*/

//---------------------------------------------------------------------------------------------------
//method11
//wrong
//need to debug
/*

#include <cuda_runtime.h>

// A more standard and correct ceiling division macro
#define cdiv(a, b) (a + b - 1) / b 

#define BLOCK 256

__global__ void scanBlock(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;

    if (n >= N) return;
    __shared__ float sh[BLOCK];
    float my_val; // To store the original value

    // 1. Load data safely
    // Each thread loads its original value and also stores it in shared mem
    if (n < N) {
        my_val = input[n];
        sh[t] = my_val;
    } else {
        // Use 0.0 (identity for +) for out-of-bounds threads
        my_val = 0.0f; 
        sh[t] = 0.0f;
    }
    __syncthreads();

    for (unsigned int step=1; step<blockDim.x; step *= 2) {
        if (t % (2 * step) == 0) {
            sh[t] += sh[t+step];
        }
        __syncthreads();
    }

    if (t == 0) {
        output[n] = sh[t];
        block_sums[blockIdx.x] = sh[t];
    }
}

// NO CHANGES to this kernel. It's correct.
__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;
    output[n] += block_sums[b];
}

// NO CHANGES to this host function.
// (Added cudaFree for correctness)
void scan(const float* input, float* output, int N) {
    float* block_sums;
    int block_sums_len = cdiv(N, BLOCK);

    cudaMalloc((void**)&block_sums, block_sums_len * sizeof(float));
    scanBlock<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, block_sums);
    
    if (N <= BLOCK) {
        cudaFree(block_sums);
        return;
    }
    
    scan(block_sums, block_sums, block_sums_len);
    mergeWithBlockSums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, block_sums);
    
    cudaFree(block_sums); // Free the temporary memory
}

// NO CHANGES to this function.
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
}
*/


//---------------------------------------------------------------------------------------------------
//method12
//wrong
//need to debug
/*

#include <cuda_runtime.h>

// A more standard and correct ceiling division macro
#define cdiv(a, b) (a + b - 1) / b 

#define BLOCK 256

__global__ void scanBlock(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;

    if (n >= N) return;
    __shared__ float sh[BLOCK];
    float my_val; // To store the original value

    // 1. Load data safely
    // Each thread loads its original value and also stores it in shared mem
    if (n < N) {
        my_val = input[n];
        sh[t] = my_val;
    } else {
        // Use 0.0 (identity for +) for out-of-bounds threads
        my_val = 0.0f; 
        sh[t] = 0.0f;
    }
    __syncthreads();

    for (unsigned int step=blockDim.x/2; step>0; step>>=1) {
        if (t < step) {
            sh[t] += sh[t+step];
        }
        __syncthreads();
    }

    if (t == 0) {
        output[n] = sh[t];
        block_sums[blockIdx.x] = sh[t];
    }
}

// NO CHANGES to this kernel. It's correct.
__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;
    output[n] += block_sums[b];
}

// NO CHANGES to this host function.
// (Added cudaFree for correctness)
void scan(const float* input, float* output, int N) {
    float* block_sums;
    int block_sums_len = cdiv(N, BLOCK);

    cudaMalloc((void**)&block_sums, block_sums_len * sizeof(float));
    scanBlock<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, block_sums);
    
    if (N <= BLOCK) {
        cudaFree(block_sums);
        return;
    }
    
    scan(block_sums, block_sums, block_sums_len);
    mergeWithBlockSums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, block_sums);
    
    cudaFree(block_sums); // Free the temporary memory
}

// NO CHANGES to this function.
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
}
//*/


//---------------------------------------------------------------------------------------------------
//method13
//wrong
///*

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void scanBlock(const float* input, float* output, int N, float* block_sums) {
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    int t = threadIdx.x;
    if (n >= N) return;
    volatile __shared__ float sh[BLOCK];
    sh[t] = input[n];
    __syncthreads();

    for (unsigned int step=blockDim.x/2; step>32; step>>=1) {
        if (t < step) {
            sh[t] += sh[t+step];
        }
        __syncthreads();
    }

    if (t < 32) {
        warpReduce(sh, t);
    }

    if (t == 0) {
        output[n] = sh[t];
        block_sums[blockIdx.x] = sh[t];
    }
}

__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    int b = blockIdx.x;
    int n = threadIdx.x + blockDim.x * (b+1);
    if (n >= N) return;
    output[n] += block_sums[b];
}

void scan(const float* input, float* output, int N) {
    float* block_sums;
    int block_sums_len = cdiv(N, BLOCK);

    cudaMalloc((void**)&block_sums, block_sums_len * sizeof(float));
    scanBlock<<<cdiv(N, BLOCK), BLOCK>>>(input, output, N, block_sums);
    if (N <= BLOCK) return;
    scan(block_sums, block_sums, block_sums_len);
    mergeWithBlockSums<<<cdiv(N-BLOCK, BLOCK), BLOCK>>>(output, N, block_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
} 
//*/



//--------------------------------------------------------------------------------------------------
/*
question0:
不理解，请解释，用例子

您提供的这段 CUDA 代码实现了一个标准的、高效的并行算法：**并行前缀和（Parallel Prefix Sum）**，也称为 **Scan** 操作。它采用了经典的 **Blelloch Scan 算法**，并结合了 **递归分块** 来处理任意大小的数组。

前缀和（Scan）操作的目标是：给定一个输入数组 $A$，生成一个输出数组 $P$，其中 $P[i]$ 是 $A[0]$ 到 $A[i]$ 的所有元素的总和。

$$
A = [a_0, a_1, a_2, a_3, \dots] \\
P = [a_0, a_0+a_1, a_0+a_1+a_2, a_0+a_1+a_2+a_3, \dots]
$$

-----

## 1\. 核心算法：Blelloch Scan (两阶段)

`scanBlock` 核函数在一个线程块内实现了 $O(\log N)$ 复杂度的并行前缀和。

### 阶段 I: Up-Sweep (归约/构建和树)

```c++
    // This builds the tree of partial sums in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((t + 1) % (2 * stride) == 0) {
            sh[t] += sh[t - stride]; // 右侧的元素加上左侧的局部和
        }
        __syncthreads();
    }
```

  * **目的:** 从底部向上，不断地将局部和累加到共享内存 (`sh`) 的较高索引位置（想象成树的父节点）。
  * **操作:** 在循环结束时，`sh[blockDim.x - 1]`（最后一个元素）将存储整个线程块的**总和**。

### 阶段 II: Down-Sweep (分散/计算独占前缀和)

在 Phase 1 结束后，`sh[blockDim.x - 1]` 存储了总和。我们希望将这个总和（以及中间节点的局部和）向下传递，为每个位置计算它的 **独占前缀和**（Exclusive Scan，即不包含自身 $A[i]$ 的和）。

```c++
    // 关键点：将总和 sh[blockDim.x - 1] 设为 0，作为 Down-Sweep 的起始值
    if (t == (blockDim.x - 1)) {
        block_sums[blockIdx.x] = sh[t]; // 保存总和到全局内存
        sh[t] = 0.0f; // 设置为 0，用于 Exclusive Scan
    }
    __syncthreads();

    // Now we "push" the exclusive prefix sums down the tree
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if ((t + 1) % (2 * stride) == 0) {
            float left_child_val = sh[t - stride]; // 左子树的局部和
            sh[t - stride] = sh[t];                // 父节点独占和传给左子树
            sh[t] = sh[t] + left_child_val;      // 计算右子树的独占和
        }
        __syncthreads();
    }
```

  * **目的:** 将 Phase 1 的总和信息向下广播。
  * **结果:** 循环结束后，`sh[t]` 存储的是 **输入 $A[0]$ 到 $A[t-1]$ 的总和**（Exclusive Scan）。

### 最终结果 (Inclusive Scan)

```c++
    if (n < N) {
        output[n] = sh[t] + my_val; // 独占和 + 自身值 = 包含自身的前缀和 (Inclusive Scan)
    }
```

  * **计算:** 将独占前缀和 (`sh[t]`) 加上线程自己的原始值 (`my_val`)，得到最终的 **包含自身的前缀和 (Inclusive Scan)**。

-----

## 3\. 主机函数：`scan` (多块递归)

由于一个 Block 只能计算局部的前缀和，`scan` 函数通过递归解决了 **块间依赖** 的问题。

### 步骤 A: 局部前缀和与块总和 (Level 1)

`scanBlock<<<...>>>`：

  * **计算:** 每个 Block 独立地计算自己的局部前缀和，结果部分地写入 `output`。
  * **输出:** 每个 Block 的总和 $\text{SumB}_i$ 被存储在 `block_sums[i]` 中。

$$
\text{block\_sums} = [\text{SumB}_0, \text{SumB}_1, \text{SumB}_2, \dots]
$$

### 步骤 B: 递归扫描块总和 (Level 2)

`scan(block_sums, block_sums, block_sums_len)`：

  * **操作:** 对 `block_sums` 数组递归调用 `scan`，直到它变成一个完整的前缀和数组 $P_{\text{B}}$。

$$
P_{\text{B}} = [\text{SumB}_0, \text{SumB}_0+\text{SumB}_1, \text{SumB}_0+\text{SumB}_1+\text{SumB}_2, \dots]
$$

  * **意义:** 数组 $P_{\text{B}}[i]$ 存储了 **所有在 Block $i$ 之前** 的元素的总和。

### 步骤 C: 合并块总和 (`mergeWithBlockSums`)

```c++
__global__ void mergeWithBlockSums(float* output, int N, float* block_sums) {
    // ...
    int n = threadIdx.x + blockDim.x * (b + 1); // 从第二个 Block 开始
    if (n >= N) return;
    output[n] += block_sums[b]; // block_sums[b] 存储的是 SumB0 + ... + SumB(b-1)
}
```

  * **操作:** 所有的线程（除了第一个 Block）都会加上**前一个 Block 的总和**。
  * **逻辑:** 对于 Block $b$ 中的任何元素 $A[n]$：
    $$\text{Final } P[n] = \text{局部 } P[n] + P_{\text{B}}[b-1]$$
    其中 $P_{\text{B}}[b-1]$ 就是 **Block $b$ 之前所有元素的总和**。

-----

## 4\. 示例追踪 (N=8, Block Size=4)

| 索引 $i$ | $A$ | 局部前缀和 (Block 0) | 局部前缀和 (Block 1) |
| :---: | :---: | :---: | :---: |
| 0 | 1 | 1 | |
| 1 | 2 | 3 | |
| 2 | 3 | 6 | |
| 3 | 4 | 10 | |
| 4 | 5 | | 5 |
| 5 | 6 | | 11 |
| 6 | 7 | | 18 |
| 7 | 8 | | 26 |

  * **Level 1 输出:**

      * $\text{block\_sums}[0] = 10$ ($\text{SumB}_0$)
      * $\text{block\_sums}[1] = 26$ ($\text{SumB}_1$)

  * **Level 2 递归扫描 $P_{\text{B}}$:**

      * $\text{block\_sums}$ 变成 $[10, 36]$ ($\text{SumB}_0, \text{SumB}_0+\text{SumB}_1$)

  * **Level 3 合并 (`mergeWithBlockSums`):**

      * **Block 0:** 跳过（无需偏移）。
      * **Block 1** (从索引 4 开始): 所有元素加上 $\text{block\_sums}[0] = 10$。
          * $\text{Output}[4] = 5 + 10 = 15$ (正确)
          * $\text{Output}[7] = 26 + 10 = 36$ (正确)

**结论:** 这种分块递归的结构确保了所有线程并行工作，并且通过后续的合并步骤，解决了数据块之间的依赖关系，最终得到了正确的全局前缀和。


//--------------------------------------------------------------------------------------------------





//--------------------------------------------------------------------------------------------------
/*
question1:
不理解，请解释，用例子

不理解，请解释，用例子

    // --- PHASE 1: Up-Sweep (Reduction / Build Sum Tree) ---
    // This builds the tree of partial sums in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // A thread at 't' adds the value from 't - stride'
        // We only need to check if 't' is a "right-child" index
        if ((t + 1) % (2 * stride) == 0) {
            sh[t] += sh[t - stride];
        }
        __syncthreads(); // Wait for all sums at this level
    }

这段代码是 **Blelloch 并行前缀和（Scan）算法** 的第一个阶段：**Up-Sweep（上行归约）**。

这个阶段的目标是：**并行地计算所有局部子数组的总和，并将最终的整个 Block 总和存储在共享内存的最后一个元素中。**

-----

## 1\. 核心思想：构建求和树 (Summation Tree)

想象一个二叉树，输入数组 $A$ 是叶子节点。Up-Sweep 从叶子节点向上，将子节点的和存储到父节点，直到树根（即 Block 的最后一个元素）。

  * **时间复杂度:** $O(\log N)$，因为每一步 `stride` 都会使需要计算的节点数量减半。

### 代码片段

```c++
for (int stride = 1; stride < blockDim.x; stride *= 2) {
    // 只有右侧的“父节点”执行加法
    if ((t + 1) % (2 * stride) == 0) {
        sh[t] += sh[t - stride];
    }
    __syncthreads(); // 必须等待，确保本层所有加法完成
}
```

## 2\. 关键变量与条件解释

| 变量/条件 | 含义 | 目的 |
| :--- | :--- | :--- |
| **`stride`** | 步长：当前归并的两个子段的长度。每次循环翻倍（1, 2, 4, 8...）。 | 控制树的层级。 |
| **`2 * stride`** | 当前正在合并的整个子数组的长度。 | 用于模运算，确定父节点的位置。 |
| **`(t + 1)`** | 线程的 1-based 索引（因为数组是 0-indexed）。 | 简化模运算的逻辑判断。 |
| **`(t + 1) % (2 * stride) == 0`** | **核心条件：** 检查线程 $t$ 是否是当前长度为 `2 * stride` 的子数组的 **最右端元素**（即父节点）。 | 确保每个子段的局部和只计算一次，并存储到正确的位置。 |
| **`sh[t] += sh[t - stride]`** | **核心操作：** $t$ 线程（父节点）将左子段的局部和（存储在 $t - stride$ 处）加到自己的值上。 | 将两个子段的局部和合并到父节点。 |

-----

## 3\. 示例追踪 (Block Size = 8)

假设输入数组 $A = [1, 2, 3, 4, 5, 6, 7, 8]$。共享内存 `sh` 初始值与 $A$ 相同。

| Pass | `stride` | `2*stride` | 模运算条件 | 执行加法的线程 $t$ | `sh` 最终状态 (只有父节点更新) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | 2 | $(t+1) \% 2 = 0$ ($t$ 是奇数) | 1, 3, 5, 7 | $[1, \mathbf{3}, 3, \mathbf{7}, 5, \mathbf{11}, 7, \mathbf{15}]$ |
| **2** | 2 | 4 | $(t+1) \% 4 = 0$ ($t = 3, 7$) | 3, 7 | $[1, 3, 3, \mathbf{10}, 5, 11, 7, \mathbf{26}]$ |
| **3** | 4 | 8 | $(t+1) \% 8 = 0$ ($t = 7$) | 7 | $[1, 3, 3, 10, 5, 11, 7, \mathbf{36}]$ |

### 结果分析

1.  **Pass 1 (`stride=1`):** 线程 1 算出了 $1+2=3$，线程 3 算出了 $3+4=7$，等等。
2.  **Pass 2 (`stride=2`):** 线程 3 算出了 $7+3=10$ (即 $1+2+3+4$ 的总和)。
3.  **Pass 3 (`stride=4`):** 线程 7 算出了 $26+10=36$ (即 $1+2+\dots+8$ 的总和)。

在 Up-Sweep 结束时，共享内存 `sh[7]`（最后一个位置）存储了整个 Block 的总和 36。所有中间的局部和也存储在 `sh` 中，这为下一步的 Down-Sweep 提供了必要的信息。

//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question2:
不理解，请解释，用例子

这个我还是不理解，请解释，用例子

    // --- PHASE 2: Down-Sweep (Distribute Exclusive Sums) ---
    // Now we "push" the exclusive prefix sums down the tree
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        // This thread is a "parent" (a right-child one)
        if ((t + 1) % (2 * stride) == 0) {
            // Save its left child's partial sum
            float left_child_val = sh[t - stride];
            // Pass its own exclusive sum (from "above") to its left child
            sh[t - stride] = sh[t];
            // Its own value becomes the inclusive sum for its right child
            sh[t] = sh[t] + left_child_val;
        }
        __syncthreads(); // Wait for this level to finish
    }



这段代码是 **Blelloch Scan 算法** 中最精妙的部分，它实现了 **独占前缀和（Exclusive Prefix Sum）** 的递归分散。

我们使用 **树状继承** 的概念来解释这三行代码是如何工作的。

---

## 1. 核心概念：独占和的继承与更新

在 Down-Sweep 中，每个父节点 $t$ 都存储着一个来自“上方”的 **独占前缀和** $E_{\text{Parent}}$。

这个父节点 $t$ 必须完成两件事：

1.  **传递给左子树:** 左子树的独占和 $E_{\text{Left}}$ 就等于 $E_{\text{Parent}}$。
2.  **计算右子树:** 右子树的独占和 $E_{\text{Right}}$ 必须是 $E_{\text{Parent}}$ 加上**整个左子树的总和**。

### 核心三行代码的角色

| 独占和 $E_{\text{Parent}}$ | 代码行 | 存储什么? | 作用 |
| :---: | :---: | :---: | :--- |
| $E_{\text{Parent}}$ | `float left_child_val = sh[t - stride];` | 存储左子树的**局部总和** $S_{\text{Left}}$。 | 必须先保存 $S_{\text{Left}}$，因为 $sh[t - stride]$ 即将被覆盖。 |
| $E_{\text{Parent}}$ | `sh[t - stride] = sh[t];` | 存储左子树的**新独占和** $E_{\text{Left}}$。 | **传递给左子树。** 简单地将 $E_{\text{Parent}}$ 传给左子树的根节点。 |
| $E_{\text{Parent}} + S_{\text{Left}}$ | `sh[t] = sh[t] + left_child_val;` | 存储右子树的**新独占和** $E_{\text{Right}}$。 | **为右子树准备。** $E_{\text{Parent}}$ 加上左子树的总和 $S_{\text{Left}}$，成为新的 $E_{\text{Right}}$，向下传递。 |

---

## 2. 示例追踪 (Pass 2: stride = 2)

我们追踪 **Rank $t=7$** 的执行过程。

**前提状态（Pass 1 结束时）**

$$sh_{\text{start}} = [1, 3, 3, \mathbf{0}, 5, 11, 7, \mathbf{10}]$$

| 索引 $t$ | 原始值（Phase 2 开始时） | 含义 |
| :---: | :---: | :--- |
| **$t=7$** | $sh[7] = 10$ | $E_{\text{Parent}}$：这是 $A[0..3]$ 的总和，是右半部分 $A[4..7]$ 的独占和。 |
| **$t=5$** | $sh[5] = 11$ | $S_{\text{Left}}$：这是左子树 $A[4..5]$ 的局部总和 ($5+6=11$)。 |

### 线程 $t=7$ 执行步骤

`stride = 2`；$t=7$ 满足条件。

| 代码行 | $t=7$ 时的值 | $t=5$ 时的值 | 结果/目的 |
| :---: | :---: | :---: | :--- |
| 1. `float left_child_val = sh[t - stride];` | `left_child_val` $= sh[5]$ $= \mathbf{11}$ | $\mathbf{11}$ | **保存 $A[4..5]$ 的总和 (11)。** |
| 2. `sh[t - stride] = sh[t];` | $sh[7]$ $= 10$ | $sh[5]$ $\leftarrow \mathbf{10}$ | **传递给左子树：** $A[4..5]$ 的独占和是 10。 |
| 3. `sh[t] = sh[t] + left_child_val;` | $sh[7]$ $\leftarrow 10 + 11 = \mathbf{21}$ | $sh[5]$ $= 10$ | **更新右子树：** $A[6..7]$ 的独占和是 $10 + 11 = 21$。 |

### Pass 2 结束后的状态

$$sh_{\text{after P2}} = [1, 3, 3, 0, 5, \mathbf{10}, 7, \mathbf{21}]$$

* **$sh[5] = 10$:** 这将作为 $A[4..5]$ (即 $[5, 6]$) 的独占和，在下一轮 (`stride=1`) 中，用于计算 5 和 6 的独占和。
* **$sh[7] = 21$:** 这将作为 $A[6..7]$ (即 $[7, 8]$) 的独占和，在下一轮 (`stride=1`) 中，用于计算 7 和 8 的独占和。

通过这三步，一个独占和 $E_{\text{Parent}}$ 就被成功地分解并传递给了树的下一层。
//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question3:
不理解，请解释，用例子


//--------------------------------------------------------------------------------------------------
