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
///*

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




