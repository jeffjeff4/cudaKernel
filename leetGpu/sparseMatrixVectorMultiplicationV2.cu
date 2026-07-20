/*
I’m tying the explanation to the exact ELL layout in your current CUDA file and using a tiny vector example so it maps directly to the code.

In sparseMatrixVectorMultiplicationV2.cu, the phrase “transpose (column major) for DRAM efficiency” means this:

- the ELL storage is arranged so that values for the same “slot” across many rows are placed next to each other in memory,
- which is better for GPU memory access than storing the data in a row-by-row layout.

## What the code is doing

Your kernel does this for each row:

```cpp
for (int k = 0; k < max_nnz; k++) {
    int col = ell_col[k * num_rows + row];
    float val = ell_val[k * num_rows + row];
    ...
}
```

The important part is:

```cpp
k * num_rows + row
```

That index means:

- `k` = which nonzero slot we are looking at,
- `row` = which matrix row this thread is handling.

So the data is stored as:

- all entries for slot `k = 0` across all rows,
- then all entries for slot `k = 1` across all rows,
- and so on.

That is the “column-major” idea.

## Tiny example

Take this sparse matrix:

```text
A = [ 1  0  2
      0  3  4
      5  6  0 ]
```

and vector:

```text
x = [10, 20, 30]
```

The result is:

```text
y = A * x
```

So:

- row 0: $1\cdot 10 + 2\cdot 30 = 70$
- row 1: $3\cdot 20 + 4\cdot 30 = 180$
- row 2: $5\cdot 10 + 6\cdot 20 = 170$

## ELL layout for this example

Each row has at most 2 nonzeros, so `max_nnz = 2`.

If we store the nonzero columns and values in a table like this:

```text
slot 0: row0->col0, row1->col1, row2->col0
slot 1: row0->col2, row1->col2, row2->col1
```

then the arrays look like this:

```text
ell_col = [0, 1, 0, 2, 2, 1]
ell_val = [1, 3, 5, 2, 4, 6]
```

Now the kernel for row 0 uses:

```text
k=0 -> col=0, val=1
k=1 -> col=2, val=2
```

and for row 1 it uses:

```text
k=0 -> col=1, val=3
k=1 -> col=2, val=4
```

Those accesses are grouped by the same `k` across neighboring rows, so the GPU can fetch them more efficiently.

## Why this helps DRAM

Without this layout, the data might be scattered in memory in a way that causes many strided accesses. With the column-major/transpose-style layout, neighboring threads often read neighboring memory locations, so the memory system can serve them more efficiently.

## In one sentence

In your code, `k * num_rows + row` means “store the data by nonzero slot first, then by row,” which makes memory access more compact and faster on the GPU.

If you want, I can also show the same example as a small 2D table and map it directly to the `ell_col` and `ell_val` arrays line by line.
*/

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ══════════════════════════════════════════════════════════════════
// ELL SpMV Kernel
// ell_col, ell_val: column-major [num_rows × max_nnz_per_row]
// One thread per row
// ══════════════════════════════════════════════════════════════════
__global__ void spmv_ell_kernel(
    const int*   ell_col,    // [num_rows * max_nnz]  col-major
    const float* ell_val,    // [num_rows * max_nnz]  col-major
    const float* x,          // dense input vector [num_cols]
    float*       y,          // output vector      [num_rows]
    int          num_rows,
    int          max_nnz)    // max nonzeros per row
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float sum = 0.0f;

    for (int k = 0; k < max_nnz; k++) {
        // Column-major index: element (row, k) is at [k * num_rows + row]
        int   col = ell_col[k * num_rows + row];
        float val = ell_val[k * num_rows + row];

        // col == -1 means padding — skip
        if (col >= 0) {
            sum += val * x[col];
        }
    }

    y[row] = sum;
}

// ══════════════════════════════════════════════════════════════════
// Host helper: CSR → ELL conversion
// ══════════════════════════════════════════════════════════════════
void csr_to_ell(
    // CSR inputs
    const int*   csr_row_ptr,  // [num_rows + 1]
    const int*   csr_col_idx,  // [nnz]
    const float* csr_val,      // [nnz]
    int          num_rows,
    // ELL outputs (caller allocates)
    int*         ell_col,      // [num_rows * max_nnz], col-major
    float*       ell_val,      // [num_rows * max_nnz], col-major
    int&         max_nnz)      // output: max nonzeros per row
{
    // Step 1: find max nonzeros per row
    max_nnz = 0;
    for (int i = 0; i < num_rows; i++) {
        int row_len = csr_row_ptr[i + 1] - csr_row_ptr[i];
        max_nnz = std::max(max_nnz, row_len);
    }

    // Step 2: fill ELL arrays (col-major)
    // Initialize with padding
    for (int i = 0; i < num_rows * max_nnz; i++) {
        ell_col[i] = -1;
        ell_val[i] = 0.0f;
    }

    for (int row = 0; row < num_rows; row++) {
        int start = csr_row_ptr[row];
        int end   = csr_row_ptr[row + 1];
        int k = 0;
        for (int j = start; j < end; j++, k++) {
            // column-major index: (row, k) → k * num_rows + row
            ell_col[k * num_rows + row] = csr_col_idx[j];
            ell_val[k * num_rows + row] = csr_val[j];
        }
        // remaining slots already padded with -1 / 0
    }
}

// ══════════════════════════════════════════════════════════════════
// Main entry point — called by LeetGPU
// Input:  CSR sparse matrix + dense vector x (all already on GPU)
// Output: dense vector y (on GPU)
//
// LeetGPU signature (adjust to match their exact interface):
// ══════════════════════════════════════════════════════════════════
extern "C" void solve(
    const int*   d_csr_row_ptr,   // device, [num_rows + 1]
    const int*   d_csr_col_idx,   // device, [nnz]
    const float* d_csr_val,       // device, [nnz]
    const float* d_x,             // device, [num_cols]
    float*       d_y,             // device, [num_rows]
    int          num_rows,
    int          num_cols,
    int          nnz)
{
    // ── Step 1: Copy CSR from device to host for conversion ──
    int*   h_csr_row_ptr = new int[num_rows + 1];
    int*   h_csr_col_idx = new int[nnz];
    float* h_csr_val     = new float[nnz];

    cudaMemcpy(h_csr_row_ptr, d_csr_row_ptr,
               (num_rows + 1) * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csr_col_idx, d_csr_col_idx,
               nnz * sizeof(int),               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csr_val,     d_csr_val,
               nnz * sizeof(float),             cudaMemcpyDeviceToHost);

    // ── Step 2: CSR → ELL on host ──
    int max_nnz = 0;

    // First pass: find max_nnz
    for (int i = 0; i < num_rows; i++) {
        int len = h_csr_row_ptr[i + 1] - h_csr_row_ptr[i];
        max_nnz = std::max(max_nnz, len);
    }

    int   ell_size = num_rows * max_nnz;
    int*   h_ell_col = new int[ell_size];
    float* h_ell_val = new float[ell_size];

    csr_to_ell(h_csr_row_ptr, h_csr_col_idx, h_csr_val,
               num_rows, h_ell_col, h_ell_val, max_nnz);

    // ── Step 3: Upload ELL to device ──
    int*   d_ell_col;
    float* d_ell_val;
    cudaMalloc(&d_ell_col, ell_size * sizeof(int));
    cudaMalloc(&d_ell_val, ell_size * sizeof(float));

    cudaMemcpy(d_ell_col, h_ell_col,
               ell_size * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_val, h_ell_val,
               ell_size * sizeof(float), cudaMemcpyHostToDevice);

    // ── Step 4: Launch ELL kernel ──
    const int BLOCK = 256;
    int grid = (num_rows + BLOCK - 1) / BLOCK;

    spmv_ell_kernel<<<grid, BLOCK>>>(
        d_ell_col, d_ell_val, d_x, d_y, num_rows, max_nnz);

    cudaDeviceSynchronize();

    // ── Step 5: Cleanup ──
    cudaFree(d_ell_col);
    cudaFree(d_ell_val);

    delete[] h_csr_row_ptr;
    delete[] h_csr_col_idx;
    delete[] h_csr_val;
    delete[] h_ell_col;
    delete[] h_ell_val;
}