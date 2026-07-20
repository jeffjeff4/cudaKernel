#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdio>

__global__ void csrSpMVKernel(const int* __restrict__ row_ptr,
                              const int* __restrict__ col_idx,
                              const float* __restrict__ values,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];

    for (int i = start; i < end; ++i) {
        sum += values[i] * x[col_idx[i]];
    }

    y[row] = sum;
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

    // Convert the dense row-major matrix A into CSR format on the host.
    // This keeps the kernel itself CSR-based while still accepting the
    // challenge's dense input pointer.
    std::vector<float> denseA(static_cast<size_t>(M) * static_cast<size_t>(N));
    std::vector<float> denseX(static_cast<size_t>(N));
    copyToHostBuffer(A, denseA, denseA.size());
    copyToHostBuffer(x, denseX, denseX.size());

    std::vector<int> row_ptr(M + 1, 0);
    std::vector<int> col_idx(static_cast<size_t>(std::max(nnz, 0)));
    std::vector<float> values(static_cast<size_t>(std::max(nnz, 0)));

    int nz_count = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float v = denseA[static_cast<size_t>(i) * static_cast<size_t>(N) + j];
            if (v != 0.0f) {
                if (nz_count >= nnz) {
                    // The input nnz hint can be smaller than the actual count of
                    // non-zero values in the matrix. In that case we still
                    // process what we have and stop safely.
                    break;
                }
                col_idx[nz_count] = j;
                values[nz_count] = v;
                ++nz_count;
            }
        }
        row_ptr[i + 1] = nz_count;
    }

    if (nz_count == 0) {
        // If the matrix contains no non-zero entries, just clear the output.
        std::vector<float> zero_y(static_cast<size_t>(M), 0.0f);
        copyBuffer(zero_y.data(), const_cast<float*>(y), zero_y.size());
        return;
    }

    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nz_count * sizeof(int));
    cudaMalloc(&d_values, nz_count * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(), nz_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nz_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, denseX.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    csrSpMVKernel<<<blocks, threads>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, M);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_row_ptr);
        cudaFree(d_col_idx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        return;
    }

    cudaDeviceSynchronize();
    copyBuffer(d_y, const_cast<float*>(y), M);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}
