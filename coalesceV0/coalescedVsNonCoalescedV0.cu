#include <cuda_runtime.h>
#include <iostream>

#define N 1024      // rows
#define M 1024      // cols

// 非 coalesced 访问：列优先（慢）
__global__ void non_coalesced_kernel(float* input, float* output) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= N) return;

    for (int col = 0; col < M; col++) {
        output[row * M + col] = input[col * N + row];  // 不连续访问
    }
}

// coalesced 访问：行优先（快）
__global__ void coalesced_kernel(float* input, float* output) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= N) return;

    for (int col = 0; col < M; col++) {
        output[row * M + col] = input[row * M + col];  // 连续访问
    }
}

void benchmark(const char* name, void (*kernel)(float*, float*), float* d_in, float* d_out) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_out, 0, sizeof(float) * N * M);

    cudaEventRecord(start);
    kernel<<<(N + 255) / 256, 256>>>(d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << name << " time: " << ms << " ms\n";
}

int main() {
    float* h_input = new float[N * M];
    float* h_output = new float[N * M];

    // Initialize input
    for (int i = 0; i < N * M; ++i) h_input[i] = i;

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N * M);
    cudaMalloc(&d_output, sizeof(float) * N * M);
    cudaMemcpy(d_input, h_input, sizeof(float) * N * M, cudaMemcpyHostToDevice);

    // Benchmark
    std::cout << "Benchmarking...\n";

    benchmark("Coalesced", coalesced_kernel, d_input, d_output);
    benchmark("Non-Coalesced", non_coalesced_kernel, d_input, d_output);

    cudaMemcpy(h_output, d_output, sizeof(float) * N * M, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    return 0;
}
