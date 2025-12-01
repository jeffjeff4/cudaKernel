
#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_SQ_THREADS 16
#define NUM_THREADS 256

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows or col >= cols) return;

    float val = input[row * cols + col];
    output[col * rows + row] = val;
}

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    float* scalar,
    int M,
    int d,
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M or n >= N) return;

    float result = 0.0f;
    for (int i = 0; i < d; i++) {
        result += A[(m * d) + i] * B[(i * N) + n];
    }

    result *= *scalar;

    C[m * N + n] = result;
}

__device__ inline int float_to_ordered_int(float x) {
    int i = __float_as_int(x);
    return (i >= 0) ? i : i ^ 0x7fffffff;   // flip sign bit band
}
__device__ inline float ordered_int_to_float(int i) {
    return __int_as_float((i >= 0) ? i : (i ^ 0x7fffffff));
}

__device__ float atomicMaxFloat(float* address, float val) {
    int* addr_i = (int*)address;
    int old = *addr_i, assumed;
    int val_i = float_to_ordered_int(val);
    do {
        assumed = old;
        if (val_i <= old) break;                 // current max already >= val
        old = atomicCAS(addr_i, assumed, val_i); // try to install
    } while (assumed != old);
    return ordered_int_to_float(old);
}

__global__ void maxvalue_kernel(const float* input, float* output, int M, int N) {
    __shared__ float shared[NUM_THREADS];
    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    if (i < N * M) {
        shared[threadIdx.x] = input[i];
    } else {
        // need to not return on this case to maintain integrity of shared
        shared[threadIdx.x] = -INFINITY;
    }

    __syncthreads();

    if (i != 0 && i % N != 0) return;

    float maxVal = -INFINITY;
    int end = ((i / N) + 1) * N - (blockDim.x * blockIdx.x);
    for (int n = threadIdx.x; n < end && n < NUM_THREADS; n++) {
        maxVal = maxVal > shared[n] ? maxVal : shared[n];
    }

    atomicMaxFloat(&output[i / N], maxVal);
}

__global__ void add_biases_kernel(float* matrix, int M, int N, float alpha) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // "j"
    int m = blockIdx.y * blockDim.y + threadIdx.y; // "i"
    if (n >= N || m >= M) return;

    float bias = alpha * (float)(m - n);
    // printf("bias=%f, m=%d, n=%d, M=%d, N=%d\n", bias, m, n, M, N);
    matrix[m * N + n] += bias;
}

__global__ void sum_exp_kernel(const float* input, float* output, float* maxValues, int M, int N) {
    __shared__ float shared[NUM_THREADS];
    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    if (i < N * M) {
        shared[threadIdx.x] = __expf(input[i] - maxValues[i / N]);
    } else {
        // need to not return on this case to maintain integrity of shared
        shared[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (i != 0 && i % N != 0) return;

    float currSum = 0.0f;
    int end = ((i / N) + 1) * N - (blockDim.x * blockIdx.x);
    for (int n = threadIdx.x; n < end && n < NUM_THREADS; n++) {
        currSum += shared[n];
    }

    atomicAdd(&output[i / N], currSum);
}

__global__ void softmax_kernel(const float* input, float* output, float* sumValues, float* maxValues, int M, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    if (i >= N * M) return;

    float numerator = __expf(input[i] - maxValues[i / N]);
    output[i] = numerator / sumValues[i / N];
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {

    dim3 threadsPerBlock_2d(NUM_SQ_THREADS, NUM_SQ_THREADS);
    dim3 blocksPerGrid_3(
        (d + threadsPerBlock_2d.x - 1) / threadsPerBlock_2d.x, // cols
        (N + threadsPerBlock_2d.y - 1) / threadsPerBlock_2d.y // rows
    );

    float* transposed_K;
    cudaMalloc(&transposed_K, N * d * sizeof(float));
    matrix_transpose_kernel<<<blocksPerGrid_3, threadsPerBlock_2d>>>(K, transposed_K, N, d);

    threadsPerBlock_2d = dim3(NUM_SQ_THREADS, NUM_SQ_THREADS);
    blocksPerGrid_3 = dim3(
        (N + threadsPerBlock_2d.x - 1) / threadsPerBlock_2d.x,
        (M + threadsPerBlock_2d.y - 1) / threadsPerBlock_2d.y
    );

    float* inv_sqrt_d;
    cudaMalloc(&inv_sqrt_d, sizeof(float));
    float inv_sqrt_d_calc = 1 / sqrtf((float)d);
    cudaMemcpy(inv_sqrt_d, &inv_sqrt_d_calc, sizeof(float), cudaMemcpyHostToDevice);

    float* QKTD_Result;
    cudaMalloc(&QKTD_Result, M * N * sizeof(float));

    matrix_multiplication_kernel<<<blocksPerGrid_3, threadsPerBlock_2d>>>(
        Q, // m1
        transposed_K, // m2
        QKTD_Result, // result
        inv_sqrt_d, // scalar
        M, // eg 2
        d, // eg 4
        N // eg 3
    );
    cudaDeviceSynchronize();

    add_biases_kernel<<<blocksPerGrid_3, threadsPerBlock_2d>>>(QKTD_Result, M, N, alpha);
    cudaDeviceSynchronize();

    int threadsPerBlock = NUM_THREADS;
    int blocksPerGrid = (N * M + threadsPerBlock - 1) / threadsPerBlock;

    float* maxValues;
    cudaMalloc(&maxValues, M * sizeof(float));
    cudaMemset(maxValues, 0, M * sizeof(float));
    maxvalue_kernel<<<blocksPerGrid, threadsPerBlock>>>(QKTD_Result, maxValues, M, N);

    float* sumValues;
    cudaMalloc(&sumValues, M * sizeof(float));
    cudaMemset(sumValues, 0, M * sizeof(float));
    sum_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(QKTD_Result, sumValues, maxValues, M, N);

    float* softmax_result;
    cudaMalloc(&softmax_result, N * M * sizeof(float));
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(QKTD_Result, softmax_result, sumValues, maxValues, M, N);

    // float* softmax_result_2 = (float*)malloc(M*N*sizeof(float));
    // cudaMemcpy(softmax_result_2, softmax_result, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    // printf("softmax_result=");
    // for (int i = 0; i < M*N; i++) {
    //     printf("%f,", softmax_result_2[i]);
    // }
    // printf("\n");
    // fflush(stdout);

    float* no_scalar;
    cudaMalloc(&no_scalar, sizeof(float));
    float z = 1.0f;
    cudaMemcpy(no_scalar, &z, sizeof(float), cudaMemcpyHostToDevice);

    threadsPerBlock_2d = dim3(NUM_SQ_THREADS, NUM_SQ_THREADS);
    blocksPerGrid_3 = dim3(
        (d + threadsPerBlock_2d.x - 1) / threadsPerBlock_2d.x,
        (M + threadsPerBlock_2d.y - 1) / threadsPerBlock_2d.y
    );

    matrix_multiplication_kernel<<<blocksPerGrid_3, threadsPerBlock_2d>>>(
        softmax_result, // m1
        V, // m2
        output, // result
        no_scalar, // 1.0
        M, // eg 2
        N, // eg 3
        d // eg 4
    );

    cudaDeviceSynchronize();
}





//--------------------------------------------------------------------------------------------------
/*
question0:
不理解，请解释，用例子


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question1:

不理解，请解释，用例子


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question2:

不理解，请解释，用例子



//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question3:

不理解，请解释，用例子




//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question4:

不理解，请解释，用例子



//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question5:

不理解，请解释，用例子


//--------------------------------------------------------------------------------------------------




//--------------------------------------------------------------------------------------------------
/*
question6:

不理解，请解释，有例子




//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
/*
question7:

不理解，请解释，用例子




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
