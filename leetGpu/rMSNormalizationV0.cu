
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

//method0
///*
__global__ void PowerSum(const float* input, int N, float* blockSum) {
    int tid = threadIdx.x;
    int idx = tid + (blockDim.x * blockIdx.x);
    __shared__ float acc[256];
    float x_pow = 0.0f;
    if (idx == 0) acc[0] = 0.0f;
    __syncthreads();
    for (int i = idx; i < N; i+= blockDim.x*gridDim.x) {
        float x = input[i];
        x_pow += x * x;
    }
    acc[threadIdx.x] = x_pow;
    __syncthreads();
    // Using reduction to accumulate sum into acc[0]
    for (int stride = blockDim.x/2; stride >= warpSize; stride /= 2) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
        }
    // warp shuffle 

    if (tid < warpSize) {
        float val = acc[tid];
        unsigned mask = __activemask();
        val += __shfl_down_sync(mask, val, 16);
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);
        if (tid == 0) blockSum[blockIdx.x] = val;
    }

    // ensure single block has all accumulated sums:
   
}
__global__ void RMSNorm(const float* input, float gamma, float beta, float* output, int N, float eps, float rmsn) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    for (int k = idx; k < N; k+= blockDim.x*gridDim.x) {
        float x_hat = input[k] / rmsn;
        output[k] = gamma * x_hat + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
                        int threadsPerBlock = 256;
                        int blocksPerGrid = (threadsPerBlock + N - 1) / threadsPerBlock;
                        float* blockSum_d = nullptr;
                        cudaMalloc(&blockSum_d, blocksPerGrid*sizeof(float));
                        PowerSum<<<blocksPerGrid, threadsPerBlock>>>(input, N, blockSum_d);
                        cudaDeviceSynchronize();

                        std::vector<float> blockSum_h(blocksPerGrid);
                        cudaMemcpy(blockSum_h.data(), blockSum_d, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
                        float rmsn = 0.0f;
                        for (int j = 0; j < blocksPerGrid; j++) {
                            rmsn += blockSum_h[j];
                        }
                        rmsn = sqrtf((rmsn/N) + eps);
                        RMSNorm<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, N, eps, rmsn);

}
//*/


//method1
//wrong
/*
#define WARP_SIZE       32
#define THEADPERBLOCK   (WARP_SIZE*WARP_SIZE)
#define STRIDE_LENGTH   8

__device__ __forceinline__ float warp_sum(float val) {
    // 得到处于活跃状态的线程掩码
    unsigned m = __activemask();
    val += __shfl_down_sync(m , val , 16);
    val += __shfl_down_sync(m , val , 8);
    val += __shfl_down_sync(m , val , 4);
    val += __shfl_down_sync(m , val , 2);
    val += __shfl_down_sync(m , val , 1);
    return val;
}

__global__ void PowerSum(const float* input, int N, float* output) {
    int tid = threadIdx.x;
    int idx = tid + (blockDim.x * blockIdx.x);

    float x_pow = 0.0f;
    for (int i = idx; i < N; i+= blockDim.x*gridDim.x) {
        float x = input[i];
        x_pow += x * x;
    }
    __syncthreads();

    // 1.在每个 warp内规约求和，并将其部分求和结果存储到共享内存中
    __shared__ float shared_partial_sum[WARP_SIZE];
    int warp = tid >> 5;    // 当前线程所在的 warp在整个 warp数组中的下标
    int lane = tid & 31;    // 当前线程在当前 warp内的下标

    float sum_val = 0;
    int wsum = warp_sum(sum_val);
    if (lane == 0) {
        shared_partial_sum[warp] = wsum;
    }
    __syncthreads();

    // 2.将每个块内所有 warp已得到的部分求和结果再进行规约求和
    if (warp == 0) {
        float partial_sum_val = shared_partial_sum[lane];
        shared_partial_sum[0] = warp_sum(partial_sum_val);
    }
     __syncthreads();

    // 3.利用原子加操作,对所有块内的 shared_partial_sum[0]求和
    if (tid == 0) {
        atomicAdd(output, shared_partial_sum[0]);
    }

}

__global__ void RMSNorm(const float* input, float gamma, float beta, float* output, int N, float eps, float rmsn) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    for (int k = idx; k < N; k+= blockDim.x*gridDim.x) {
        float x_hat = input[k] / rmsn;
        output[k] = gamma * x_hat + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
                        int threadsPerBlock = 256;
                        int blocksPerGrid = (threadsPerBlock + N - 1) / threadsPerBlock;
                        float* blockSum_d = nullptr;
                        cudaMalloc(&blockSum_d, blocksPerGrid*sizeof(float));

                        cudaMemset(output, 0.0f, sizeof(float));
                        PowerSum<<<blocksPerGrid, threadsPerBlock>>>(input, N, output);
                        cudaDeviceSynchronize();

                        float rmsn = *output;
                        cudaMemset(output, 0.0f, sizeof(float));
                        RMSNorm<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, N, eps, rmsn);

}
//*/



//--------------------------------------------------------------------------------------------------
/*
question0:
不理解，请解释，用例子

method0

这段代码实现了一个计算 **RMS 归一化（Root Mean Square Normalization）** 的流程。它利用 CUDA 并行计算均方根 (RMS) 的平方和部分，然后由 CPU 计算最终的 RMS 值，最后再由 GPU 完成归一化变换。

-----

## ⚙️ I. 核心原理：RMS 归一化

RMS 归一化的基本公式是：

RMS(x) = sqrt1/N * sum_(=1)^N x_i^2 + epsilon

Output_i = gamma * x_i / RMS(x) + beta

这段代码的流程分解为：

1.  **GPU (`PowerSum`):** 并行计算 sum x_i^2 的总和，按 Block 分片存储。
2.  **CPU (`solve`):** 收集所有 Block 的 sum x_i^2 总和，计算最终的 RMS(x) 值。
3.  **GPU (`RMSNorm`):** 使用计算出的 RMS(x) 完成最终的归一化和缩放 (gamma, \beta) 变换。

-----

## 🚀 II. Kernel 1: 平方和计算 (`PowerSum`)

这个 Kernel 负责计算所有输入元素 x 的平方和 sum x^2，并将其归约到每个 Block 的结果中。

### 1\. 局部平方和计算 (Grid-Stride Loop)

c
// ...
for (int i = idx; i < N; i+= blockDim.x*gridDim.x) 
    float x = input[i];
    x_pow += x * x; // 局部累积平方和

acc[threadIdx.x] = x_pow;
__syncthreads();


  * **线程分工:** 使用 **Grid-Stride Loop** 模式。每个线程 idx 负责处理 N 个数据中，以 (blockDim.x * gridDim.x) 为步长的一系列元素。
  * **目的:** 保证整个数据集 N 的每个元素都被一个线程处理到。
  * **结果:** 每个线程 tx 的局部平方和 x_pow 被写入 Shared Memory 数组 acc[tx] 中。

### 2\. 块内归约 (两级归约)

**A. Shared Memory 归约 (粗粒度):**

c
for (int stride = blockDim.x/2; stride >= warpSize; stride /= 2) 
    if (tid < stride) acc[tid] += acc[tid + stride];
    __syncthreads();



  * **目的:** 将 256 个局部和归约到 acc[0] 到 acc[31] (第一个 Warp 的区域) 中。
  * **步长:** 归约到 warpSize=32 处停止。

**B. Warp Shuffle 归约 (细粒度):**

c
if (tid < warpSize) 
    // ... Shuffle down reduction ...
    val += __shfl_down_sync(mask, val, 1);
    if (tid == 0) blockSum[blockIdx.x] = val; // 线程 0 写入全局结果



  * **目的:** 使用最快的 **Warp Shuffle 指令** 将剩下的 32 个值归约成一个总和 val。
  * **结果:** 最终的总和存储在线程 0 的 val 变量中，然后被写入全局数组 blockSum[blockIdx.x]。

-----

## 💻 III. Host 端计算 RMS (solve 函数)

主机端负责收集 GPU 的部分结果，并计算最终的 RMS 值。

c
// ... Copy back to host
std::vector<float> blockSum_h(blocksPerGrid);
cudaMemcpy(blockSum_h.data(), blockSum_d, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

float rmsn = 0.0f;
for (int j = 0; j < blocksPerGrid; j++) 
    rmsn += blockSum_h[j]; // 累加所有 Block 的平方和


// 最终 RMS 公式计算
rmsn = sqrtf((rmsn/N) + eps);


  * **目的:** 串行累加所有 Block 的平方和 (rmsn = sum (sum x^2))。
  * **RMS 计算:** 应用 RMS 公式：rmsn = sqrt(sum x^2 / N) + epsilon。

> **示例:** 假设 N=100，Block 0 的平方和是 800，Block 1 的平方和是 200。
>
> 1.  rmsn (累加) = 800 + 200 = 1000.
> 2.  rmsn = sqrt(1000 / 100) + epsilon = sqrt(10 + epsilon)。

-----

## 🏁 IV. Kernel 2: RMS 归一化变换 (`RMSNorm`)

这个 Kernel 使用 CPU 计算出的 RMS 值，完成最终的标准化和缩放。

c
for (int k = idx; k < N; k+= blockDim.x*gridDim.x) 
    float x_hat = input[k] / rmsn; // 归一化 (x / RMS)
    output[k] = gamma * x_hat + beta; // 缩放和偏移



  * **分工:** 再次使用 **Grid-Stride Loop** 模式，每个线程 k 负责处理 input 数组中的多个元素。
  * **计算:** 对每个元素 x_k 执行 RMS 归一化公式 y_k = gamma ... (x_k / rmsn) + \beta。
  * **写回:** 结果写入 output 数组。

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
