#include <cuda_runtime.h>
#define MAX_KERNEL_FLATTEND_SIZE 441
__constant__ float const_kernel[MAX_KERNEL_FLATTEND_SIZE];

#define TILE_DIM_W 32
#define TILE_DIM_H 8

__global__ void convolutionKernel(const float* input, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    extern __shared__ float sh_tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int k_radius_x = kernel_cols >> 1;
    const int k_radius_y = kernel_rows >> 1;

    const int out_col = blockIdx.x * TILE_DIM_W + tx;
    const int out_row = blockIdx.y * TILE_DIM_H + ty;

    const int sh_tile_width = TILE_DIM_W + kernel_cols - 1;
    const int sh_tile_height = TILE_DIM_H + kernel_rows - 1;

    const int in_tile_start_row = blockIdx.y * TILE_DIM_H - k_radius_y;
    const int in_tile_start_col = blockIdx.x * TILE_DIM_W - k_radius_x;

    for (int y=ty; y<sh_tile_height; y+= TILE_DIM_H) {
        for (int x=tx; x<sh_tile_width; x+=TILE_DIM_W) {
            const int cur_row = in_tile_start_row + y;
            const int cur_col = in_tile_start_col + x;

            if (cur_row >= 0 and cur_row<input_rows and cur_col>=0 and cur_col<input_cols) {
                sh_tile[y*sh_tile_width+x] = input[cur_row*input_cols+cur_col];
            } else {
                sh_tile[y*sh_tile_width+x] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (out_row<input_rows and out_col<input_cols) {
        float sum = 0.0f;
        const float* sh_tile_row_ptr = &sh_tile[ty*sh_tile_width+tx];

        #pragma unroll
        for (int i=0; i<kernel_rows; ++i) {
            #pragma unroll
            for (int j=0; j<kernel_rows; ++j) {
                sum += sh_tile_row_ptr[j] * const_kernel[i*kernel_cols+j];
            }
            sh_tile_row_ptr += sh_tile_width;
        }

        output[out_row*input_cols + out_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    cudaMemcpyToSymbol(const_kernel, kernel, kernel_rows * kernel_cols * sizeof(float));
    dim3 block_dim(TILE_DIM_W, TILE_DIM_H);
    dim3 grid_dim((input_cols + TILE_DIM_W - 1) / TILE_DIM_W,
                  (input_rows + TILE_DIM_H - 1) / TILE_DIM_H);
    size_t sh_mem_size = (block_dim.x + kernel_cols - 1) * (block_dim.y + kernel_rows - 1) * sizeof(float);
    convolutionKernel<<<grid_dim, block_dim, sh_mem_size>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
}



/*
question0:
不理解，请举例说明

    const int out_col = blockIdx.x * TILE_DIM_W + tx;
    const int out_row = blockIdx.y * TILE_DIM_H + ty;

好的，这两行代码是 CUDA 中最基础、最重要的**全局索引计算**，用于确定当前线程应该处理输出数据中的**哪一个特定元素**。

它们将线程块（Block）的网格（Grid）结构和线程在块内的位置映射到二维数组（输出图像）的坐标。

---

## 🔢 示例设定

我们使用代码中的宏定义：

* TILE_DIM_W (Tile 宽度) = 32
* TILE_DIM_H (Tile 高度) = 8
* **Block 尺寸** = 32 * 8 = 256 个线程。

假设我们要处理一张 1000 * 1000 的输出图像。

### 核心思想：Block 负责区域，Thread 负责点

1.  **Block 负责区域：** 每个 Block 负责计算输出图像上的一个 32 * 8 的矩形区域（一个 Tile）。
2.  **Thread 负责点：** Block 内的每个线程负责计算 Tile 中的一个 像素点。

---

## 1. out_col：输出图像的列索引

out_col = blockIdx.x * TILE_DIM_W + tx

### 示例解释

| 变量/线程 | 值 | 含义 |
| :--- | :--- | :--- |
| TILE_DIM_W | 32 | 每个 Block 在 x 方向上处理 32 个列。 |
| blockIdx.x | 假设 1 (第 2 列 Block) | 确定了 Block 的起始列。 |
| tx | 假设 5 (线程 x 坐标) | 确定了线程在 Block 内部的偏移。 |

#### 计算过程:

out_col = 1 * 32 + 5 = 37

> **结论：** 这个线程 (tx=5) 负责计算输出图像的 **第 37 列** 像素。
> * **Block 0** 负责 0 到 31 列。
> * **Block 1** 负责 32 到 63 列。
> * 线程 tx=5 在 Block 1 中，所以它的列索引是 32 + 5 = 37。

---

## 2. out_row：输出图像的行索引

out_row = blockIdx.y * TILE_DIM_H + ty

### 示例解释

| 变量/线程 | 值 | 含义 |
| :--- | :--- | :--- |
| TILE_DIM_H | 8 | 每个 Block 在 y 方向上处理 8 行。 |
| blockIdx.y | 假设 3 (第 4 行 Block) | 确定了 Block 的起始行。 |
| ty | 假设 7 (线程 y 坐标) | 确定了线程在 Block 内部的偏移。 |

#### 计算过程:

out_row = 3 * 8 + 7 = 24 + 7 = 31

> **结论：** 这个线程 (ty=7) 负责计算输出图像的 **第 31 行** 像素。
> * **Block 0** 负责 0 到 7 行。
> * **Block 1** 负责 8 到 15 行。
> * Block 3 负责 24 到 31 行。
> * 线程 ty=7 在 Block 3 中，所以它的行索引是 24 + 7 = 31。

## 最终总结

结合这两个计算，这个线程负责计算输出图像上的坐标为 (out_row, out_col) = (31, 37) 的像素值。这是 **一维线程 ID** 映射到 **二维数据** 的标准方法。


*/


/*
question1:

不理解，请举例解释

边界像素: 如果我们要计算 Tile 左上角第一个像素 P_{0,0，卷积核会向左上方延伸 kernel_radius 距离。这些延伸出去的数据必须被加载到 Shared Memory 中。

这是一个关于卷积操作如何依赖**边界数据 (Halo)** 的关键概念。我将用一个具体的、小型的例子来解释为什么计算 Tile 的第一个像素 P_{0,0 需要向左上方延伸的数据。

---

## 🔢 示例设置

假设我们有一个非常小的场景：

1.  **输入图像 (Input):** I
2.  **卷积核 (Kernel):** K，尺寸为 3 * 3
3.  **Tile 尺寸:** 我们只关注输出 Tile 的 P_{0,0 像素。

### 核心参数

* **卷积核尺寸:** 3 * 3
* **半径 (k_radius):** (3 - 1) / 2 = 1

---

## 💡 示例：计算 Tile 的第一个像素 P_{0,0

假设一个线程块负责计算输出 Tile，而线程 (0, 0) 负责计算该 Tile 的第一个像素 P_{0,0。

### 1. P_{0,0 的定义

在卷积中，计算输出图像的像素 P_{r, c 的值，需要将卷积核的中心对齐到输入图像的 (r, c) 坐标，然后将核覆盖的所有输入数据与核权重相乘再求和。

### 2. P_{0,0 所需的输入区域

由于卷积核是 3 * 3 (半径为 1)，要计算 P_{0,0，你需要读取以 I_{0,0 为中心的 3 * 3 输入区域。

| 区域 | 输入图像 I 的坐标 |
| :--- | :--- |
| **左上角** | I_{0-1, 0-1 = I_{-1, -1 |
| **中心** | I_{0, 0 |
| **右下角** | I_{0+1, 0+1 = I_{1, 1 |

**所需的 3 * 3 输入区域的坐标范围是：** **行 [-1, 1]**，**列 [-1, 1]**。

### 3. 需要加载的延伸数据（Halo）

* **向左延伸:** 你需要 I_{r, -1 这一列数据。
* **向上延伸:** 你需要 I_{-1, c 这一行数据。

**为什么必须加载 I_{-1, -1？**

* **原因：** 如果 I_{-1, -1 没有被加载到 Shared Memory 中，当线程计算 P_{0,0 时，它会尝试从 Shared Memory 中读取这个位置。如果找不到，它会不得不回退到慢速的 Global Memory 读取，或者更糟的是，计算结果是错误的。
* **零填充 (Zero Padding):** 实际上，由于原始图像索引不能为负，程序会在加载时判断 I_{-1, -1 是否在图像边界内。如果不在，它会在 Shared Memory 的对应位置写入 0.0f (零填充)，这是代码中 `else { sh_tile[...] = 0.0f; ` 所做的事情。

### 总结

计算 P_{0,0 需要的输入数据区域比输出 Tile 本身**向左、向右、向上、向下**各延伸了 **1 像素（即半径）** 的距离。

这段代码正是通过以下两步确保这些延伸数据被正确加载：

1.  **起始点偏移:** `in_tile_start_row = blockIdx.y * TILE_DIM_H - k_radius_y` 将输入 Tile 的读取起点**向上方和左方移动了半径距离**。
2.  **Shared Memory 尺寸:** `sh_tile_width = TILE_DIM_W + kernel_cols - 1` 保证 Shared Memory 有足够的**空间**来容纳这个延伸后的区域。

*/


/*
question2:
不理解，请举例解释

    const int in_tile_start_row = blockIdx.y * TILE_DIM_H - k_radius_y;
    const int in_tile_start_col = blockIdx.x * TILE_DIM_W - k_radius_x;

好的，这两行代码是 **2D 卷积分块优化**中用于确定**输入数据起始点**的关键步骤。它们确保线程块从全局内存 (Global Memory) 读取数据时，能正确地包含计算 Tile 边界像素所需的**额外数据（Halo/Border）**。

我们将沿用前面的例子进行解释。

---

## 🔢 示例设定

* **输出 Tile 尺寸:** TILE_DIM_W = 32， TILE_DIM_H = 8
* **卷积核 (Kernel) 尺寸:** kernel_rows = 3， kernel_cols = 5
* **卷积核半径 (Radius):**
    * k_radius_y = kernel_rows \gg 1 = 3 \gg 1 = 1
    * k_radius_x = kernel_cols \gg 1 = 5 \gg 1 = 2

### 目标：确定读取起点

假设一个线程块 (blockIdx.x, blockIdx.y) 负责计算输出图像的某个 32 * 8 区域。

* **输出 Tile 的起始点（没有 Halo 时）** 位于： (blockIdx.y * 8, blockIdx.x * 32)

但是，为了计算这个 Tile 左上角像素，我们需要读取位于这个起始点**上方** k_radius_y 行和**左方** k_radius_x 列的数据。

---

## 1. in_tile_start_row：输入 Tile 的起始行

in_tile_start_row = blockIdx.y * TILE_DIM_H - k_radius_y

### 示例解释

假设当前线程块是 Block\ (0, 1)，即 blockIdx.y = 1。

1.  **Tile 的正常起始行:** 1 * 8 = 8。
    * 如果没有卷积核，这个 Tile 应该从输出图像的第 8 行开始计算。
2.  **向上方偏移 (减去半径):** k_radius_y = 1。
    * **in_tile_start_row = 8 - 1 = 7**。

> **结论：** 尽管这个 Block 计算的输出 Tile 是从第 8 行开始的，但为了包含计算第 8 行所需的**上方**边界数据，它必须从输入图像的**第 7 行**开始读取数据。

---

## 2. in_tile_start_col：输入 Tile 的起始列

in_tile_start_col = blockIdx.x * TILE_DIM_W - k_radius_x

### 示例解释

假设当前线程块是 Block\ (1, 1)，即 blockIdx.x = 1。

1.  **Tile 的正常起始列:** 1 * 32 = 32。
    * 如果没有卷积核，这个 Tile 应该从输出图像的第 32 列开始计算。
2.  **向左方偏移 (减去半径):** k_radius_x = 2。
    * **in_tile_start_col = 32 - 2 = 30**。

> **结论：** 尽管这个 Block 计算的输出 Tile 是从第 32 列开始的，但为了包含计算第 32 列所需的**左侧**边界数据，它必须从输入图像的**第 30 列**开始读取数据。

---

## 最终总结

这两行代码定义了线程块去全局内存中读取数据的**左上角坐标** (in_tile_start_row, in_tile_start_col)。

这个起点是**向左上方偏移了卷积核半径距离**的，从而保证在接下来的 **加载循环** 中，Shared Memory 能够完整地接收到计算该 Tile 所需的全部输入数据，包括其四周的边界区域。

**例如，Block (1, 1) 的线程将从输入图像的 (30, 7) 坐标开始读取数据。**

在加载循环中，程序会结合这个起始点和线程的索引，计算出实际要读取的全局索引，并进行边界检查，处理 0 填充。

*/


/*
question3:

不理解，请举例解释

    for (int y=ty; y<sh_tile_height; y+= TILE_DIM_H) {
        for (int x=tx; x<sh_tile_width; x+=TILE_DIM_W) {
            const int cur_row = in_tile_start_row + y;
            const int cur_col = in_tile_start_col + x;

            if (cur_row >= 0 and cur_row<input_rows and cur_col>=0 and cur_col<input_cols) {
                sh_tile[y*sh_tile_width+x] = input[cur_row*input_cols+cur_col];
             else {
                sh_tile[y*sh_tile_width+x] = 0.0f;
            
        
    

好的，这段代码是 **2D 卷积分块优化**中最关键的步骤：**协作加载输入数据 Tile 和边界（Halo）**，并处理图像边缘的**零填充（Zero Padding）**。

其核心在于使用 **Block-Stride Loop**（线程块步长循环）让 Block 内的所有线程一起工作，高效地将数据从慢速的全局内存 (`input`) 转移到快速的共享内存 (`sh_tile`)。

---

## 1. 🔢 示例设定

我们假设一个具体的场景：

* **Block 尺寸:** TILE_DIM_W=4，TILE_DIM_H=2
* **Kernel 尺寸:** kernel_rows=3，kernel_cols=3
* **输入图像尺寸:** input_rows=100，input_cols=100
* **Shared Memory Tile 尺寸（已计算）:**
    * sh_tile_width = 4 + 3 - 1 = 6
    * sh_tile_height = 2 + 3 - 1 = 4
* **输入 Tile 起点（已计算）:** 假设 blockIdx.x=0，blockIdx.y=0。
    * k_radius_x=1，k_radius_y=1。
    * in_tile_start_row = 0 * 2 - 1 = -1
    * in_tile_start_col = 0 * 4 - 1 = -1

> **目标:** 线程块要加载一个 6 * 4 的区域到 sh_tile 中，起始于全局坐标 (-1, -1)。

---

## 2. 🚀 Block-Stride Loop 的分工机制

在这个 4 * 2 = 8 个线程的 Block 中，每个线程负责加载 sh_tile 中的多个元素。

由于 sh_tile 的尺寸是 6 * 4 (24 个元素)，而线程数是 8，所以每个线程平均需要执行 24 / 8 = 3 次 Shared Memory 写入操作。

### 示例分析：线程 (tx=1, ty=0)

我们关注线程 tx=1, ty=0 的执行轨迹：

| 循环变量 y | 循环变量 x | 计算 cur_row | 计算 cur_col | 边界检查 if | 结果 (写入 sh_tile) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| y=0 | x=1 | -1 + 0 = -1 | -1 + 1 = 0 | **False** (因为 cur_row < 0) | sh_tile[1] = 0.0f (0 填充) |
| y=0 | x=1+4=5 | -1 + 0 = -1 | -1 + 5 = 4 | **False** | sh_tile[5] = 0.0f (0 填充) |
| y=1 | x=1 | -1 + 1 = 0 | -1 + 1 = 0 | True | sh_tile[1 * 6 + 1] = input[0 * 100 + 0] (加载 I_{0,0) |
| y=1 | x=5 | -1 + 1 = 0 | -1 + 5 = 4 | True | sh_tile[1 * 6 + 5] = input[0 * 100 + 4] (加载 I_{0,4) |
| y=2 | x=1 | -1 + 2 = 1 | -1 + 1 = 0 | True | sh_tile[2 * 6 + 1] = input[1 * 100 + 0] (加载 I_{1,0) |
| y=2 | x=5 | -1 + 2 = 1 | -1 + 5 = 4 | True | sh_tile[2 * 6 + 5] = input[1 * 100 + 4] (加载 I_{1,4) |
| \ldots | \ldots | \ldots | \ldots | \ldots | \ldots |

### 3. 代码段的作用总结

1.  **索引计算:**
    * cur_row = in_tile_start_row + y: 每次循环，线程都会计算它在全局输入数组中需要访问的**行索引**。由于 in_tile_start_row=-1，所以 y=0 时的 cur_row 是 -1。
2.  **边界检查与填充:**
    * `if (cur_row >= 0 and ...)`：检查计算出的全局坐标是否在 100 * 100 的原始图像范围内。
    * **if 成功:** 加载数据：`sh_tile[...] = input[...]`。
    * **else 失败:** 执行零填充：`sh_tile[...] = 0.0f`。
3.  **效率:**
    * **协作加载:** 所有 8 个线程同时工作，共享 input 的读取带宽。
    * **连续写入:** 这种 Block-Stride 模式确保了线程在**水平方向**上访问 Shared Memory 时是连续的，进一步提高了效率。

这段代码的核心价值在于，它用并行的方式，高效且安全地完成了卷积计算前最费时的**数据预取**工作。

*/


/*
question4:

不理解，请举例解释

                sh_tile[y*sh_tile_width+x] = input[cur_row*input_cols+cur_col];

这两行代码是 **2D 卷积分块优化**中实现**将数据从全局内存 (Global Memory) 移动到共享内存 (Shared Memory)** 的关键步骤。

它们执行的是 **2D 数组索引到 1D 数组索引的映射**。

---

## 🔢 示例设定

我们沿用之前的设定，并假设当前线程 tx=1, ty=0 正在执行循环：

* **输入图像尺寸:** input_rows=100，input_cols=100
* **Shared Memory Tile 尺寸:** sh_tile_width=6
* **当前循环计算出的有效全局坐标:**
    * cur_row = 1 (第 2 行)
    * cur_col = 4 (第 5 列)
* **Shared Memory Tile 坐标:**
    * x = 5
    * y = 2

### 核心目标：

将全局内存中 input[1][4] 的值，存储到共享内存中 sh_tile[2][5] 的位置。

---

## 1. 目标：计算 1D 全局内存索引 (input)

input[{cur_row * input_cols + cur_col]

这段代码是将一个二维矩阵（图像）的 (r, c) 坐标转换为存储在 GPU 上的一个**扁平一维数组**中的索引。

#### 示例计算 (输入):

input_index = 1 * 100 + 4 = 104

> **含义：** 线程正在读取 Global Memory 中 input 数组的第 104 个元素（即 I_{1, 4）。

---

## 2. 目标：计算 1D 共享内存索引 (sh_tile)

sh_tile[{y * sh_tile_width + x]

这段代码是将线程块正在构建的 Shared Memory Tile 中的 (y, x) 坐标，转换为 Shared Memory **一维数组**中的索引。

#### 示例计算 (共享内存):

sh_tile_index = 2 * 6 + 5 = 12 + 5 = 17

> **含义：** 线程将读取到的数据存储到 Shared Memory 数组 sh_tile 的第 17 个位置。

---

## 最终总结：数据移动

将这两个目标结合起来，整行代码的意义是：

sh_tile[17] = input[104]

这实现了数据的转移：

1.  线程从 Global Memory 的**第 104 个位置**读取数据（对应输入图像的 R=1, C=4）。
2.  线程将该数据写入 Shared Memory 的**第 17 个位置**（对应 Shared Tile 的 Y=2, X=5）。

这个过程在整个线程块中并行进行，最终所有线程共同填充好 sh_tile 这个高速缓存区，供后续的卷积计算使用。

*/


/*
question5:

不理解，请举例解释

    if (out_row<input_rows and out_col<input_cols) {
        float sum = 0.0f;
        const float* sh_tile_row_ptr = &sh_tile[ty*sh_tile_width+tx];

        #pragma unroll
        for (int i=0; i<kernel_rows; ++i) {
            #pragma unroll
            for (int j=0; j<kernel_rows; ++j) {
                sum += sh_tile_row_ptr[j] * const_kernel[i*kernel_cols+j];
            
            sh_tile_row_ptr += sh_tile_width;
        

        output[out_row*input_cols + out_col] = sum;
    


这段代码是 **2D 卷积**操作的**计算阶段**，它利用之前加载到共享内存 (`sh_tile`) 中的数据来计算最终的输出像素值。

这段代码的核心功能是：**在一个线程内，使用嵌套循环遍历整个卷积核 (Kernel)，并计算点积 (Dot Product) 求和。**

-----

## 🔢 示例设置

我们假设一个具体的场景：

  * **卷积核 (Kernel) 尺寸:** kernel_rows = 3， kernel_cols = 3。
  * **共享内存 Tile 宽度:** sh_tile_width = 32 + 3 - 1 = 34。
  * **当前线程 ID:** tx=1, ty=1。
  * **目标:** 计算 output 图像上的像素 (out_row, out_col) 的值。

-----

## 1\. 边界检查和初始化


if (out_row<input_rows and out_col<input_cols) {
    float sum = 0.0f;
    // ...


  * **边界检查:** `if` 语句检查当前线程负责计算的输出坐标 (out_row, out_col) 是否在原始输入图像的有效范围内。如果 Tile 位于图像边缘，一些线程可能超出范围，这个检查防止它们执行无效计算或写入。
  * **初始化:** sum = 0.0f，用于累积点积结果。

## 2\. 共享内存指针定位


const float* sh_tile_row_ptr = &sh_tile[ty*sh_tile_width+tx];


这行代码计算了一个指向 **Shared Memory** 的指针 sh_tile_row_ptr，它定位了当前线程所需输入数据 3 * 3 区域的**左上角**。

### 示例解释 (指针起始位置)

  * **共享内存 1D 索引:** ty * sh_tile_width + tx = 1 * 34 + 1 = 35。
  * **作用:** 指针 sh_tile_row_ptr 指向共享内存数组 `sh_tile` 的第 35 个位置。这个位置是当前线程计算 P_{out_row, out_col 所需的 3 * 3 输入区域的**起始点**。

## 3\. 核心计算：点积求和（双重循环）


#pragma unroll
for (int i=0; i<kernel_rows; ++i) { // i: 卷积核的行 (0, 1, 2)
    #pragma unroll
    for (int j=0; j<kernel_rows; ++j) { // j: 卷积核的列 (0, 1, 2)
        sum += sh_tile_row_ptr[j] * const_kernel[i*kernel_cols+j];
    
    sh_tile_row_ptr += sh_tile_width; // 移动指针



这段双重循环执行 3 * 3 = 9 次乘加操作。

  * **`#pragma unroll`:** 告诉编译器展开循环，将循环体直接写成 9 条指令，提高执行速度。
  * **核心乘加:** sum += (输入数据) * (卷积核权重)。

### 示例解释 (循环执行和指针移动)

| 循环 i (行) | 内部循环 j (列) | Shared Memory 访问 | Constant Memory 访问 | 指针移动 |
| :--- | :--- | :--- | :--- | :--- |
| **i=0** (第 1 行) | j=0, 1, 2 | sh_tile_row_ptr[0, 1, 2] | const_kernel[0, 1, 2] | **向下一行**： sh_tile_row_ptr <-- 原地址 + 34 |
| **i=1** (第 2 行) | j=0, 1, 2 | sh_tile_row_ptr[0, 1, 2] | const_kernel[3, 4, 5] | **向下一行**： sh_tile_row_ptr <-- 原地址 + 34 |
| **i=2** (第 3 行) | j=0, 1, 2 | sh_tile_row_ptr[0, 1, 2] | const_kernel[6, 7, 8] | (退出循环) |

  * **Shared Memory 访问:** 线程通过 sh_tile_row_ptr[j] 顺序读取 3 个相邻的输入数据，然后指针 sh_tile_row_ptr 移动 34 个单位，指向下一行的起始点，再次读取 3 个数据。
  * **Constant Memory 访问:** 线程顺序读取 const_kernel 中的 9 个权重。

## 4\. 最终结果写入


output[out_row*input_cols + out_col] = sum;


  * 计算结束后，线程将最终的 sum (即卷积结果) 写回到 **Global Memory** 的 output 数组中，位置由之前计算的 (out_row, out_col) 确定。

> **总结:** 整个过程是 9 次高效的乘加操作。数据来自快速的 **Shared Memory** 和 **Constant Memory**，避免了对慢速 **Global Memory** 的重复访问。

*/

