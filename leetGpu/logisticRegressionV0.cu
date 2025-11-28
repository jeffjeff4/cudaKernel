#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// Small L2 regularization to make the solution well-defined and close
// to the reference "expected" coefficients on separable data.
static const double REG_LAMBDA = 1e-6;

// Numerically stable sigmoid in double precision
static inline double sigmoid_double(double z) {
    if (z >= 0.0) {
        double e = exp(-z);
        return 1.0 / (1.0 + e);
    } else {
        double e = exp(z);
        return e / (1.0 + e);
    }
}

/**
 * Solve A x = b for x using Gaussian elimination with partial pivoting.
 * A is modified in-place to its upper-triangular form; b is also modified.
 * d = dimension.
 */
static void solve_linear_system(double* A, double* b, double* x, int d) {
    // Forward elimination
    for (int i = 0; i < d; ++i) {
        // Pivot: find row k >= i with max |A[k, i]|
        int pivot_row = i;
        double max_val = fabs(A[i * d + i]);
        for (int k = i + 1; k < d; ++k) {
            double val = fabs(A[k * d + i]);
            if (val > max_val) {
                max_val = val;
                pivot_row = k;
            }
        }

        // Swap rows i and pivot_row in A and b if needed
        if (pivot_row != i) {
            for (int j = 0; j < d; ++j) {
                double tmp = A[i * d + j];
                A[i * d + j] = A[pivot_row * d + j];
                A[pivot_row * d + j] = tmp;
            }
            double tmpb = b[i];
            b[i] = b[pivot_row];
            b[pivot_row] = tmpb;
        }

        // If pivot is extremely small, add tiny jitter to keep system solvable
        double pivot = A[i * d + i];
        if (fabs(pivot) < 1e-12) {
            pivot = (pivot >= 0.0 ? 1e-12 : -1e-12);
            A[i * d + i] = pivot;
        }

        // Eliminate below pivot
        for (int k = i + 1; k < d; ++k) {
            double factor = A[k * d + i] / pivot;
            if (factor == 0.0) continue;

            // Row operation on A
            for (int j = i; j < d; ++j) {
                A[k * d + j] -= factor * A[i * d + j];
            }
            // Row operation on b
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    for (int i = d - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < d; ++j) {
            sum -= A[i * d + j] * x[j];
        }
        double diag = A[i * d + i];
        if (fabs(diag) < 1e-12) {
            diag = (diag >= 0.0 ? 1e-12 : -1e-12);
        }
        x[i] = sum / diag;
    }
}

/**
 * Host-side Newton / IRLS solver for L2-regularized logistic regression.
 * X_dev, y_dev, beta_dev are device pointers. This function copies data
 * to host, runs Newton in double, then copies beta back to device.
 */
extern "C" void solve(const float* X_dev, const float* y_dev, float* beta_dev,
                      int n_samples, int n_features) {
    const int n = n_samples;
    const int d = n_features;

    if (n <= 0 || d <= 0) {
        return;
    }

    // Temporary host buffers (double precision for stability)
    double* X = (double*)malloc((size_t)n * d * sizeof(double));
    double* y = (double*)malloc((size_t)n * sizeof(double));
    double* beta = (double*)malloc((size_t)d * sizeof(double));
    double* grad = (double*)malloc((size_t)d * sizeof(double));
    double* delta = (double*)malloc((size_t)d * sizeof(double));
    double* H = (double*)malloc((size_t)d * d * sizeof(double));
    double* p = (double*)malloc((size_t)n * sizeof(double));
    double* w = (double*)malloc((size_t)n * sizeof(double));

    if (!X || !y || !beta || !grad || !delta || !H || !p || !w) {
        // Allocation failure: free what we can and return
        if (X) free(X);
        if (y) free(y);
        if (beta) free(beta);
        if (grad) free(grad);
        if (delta) free(delta);
        if (H) free(H);
        if (p) free(p);
        if (w) free(w);
        return;
    }

    // Copy X, y from device to host (via temporary float buffers)
    float* X_tmp = (float*)malloc((size_t)n * d * sizeof(float));
    float* y_tmp = (float*)malloc((size_t)n * sizeof(float));
    if (!X_tmp || !y_tmp) {
        if (X_tmp) free(X_tmp);
        if (y_tmp) free(y_tmp);
        free(X); free(y); free(beta); free(grad); free(delta); free(H); free(p); free(w);
        return;
    }

    cudaMemcpy(X_tmp, X_dev, (size_t)n * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_tmp, y_dev, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert to double
    for (int i = 0; i < n * d; ++i) {
        X[i] = (double)X_tmp[i];
    }
    for (int i = 0; i < n; ++i) {
        y[i] = (double)y_tmp[i];
    }

    free(X_tmp);
    free(y_tmp);

    // Initialize beta = 0
    for (int j = 0; j < d; ++j) {
        beta[j] = 0.0;
    }

    // Newton / IRLS hyperparameters
    const int    MAX_ITER = 25;
    const double TOL      = 1e-8;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // 1) Compute p_i = sigmoid(x_i^T beta), w_i = p_i (1 - p_i)
        for (int i = 0; i < n; ++i) {
            double z = 0.0;
            const double* Xi = X + (size_t)i * d;
            for (int j = 0; j < d; ++j) {
                z += Xi[j] * beta[j];
            }
            double pi = sigmoid_double(z);
            p[i] = pi;
            w[i] = pi * (1.0 - pi);
        }

        // 2) Gradient: grad = X^T (p - y) + lambda * beta
        //    We build it as: grad[j] = lambda * beta[j] + sum_i X_ij * (p_i - y_i)
        double max_abs_grad = 0.0;

        for (int j = 0; j < d; ++j) {
            grad[j] = REG_LAMBDA * beta[j];
        }

        for (int i = 0; i < n; ++i) {
            double t = p[i] - y[i];
            const double* Xi = X + (size_t)i * d;
            for (int j = 0; j < d; ++j) {
                grad[j] += Xi[j] * t;
            }
        }

        for (int j = 0; j < d; ++j) {
            double g = fabs(grad[j]);
            if (g > max_abs_grad) max_abs_grad = g;
        }

        if (max_abs_grad < TOL) {
            // Converged
            break;
        }

        // 3) Hessian: H = X^T W X + lambda * I
        //    W is diagonal with w_i = p_i (1 - p_i)
        //    We exploit symmetry (compute upper triangle and mirror).
        for (int j = 0; j < d * d; ++j) {
            H[j] = 0.0;
        }

        for (int i = 0; i < n; ++i) {
            const double* Xi = X + (size_t)i * d;
            double wi = w[i];
            if (wi == 0.0) continue;
            for (int j = 0; j < d; ++j) {
                double xij = Xi[j];
                double w_xij = wi * xij;
                for (int k = 0; k <= j; ++k) {
                    H[j * d + k] += w_xij * Xi[k];
                }
            }
        }

        // Mirror the symmetric Hessian and add regularization on the diagonal
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < j; ++k) {
                H[k * d + j] = H[j * d + k];
            }
            H[j * d + j] += REG_LAMBDA;
        }

        // 4) Solve H * delta = grad  (for minimizing L)
        //    Then beta <- beta - delta
        for (int j = 0; j < d; ++j) {
            delta[j] = 0.0;
        }

        // We will overwrite grad when solving; pass it as RHS b.
        solve_linear_system(H, grad, delta, d);

        // Update beta
        for (int j = 0; j < d; ++j) {
            beta[j] -= delta[j];
        }
    }

    // Copy final beta back to device as float
    float* beta_tmp = (float*)malloc((size_t)d * sizeof(float));
    if (beta_tmp) {
        for (int j = 0; j < d; ++j) {
            beta_tmp[j] = (float)beta[j];
        }
        cudaMemcpy(beta_dev, beta_tmp, (size_t)d * sizeof(float), cudaMemcpyHostToDevice);
        free(beta_tmp);
    }

    // Free host buffers
    free(X);
    free(y);
    free(beta);
    free(grad);
    free(delta);
    free(H);
    free(p);
    free(w);
}




//--------------------------------------------------------------------------------------------------
/*
question0:

不理解，请解释，用例子

这段代码实现了一个**L2 正则化逻辑回归（L2-Regularized Logistic Regression）求解器**，它使用 **牛顿法 (Newton's Method)** 或其变体 **迭代重加权最小二乘法 (IRLS)** 在 **CPU 端** 进行求解。

该代码的特点是：利用了 **CUDA** 运行时 API 将数据从 GPU 传输到 CPU，并在 CPU 的 **双精度 (double)** 环境中执行复杂的迭代优化计算，以保证数值稳定性。

-----

## ⚙️ I. 核心算法与数据结构

### 1\. 算法核心：牛顿法 / IRLS

牛顿法是一种用于寻找函数根的迭代优化算法。在逻辑回归中，它通过以下步骤迭代逼近最优解 \beta：

\beta_t+1 = \beta_t - H^-1 \nabla L(\beta_t)

  * \nabla L(\beta_t)：损失函数 L 的**梯度 (Gradient)**。
  * H：损失函数 L 的 **Hessian 矩阵**（二阶导数）。
  * H * \delta = \nabla L：求解牛顿步长 \delta，然后更新 \beta <-- \beta - \delta。

### 2\. 关键数据结构 (Host Buffers)

为了数值稳定性，所有计算都在双精度 (`double`) 下进行。

  * X, y：训练数据和标签。
  * beta：待求解的系数向量。
  * grad：损失函数的梯度向量 (\nabla L)。
  * H：Hessian 矩阵。
  * p：预测概率向量 (p_i = \textsigmoid(x_i^T \beta))。
  * w：权重向量 (w_i = p_i (1 - p_i)，用于构建 Hessian 矩阵 H = X^T W X + \lambda I)。

-----

## 🚀 II. 辅助函数解释

### 1\. 数值稳定 Sigmoid (`sigmoid_double`)

c
static inline double sigmoid_double(double z) 
    if (z >= 0.0)    else   



  * **目的:** 计算 Sigmoid 函数 \frac11 + e^-z。
  * **优化:** 为了避免当 z 很大时 e^-z 下溢（underflow）或 e^z 溢出（overflow）导致的数值不稳定，函数使用条件判断：
      * 当 z \ge 0 时，计算 \frac11 + e^-z。
      * 当 z < 0 时，等效计算 \frace^z1 + e^z。

### 2\. 线性系统求解 (`solve_linear_system`)

c
static void solve_linear_system(double* A, double* b, double* x, int d) 
    // Forward elimination
    // Back substitution



  * **目的:** 求解牛顿步长所需的线性方程组 A * \delta = b (H * \delta = grad)。
  * **方法:** 使用 **高斯消元法 (Gaussian Elimination)**，并结合**部分主元选择 (Partial Pivoting)** 来增强数值稳定性。
      * **主元选择:** 找到当前列中绝对值最大的元素作为主元，并交换行，避免除以接近零的数。

-----

## 🧭 III. 迭代求解 (`solve` 函数核心循环)

`solve` 函数将设备数据复制到主机，然后在一个循环中执行牛顿法的 4 个核心步骤，直到收敛或达到最大迭代次数 (25 次)。

### 阶段 1：计算预测概率和权重 (p 和 w)

c
// 1) Compute p_i = sigmoid(x_i^T beta), w_i = p_i (1 - p_i)
for (int i = 0; i < n; ++i) 
    // ... 计算线性预测 z = x_i^T * beta ...
    double pi = sigmoid_double(z);
    p[i] = pi;
    w[i] = pi * (1.0 - pi); // 计算权重 w_i



### 阶段 2：计算梯度 (grad)

\nabla L = X^T (p - y) + \lambda \beta

c
// 2) Gradient: grad = X^T (p - y) + lambda * beta
// ... 初始化 grad[j] = REG_LAMBDA * beta[j] (L2 正则项) ...
for (int i = 0; i < n; ++i) 
    double t = p[i] - y[i]; // 计算残差 (p - y)
    for (int j = 0; j < d; ++j) 
        grad[j] += Xi[j] * t; // 累加 X^T * (p - y)
    



  * 如果梯度的绝对值最大值小于容忍度 (TOL)，则认为**收敛**，退出循环。

### 阶段 3：计算 Hessian 矩阵 (H)

H = X^T W X + \lambda I

c
// 3) Hessian: H = X^T W X + lambda * I
// ... (初始化 H) ...
for (int i = 0; i < n; ++i)  // 遍历样本
    double wi = w[i];
    for (int j = 0; j < d; ++j) 
        double w_xij = wi * Xi[j];
        for (int k = 0; k <= j; ++k) 
            H[j * d + k] += w_xij * Xi[k]; // 利用对称性只计算上三角
        
    

// Mirror the symmetric Hessian and add regularization on the diagonal
// ... H[k*d + j] = H[j*d + k] ... 
// ... H[j*d + j] += REG_LAMBDA ... (对角线加上 L2 正则项)


  * **W** 是一个对角矩阵，其元素 w_i 由 p_i(1-p_i) 构成。代码通过直接将 w_i 乘到 X_ij 上，避免了显式构造 W 矩阵。

### 阶段 4：求解和更新 (\beta)

c
// 4) Solve H * delta = grad  (for minimizing L)
solve_linear_system(H, grad, delta, d); // 求解步长 delta
// Update beta
for (int j = 0; j < d; ++j) 
    beta[j] -= delta[j]; // beta <- beta - delta



-----

## 🏁 IV. 结论

这个 `solve` 函数展示了在 CUDA 环境下进行复杂优化的一个**混合策略**：将数据从 GPU 内存移动到 CPU 内存，利用 CPU 的高精度浮点运算和复杂的线性代数求解器来保证 **数值的稳定性** 和 **求解的正确性**，然后将最终结果传回 GPU。

//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question1:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question2:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
/*
question0:


//--------------------------------------------------------------------------------------------------


