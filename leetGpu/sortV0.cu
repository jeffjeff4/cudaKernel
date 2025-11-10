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
////    }
////    // ... Final Copy (if needed) ...
////}
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

#define BLOCK_SIZE 256


/**
 * @brief Finds the first element *not less than* value. (std::lower_bound)
 *
 * This function calculates the "rank" of 'value'. It returns the index 'l'
 * such that all elements arr[s...l-1] are < value.
 *
 * @param arr   The array to search in.
 * @param s     The starting index (inclusive).
 *CH* @param e     The ending index (exclusive).
 * @param value The value to search for.
 * @return The first index 'l' where 'value' could be inserted.
 */
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

/**
 * @brief Finds the first element *greater than* value. (std::upper_bound)
 * This is the key to a stable merge for handling equal values.
 */
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

/**
 * @brief An iterative, bottom-up parallel merge kernel.
 *
 * This single kernel is launched for each "pass" of the merge sort.
 * Each thread is responsible for placing one element from `d_src` into
 * its final sorted position in `d_dst`.
 *
 * @param d_src The source array (contains sorted sub-arrays of size 'width').
 * @param d_dst The destination array (will contain sorted sub-arrays of size '2*width').
 * @param N     The total number of elements in the array.
 * @param width The size of the sorted sub-arrays in `d_src` (e.g., 1, 2, 4, 8...).
 */
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

/**
 * @brief High-performance iterative, bottom-up merge sort.
 * This is the main "solve" function.
 */
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

//---------------------------------------------------------------------------
//method1
//wrong

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> // Added for malloc/free

#define BLOCK_SIZE 256

/**
 * @brief Finds the insertion point for 'value' in the sorted array 'arr'.
 *
 * This function calculates the "rank" of 'value'. It returns the index 'l'
 * such that all elements arr[s...l-1] are < value. This is equivalent
 * to C++'s std::lower_bound.
 *
 * @param arr   The sorted array to search in.
 * @param s     The starting index (inclusive).
 * @param e     The ending index (exclusive).
 * @param value The value to search for.
 * @return The first index 'l' where 'value' could be inserted.
 */
__device__ int lower_bound_rank(const float* arr, int s, int e, float value) {
    int l = s;
    int r = e; // 'e' is exclusive (one past the end)
    while (l < r) {
        int m = l + (r - l) / 2;
        if (arr[m] < value) {
            l = m + 1; // value must be in the right half
        } else {
            r = m;     // value is in the left half (or at m)
        }
    }
    return l; // 'l' is the insertion point (rank)
}

/**
 * @brief Parallel merge kernel.
 *
 * Each thread takes one element from one of the two sorted halves
 * (arr[0...m-1] and arr[m...N-1]), finds its final sorted position in the
 * 'helper' array using a binary search (rank), and places it there.
 *
 * @param arr     Input array, containing two sorted halves (split at N/2).
 * @param helper  Output (helper) array to write the merged result.
 * @param N       Total number of elements in this merge.
 */
__global__ void merge_kernel(const float* arr, float* helper, int N) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n >= N) return;

    // 'm' is the split point, dividing arr into [0...m-1] and [m...N-1]
    int m = N / 2;
    float value = arr[n];

    if (n < m) {
        // This thread is in the LEFT half (arr[0...m-1])
        // 1. Find my index within my half: (n)
        // 2. Find my rank (insertion point) in the OTHER half:
        //    lower_bound_rank(arr, m, N, value)
        // 3. The rank's offset from the start of its half is (rank - m)
        // 4. My final position = (my index in half) + (other half's rank)
        int rank_in_other = lower_bound_rank(arr, m, N, value);
        int dest_idx = n + (rank_in_other - m);
        helper[dest_idx] = value;
    } else {
        // This thread is in the RIGHT half (arr[m...N-1])
        // 1. Find my index within my half: (n - m)
        // 2. Find my rank (insertion point) in the OTHER half:
        //    lower_bound_rank(arr, 0, m, value)
        // 3. The rank is already relative to index 0.
        // 4. My final position = (my index in half) + (other half's rank)
        int rank_in_other = lower_bound_rank(arr, 0, m, value);
        int dest_idx = (n - m) + rank_in_other;
        helper[dest_idx] = value;
    }
}

// Helper function to print a device array from the host
void print_darr(const char* title, float* d_arr, int n) {
    float* h_arr = (float*)malloc(n * sizeof(float));
    if (h_arr == NULL) {
        fprintf(stderr, "Failed to allocate host memory for printing\n");
        return;
    }
    cudaError_t err = cudaMemcpy(h_arr, d_arr, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return;
    }

    printf("%s", title);
    for (int i = 0; i < n; i++) {
        fprintf(stdout, "%f ", h_arr[i]);
    }
    fprintf(stdout, "\n");
    free(h_arr);
}

/**
 * @brief Host-recursive merge sort function.
 *
 * @param arr    The device array to be sorted.
 * @param helper A temporary device buffer of the same size.
 * @param s      The starting index (inclusive).
 * @param e      The ending index (inclusive).
 */
void merge_sort(float* arr, float* helper, int s, int e) {
    // FIX 1: Base case. Stop if 0 or 1 elements.
    if (s >= e) {
        return;
    }

    // FIX 2: Correct recursive split.
    // We split the N elements into N/2 and (N - N/2)
    // to match the kernel's `m = N/2` assumption.
    int N_total = e - s + 1;
    int N_left = N_total / 2;
    int m_abs = s + N_left - 1; // The absolute index of the split
    
    // Recurse on the two halves *before* merging
    merge_sort(arr, helper, s, m_abs);     // First half: [s ... m_abs]
    merge_sort(arr, helper, m_abs + 1, e); // Second half: [m_abs+1 ... e]
    
    // --- Merge the two sorted halves ---
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N_total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // We launch the kernel on the sub-array `arr + s`.
    // The kernel will see this as an array of size `N_total`.
    // Its internal `m = N_total / 2` (which is N_left) logic will
    // correctly match our split, so the merge will work.
    merge_kernel<<<blocks, threads>>>(arr + s, helper + s, N_total);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    // FIX 3: Copy the merged data from helper back to arr
    cudaMemcpy(arr + s, helper + s, N_total * sizeof(float), cudaMemcpyDeviceToDevice);
}

// data is device pointer
extern "C" void solve(float* data, int N) {
     float* d_helper;
     cudaMalloc((void**)&d_helper, N * sizeof(float));

     // Call sort on the full range of indices [0 ... N-1]
     merge_sort(data, d_helper, 0, N - 1);

     cudaFree(d_helper);
     cudaDeviceSynchronize();
}

//---------------------------------------------------------------------------
//method2
//wrong

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> // Added for malloc/free

#define BLOCK_SIZE 256

/**
 * @brief Finds the first element *not less than* value. (std::lower_bound)
 */
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

/**
 * @brief BUG FIX: Finds the first element *greater than* value. (std::upper_bound)
 * This is the key to a stable merge.
 */
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


/**
 * @brief Parallel merge kernel (now stable).
 */
__global__ void merge_kernel(const float* arr, float* helper, int N) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n >= N) return;

    int m = N / 2;
    float value = arr[n];

    if (n < m) {
        // This is a thread from the LEFT half.
        // Find how many elements in the RIGHT half are < me.
        // Use lower_bound.
        int rank_in_other = lower_bound_rank(arr, m, N, value);
        int dest_idx = n + (rank_in_other - m);
        helper[dest_idx] = value;
    } else {
        // This is a thread from the RIGHT half.
        // Find how many elements in the LEFT half are <= me
        // to ensure I place myself *after* them.
        // Use upper_bound.
        int rank_in_other = upper_bound_rank(arr, 0, m, value); // <-- BUG FIX
        int dest_idx = (n - m) + rank_in_other;
        helper[dest_idx] = value;
    }
}

// Helper function to print a device array from the host
void print_darr(const char* title, float* d_arr, int n) {
    float* h_arr = (float*)malloc(n * sizeof(float));
    if (h_arr == NULL) {
        fprintf(stderr, "Failed to allocate host memory for printing\n");
        return;
    }
    cudaError_t err = cudaMemcpy(h_arr, d_arr, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return;
    }

    printf("%s", title);
    for (int i = 0; i < n; i++) {
        fprintf(stdout, "%f ", h_arr[i]);
    }
    fprintf(stdout, "\n");
    free(h_arr);
}

/**
 * @brief Host-recursive merge sort function.
 */
void merge_sort(float* arr, float* helper, int s, int e) {
    if (s >= e) {
        return;
    }

    // Split logic to match the kernel's N/2 assumption
    int N_total = e - s + 1;
    int N_left = N_total / 2;
    int m_abs = s + N_left - 1; // The absolute index of the split
    
    // Recurse on the two halves *before* merging
    merge_sort(arr, helper, s, m_abs);     // First half: [s ... m_abs]
    merge_sort(arr, helper, m_abs + 1, e); // Second half: [m_abs+1 ... e]
    
    // --- Merge the two sorted halves ---
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N_total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch the kernel on the sub-array `arr + s`
    // The kernel sees this as an array of size `N_total`.
    // Its internal `m = N_total / 2` (which is N_left) logic
    // correctly matches our split.
    merge_kernel<<<blocks, threads>>>(arr + s, helper + s, N_total);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    // Copy the merged data from helper back to arr
    cudaMemcpy(arr + s, helper + s, N_total * sizeof(float), cudaMemcpyDeviceToDevice);
}

// data is device pointer
extern "C" void solve(float* data, int N) {
     float* d_helper;
     cudaMalloc((void**)&d_helper, N * sizeof(float));

     // It's good practice to zero-initialize the helper buffer
     // in case of bugs like the one we found, although
     // a correct algorithm doesn't strictly need it.
     cudaMemset(d_helper, 0, N * sizeof(float));

     // Call sort on the full range of indices [0 ... N-1]
     merge_sort(data, d_helper, 0, N - 1);

     cudaFree(d_helper);
     cudaDeviceSynchronize();
}


//---------------------------------------------------------------------------
//method3
//correct, but timeout

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> // For malloc/free
#include <math.h>   // For fminf

#define BLOCK_SIZE 256

/**
 * @brief Finds the first element *not less than* value. (std::lower_bound)
 *
 * This function calculates the "rank" of 'value'. It returns the index 'l'
 * such that all elements arr[s...l-1] are < value.
 *
 * @param arr   The array to search in.
 * @param s     The starting index (inclusive).
 *CH* @param e     The ending index (exclusive).
 * @param value The value to search for.
 * @return The first index 'l' where 'value' could be inserted.
 */
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

/**
 * @brief Finds the first element *greater than* value. (std::upper_bound)
 * This is the key to a stable merge for handling equal values.
 */
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

/**
 * @brief An iterative, bottom-up parallel merge kernel.
 *
 * This single kernel is launched for each "pass" of the merge sort.
 * Each thread is responsible for placing one element from `d_src` into
 * its final sorted position in `d_dst`.
 *
 * @param d_src The source array (contains sorted sub-arrays of size 'width').
 * @param d_dst The destination array (will contain sorted sub-arrays of size '2*width').
 * @param N     The total number of elements in the array.
 * @param width The size of the sorted sub-arrays in `d_src` (e.g., 1, 2, 4, 8...).
 */
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

/**
 * @brief High-performance iterative, bottom-up merge sort.
 * This is the main "solve" function.
 */
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


