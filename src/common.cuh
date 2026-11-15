#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(call)                                                                          \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,     \
                         cudaGetErrorString(err_));                                       \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

#define CB(call)                                                                          \
    do {                                                                                  \
        cublasStatus_t st_ = (call);                                                      \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                               \
            std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call,     \
                         static_cast<int>(st_));                                          \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { CK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& h) { CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

inline std::vector<float> random_matrix(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) x = u(rng);
    return v;
}

// cuBLAS is column-major. A row-major M x K matrix is a column-major K x M one, so the
// row-major C = A * B is the column-major C^T = B^T * A^T: swap the operands and M, N.
inline void cublas_sgemm(cublasHandle_t h, int M, int N, int K, float alpha, const float* A, const float* B,
                         float beta, float* C) {
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

inline void cublas_hgemm_f32(cublasHandle_t h, int M, int N, int K, const half* A, const half* B, float* C) {
    const float alpha = 1.0f, beta = 0.0f;
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C,
                    CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

__global__ void to_half(const float* x, half* y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = __float2half(x[i]);
}

__global__ void half_to_float_abs(const half* x, float* y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = fabsf(__half2float(x[i]));
}
