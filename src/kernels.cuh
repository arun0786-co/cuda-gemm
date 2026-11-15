#pragma once
// Matrix multiply C = alpha * A * B + beta * C, row-major, one step at a time.
//
//   1 naive_rows     one thread per element of C, neighbouring threads on neighbouring rows:
//                    a warp reads 32 different rows of A, so loads are not coalesced
//   2 naive_cols     neighbouring threads on neighbouring columns: B and C are read in
//                    order and a warp shares one element of A
//   3 smem_32        32 x 32 tiles of A and B staged in shared memory, each element loaded
//                    from DRAM once per tile instead of once per use
//   4 tile_1d        64 x 64 block tiles, each thread computes 8 elements of a column, so
//                    one value of B from shared memory feeds 8 multiply-adds
//   5 tile_2d        128 x 128 block tiles, each thread an 8 x 8 block of C: per k step it
//                    reads 16 values from shared memory and does 64 multiply-adds
//   6 vector         as 5, with 16-byte loads from global memory, and A stored transposed
//                    in shared memory so the 8 values a thread needs sit together
//   7 no_conflicts   as 6, but each thread's 8 x 8 outputs are split into four 4 x 4
//                    blocks 64 apart, so neighbouring lanes read neighbouring 16-byte
//                    values; in 6 they are 32 bytes apart and two of every 8 lanes share
//                    banks
//   8 double_buffer  as 7, with two shared buffers: the next tile is fetched into
//                    registers while the current one is used, and one barrier per tile
//                    instead of two
//   9 wmma           fp16 inputs, fp32 results, on the tensor cores with WMMA 16x16x16
//                    fragments; 128 x 128 block tiles, 8 warps of 64 x 32 each
//
// Kernels 1 to 5 take any sizes. 6 to 8 need K and N to be multiples of 4 (16-byte loads);
// 9 needs M and N multiples of 128 and K a multiple of 32, and computes C = A * B only.

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>

namespace gemm {

__device__ __forceinline__ float out(float acc, float alpha, float beta, const float* c) {
    return beta == 0.0f ? alpha * acc : alpha * acc + beta * *c;  // beta 0: C is not read
}

// 1 -------------------------------------------------------------------------------------
__global__ void naive_rows(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;  // x is the fast index: rows
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = out(acc, alpha, beta, &C[row * N + col]);
}

// 2 -------------------------------------------------------------------------------------
__global__ void naive_cols(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x is the fast index: columns
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = out(acc, alpha, beta, &C[row * N + col]);
}

// 3 -------------------------------------------------------------------------------------
template <int T>
__global__ void smem_tiled(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ float As[T][T], Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = out(acc, alpha, beta, &C[row * N + col]);
}

// 4 -------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__(BM * BN / TM)
tile_1d(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    constexpr int NT = BM * BN / TM;
    __shared__ float As[BM * BK], Bs[BK * BN];
    const int tid = threadIdx.x;
    const int tcol = tid % BN, trow = tid / BN;  // rows trow * TM .. + TM of column tcol
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    float acc[TM] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += NT) {
            int r = i / BK, c = i % BK;
            As[i] = (r0 + r < M && k0 + c < K) ? A[(r0 + r) * K + k0 + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += NT) {
            int r = i / BN, c = i % BN;
            Bs[i] = (k0 + r < K && c0 + c < N) ? B[(k0 + r) * N + c0 + c] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float b = Bs[k * BN + tcol];
#pragma unroll
            for (int t = 0; t < TM; ++t) acc[t] += As[(trow * TM + t) * BK + k] * b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        int r = r0 + trow * TM + t, c = c0 + tcol;
        if (r < M && c < N) C[r * N + c] = out(acc[t], alpha, beta, &C[r * N + c]);
    }
}

// 5 -------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
tile_2d(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    constexpr int NT = (BM / TM) * (BN / TN);
    __shared__ float As[BM * BK], Bs[BK * BN];
    const int tid = threadIdx.x;
    const int trow = tid / (BN / TN), tcol = tid % (BN / TN);
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    float acc[TM][TN] = {};
    float rm[TM], rn[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += NT) {
            int r = i / BK, c = i % BK;
            As[i] = (r0 + r < M && k0 + c < K) ? A[(r0 + r) * K + k0 + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += NT) {
            int r = i / BN, c = i % BN;
            Bs[i] = (k0 + r < K && c0 + c < N) ? B[(k0 + r) * N + c0 + c] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) rm[i] = As[(trow * TM + i) * BK + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) rn[j] = Bs[k * BN + tcol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += rm[i] * rn[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = r0 + trow * TM + i, c = c0 + tcol * TN + j;
            if (r < M && c < N) C[r * N + c] = out(acc[i][j], alpha, beta, &C[r * N + c]);
        }
}

// 6, 7, 8 -------------------------------------------------------------------------------
// 128 x 128 x 8 block tiles, 256 threads, an 8 x 8 block of C per thread. Global loads are
// float4: the A tile (128 rows x 8) is one float4 per thread, as is the B tile (8 x 128).
// A is stored transposed in shared memory (As[k][m]) so a thread's 8 values of A for one
// k are contiguous.
constexpr int VBM = 128, VBN = 128, VBK = 8, VT = 256;

struct Tile {
    float4 a, b;
};

__device__ __forceinline__ Tile fetch(int M, int N, int K, const float* A, const float* B, int r0, int c0, int k0) {
    const int tid = threadIdx.x;
    const int ar = tid / 2, ac = (tid % 2) * 4;    // A: row ar, columns ac .. ac + 3
    const int br = tid / 32, bc = (tid % 32) * 4;  // B: row br, columns bc .. bc + 3
    const float4 zero = make_float4(0, 0, 0, 0);
    Tile t;
    t.a = (r0 + ar < M && k0 + ac < K) ? *reinterpret_cast<const float4*>(&A[(r0 + ar) * K + k0 + ac]) : zero;
    t.b = (k0 + br < K && c0 + bc < N) ? *reinterpret_cast<const float4*>(&B[(k0 + br) * N + c0 + bc]) : zero;
    return t;
}

__device__ __forceinline__ void put(const Tile& t, float* As, float* Bs) {
    const int tid = threadIdx.x;
    const int ar = tid / 2, ac = (tid % 2) * 4;
    As[(ac + 0) * VBM + ar] = t.a.x;  // transposed: As[k][m]
    As[(ac + 1) * VBM + ar] = t.a.y;
    As[(ac + 2) * VBM + ar] = t.a.z;
    As[(ac + 3) * VBM + ar] = t.a.w;
    *reinterpret_cast<float4*>(&Bs[(tid / 32) * VBN + (tid % 32) * 4]) = t.b;
}

// rows and columns of C that thread (trow, tcol) owns, for the two layouts
template <bool Split>
__device__ __forceinline__ int own(int t, int i) {
    // contiguous: t*8 + i. split: four values at t*4, four more 64 further on
    return Split ? (i < 4 ? t * 4 + i : 64 + t * 4 + (i - 4)) : t * 8 + i;
}

template <bool Split>
__device__ __forceinline__ void compute(const float* As, const float* Bs, int trow, int tcol, float (&acc)[8][8]) {
#pragma unroll
    for (int k = 0; k < VBK; ++k) {
        float rm[8], rn[8];
        const float4* a = reinterpret_cast<const float4*>(&As[k * VBM]);
        const float4* b = reinterpret_cast<const float4*>(&Bs[k * VBN]);
        float4 a0 = a[own<Split>(trow, 0) / 4], a1 = a[own<Split>(trow, 4) / 4];
        float4 b0 = b[own<Split>(tcol, 0) / 4], b1 = b[own<Split>(tcol, 4) / 4];
        rm[0] = a0.x; rm[1] = a0.y; rm[2] = a0.z; rm[3] = a0.w;
        rm[4] = a1.x; rm[5] = a1.y; rm[6] = a1.z; rm[7] = a1.w;
        rn[0] = b0.x; rn[1] = b0.y; rn[2] = b0.z; rn[3] = b0.w;
        rn[4] = b1.x; rn[5] = b1.y; rn[6] = b1.z; rn[7] = b1.w;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] += rm[i] * rn[j];
    }
}

template <bool Split>
__device__ __forceinline__ void store(int M, int N, float alpha, float beta, float* C, int r0, int c0, int trow,
                                      int tcol, const float (&acc)[8][8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = r0 + own<Split>(trow, i);
        if (r >= M) continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = c0 + own<Split>(tcol, 4 * h);
            if (c >= N) continue;  // N is a multiple of 4, so the 4 columns are all in or all out
            float4* p = reinterpret_cast<float4*>(&C[r * N + c]);
            float4 v = beta == 0.0f ? make_float4(0, 0, 0, 0) : *p;
            v.x = alpha * acc[i][4 * h + 0] + beta * v.x;
            v.y = alpha * acc[i][4 * h + 1] + beta * v.y;
            v.z = alpha * acc[i][4 * h + 2] + beta * v.z;
            v.w = alpha * acc[i][4 * h + 3] + beta * v.w;
            *p = v;
        }
    }
}

template <bool Split>
__global__ void __launch_bounds__(VT)
vector_tiled(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ __align__(16) float As[VBK * VBM], Bs[VBK * VBN];
    const int tid = threadIdx.x, trow = tid / 16, tcol = tid % 16;
    const int r0 = blockIdx.y * VBM, c0 = blockIdx.x * VBN;
    float acc[8][8] = {};
    for (int k0 = 0; k0 < K; k0 += VBK) {
        put(fetch(M, N, K, A, B, r0, c0, k0), As, Bs);
        __syncthreads();
        compute<Split>(As, Bs, trow, tcol, acc);
        __syncthreads();
    }
    store<Split>(M, N, alpha, beta, C, r0, c0, trow, tcol, acc);
}

__global__ void __launch_bounds__(VT)
double_buffered(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ __align__(16) float As[2][VBK * VBM], Bs[2][VBK * VBN];
    const int tid = threadIdx.x, trow = tid / 16, tcol = tid % 16;
    const int r0 = blockIdx.y * VBM, c0 = blockIdx.x * VBN;
    float acc[8][8] = {};
    put(fetch(M, N, K, A, B, r0, c0, 0), As[0], Bs[0]);
    __syncthreads();
    const int tiles = (K + VBK - 1) / VBK;
    for (int t = 0; t < tiles; ++t) {
        const int cur = t & 1;
        Tile next;
        const bool more = t + 1 < tiles;
        if (more) next = fetch(M, N, K, A, B, r0, c0, (t + 1) * VBK);  // in flight during compute
        compute<true>(As[cur], Bs[cur], trow, tcol, acc);
        // the other buffer was last read before the previous barrier, so it is free
        if (more) put(next, As[cur ^ 1], Bs[cur ^ 1]);
        __syncthreads();
    }
    store<true>(M, N, alpha, beta, C, r0, c0, trow, tcol, acc);
}

// 9 -------------------------------------------------------------------------------------
constexpr int WBM = 128, WBN = 128, WBK = 32;
constexpr int WPA = WBK + 8, WPB = WBN + 8;  // padded rows: fewer bank conflicts, 16-byte aligned

__global__ void __launch_bounds__(256) wmma_gemm(int M, int N, int K, const half* A, const half* B, float* C) {
#if __CUDA_ARCH__ >= 700  // mma.h only declares nvcuda::wmma when compiling for 7.0 and up
    namespace wmma = nvcuda::wmma;
    __shared__ __align__(32) half As[WBM * WPA];
    __shared__ __align__(32) half Bs[WBK * WPB];
    const int tid = threadIdx.x, warp = tid / 32;
    const int wm = warp / 4, wn = warp % 4;  // 2 x 4 warps, each a 64 x 32 piece of C
    const int r0 = blockIdx.y * WBM, c0 = blockIdx.x * WBN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    for (int k0 = 0; k0 < K; k0 += WBK) {
        // 16-byte chunks of 8 halves: 512 for the A tile and 512 for the B tile, 2 each per thread
        for (int i = tid; i < WBM * WBK / 8; i += 256) {
            int r = i / (WBK / 8), c = (i % (WBK / 8)) * 8;
            *reinterpret_cast<uint4*>(&As[r * WPA + c]) = *reinterpret_cast<const uint4*>(&A[(r0 + r) * K + k0 + c]);
        }
        for (int i = tid; i < WBK * WBN / 8; i += 256) {
            int r = i / (WBN / 8), c = (i % (WBN / 8)) * 8;
            *reinterpret_cast<uint4*>(&Bs[r * WPB + c]) = *reinterpret_cast<const uint4*>(&B[(k0 + r) * N + c0 + c]);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < WBK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(a[i], &As[(wm * 64 + i * 16) * WPA + kk], WPA);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[kk * WPB + wn * 32 + j * 16], WPB);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&C[(r0 + wm * 64 + i * 16) * N + c0 + wn * 32 + j * 16], acc[i][j], N,
                                    wmma::mem_row_major);
#endif
}

// ---------------------------------------------------------------------------------------
constexpr int kFp32Kernels = 8;
inline const char* name(int k) {
    static const char* n[] = {"", "naive_rows", "naive_cols", "smem_32", "tile_1d", "tile_2d",
                              "vector", "no_conflicts", "double_buffer", "wmma"};
    return n[k];
}

// can kernel k handle these sizes?
inline bool supports(int k, int M, int N, int K) {
    if (k >= 6 && k <= 8) return K % 4 == 0 && N % 4 == 0;
    if (k == 9) return M % WBM == 0 && N % WBN == 0 && K % WBK == 0;
    return true;
}

inline void launch(int k, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C,
                   cudaStream_t s = 0) {
    auto cdiv = [](int a, int b) { return (a + b - 1) / b; };
    switch (k) {
        case 1: naive_rows<<<dim3(cdiv(M, 32), cdiv(N, 32)), dim3(32, 32), 0, s>>>(M, N, K, alpha, A, B, beta, C); break;
        case 2: naive_cols<<<dim3(cdiv(N, 32), cdiv(M, 32)), dim3(32, 32), 0, s>>>(M, N, K, alpha, A, B, beta, C); break;
        case 3: smem_tiled<32><<<dim3(cdiv(N, 32), cdiv(M, 32)), dim3(32, 32), 0, s>>>(M, N, K, alpha, A, B, beta, C); break;
        case 4: tile_1d<64, 64, 8, 8><<<dim3(cdiv(N, 64), cdiv(M, 64)), 512, 0, s>>>(M, N, K, alpha, A, B, beta, C); break;
        case 5: tile_2d<128, 128, 8, 8, 8><<<dim3(cdiv(N, 128), cdiv(M, 128)), 256, 0, s>>>(M, N, K, alpha, A, B, beta, C); break;
        case 6: vector_tiled<false><<<dim3(cdiv(N, VBN), cdiv(M, VBM)), VT, 0, s>>>(M, N, K, alpha, A, B, beta, C); break;
        case 7: vector_tiled<true><<<dim3(cdiv(N, VBN), cdiv(M, VBM)), VT, 0, s>>>(M, N, K, alpha, A, B, beta, C); break;
        case 8: double_buffered<<<dim3(cdiv(N, VBN), cdiv(M, VBM)), VT, 0, s>>>(M, N, K, alpha, A, B, beta, C); break;
        default: std::fprintf(stderr, "no fp32 kernel %d\n", k);
    }
}

inline void launch_wmma(int M, int N, int K, const half* A, const half* B, float* C, cudaStream_t s = 0) {
    wmma_gemm<<<dim3(N / WBN, M / WBM), 256, 0, s>>>(M, N, K, A, B, C);
}

// registers per thread and static shared memory of each kernel, for the results table
inline cudaFuncAttributes attributes(int k) {
    cudaFuncAttributes a{};
    switch (k) {
        case 1: cudaFuncGetAttributes(&a, naive_rows); break;
        case 2: cudaFuncGetAttributes(&a, naive_cols); break;
        case 3: cudaFuncGetAttributes(&a, smem_tiled<32>); break;
        case 4: cudaFuncGetAttributes(&a, tile_1d<64, 64, 8, 8>); break;
        case 5: cudaFuncGetAttributes(&a, tile_2d<128, 128, 8, 8, 8>); break;
        case 6: cudaFuncGetAttributes(&a, vector_tiled<false>); break;
        case 7: cudaFuncGetAttributes(&a, vector_tiled<true>); break;
        case 8: cudaFuncGetAttributes(&a, double_buffered); break;
        case 9: cudaFuncGetAttributes(&a, wmma_gemm); break;
    }
    return a;
}

inline int threads(int k) {
    static const int t[] = {0, 1024, 1024, 1024, 512, 256, 256, 256, 256, 256};
    return t[k];
}

}  // namespace gemm
