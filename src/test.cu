// gemm_test: every kernel against cuBLAS on sizes that are and are not multiples of the
// tiles. Exits non-zero on the first failure.
//
// The tolerance is the textbook bound for a dot product in floating point: computing
// sum_k a_k b_k in any order is off by at most about K * u * sum_k |a_k b_k| (u = 2^-24),
// so two correct implementations differ by at most twice that. R = |A| |B| gives the sum of
// absolute products for every element at once. An indexing bug is off by about R itself,
// thousands of times more than the bound for these sizes.

#include <cmath>
#include <limits>

#include "common.cuh"
#include "kernels.cuh"

static int failures = 0;

static void check(const char* what, int M, int N, int K, const std::vector<float>& got, const std::vector<float>& ref,
                  const std::vector<float>& R, const std::vector<float>& C0, float alpha, float beta, double slack) {
    const double u = std::ldexp(1.0, -24);
    double worst = 0;  // largest error as a fraction of the allowed error
    for (size_t i = 0; i < got.size(); ++i) {
        double allowed = slack * (2.0 * K + 4) * u * (std::fabs(alpha) * R[i] + std::fabs(beta) * std::fabs(C0[i])) + 1e-30;
        double e = std::fabs(double(got[i]) - double(ref[i]));
        if (!(e <= allowed)) {  // also catches NaN
            if (failures < 20)
                std::printf("FAIL %-14s %5d x %5d x %5d  element %zu: %g vs %g (allowed %g)\n", what, M, N, K, i,
                            got[i], ref[i], allowed);
            ++failures;
            return;
        }
        worst = std::max(worst, e / allowed);
    }
    std::printf("ok   %-14s %5d x %5d x %5d  worst error %.3f of the bound\n", what, M, N, K, worst);
}

static void fp32_case(cublasHandle_t h, int M, int N, int K, float alpha, float beta, unsigned seed) {
    auto a = random_matrix(size_t(M) * K, seed), b = random_matrix(size_t(K) * N, seed + 1);
    auto c0 = random_matrix(size_t(M) * N, seed + 2);
    if (beta == 0.0f) std::fill(c0.begin(), c0.end(), std::numeric_limits<float>::quiet_NaN());  // must not be read
    Dev<float> A(a.size()), B(b.size()), C(c0.size()), Cref(c0.size());
    A.up(a);
    B.up(b);
    Cref.up(beta == 0.0f ? std::vector<float>(c0.size(), 0.0f) : c0);
    cublas_sgemm(h, M, N, K, alpha, A.p, B.p, beta, Cref.p);
    // R = |A| |B|
    std::vector<float> aa(a), bb(b);
    for (float& x : aa) x = std::fabs(x);
    for (float& x : bb) x = std::fabs(x);
    Dev<float> AA(aa.size()), BB(bb.size()), RR(c0.size());
    AA.up(aa);
    BB.up(bb);
    cublas_sgemm(h, M, N, K, 1.0f, AA.p, BB.p, 0.0f, RR.p);
    auto ref = Cref.down(), R = RR.down();
    std::vector<float> c0_for_bound = beta == 0.0f ? std::vector<float>(c0.size(), 0.0f) : c0;
    for (int k = 1; k <= gemm::kFp32Kernels; ++k) {
        if (!gemm::supports(k, M, N, K)) continue;
        C.up(c0);
        gemm::launch(k, M, N, K, alpha, A.p, B.p, beta, C.p);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        check(gemm::name(k), M, N, K, C.down(), ref, R, c0_for_bound, alpha, beta, 1.0);
    }
}

static void wmma_case(cublasHandle_t h, int M, int N, int K, unsigned seed) {
    auto a = random_matrix(size_t(M) * K, seed), b = random_matrix(size_t(K) * N, seed + 1);
    Dev<float> A(a.size()), B(b.size());
    A.up(a);
    B.up(b);
    Dev<half> Ah(a.size()), Bh(b.size());
    to_half<<<256, 256>>>(A.p, Ah.p, a.size());
    to_half<<<256, 256>>>(B.p, Bh.p, b.size());
    Dev<float> C(size_t(M) * N), Cref(size_t(M) * N), AA(a.size()), BB(b.size()), RR(size_t(M) * N);
    cublas_hgemm_f32(h, M, N, K, Ah.p, Bh.p, Cref.p);
    half_to_float_abs<<<256, 256>>>(Ah.p, AA.p, a.size());
    half_to_float_abs<<<256, 256>>>(Bh.p, BB.p, b.size());
    cublas_sgemm(h, M, N, K, 1.0f, AA.p, BB.p, 0.0f, RR.p);
    CK(cudaMemset(C.p, 0xff, C.n * sizeof(float)));  // NaN everywhere: every element must be written
    gemm::launch_wmma(M, N, K, Ah.p, Bh.p, C.p);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> zero(C.n, 0.0f);
    // products of halves are exact in fp32, so the same accumulation bound applies; the slack
    // allows for the tensor cores' own order of additions inside each 16 x 16 x 16 step
    check("wmma", M, N, K, C.down(), Cref.down(), RR.down(), zero, 1.0f, 0.0f, 4.0);
}

int main() {
    cublasHandle_t h;
    CB(cublasCreate(&h));
    struct S {
        int M, N, K;
    };
    // tile multiples, odd sizes, sizes just past a tile, one-element and thin cases
    const S sizes[] = {{1, 1, 1},       {7, 5, 3},        {33, 65, 17},     {100, 100, 100}, {128, 128, 128},
                       {257, 129, 64},  {257, 132, 68},   {64, 1024, 4},    {1024, 8, 1024}, {513, 260, 1000},
                       {1024, 1024, 1024}};
    unsigned seed = 1;
    for (S s : sizes) fp32_case(h, s.M, s.N, s.K, 1.0f, 0.0f, seed += 3);
    for (S s : {S{100, 100, 100}, S{257, 132, 68}, S{1024, 1024, 1024}}) fp32_case(h, s.M, s.N, s.K, 1.5f, -0.5f, seed += 3);

    int dev = 0, major = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    if (major >= 7) {
        for (S s : {S{128, 128, 32}, S{256, 384, 96}, S{1024, 1024, 1024}, S{2048, 1024, 512}}) wmma_case(h, s.M, s.N, s.K, seed += 3);
        if (gemm::supports(9, 100, 128, 32) || gemm::supports(9, 128, 128, 48)) {
            std::printf("FAIL wmma claims to support sizes it cannot do\n");
            ++failures;
        }
    } else {
        std::printf("skip wmma: no tensor cores before compute 7.0\n");
    }
    CB(cublasDestroy(h));
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
