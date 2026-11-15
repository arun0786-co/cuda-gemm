// gemm_bench: speed of every kernel and of cuBLAS on square matrices.
//
//   gemm_bench --out results/rtx2080ti
//
// writes gemm.csv (kernel, n, milliseconds, GFLOP/s; the median of repeated runs after a
// warm-up, with the SM clock and board power NVML reports right after each run) and
// kernels.csv (registers, shared memory and blocks per SM of each kernel).

#include <nvml.h>
#include <sys/stat.h>

#include <cctype>
#include <cstring>
#include <string>

#include "common.cuh"
#include "kernels.cuh"

// NVML numbers its GPUs on its own and ignores CUDA_VISIBLE_DEVICES, so the device is
// found by its PCI bus id
static nvmlDevice_t nvml_device() {
    char bus[32];
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
    nvmlDevice_t d{};
    if (nvmlInit() != NVML_SUCCESS || nvmlDeviceGetHandleByPciBusId_v2(bus, &d) != NVML_SUCCESS) {
        std::fprintf(stderr, "NVML unavailable\n");
        std::exit(1);
    }
    return d;
}

struct Timing {
    double ms, clock_mhz, power_w;
};

template <class F>
static Timing measure(nvmlDevice_t dev, int reps, F&& f) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    f();  // warm up
    CK(cudaDeviceSynchronize());
    std::vector<double> t, clk, pw;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms);
        unsigned mhz = 0, mw = 0;  // read straight after the run, while the clock still reflects it
        nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
        nvmlDeviceGetPowerUsage(dev, &mw);
        clk.push_back(mhz);
        pw.push_back(mw / 1000.0);
    }
    CK(cudaGetLastError());
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    return {med(t), med(clk), med(pw)};
}

static std::string slug(std::string s) {
    std::string out;
    for (char c : s)
        if (std::isalnum(static_cast<unsigned char>(c))) out += static_cast<char>(std::tolower(c));
    for (const char* drop : {"nvidia", "geforce"}) {
        auto p = out.find(drop);
        if (p != std::string::npos) out.erase(p, std::strlen(drop));
    }
    return out;
}

int main(int argc, char** argv) {
    std::string out;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        if (!std::strcmp(argv[i], "--quick")) quick = true;
    }
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    if (out.empty()) out = "results/" + slug(p.name);
    mkdir("results", 0755);
    mkdir(out.c_str(), 0755);
    const bool tensor = p.major >= 7;

    FILE* kf = std::fopen((out + "/kernels.csv").c_str(), "w");
    std::fprintf(kf, "kernel,registers,static_shared_bytes,threads,blocks_per_sm,warps_per_sm\n");
    for (int k = 1; k <= 9; ++k) {
        if (k == 9 && !tensor) continue;
        cudaFuncAttributes a = gemm::attributes(k);
        int per_sm = 0;
        switch (k) {  // the occupancy query needs the kernel itself
            case 1: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::naive_rows, gemm::threads(k), 0)); break;
            case 2: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::naive_cols, gemm::threads(k), 0)); break;
            case 3: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::smem_tiled<32>, gemm::threads(k), 0)); break;
            case 4: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::tile_1d<64, 64, 8, 8>, gemm::threads(k), 0)); break;
            case 5: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::tile_2d<128, 128, 8, 8, 8>, gemm::threads(k), 0)); break;
            case 6: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::vector_tiled<false>, gemm::threads(k), 0)); break;
            case 7: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::vector_tiled<true>, gemm::threads(k), 0)); break;
            case 8: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::double_buffered, gemm::threads(k), 0)); break;
            case 9: CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemm::wmma_gemm, gemm::threads(k), 0)); break;
        }
        std::fprintf(kf, "%s,%d,%zu,%d,%d,%d\n", gemm::name(k), a.numRegs, a.sharedSizeBytes, gemm::threads(k), per_sm,
                     per_sm * gemm::threads(k) / 32);
    }
    std::fclose(kf);

    nvmlDevice_t dev = nvml_device();
    cublasHandle_t h;
    CB(cublasCreate(&h));
    FILE* f = std::fopen((out + "/gemm.csv").c_str(), "w");
    std::fprintf(f, "kernel,precision,n,ms,gflops,sm_clock_mhz,power_w\n");
    std::vector<int> sizes = quick ? std::vector<int>{512, 1024} : std::vector<int>{256, 512, 1024, 2048, 4096, 8192};
    for (int n : sizes) {
        const size_t nn = size_t(n) * n;
        Dev<float> A(nn), B(nn), C(nn);
        A.up(random_matrix(nn, 1));
        B.up(random_matrix(nn, 2));
        const double flop = 2.0 * n * n * double(n);
        const int reps = n <= 1024 ? 20 : n <= 4096 ? 7 : 3;
        auto rec = [&](const char* name, const char* prec, Timing t) {
            std::fprintf(f, "%s,%s,%d,%.4f,%.1f,%.0f,%.1f\n", name, prec, n, t.ms, flop / (t.ms * 1e6), t.clock_mhz,
                         t.power_w);
            std::fflush(f);
            std::printf("%6d %-14s %-5s %10.3f ms %9.1f GFLOP/s %5.0f MHz %4.0f W\n", n, name, prec, t.ms,
                        flop / (t.ms * 1e6), t.clock_mhz, t.power_w);
            std::fflush(stdout);
        };
        rec("cublas", "fp32", measure(dev, reps, [&] { cublas_sgemm(h, n, n, n, 1.0f, A.p, B.p, 0.0f, C.p); }));
        for (int k = 1; k <= gemm::kFp32Kernels; ++k) {
            if (k <= 2 && n > 4096) continue;  // the naive kernels take minutes there
            rec(gemm::name(k), "fp32", measure(dev, k <= 2 && n >= 4096 ? 1 : reps,
                                                 [&] { gemm::launch(k, n, n, n, 1.0f, A.p, B.p, 0.0f, C.p); }));
        }
        if (tensor && n >= 256) {
            Dev<half> Ah(nn), Bh(nn);
            to_half<<<256, 256>>>(A.p, Ah.p, nn);
            to_half<<<256, 256>>>(B.p, Bh.p, nn);
            rec("cublas", "fp16", measure(dev, reps, [&] { cublas_hgemm_f32(h, n, n, n, Ah.p, Bh.p, C.p); }));
            rec("wmma", "fp16", measure(dev, reps, [&] { gemm::launch_wmma(n, n, n, Ah.p, Bh.p, C.p); }));
        }
    }
    std::fclose(f);
    CB(cublasDestroy(h));
    nvmlShutdown();
    return 0;
}
