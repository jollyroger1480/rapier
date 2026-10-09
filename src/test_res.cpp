// Parity + fault test for the new Q5_K/Q4_K residual-epilogue kernels.
// Compares: res-kernel output vs (non-res kernel output + host residual add),
// and the multi-column verify path vs two single-column calls.
// Synthetic tensors, small enough for the 4 GB 6500 XT. No model required.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "strata/kernels/native_mmvq.hpp"

#define CK(x) do { auto e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

static const int N_IN = 4096;           // 16 Q5_K/Q4_K blocks per row
static const int N_OUT = 256;
static const int ROWS_PER_BLOCK = 4;

int main() {
    CK(hipInit(0));
    int dev = 0; CK(hipSetDevice(dev));   // sacrificial 6500 XT
    hipStream_t st; CK(hipStreamCreate(&st));
    srand(7);

    // Q5_K: 176 B per 256; Q4_K: 144 B per 256
    const size_t q5_row = (size_t)(N_IN / 256) * 176;
    const size_t q4_row = (size_t)(N_IN / 256) * 144;
    std::vector<unsigned char> w5((size_t)N_OUT * q5_row), w4((size_t)N_OUT * q4_row);
    for (auto& b : w5) b = (unsigned char)(rand() & 0xff);
    for (auto& b : w4) b = (unsigned char)(rand() & 0xff);
    // activations: one Q8_1 row (36 B per 32) x N_IN/32 blocks; x2 = two columns
    const size_t q81_row = (size_t)(N_IN / 32) * 36;
    std::vector<unsigned char> x1(q81_row), x2(2 * q81_row);
    for (auto& b : x1) b = (unsigned char)(rand() & 0xff);
    std::memcpy(x2.data(), x1.data(), q81_row);
    for (size_t i = q81_row; i < x2.size(); ++i) x2[i] = (unsigned char)(rand() & 0xff);
    std::vector<float> res(N_OUT);
    for (auto& v : res) v = (float)(rand() % 1000) / 1000.0f;

    void *dw5, *dw4, *dx1, *dx2, *dres, *dpart;
    float *dy, *dy2;
    CK(hipMalloc(&dw5, w5.size())); CK(hipMalloc(&dw4, w4.size()));
    CK(hipMalloc(&dx1, x1.size())); CK(hipMalloc(&dx2, x2.size()));
    CK(hipMalloc(&dy, (size_t)N_OUT * 4)); CK(hipMalloc(&dy2, (size_t)N_OUT * 4 * 2));
    CK(hipMalloc(&dres, (size_t)N_OUT * 4));
    CK(hipMalloc(&dpart, (size_t)(N_OUT / ROWS_PER_BLOCK + 4) * 4));
    CK(hipMemcpy(dw5, w5.data(), w5.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dw4, w4.data(), w4.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dx1, x1.data(), x1.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dx2, x2.data(), x2.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dres, res.data(), res.size(), hipMemcpyHostToDevice));
    CK(hipDeviceSynchronize());

    std::vector<float> y_ref(N_OUT), y_res(N_OUT), y_multi(N_OUT), y_multi1(N_OUT);

    // --- Q5_K: non-res reference (col0), then the res kernel on the same column
    strata::kernels::native_q5_k_mmvq(dw5, dx1, dy, N_IN, N_OUT, 1, st);
    CK(hipMemcpy(y_ref.data(), dy, (size_t)N_OUT * 4, hipMemcpyDeviceToHost));
    CK(hipMemset(dpart, 0, (size_t)(N_OUT / ROWS_PER_BLOCK + 4) * 4));
    strata::kernels::native_q5_k_mmvq_res(dw5, dx1, dy, (const float*)dres, (float*)dpart, N_IN, N_OUT, st);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(y_res.data(), dy, (size_t)N_OUT * 4, hipMemcpyDeviceToHost));
    int bad5 = 0; double maxd5 = 0;
    for (int i = 0; i < N_OUT; ++i) {
        const float want = y_ref[i] + res[i];
        const double d = std::fabs((double)want - (double)y_res[i]);
        if (d > 1e-3 && std::fabs(want) > 1e-6) { if (bad5 < 3) std::printf("Q5 res mismatch row %d: want %g got %g\n", i, want, y_res[i]); ++bad5; }
        if (d > maxd5) maxd5 = d;
    }
    std::printf("Q5 res kernel: %d mismatches, max delta %g\n", bad5, maxd5);

    // --- Q5_K: multi-column (ncols=2) vs two single-column calls
    strata::kernels::native_q5_k_mmvq(dw5, dx2, dy, N_IN, N_OUT, 1, st);
    CK(hipMemcpy(y_multi1.data(), dy, (size_t)N_OUT * 4, hipMemcpyDeviceToHost));
    strata::kernels::native_q5_k_mmvq(dw5, dx1, dy2, N_IN, N_OUT, 1, st);
    strata::kernels::native_q5_k_mmvq(dw5, (const char*)dx2 + q81_row, (float*)((char*)dy2 + N_OUT * 4), N_IN, N_OUT, 1, st);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(y_multi.data(), dy2, (size_t)N_OUT * 2 * 4, hipMemcpyDeviceToHost));
    strata::kernels::native_q5_k_mmvq(dw5, dx2, dy2, N_IN, N_OUT, 2, st);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(y_multi1.data(), dy2, (size_t)N_OUT * 2 * 4, hipMemcpyDeviceToHost));
    int badm = 0;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < N_OUT; ++i) {
            const float a = y_multi[c * N_OUT + i];
            const float b = y_multi1[c * N_OUT + i];
            if (std::fabs((double)a - (double)b) > 1e-3) { if (badm < 3) std::printf("Q5 multi col%d row %d: single %g multi %g\n", c, i, b, a); ++badm; }
        }
    std::printf("Q5 multi(2col) vs single: %d mismatches\n", badm);

    // --- Q4_K res kernel
    strata::kernels::native_q4_k_mmvq(dw4, dx1, dy, N_IN, N_OUT, 1, st);
    CK(hipMemcpy(y_ref.data(), dy, (size_t)N_OUT * 4, hipMemcpyDeviceToHost));
    CK(hipMemset(dpart, 0, (size_t)(N_OUT / ROWS_PER_BLOCK + 4) * 4));
    strata::kernels::native_q4_k_mmvq_res(dw4, dx1, dy, (const float*)dres, (float*)dpart, N_IN, N_OUT, st);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(y_res.data(), dy, (size_t)N_OUT * 4, hipMemcpyDeviceToHost));
    int bad4 = 0; double maxd4 = 0;
    for (int i = 0; i < N_OUT; ++i) {
        const float want = y_ref[i] + res[i];
        const double d = std::fabs((double)want - (double)y_res[i]);
        if (d > 1e-3 && std::fabs(want) > 1e-6) { if (bad4 < 3) std::printf("Q4 res mismatch row %d: want %g got %g\n", i, want, y_res[i]); ++bad4; }
        if (d > maxd4) maxd4 = d;
    }
    std::printf("Q4 res kernel: %d mismatches, max delta %g\n", bad4, maxd4);

    // --- partials sanity: block 0's partial = sum of squares of rows 0..3 of y_ref+res
    float part0 = 0; CK(hipMemcpy(&part0, dpart, 4, hipMemcpyDeviceToHost));
    float want0 = 0;
    for (int i = 0; i < 4; ++i) { const float v = y_ref[i] + res[i]; want0 += v * v; }
    std::printf("Q5 partials[0]: got %g want %g\n", part0, want0);
    std::printf("TEST DONE\n");

    // ---- model-shape sweep: every distinct GEMV shape the Q5 file exercises ----
    struct Shape { int n_in, n_out, type; const char* name; };
    Shape shapes[] = {
        {4096, 8192, 13, "upgate"},
        {12288, 4096, 14, "ffn_down (q6 res)"},
        {4096, 4096, 13, "ssm_out/attn_output"},
        {8192, 4096, 13, "eh_proj"},
    };
    y_ref.resize(32768); y_res.resize(32768); y_multi.resize(32768); y_multi1.resize(32768);
    for (const Shape& sh : shapes) {
        const bool q6 = (sh.type == 14);
        const size_t row = (size_t)(sh.n_in / 256) * (q6 ? 210 : (sh.type == 13 ? 176 : 144));
        std::vector<unsigned char> w((size_t)sh.n_out * row);
        for (auto& b : w) b = (unsigned char)(rand() & 0xff);
        std::vector<float> rr(sh.n_out);
        for (auto& v : rr) v = (float)(rand() % 997) / 997.0f;
        void* dw; float* dy_; void* dr; float* dp;
        if (hipMalloc(&dw, w.size()) || hipMalloc(&dy_, (size_t)sh.n_out * 4) ||
            hipMalloc(&dr, (size_t)sh.n_out * 4) || hipMalloc(&dp, (size_t)sh.n_out / 2 * 4 + 64)) {
            std::printf("%s: alloc failed\n", sh.name); continue;
        }
        CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dr, rr.data(), (size_t)sh.n_out * 4, hipMemcpyHostToDevice));
        CK(hipMemset(dp, 0, (size_t)sh.n_out / 2 * 4 + 64));
        if (q6) strata::kernels::native_q6_k_mmvq(dw, dx1, dy_, sh.n_in, sh.n_out, 1, st);
        else if (sh.type == 13) strata::kernels::native_q5_k_mmvq(dw, dx1, dy_, sh.n_in, sh.n_out, 1, st);
        else strata::kernels::native_q4_k_mmvq(dw, dx1, dy_, sh.n_in, sh.n_out, 1, st);
        CK(hipMemcpy(y_ref.data(), dy_, (size_t)sh.n_out * 4, hipMemcpyDeviceToHost));
        if (q6) strata::kernels::native_q6_k_mmvq_res(dw, dx1, dy_, (const float*)dr, (float*)dp, sh.n_in, sh.n_out, st);
        else if (sh.type == 13) strata::kernels::native_q5_k_mmvq_res(dw, dx1, dy_, (const float*)dr, (float*)dp, sh.n_in, sh.n_out, st);
        else strata::kernels::native_q4_k_mmvq_res(dw, dx1, dy_, (const float*)dr, (float*)dp, sh.n_in, sh.n_out, st);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(y_res.data(), dy_, (size_t)sh.n_out * 4, hipMemcpyDeviceToHost));
        int bad = 0; float maxd = 0;
        for (int i = 0; i < sh.n_out; ++i) {
            const float want = y_ref[i] + rr[i];
            const float d = std::fabs(want - y_res[i]);
            if (d > 1e-2f) { if (bad < 2) std::printf("  %s row %d: want %g got %g\n", sh.name, i, want, y_res[i]); ++bad; }
            if (d > maxd) maxd = d;
        }
        std::printf("%-24s: %d mismatches, maxd %g\n", sh.name, bad, maxd);
        CK(hipFree(dw)); CK(hipFree(dy_)); CK(hipFree(dr)); CK(hipFree(dp));
    }
    return 0;
}
