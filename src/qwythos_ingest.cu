// src/program/qwythos_ingest.cu - row-batched ingest kernels for the qwen35 serve path.
//
// Every kernel here is a row/column replication of the per-column kernel it replaces in
// serve_ingest_kernels: the per-row (per-column) arithmetic, reduction trees and update
// order are IDENTICAL, so results are bit-equal to the per-column path by construction.
// The stateful pair (conv, step-norm) keeps the GDN state in registers across the B
// tokens and processes tokens strictly in order - the same recurrence the per-column
// launches perform, with the CORRECT s[RPG] state indexing from fused_gdn.cu (an earlier
// batched attempt kept 4 of the 32 rows per thread, which diverged ids - 2026-10-15).
#include <cuda_runtime.h>
#include <cstdint>

#include "strata/kernels/q8_1_finite.hpp"

using strata::kernels::q8_1_finite;
using strata::kernels::q8_1_quant;
using strata::kernels::q8_1_ds;

namespace {
constexpr int S = 128;            // GDN state size
constexpr int RG = 4;             // row groups
constexpr int RPG = S / RG;       // 32 rows per thread

__device__ __forceinline__ float iq_warp_max32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float iq_warp_sum32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

struct Q8OutBlock {
    uint32_t ds;
    int8_t qs[32];
};
static_assert(sizeof(Q8OutBlock) == 36);
}  // namespace

// ---- row_sum_partials: per-column kEmb/4 1-thread blocks; batched grid (kEmb/4, B).
__global__ void iq_row_sum_partials_kernel(const float* __restrict__ x, float* __restrict__ partials, int n) {
    const int row0 = blockIdx.x * 4;
    const int b = blockIdx.y;
    const size_t off = (size_t) b * n;
    if (threadIdx.x == 0) {
        float p = 0.0f;
        for (int i = row0; i < row0 + 4 && i < n; ++i) p += x[off + i] * x[off + i];
        partials[(size_t) b * (n / 4) + blockIdx.x] = p;
    }
}
extern "C" void iq_row_sum_partials(const float* x, float* partials, int n, int B, void* stream) {
    iq_row_sum_partials_kernel<<<dim3(n / 4, B), 1, 0, (cudaStream_t) stream>>>(x, partials, n);
}

// ---- rms_quant_apply: per-column 16 blocks x 256 threads; batched grid dim3(16, B).
// Same per-row partials reduction (shfl_down then 8-lane sum) and grid-stride block walk.
__global__ void iq_rms_quant_rows_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                         const float* __restrict__ partials, unsigned char* __restrict__ y,
                                         int cols, float eps, int q8_stride) {
    const int b = blockIdx.y;
    const float* xr = x + (size_t) b * cols;
    const float* pr = partials + (size_t) b * (cols / 4);
    unsigned char* yr = y + (size_t) b * q8_stride;
    __shared__ float red[8];
    float part = 0.0f;
    for (int p = threadIdx.x; p < cols / 4; p += blockDim.x) part += pr[p];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffff, part, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = part;
    __syncthreads();
    const float ss = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    const float inv = rsqrtf(ss / (float) cols + eps);
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    const int nblk = cols >> 5;
    for (int blk = blockIdx.x * warps + (threadIdx.x >> 5); blk < nblk; blk += gridDim.x * warps) {
        const int base = blk << 5;
        const float v = xr[base + lane] * w[base + lane] * inv;
        const float amax = iq_warp_max32(fabsf(v));
        const float sum = iq_warp_sum32(v);
        const float d = q8_1_finite(amax / 127.0f);
        const int8_t q = q8_1_quant(v, d, amax);
        unsigned char* blkp = yr + (size_t) blk * 36;
        reinterpret_cast<int8_t*>(blkp + 4)[lane] = q;
        if (lane == 0) {
            const __half2 ds = q8_1_ds(d, sum);
            *reinterpret_cast<uint32_t*>(blkp) = *reinterpret_cast<const uint32_t*>(&ds);
        }
    }
}
extern "C" void iq_rms_quant_rows(const float* x, const float* w, const float* partials,
                                  unsigned char* y, int cols, float eps, int q8_stride, void* stream) {
    iq_rms_quant_rows_kernel<<<dim3(16, 8), 256, 0, (cudaStream_t) stream>>>(x, w, partials, y, cols, eps, q8_stride);
}

// ---- beta_gate: per-column 1 block x 64 threads; batched grid (B).
__global__ void iq_beta_gate_rows_kernel(float* __restrict__ beta, float* __restrict__ ggate,
                                         const float* __restrict__ dt, const float* __restrict__ ssm_a,
                                         int n) {
    const int b = blockIdx.x;
    float* beta_c = beta + (size_t) b * 2 * n;
    const float* alpha_c = beta_c + n;
    float* ggate_c = ggate + (size_t) b * n;
    const int i = threadIdx.x;
    if (i < n) {
        beta_c[i] = 1.0f / (1.0f + expf(-beta_c[i]));
        const float v = alpha_c[i] + dt[i];
        ggate_c[i] = (v > 20.0f ? v : log1pf(expf(v))) * ssm_a[i];
    }
}
extern "C" void iq_beta_gate_rows(float* beta, float* ggate, const float* dt, const float* ssm_a,
                                  int n, int B, void* stream) {
    iq_beta_gate_rows_kernel<<<B, 64, 0, (cudaStream_t) stream>>>(beta, ggate, dt, ssm_a, n);
}

// ---- conv+silu+L2: per-column kQkv/128 blocks x 128 threads; batched keeps the conv
// window in registers across the B tokens (channel threads process t sequentially).
__global__ void iq_conv_l2_rows_kernel(float* __restrict__ hist, float* __restrict__ P, int row,
                                       const float* __restrict__ w, int qk_heads, float eps, int B) {
    __shared__ float part[S / 32];
    const int c = blockIdx.x * S + threadIdx.x;
    float h0 = hist[c * 3], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2];
    const bool norm = (int) blockIdx.x < qk_heads;
    for (int t = 0; t < B; ++t) {
        float* pc = P + (size_t) t * row + c;
        const float x = *pc;
        const float sum = h0 * w[c * 4] + h1 * w[c * 4 + 1] + h2 * w[c * 4 + 2] + x * w[c * 4 + 3];
        h0 = h1; h1 = h2; h2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (norm) {
            float sq = y * y;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            y *= rsqrtf(part[0] + part[1] + part[2] + part[3] + eps);
            __syncthreads();
        }
        *pc = y;
    }
    hist[c * 3] = h0; hist[c * 3 + 1] = h1; hist[c * 3 + 2] = h2;
}
extern "C" void iq_conv_l2_rows(float* hist, float* P, int row, const float* w,
                                int channels, int qk_heads, float eps, int B, void* stream) {
    iq_conv_l2_rows_kernel<<<channels / S, S, 0, (cudaStream_t) stream>>>(hist, P, row, w, qk_heads, eps, B);
}

// ---- step+RMS+SiLU+Q8: per-column one block per v-head; batched keeps the state in
// registers across the B tokens (sequential t - same recurrence, same reductions).
__global__ void iq_step_norm_rows_q8_kernel(float* __restrict__ state, const float* __restrict__ P, int row,
                                            const float* __restrict__ gamma, const float* __restrict__ bgate,
                                            const float* __restrict__ ggate,
                                            float* __restrict__ y_out, unsigned char* __restrict__ q8,
                                            int h_k, int h_v, int z_off, float eps, int B, int q8_stride) {
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * S + col;
    const int qh = head % h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < B; ++t) {
        const float* qt = P + (size_t) t * row + qh * S;
        const float* kt = P + (size_t) t * row + h_k * S + qh * S;
        const float* vt = P + (size_t) t * row + 2 * h_k * S + head * S;
        const float* zt = P + (size_t) t * row + z_off + head * S;
        const float* gate = ggate + (size_t) t * h_v + head;
        const float* beta = bgate + (size_t) t * 2 * h_v + head;
        if (tid < S) { sk[tid] = kt[tid]; sq[tid] = qt[tid]; }
        __syncthreads();
        const float g = __expf(gate[0]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (vt[col] - g * kv_col) * beta[0];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = zt[col];
            const float yv = oc * scale * gamma[col] * (zz / (1.0f + expf(-zz)));
            y_out[(size_t) t * h_v * S + head * S + col] = yv;
            const float amax = iq_warp_max32(fabsf(yv));
            const float sum = iq_warp_sum32(yv);
            const float d = q8_1_finite(amax / 127.0f);
            const int8_t qc = q8_1_quant(yv, d, amax);
            unsigned char* q8t = q8 + (size_t) t * q8_stride;
            Q8OutBlock* blk = reinterpret_cast<Q8OutBlock*>(q8t) + (size_t) head * 4 + (col >> 5);
            blk->qs[col & 31] = qc;
            if ((col & 31) == 0) {
                const __half2 ds = q8_1_ds(d, sum);
                blk->ds = *reinterpret_cast<const uint32_t*>(&ds);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}
extern "C" void iq_step_norm_rows_q8(float* state, const float* P, int row, const float* gamma,
                                     const float* bgate, const float* ggate, float* y_out,
                                     unsigned char* q8, int h_k, int h_v, int z_off, float eps,
                                     int B, int q8_stride, void* stream) {
    iq_step_norm_rows_q8_kernel<<<dim3(h_v), dim3(S, RG), 0, (cudaStream_t) stream>>>(
        state, P, row, gamma, bgate, ggate, y_out, q8, h_k, h_v, z_off, eps, B, q8_stride);
}

// ---- split q|gate: per-column grid (heads); batched grid (heads, B).
__global__ void iq_split_qg_rows_kernel(const float* qg, int row, float* q, float* g, int heads, int hd) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int i = threadIdx.x;
    if (i < hd) {
        const float* src = qg + (size_t) t * row + h * (hd * 2);
        q[(size_t) t * heads * hd + (size_t) h * hd + i] = src[i];
        g[(size_t) t * heads * hd + (size_t) h * hd + i] = src[hd + i];
    }
}
extern "C" void iq_split_qg_rows(const float* qg, int row, float* q, float* g, int heads, int hd, int B, void* stream) {
    iq_split_qg_rows_kernel<<<dim3(heads, B), hd, 0, (cudaStream_t) stream>>>(qg, row, q, g, heads, hd);
}

// ---- store kv: per-column one launch with its own meta; batched reads dpos per row.
__global__ void iq_store_kv_rows_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                        float* __restrict__ kcache, float* __restrict__ vcache,
                                        const int* __restrict__ dpos, int n, int stride) {
    const int t = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= 2 * n) return;
    const int pos = dpos[t];
    if (r < n) kcache[(size_t) pos * n + r] = k[(size_t) t * stride + r];
    else vcache[(size_t) pos * n + (r - n)] = v[(size_t) t * stride + (r - n)];
}
extern "C" void iq_store_kv_rows(const float* k, const float* v, float* kcache, float* vcache,
                                 const int* dpos, int n, int stride, int B, void* stream) {
    iq_store_kv_rows_kernel<<<dim3((2 * n + 255) / 256, B), 256, 0, (cudaStream_t) stream>>>(
        k, v, kcache, vcache, dpos, n, stride);
}

// ---- swiglu + q8: per-column one launch over kFF; batched over B*kFF (row = i / kFF;
// kFF is a multiple of 256 so every warp stays inside one row and one q8 block run).
__global__ void iq_swiglu_rows_kernel(const float* __restrict__ ug, unsigned char* __restrict__ y,
                                      int kff, int q8_stride) {
    const int t = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= kff) return;
    const float g = ug[(size_t) t * 2 * kff + kff + j];
    const float v = g / (1.0f + expf(-g)) * ug[(size_t) t * 2 * kff + j];
    const float amax = iq_warp_max32(fabsf(v));
    const float sum = iq_warp_sum32(v);
    const float d = q8_1_finite(amax / 127.0f);
    const int lane = threadIdx.x & 31;
    const int8_t q = q8_1_quant(v, d, amax);
    unsigned char* yt = y + (size_t) t * q8_stride + (size_t)(j >> 5) * 36;
    reinterpret_cast<int8_t*>(yt + 4)[j & 31] = q;
    if (lane == 0) {
        const __half2 ds = q8_1_ds(d, sum);
        *reinterpret_cast<uint32_t*>(yt) = *reinterpret_cast<const uint32_t*>(&ds);
    }
}
extern "C" void iq_swiglu_rows(const float* ug, unsigned char* y, int kff, int B, int q8_stride, void* stream) {
    dim3 grid((kff + 255) / 256, B);
    iq_swiglu_rows_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(ug, y, kff, q8_stride);
}

// ---- o * sigmoid(z) + q8: same row batching over B*kEmb.
__global__ void iq_sig_mul_rows_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                       unsigned char* __restrict__ y, int kemb, int q8_stride) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= kemb) return;
    const int t = blockIdx.y;
    const float v = o[(size_t) t * kemb + j] * (1.0f / (1.0f + expf(-z[(size_t) t * kemb + j])));
    const float amax = iq_warp_max32(fabsf(v));
    const float sum = iq_warp_sum32(v);
    const float d = q8_1_finite(amax / 127.0f);
    const int lane = threadIdx.x & 31;
    const int8_t q = q8_1_quant(v, d, amax);
    unsigned char* yt = y + (size_t) t * q8_stride + (size_t)(j >> 5) * 36;
    reinterpret_cast<int8_t*>(yt + 4)[j & 31] = q;
    if (lane == 0) {
        const __half2 ds = q8_1_ds(d, sum);
        *reinterpret_cast<uint32_t*>(yt) = *reinterpret_cast<const uint32_t*>(&ds);
    }
}
extern "C" void iq_sig_mul_rows(const float* o, const float* z, unsigned char* y, int kemb, int B, int q8_stride, void* stream) {
    dim3 grid((kemb + 255) / 256, B);
    iq_sig_mul_rows_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(o, z, y, kemb, q8_stride);
}

// ---- residual add + partials: per-column kEmb/4 1-thread blocks; batched grid (kEmb/4, B).
__global__ void iq_add_partials_rows_kernel(float* __restrict__ x, const float* __restrict__ o,
                                            float* __restrict__ partials, int n) {
    const int row0 = blockIdx.x * 4;
    const int b = blockIdx.y;
    const size_t off = (size_t) b * n;
    if (threadIdx.x == 0) {
        float p = 0.0f;
        for (int i = row0; i < row0 + 4 && i < n; ++i) {
            const float v = x[off + i] + o[off + i];
            x[off + i] = v;
            p += v * v;
        }
        partials[(size_t) b * (n / 4) + blockIdx.x] = p;
    }
}
extern "C" void iq_add_partials_rows(float* x, const float* o, float* partials, int n, int B, void* stream) {
    iq_add_partials_rows_kernel<<<dim3(n / 4, B), 1, 0, (cudaStream_t) stream>>>(x, o, partials, n);
}

// ---- fill positions: per-column 1-block launches; batched grid (B).
__global__ void iq_fill_pos_rows_kernel(int* __restrict__ p, const int* __restrict__ meta, int n) {
    const int t = blockIdx.x;
    const int i = threadIdx.x;
    if (i < n) p[(size_t) t * n + i] = meta[t];
}
extern "C" void iq_fill_pos_rows(int* dpos, const int* dmeta, int heads, int B, void* stream) {
    iq_fill_pos_rows_kernel<<<B, 32, 0, (cudaStream_t) stream>>>(dpos, dmeta, heads);
}
