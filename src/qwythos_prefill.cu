#include "strata/kernels/f16_bits.hpp"

constexpr int S_PF = 128;   // head dim (prefill batched kernels)
#include "strata/hip_compat/cuda_runtime.h"

__global__ void rms_rows_f16_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    uint16_t* __restrict__ y, int T, int K, float eps) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const float* row = x + (size_t) t * K;
    uint16_t* out = y + (size_t) t * K;
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = row[i];
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        const int nwarps = blockDim.x >> 5;
        for (int i = 0; i < nwarps; ++i) tot += red[i];
        red[0] = rsqrtf(tot / (float) K + eps);
    }
    __syncthreads();
    const float inv = red[0];
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        out[i] = strata::kernels::f16_from_f32(row[i] * (w ? w[i] : 1.f) * inv);
}

__global__ void add_rows_kernel(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

extern "C" void qwythos_rms_rows_f16(const float* x, const float* w, uint16_t* y, int T, int K, float eps, void* stream) {
    rms_rows_f16_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(x, w, y, T, K, eps);
}

extern "C" void qwythos_add_rows(float* x, const float* y, int n, void* stream) {
    add_rows_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, y, n);
}

__global__ void transpose_nt_kernel(const float* __restrict__ src, float* __restrict__ dst, int T, int N) {
    const int t = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (t < T && n < N) dst[(size_t) t * N + n] = src[(size_t) t * N + n];
}
extern "C" void qwythos_transpose_nt(const float* src, float* dst, int T, int N, void* stream) {
    transpose_nt_kernel<<<dim3(T, (N + 255) / 256), 256, 0, (cudaStream_t) stream>>>(src, dst, T, N);
}

// ==================== batched prefill kernels: T tokens per launch ====================
// qkv3 row layout: [q,g interleaved per head (2*heads*hd)] [k (kv_n)] [v (kv_n)]

__global__ void pf_split_qkv_kernel(const float* __restrict__ P, int row, float* __restrict__ q,
                                    float* __restrict__ z, float* __restrict__ k, float* __restrict__ v,
                                    int heads, int hd, int kv_n, int T) {
    const int qn = heads * hd;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * qn) {
        const int t = i / qn; const int r = i - t * qn;
        const int h = r / hd; const int l = r - h * hd;
        const size_t off = (size_t) t * row + h * 2 * hd;
        q[i] = P[off + l];
        z[i] = P[off + hd + l];
        return;
    }
    const int j = i - T * qn;
    if (j < T * kv_n) {
        const int t = j / kv_n; const int r = j - t * kv_n;
        k[j] = P[(size_t) t * row + 2 * qn + r];
        return;
    }
    const int j2 = j - T * kv_n;
    if (j2 < T * kv_n) {
        const int t = j2 / kv_n; const int r = j2 - t * kv_n;
        v[j2] = P[(size_t) t * row + 2 * qn + kv_n + r];
    }
}
extern "C" void pf_split_qkv(const float* P, int row, float* q, float* z, float* k, float* v,
                             int heads, int hd, int kv_n, int T, void* stream) {
    const int total = T * (heads * hd + 2 * kv_n);
    pf_split_qkv_kernel<<<(total + 255) / 256, 256, 0, (cudaStream_t) stream>>>(P, row, q, z, k, v, heads, hd, kv_n, T);
}

__global__ void pf_store_kv_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                   float* __restrict__ kcache, float* __restrict__ vcache,
                                   const int* __restrict__ dpos, int kv_n, int heads, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 2 * kv_n) return;
    const int t = i / (2 * kv_n);
    const int r = i - t * (2 * kv_n);
    const int pos = dpos[t * heads];
    if (r < kv_n) kcache[(size_t) pos * kv_n + r] = k[t * kv_n + r];
    else vcache[(size_t) pos * kv_n + (r - kv_n)] = v[t * kv_n + (r - kv_n)];
}
extern "C" void pf_store_kv(const float* k, const float* v, float* kcache, float* vcache,
                            const int* dpos, int kv_n, int heads, int T, void* stream) {
    const int total = T * 2 * kv_n;
    pf_store_kv_kernel<<<(total + 255) / 256, 256, 0, (cudaStream_t) stream>>>(k, v, kcache, vcache, dpos, kv_n, heads, T);
}

// Flash-decode attention per (token, head): keys 0..pos0+t.  Same math/order as gqa_head_kernel.
__global__ void pf_gqa_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                              const float* __restrict__ vc, float* __restrict__ o,
                              float* __restrict__ scores, int heads, int n_kv, int hd,
                              const int* dpos, float scale, int pos0, int score_stride) {
    __shared__ float red[8];
    __shared__ float ps[256];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int n_tok = pos0 + t + 1;
    const int kv = h / (heads / n_kv);
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const float* qq = q + (size_t)(t * heads + h) * hd;
    float* sc = scores + (size_t)(t * heads + h) * score_stride;

    for (int tt = warp; tt < n_tok; tt += 8) {
        const float* kk = kc + ((size_t) tt * n_kv + kv) * hd;
        float dot = 0.0f;
        for (int i = lane; i < hd; i += 32) dot += qq[i] * kk[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (lane == 0) sc[tt] = dot * scale;
    }
    __syncthreads();

    float m = -1e30f;
    for (int tt = tid; tt < n_tok; tt += 256) m = fmaxf(m, sc[tt]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    const float gmax = fmaxf(fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])),
                             fmaxf(fmaxf(red[4], red[5]), fmaxf(red[6], red[7])));
    float s = 0.0f;
    for (int tt = tid; tt < n_tok; tt += 256) {
        const float e = expf(sc[tt] - gmax);
        sc[tt] = e;
        s += e;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    const float gsum = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    const float inv = 1.0f / gsum;

    float acc = 0.0f;
    const float* vbase = vc + (size_t) kv * hd + tid;
    for (int t0 = 0; t0 < n_tok; t0 += 256) {
        const int tn = min(256, n_tok - t0);
        if (tid < tn) ps[tid] = sc[t0 + tid];
        __syncthreads();
        for (int tt = 0; tt < tn; ++tt) acc += ps[tt] * vbase[(size_t)(t0 + tt) * n_kv * hd];
        __syncthreads();
    }
    o[(size_t)(t * heads + h) * hd + tid] = acc * inv;
}
extern "C" void pf_gqa(const float* q, const float* kc, const float* vc, float* o, float* scores,
                       int heads, int n_kv, int hd, const int* dpos, float scale, int pos0,
                       int g, int score_stride, void* stream) {
    pf_gqa_kernel<<<dim3(heads, g), 256, 0, (cudaStream_t) stream>>>(
        q, kc, vc, o, scores, heads, n_kv, hd, dpos, scale, pos0, score_stride);
}

// o * sigmoid(z), packed straight to fp16 for the wo GEMM (the batched path's wo input).
__global__ void pf_sig_mul_f16_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                      uint16_t* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = strata::kernels::f16_from_f32(o[i] * (1.0f / (1.0f + expf(-z[i]))));
}
extern "C" void pf_sig_mul_f16(const float* o, const float* z, uint16_t* out, int n, void* stream) {
    pf_sig_mul_f16_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(o, z, out, n);
}

// swiglu then per-row RMS (eps=1e-6, no weight), packed to fp16 - mirrors
// swiglu_kernel + rms_rows_f16(eps=1e-6) exactly, one launch for T rows.
__global__ void pf_swiglu_norm_f16_kernel(const float* __restrict__ R, uint16_t* __restrict__ out,
                                          int kff, float eps, int T) {
    const int t = blockIdx.x;
    const float* g = R + (size_t) t * 2 * kff;
    const float* u = g + kff;
    float ss = 0.f;
    for (int i = threadIdx.x; i < kff; i += blockDim.x) {
        const float gi = g[i];
        const float v = gi / (1.0f + expf(-gi)) * u[i];
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        const int nwarps = blockDim.x >> 5;
        for (int i = 0; i < nwarps; ++i) tot += red[i];
        red[0] = rsqrtf(tot / (float) kff + eps);
    }
    __syncthreads();
    const float inv = red[0];
    uint16_t* o16 = out + (size_t) t * kff;
    for (int i = threadIdx.x; i < kff; i += blockDim.x) {
        const float gi = g[i];
        const float v = gi / (1.0f + expf(-gi)) * u[i];
        o16[i] = strata::kernels::f16_from_f32(v * inv);
    }
}
extern "C" void pf_swiglu_norm_f16(const float* R, uint16_t* out, int kff, float eps, int T, void* stream) {
    pf_swiglu_norm_f16_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(R, out, kff, eps, T);
}

// GDN beta|alpha gate over T rows of R ([beta|alpha] per row) -> bgate ([beta][gate] per token).
__global__ void pf_beta_gate_kernel(const float* __restrict__ R, float* __restrict__ bgate,
                                    const float* __restrict__ dt, const float* __restrict__ ssm_a,
                                    int hv, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * hv) return;
    const int t = i / hv; const int h = i - t * hv;
    const float b = R[(size_t) t * 2 * hv + h];
    const float a = R[(size_t) t * 2 * hv + hv + h];
    bgate[i] = 1.0f / (1.0f + expf(-b));
    const float v = a + dt[h];
    bgate[T * hv + i] = (v > 20.0f ? v : log1pf(expf(v))) * ssm_a[h];
}
extern "C" void pf_beta_gate(const float* R, float* bgate, const float* dt, const float* ssm_a,
                             int hv, int T, void* stream) {
    pf_beta_gate_kernel<<<(T * hv + 63) / 64, 64, 0, (cudaStream_t) stream>>>(R, bgate, dt, ssm_a, hv, T);
}

// conv+SiLU+L2 over T tokens: one thread per channel, conv window in registers.
// Same math/order as gdn_conv_l2_qk_kernel per token; hist store folded to the end.
__global__ void pf_conv_l2_qk_kernel(float* __restrict__ hist, float* __restrict__ P, int row,
                                     const float* __restrict__ w, int channels, int qk_heads,
                                     float eps, int T) {
    __shared__ float part[S_PF / 32];
    const int c = blockIdx.x * S_PF + threadIdx.x;
    if (c >= channels) return;
    float h0 = hist[c * 3], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2];
    const bool norm = (int) blockIdx.x < qk_heads;
    for (int t = 0; t < T; ++t) {
        const float x = P[(size_t) t * row + c];
        const float sum = h0 * w[c * 4] + h1 * w[c * 4 + 1] + h2 * w[c * 4 + 2] + x * w[c * 4 + 3];
        h0 = h1; h1 = h2; h2 = x;
        float y = sum / (1.0f + expf(-sum));
        if (norm) {
            float sq = y * y;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            y *= rsqrtf(part[0] + part[1] + part[2] + part[3] + eps);
            __syncthreads();
        }
        P[(size_t) t * row + c] = y;
    }
    hist[c * 3] = h0; hist[c * 3 + 1] = h1; hist[c * 3 + 2] = h2;
}
extern "C" void pf_conv_l2_qk(float* hist, const float* P, int row, const float* w,
                              int channels, int qk_heads, float eps, int T, void* stream) {
    pf_conv_l2_qk_kernel<<<channels / S_PF, S_PF, 0, (cudaStream_t) stream>>>(hist, (float*) P, row, w, channels, qk_heads, eps, T);
}

// step+RMS+SiLU over T tokens: one block per v-head, the GDN state lives in registers
// across the whole chunk (same update order as gdn_step_norm_kernel per token - bit-exact).
__global__ void pf_step_norm_silu_kernel(float* __restrict__ state, const float* __restrict__ P, int row,
                                         const float* __restrict__ gamma, const float* __restrict__ bgate,
                                         float* __restrict__ S, int hk, int hv, int z_off,
                                         float eps, int T) {
    __shared__ float sk[S_PF], sq[S_PF];
    __shared__ float red_kv[4][S_PF];
    __shared__ float red_o[4][S_PF];
    __shared__ float wsum[S_PF * 4 / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * S_PF + col;
    const int qh = head % hk;
    float s[4];
    float* base = state + ((size_t) (rg * 32) * hv + head) * S_PF + col;
    const size_t row_stride = (size_t) hv * S_PF;
#pragma unroll
    for (int r = 0; r < 4; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* qt = P + (size_t) t * row + qh * S_PF;
        const float* kt = P + (size_t) t * row + hk * S_PF + qh * S_PF;
        const float* vt = P + (size_t) t * row + 2 * hk * S_PF + head * S_PF;
        const float* zt = P + (size_t) t * row + z_off + head * S_PF;
        const float* gate = bgate + T * hv + t * hv + head;
        const float* beta = bgate + t * hv + head;
        if (tid < S_PF) { sk[tid] = kt[tid]; sq[tid] = qt[tid]; }
        __syncthreads();
        const float g = expf(gate[0]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) kv = fmaf(s[r], sk[rg * 32 + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (vt[col] - g * kv_col) * beta[0];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * 32 + r] * delta);
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) o = fmaf(s[r], sq[rg * 32 + r], o);
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S_PF);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S_PF + eps);
            const float zz = zt[col];
            const float yv = oc * scale * gamma[col] * (zz / (1.0f + expf(-zz)));
            S[(size_t) t * hv * S_PF + head * S_PF + col] = yv;
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) base[r * row_stride] = s[r];
}
extern "C" void pf_step_norm_silu(float* state, const float* P, int row, const float* gamma,
                                  const float* bgate, float* S, int hk, int hv, int z_off,
                                  float eps, int T, void* stream) {
    pf_step_norm_silu_kernel<<<dim3(hv, 4), dim3(128, 4), 0, (cudaStream_t) stream>>>(
        state, P, row, gamma, bgate, S, hk, hv, z_off, eps, T);
}
