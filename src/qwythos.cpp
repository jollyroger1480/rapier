// Qwythos-9B-v2 trunk decode on the HIP build.
// 24 gated-delta layers and 8 full-attention layers, then the Q6_K LM head.
// Block 32 (MTP) is not run. Default is the largest GPU only. --split is optional.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_rope.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kEmb = 4096;
constexpr int kFF = 12288;
constexpr int kVocab = 248320;
constexpr int kLayers = 32;
constexpr int kS = 128;
constexpr int kHk = 16;
constexpr int kHv = 32;
constexpr int kQkv = 8192;
constexpr int kHd = 256;
constexpr int kHeads = 16;
constexpr int kKv = 4;
constexpr int kRot = 64;
constexpr int kMaxCtx = 16384;
// set at parse time (--serve); silences stdout chatter that would corrupt the JSON protocol
static bool g_serve_mode = false;
constexpr float kInvSqrtS = 0.0883883476f; // 1/sqrt(128)
constexpr float kAttnScale = 0.0625f;      // 1/sqrt(256)

#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #e, cudaGetErrorString(_e)); std::exit(1); } } while (0)

size_t row_bytes(int type, int n_in) {
    if (type == 0) return (size_t) n_in * 4;
    if (type == 8) return (size_t) (n_in / 32) * 34;
    if (type == 14) return (size_t) (n_in / 256) * 210;
    return 0;
}

struct Tensor {
    int type = 0;
    int n_in = 0;
    int n_out = 1;
    const uint8_t* data = nullptr;
};

struct Mat {
    int type = 0;
    int n_in = 0;
    int n_out = 0;
    int n_a = 0;
    void* a = nullptr;
    void* b = nullptr;
};

struct Layer {
    bool recr = true;
    Mat attn_norm, post_norm;
    Mat qkv, z_w, alpha_w, beta_w, ssm_out;
    Mat up, gate, down;
    Mat wq, wk, wv, wo, qn, kn;
    // concatenated projections (big-card path only): one GEMV launch per group instead of 2-4
    Mat qkvz, ab, upgate, qkv3;
    bool cat_qkvz = false, cat_ab = false, cat_upgate = false, cat_qkv3 = false;
    float* conv_w = nullptr;
    float* ssm_a = nullptr;
    float* ssm_dt = nullptr;
    float* ssm_norm = nullptr;
    float* conv_state = nullptr;
    float* gdn_state = nullptr;
    float* kcache = nullptr;
    float* vcache = nullptr;
};

struct Engine {
    int primary = 1;
    int secondary = 0;
    float split = 0;
    float eps = 1e-6f;
    bool peer = false;
    cudaStream_t sp = nullptr;
    cudaStream_t ss = nullptr;
    cudaEvent_t ev_x = nullptr;
    cudaEvent_t ev_s = nullptr;
    cudaEvent_t ev_layer = nullptr;
    cudaEvent_t ev_done = nullptr;
    cudaGraphExec_t graph = nullptr;
    int* hmeta = nullptr;
    int* dmeta = nullptr;
    double layer_ms = 0;
    cudaEvent_t* seg_b = nullptr;
    cudaEvent_t* seg_e = nullptr;
    char* seg_tag = nullptr;
    int seg_n = 0;
    int seg_cap = 0;
    void* scratch_p = nullptr;
    void* scratch_s = nullptr;
    float* x = nullptr;
    float* x_s = nullptr;
    float* y_s = nullptr;
    float* res = nullptr;
    float* o = nullptr;
    float* qkv = nullptr;
    float* z = nullptr;
    float* up = nullptr;
    float* gate = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* beta = nullptr;
    float* alpha = nullptr;
    float* ggate = nullptr;
    float* logits = nullptr;
    float* ascores = nullptr;
    float* hlogits = nullptr;
    float* hrow = nullptr;
    float* qkvz_o = nullptr;   // qkv‖z GEMV output (12288)
    float* ab_o = nullptr;     // alpha‖beta GEMV output (64)
    float* ug_o = nullptr;     // up‖gate GEMV output (24576)
    float* q3_o = nullptr;     // wq‖wk‖wv GEMV output (10240)
    float* partials = nullptr; // kEmb/4 row sum-of-squares partials (GEMV epilogue layout)
    void* scratch_up8 = nullptr;  // Q8_1 of the 4096-wide ssm_out/wo input
    void* scratch_dn8 = nullptr;  // Q8_1 of the 12288-wide ffn down input
    int* dpos = nullptr;
    int seed_pos = 0;
    float thresh = 0.0f;
    // MTP speculative decode: draft = blk.32 (an attention-type layer) + the nextn glue, and a
    // 2-column batched verify pass whose GEMVs stream the weights once for both tokens.
    Layer d32;
    Mat eh_proj, hnorm, enorm, shnorm;
    float* kcache32 = nullptr;
    float* vcache32 = nullptr;
    float* emb_row = nullptr;      // host pinned: dequantized embedding row for the draft input
    float* emb_row2 = nullptr;     // host pinned: BOTH verify embedding rows (one H2D copy, no race)
    float* h_zero = nullptr;       // draft position-0 input: zero hidden (shift-right pad)
    float* h_normed = nullptr;     // FINAL-NORMED trunk hidden (what llama.cpp feeds the MTP: t_h_nextn)
    float* hn_partials = nullptr;  // its sum-of-squares partials
    float* p_zero = nullptr;       // matching zero partials
    float* emb_dev = nullptr;      // device copy of the embedding row
    float* d_partials = nullptr;   // draft-internal row sum-of-squares partials
    float* eh_cat = nullptr;       // concat(hnorm(h_prev), enorm(emb)) = 2*kEmb
    float* eh_cat_q8 = nullptr;    // Q8_1 of the concat (8192)
    float* d_x = nullptr;          // draft layer hidden (kEmb)
    float* d_up = nullptr;         // draft ffn up (kFF)
    float* d_gate = nullptr;
    float* d_o = nullptr;
    float* d_qkv = nullptr;        // draft qkv+g (kQkv)
    float* d_z = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_up8 = nullptr;        // Q8 of draft out_norm output (wo/... here: ssm-less layer)
    float* d_ugq8 = nullptr;       // Q8 of draft ffn down input (kFF)
    float* d_logits = nullptr;     // draft lm_head output (kVocab)
    float* hlogits2 = nullptr;     // pinned host copy of both verify logits rows
    // 2-column verify buffers (column stride = size)
    float* x2 = nullptr;
    float* o2 = nullptr;
    float* qkvz_o2 = nullptr;
    float* ab_o2 = nullptr;
    float* ug_o2 = nullptr;
    float* q3_o2 = nullptr;
    float* up2 = nullptr;          // step_norm / gqa output, 2*kFF
    float* partials2 = nullptr;    // 2 * kEmb/4
    void* scratch_p2 = nullptr;    // 2 * Q8(kEmb)
    void* scratch_up8_2 = nullptr; // 2 * Q8(kEmb)
    void* scratch_dn8_2 = nullptr; // 2 * Q8(kFF)
    float* logits2 = nullptr;      // 2 * kVocab
    float* gdn_snap = nullptr;     // 24 * (S*hv*S) state rollback snapshots
    float* gdn_pre = nullptr;      // CMPV diagnostic: full state before a verify
    int vstop = -1;                // CMPV diagnostic: layer limit for verify/single paths
    float* conv_pre = nullptr;
    float* conv_snap = nullptr;    // 24 * 3*kQkv
    float* h_save = nullptr;       // trunk hidden(s) for the draft: [h_pos, h_pos1]
    float* h_partials2 = nullptr;  // matching partials for both hidden columns
    float* z2 = nullptr;           // attention g/z, 2*kEmb
    float* ggate2 = nullptr;       // draft gates, 2*kHv
    int* dmeta1 = nullptr;         // col1 position
    int* hmeta1 = nullptr;
    const uint8_t* emb_host = nullptr;
    size_t emb_stride = 0;
    int max_tail = 0;
    Mat out_w, out_n;
    Layer layers[kLayers];
    strata::kernels::RopeScaling rope;
};

void ck_launch(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void swiglu_kernel(float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = g / (1.0f + expf(-g));
    gate[i] = s * up[i];
}

__global__ void sig_mul_kernel(float* o, const float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= 1.0f / (1.0f + expf(-g[i]));
}

__global__ void split_qg_kernel(const float* qg, float* q, float* g, int heads, int hd) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (h < heads && i < hd) {
        q[h * hd + i] = qg[h * (hd * 2) + i];
        g[h * hd + i] = qg[h * (hd * 2) + hd + i];
    }
}

__global__ void fill_pos_kernel(int* p, int n, const int* meta) {
    int i = threadIdx.x;
    if (i < n) p[i] = meta[0];
}

__global__ void store_kv_kernel(float* cache, const float* src, const int* meta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cache[(size_t) meta[0] * n + i] = src[i];
}

// k and v land in their caches with one launch (k occupies the first kv_n lanes).
__global__ void store_kv2_kernel(float* __restrict__ kcache, float* __restrict__ vcache,
                                 const float* __restrict__ k, const float* __restrict__ v, const int* meta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) kcache[(size_t) meta[0] * n + i] = k[i];
    else if (i < 2 * n) vcache[(size_t) meta[0] * n + (i - n)] = v[i - n];
}

__global__ void gqa_kernel(const float* q, const float* kc, const float* vc, float* o, float* scores,
                           int n_head, int n_kv, int hd, const int* meta, float scale) {
    int h = threadIdx.x;
    if (h >= n_head) return;
    int n_tok = meta[0] + 1;
    int kv = h / (n_head / n_kv);
    const float* qq = q + (size_t) h * hd;
    float* sc = scores + (size_t) h * n_tok;
    float maxs = -1e30f;
    for (int t = 0; t < n_tok; ++t) {
        const float* kk = kc + ((size_t) t * n_kv + kv) * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; ++i) dot += qq[i] * kk[i];
        float s = dot * scale;
        sc[t] = s;
        maxs = fmaxf(maxs, s);
    }
    float sum = 0.0f;
    for (int t = 0; t < n_tok; ++t) {
        float e = expf(sc[t] - maxs);
        sc[t] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < hd; ++i) {
        float acc = 0.0f;
        const float* col = vc + (size_t) kv * hd + i;
        for (int t = 0; t < n_tok; ++t)
            acc += sc[t] * col[((size_t) t * n_kv) * hd];
        o[(size_t) h * hd + i] = acc * inv;
    }
}

// Flash-decode style GQA: one block per head, 256 threads.  The old kernel ran the whole layer on ONE warp, so
// per-token cost grew with context until attention dominated the step (gpu_ms rose from ~18 ms toward 80+ over
// a 400-token generation).  Same math; reductions are parallel trees with fixed order, so results stay
// deterministic run to run (fp accumulation order differs from the single-warp version at the last-bit level).
__global__ void __launch_bounds__(256) gqa_head_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                                       const float* __restrict__ vc, float* __restrict__ o,
                                                       float* __restrict__ scores, int n_head, int n_kv, int hd,
                                                       const int* meta, float scale) {
    __shared__ float red[8];
    __shared__ float ps[256];
    const int h = blockIdx.x;
    const int n_tok = meta[0] + 1;
    const int kv = h / (n_head / n_kv);
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const float* qq = q + (size_t) h * hd;
    float* sc = scores + (size_t) h * n_tok;

    // scores: warp per token stripe, lanes strided over hd, xor tree within the warp
    for (int t = warp; t < n_tok; t += 8) {
        const float* kk = kc + ((size_t) t * n_kv + kv) * hd;
        float dot = 0.0f;
        for (int i = lane; i < hd; i += 32) dot += qq[i] * kk[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (lane == 0) sc[t] = dot * scale;
    }
    __syncthreads();

    // softmax over n_tok: block max then block sum, fixed two-stage trees
    float m = -1e30f;
    for (int t = tid; t < n_tok; t += 256) m = fmaxf(m, sc[t]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    const float gmax = fmaxf(fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])),
                             fmaxf(fmaxf(red[4], red[5]), fmaxf(red[6], red[7])));
    float s = 0.0f;
    for (int t = tid; t < n_tok; t += 256) {
        const float e = expf(sc[t] - gmax);
        sc[t] = e;
        s += e;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    const float gsum = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    const float inv = 1.0f / gsum;

    // weighted V: thread per output element, probabilities staged through shared, V reads coalesced
    float acc = 0.0f;
    const float* vbase = vc + (size_t) kv * hd + tid;
    for (int t0 = 0; t0 < n_tok; t0 += 256) {
        const int tn = min(256, n_tok - t0);
        if (tid < tn) ps[tid] = sc[t0 + tid];
        __syncthreads();
        for (int t = 0; t < tn; ++t) acc += ps[t] * vbase[(size_t)(t0 + t) * n_kv * hd];
        __syncthreads();
    }
    o[(size_t) h * hd + tid] = acc * inv;
}

struct Q81Row {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Row) == 36, "q8_1");

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off, 32);
    return v;
}

__device__ __forceinline__ float warp_max32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off, 32));
    return v;
}

// Same tree as rms_norm_weighted: shfl_down, rsqrt on lane 0, then broadcast.
// An xor-sum computed on every lane disagrees in the last bit and drifts tokens.
__device__ __forceinline__ float warp_sum_down(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

static bool g_vtrace = false;
static float g_vtrace_buf[16][6] = {};
static int g_vtrace_n = 0;
static int g_vtrace_layer = -1;
static void vtrace(int slot, const float* dev) {
    if (!g_vtrace || g_vtrace_layer != 0 || slot >= 16) return;
    float v = 0.0f;
    cudaMemcpy(&v, dev, 4, cudaMemcpyDeviceToHost);
    g_vtrace_buf[slot][g_vtrace_n] = v;
}

// The residual add and the row's sum-of-squares partials now ride the GEMV epilogue (mm_q_res): 1024 fixed
// partial slots cover the 4096-wide row.  row_sum_partials seeds the same layout for the fresh embedding row.
static const bool old_gdn = std::getenv("QWYTHOS_OLDGDN") != nullptr;  // diagnostic: pre-fusion GDN chain

__global__ void add_partials_1024_kernel(float* __restrict__ x, const float* __restrict__ o,
                                         float* __restrict__ partials, int n) {
    const int row0 = blockIdx.x * 4;
    if (threadIdx.x == 0) {
        float p = 0.0f;
        for (int i = row0; i < row0 + 4 && i < n; ++i) {
            const float v = x[i] + o[i];
            x[i] = v;
            p += v * v;
        }
        partials[blockIdx.x] = p;
    }
}

__global__ void row_sum_partials_kernel(const float* __restrict__ x, float* __restrict__ partials, int n) {
    const int row0 = blockIdx.x * 4;
    float p = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = row0; i < row0 + 4 && i < n; ++i) p += x[i] * x[i];
        partials[blockIdx.x] = p;
    }
}

// One warp per Q8 block: v = x * w * inv quantized exactly like native_quantize_q8_1 (xor-tree amax/sum over
// the 32 consecutive lanes, d = amax/127, roundf).  x is NOT rewritten: the caller keeps the pre-norm row.
__global__ void rms_quant_apply_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                       const float* __restrict__ partials, Q81Row* __restrict__ y, int cols,
                                       float eps) {
    // kEmb/4 fixed partial slots, written by the preceding GEMV residual epilogue (or the seed kernel).
    // Block-level reduction: each thread strides the partials, then a fixed two-stage tree - same order in
    // every block, so the result is deterministic.
    __shared__ float red[8];
    float part = 0.0f;
    for (int p = threadIdx.x; p < kEmb / 4; p += blockDim.x) part += partials[p];
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
        const float v = x[base + lane] * w[base + lane] * inv;
        const float amax = warp_max32(fabsf(v));
        const float sum = warp_sum32(v);
        const float d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
        y[blk].qs[lane] = q;
        if (lane == 0) y[blk].ds = make_half2(d, sum);
    }
}

// RMS-normalize a row into an FP32 output buffer (draft eh_proj halves): out[i] = x[i] * w[i] * inv.
__global__ void rms_norm_out_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    const float* __restrict__ partials, float* __restrict__ out, int cols,
                                    float eps) {
    __shared__ float red[8];
    float part = 0.0f;
    for (int p = threadIdx.x; p < kEmb / 4; p += blockDim.x) part += partials[p];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffff, part, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = part;
    __syncthreads();
    const float ss = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    const float inv = rsqrtf(ss / (float) cols + eps);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) out[i] = x[i] * w[i] * inv;
}

// swiglu + Q8_1 quantize in one pass: v = silu(gate) * up, one warp per 32-column block.  The float product is
// consumed only by the down GEMV, which reads the Q8 blocks, so no float output is written at all.
__global__ void swiglu_quant_kernel(const float* __restrict__ up, const float* __restrict__ gate_w,
                                    Q81Row* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate_w[i];
    const float v = g / (1.0f + expf(-g)) * up[i];
    const float amax = warp_max32(fabsf(v));
    const float sum = warp_sum32(v);
    const float d = amax / 127.0f;
    const int lane = threadIdx.x & 31;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
    y[i >> 5].qs[lane] = q;
    if (lane == 0) y[i >> 5].ds = make_half2(d, sum);
}

// o *= sigmoid(z) + Q8_1 quantize, one warp per 32-column block (the wo GEMV input).
__global__ void sig_mul_quant_kernel(const float* __restrict__ o, const float* __restrict__ g,
                                     Q81Row* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = o[i] * (1.0f / (1.0f + expf(-g[i])));
    const float amax = warp_max32(fabsf(v));
    const float sum = warp_sum32(v);
    const float d = amax / 127.0f;
    const int lane = threadIdx.x & 31;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
    y[i >> 5].qs[lane] = q;
    if (lane == 0) y[i >> 5].ds = make_half2(d, sum);
}

// beta = sigmoid(beta_proj) and gate = softplus(alpha + dt) * ssm_a in one launch (two 32-wide vectors).
__global__ void beta_gate_kernel(float* __restrict__ beta, const float* __restrict__ alpha,
                                 const float* __restrict__ dt, const float* __restrict__ ssm_a,
                                 float* __restrict__ ggate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    beta[i] = 1.0f / (1.0f + expf(-beta[i]));
    const float v = alpha[i] + dt[i];
    ggate[i] = (v > 20.0f ? v : log1pf(expf(v))) * ssm_a[i];
}

// DRAM streaming-read ceiling probe: consume 16-B words grid-stride, one float per block written.
__global__ void stream_read_kernel(const float4* __restrict__ src, float* __restrict__ sink, int n4) {
    const int stride = gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = __ldg(src + i);
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = acc.x + acc.y + acc.z + acc.w;
}

void mm_q(int type, const void* w, const void* xq, float* y, int n_in, int n_out, cudaStream_t st) {
    if (n_out <= 0) return;
    if (type == 14) strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, 1, st);
    else if (type == 8) strata::kernels::native_q8_0_mmvq(w, xq, y, n_in, n_out, 1, st);
    else {
        std::fprintf(stderr, "bad gemv type %d\n", type);
        std::exit(1);
    }
}

void copy_tail(Engine& e, float* dst, int n) {
    size_t bytes = (size_t) n * 4;
    if (e.peer) {
        CK(hipMemcpyPeerAsync(dst, e.primary, e.y_s, e.secondary, bytes, e.ss));
        return;
    }
    std::vector<float> tmp((size_t) n);
    CK(cudaStreamSynchronize(e.ss));
    CK(cudaMemcpy(tmp.data(), e.y_s, bytes, cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(e.primary));
    CK(cudaMemcpyAsync(dst, tmp.data(), bytes, cudaMemcpyHostToDevice, e.sp));
    CK(cudaStreamSynchronize(e.sp));
    CK(cudaSetDevice(e.secondary));
}

void seg_open(Engine& e, char tag) {
    if (!e.seg_b || e.seg_n >= e.seg_cap) return;
    e.seg_tag[e.seg_n] = tag;
    CK(cudaEventRecord(e.seg_b[e.seg_n], e.sp));
}

void seg_close(Engine& e) {
    if (!e.seg_b || e.seg_n >= e.seg_cap) return;
    CK(cudaEventRecord(e.seg_e[e.seg_n], e.sp));
    e.seg_n++;
}

void apply_mms(Engine& e, const Mat* ms, float* const* ys, int nmat, const float* x, bool ready) {
    seg_open(e, 'M');
    int n_in = ms[0].n_in;
    bool any = false;
    for (int i = 0; i < nmat; ++i) {
        if (ms[i].n_in != n_in) {
            std::fprintf(stderr, "mixed n_in in one gemv group\n");
            std::exit(1);
        }
        if (ms[i].b) any = true;
    }
    if (e.split > 0.0f) CK(cudaSetDevice(e.primary));
    if (any) CK(cudaEventRecord(e.ev_x, e.sp));
    if (!ready) strata::kernels::native_quantize_q8_1(x, e.scratch_p, n_in, 1, e.sp);
    for (int i = 0; i < nmat; ++i) {
        int rows = ms[i].b ? ms[i].n_a : ms[i].n_out;
        mm_q(ms[i].type, ms[i].a, e.scratch_p, ys[i], n_in, rows, e.sp);
    }
    if (!any) {
        seg_close(e);
        return;
    }
    CK(cudaSetDevice(e.secondary));
    CK(cudaStreamWaitEvent(e.ss, e.ev_x, 0));
    const size_t xbytes = (size_t) n_in * 4;
    if (e.peer) {
        CK(hipMemcpyPeerAsync(e.x_s, e.secondary, x, e.primary, xbytes, e.ss));
    } else {
        std::vector<float> tmp((size_t) n_in);
        CK(cudaSetDevice(e.primary));
        CK(cudaStreamSynchronize(e.sp));
        CK(cudaMemcpy(tmp.data(), x, xbytes, cudaMemcpyDeviceToHost));
        CK(cudaSetDevice(e.secondary));
        CK(cudaMemcpyAsync(e.x_s, tmp.data(), xbytes, cudaMemcpyHostToDevice, e.ss));
        CK(cudaStreamSynchronize(e.ss));
    }
    strata::kernels::native_quantize_q8_1(e.x_s, e.scratch_s, n_in, 1, e.ss);
    for (int i = 0; i < nmat; ++i) {
        if (!ms[i].b) continue;
        int tail = ms[i].n_out - ms[i].n_a;
        mm_q(ms[i].type, ms[i].b, e.scratch_s, e.y_s, n_in, tail, e.ss);
        copy_tail(e, ys[i] + ms[i].n_a, tail);
    }
    if (e.peer) {
        CK(cudaEventRecord(e.ev_s, e.ss));
        CK(cudaSetDevice(e.primary));
        CK(cudaStreamWaitEvent(e.sp, e.ev_s, 0));
    } else {
        CK(cudaSetDevice(e.primary));
    }
    seg_close(e);
}

void apply_mm(Engine& e, const Mat& m, const float* x, float* y, bool ready = false) {
    float* ys[1] = {y};
    apply_mms(e, &m, ys, 1, x, ready);
}

// Big-card path only: the Q8_1 input is already built (fused step-norm / swiglu / sigmoid-gate epilogues).
void apply_mm_q8(Engine& e, const Mat& m, const void* xq8, float* y) {
    seg_open(e, 'M');
    mm_q(m.type, m.a, xq8, y, m.n_in, m.n_out, e.sp);
    seg_close(e);
}

// Residual GEMV: y = weights @ xq8 + res written row-by-row, plus the row's sum-of-squares partials for the
// next rms-norm.  Used when y and res are the same buffer (the pre-norm residual row).
void mm_q_res(int type, const void* w, const void* xq8, float* y, float* partials, int n_in, int n_out,
              cudaStream_t st) {
    if (type == 14) strata::kernels::native_q6_k_mmvq_res(w, xq8, y, y, partials, n_in, n_out, st);
    else if (type == 8) strata::kernels::native_q8_0_mmvq_res(w, xq8, y, y, partials, n_in, n_out, st);
    else {
        std::fprintf(stderr, "bad res gemv type %d\n", type);
        std::exit(1);
    }
}

void rms_quant_row(Engine& e, float* x, const Mat& w) {
    seg_open(e, 'Q');
    rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(x, (const float*) w.a, e.partials,
                                                 static_cast<Q81Row*>(e.scratch_p), kEmb, e.eps);
    ck_launch("rms_quant_apply");
    seg_close(e);
}

// Concatenate two same-type, same-n_in weight mats into one device buffer (big-card path only), then drop the
// originals, so a projection group rides one GEMV launch instead of two to four.  Returns false when the mats
// differ in type or the split path needs them whole.
bool try_cat2(Engine& e, Mat& dst, Mat& a, Mat& b) {
    if (e.split > 0.0f || a.a == nullptr || b.a == nullptr) return false;
    if (a.type != b.type || a.n_in != b.n_in || a.b || b.b) return false;
    const size_t rb = row_bytes(a.type, a.n_in);
    dst.type = a.type;
    dst.n_in = a.n_in;
    dst.n_out = a.n_out + b.n_out;
    dst.n_a = dst.n_out;
    CK(cudaSetDevice(e.primary));
    CK(cudaMalloc(&dst.a, rb * (size_t) dst.n_out));
    CK(cudaMemcpy(dst.a, a.a, rb * (size_t) a.n_out, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy((char*) dst.a + rb * (size_t) a.n_out, b.a, rb * (size_t) b.n_out, cudaMemcpyDeviceToDevice));
    CK(cudaFree(a.a));
    CK(cudaFree(b.a));
    a.a = nullptr;
    b.a = nullptr;
    return true;
}

bool try_cat3(Engine& e, Mat& dst, Mat& a, Mat& b, Mat& c) {
    if (e.split > 0.0f || a.a == nullptr || b.a == nullptr || c.a == nullptr) return false;
    if (a.type != b.type || a.type != c.type || a.n_in != b.n_in || a.n_in != c.n_in || a.b || b.b || c.b) {
        return false;
    }
    const size_t rb = row_bytes(a.type, a.n_in);
    dst.type = a.type;
    dst.n_in = a.n_in;
    dst.n_out = a.n_out + b.n_out + c.n_out;
    dst.n_a = dst.n_out;
    CK(cudaSetDevice(e.primary));
    CK(cudaMalloc(&dst.a, rb * (size_t) dst.n_out));
    CK(cudaMemcpy(dst.a, a.a, rb * (size_t) a.n_out, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy((char*) dst.a + rb * (size_t) a.n_out, b.a, rb * (size_t) b.n_out, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy((char*) dst.a + rb * (size_t) (a.n_out + b.n_out), c.a, rb * (size_t) c.n_out,
                  cudaMemcpyDeviceToDevice));
    CK(cudaFree(a.a));
    CK(cudaFree(b.a));
    CK(cudaFree(c.a));
    a.a = nullptr;
    b.a = nullptr;
    c.a = nullptr;
    return true;
}

void upload_mat(Engine& e, Mat& m, const Tensor& t) {
    m.type = t.type;
    m.n_in = t.n_in;
    m.n_out = t.n_out;
    if (t.type != 14 && t.type != 8) {
        std::fprintf(stderr, "upload_mat type %d\n", t.type);
        std::exit(1);
    }
    size_t rb = row_bytes(t.type, t.n_in);
    int n_a = t.n_out;
    if (e.split > 0.0f && t.n_out >= 1024) {
        n_a = (int) std::lround((1.0 - (double) e.split) * t.n_out);
        if (n_a < 1) n_a = 1;
        if (n_a > t.n_out - 1) n_a = t.n_out;
    }
    m.n_a = n_a;
    CK(cudaSetDevice(e.primary));
    CK(cudaMalloc(&m.a, rb * (size_t) n_a));
    CK(cudaMemcpy(m.a, t.data, rb * (size_t) n_a, cudaMemcpyHostToDevice));
    if (n_a < t.n_out) {
        int tail = t.n_out - n_a;
        if (tail > e.max_tail) e.max_tail = tail;
        CK(cudaSetDevice(e.secondary));
        CK(cudaMalloc(&m.b, rb * (size_t) tail));
        CK(cudaMemcpy(m.b, t.data + rb * (size_t) n_a, rb * (size_t) tail, cudaMemcpyHostToDevice));
        CK(cudaSetDevice(e.primary));
    }
}

float* upload_f32(int dev, const void* src, size_t n) {
    float* p = nullptr;
    CK(cudaSetDevice(dev));
    CK(cudaMalloc(&p, n * 4));
    if (src && n) CK(cudaMemcpy(p, src, n * 4, cudaMemcpyHostToDevice));
    return p;
}

Tensor take(const strata::GgufFile& file, const std::string& name) {
    const strata::TensorInfo* info = file.find(name);
    if (!info) {
        std::fprintf(stderr, "missing tensor %s\n", name.c_str());
        std::exit(1);
    }
    Tensor t;
    t.type = (int) info->type;
    t.n_in = (int) info->shape[0];
    t.n_out = info->shape.size() > 1 ? (int) info->shape[1] : 1;
    t.data = file.tensor_data(*info);
    return t;
}

void expect_f32(Engine& e, Mat& m, const Tensor& t, int n) {
    if (t.type != 0 || t.n_in != n || t.n_out != 1) {
        std::fprintf(stderr, "bad f32 norm %d x %d (type %d)\n", t.n_in, t.n_out, t.type);
        std::exit(1);
    }
    m.type = 0;
    m.n_in = n;
    m.n_out = 1;
    m.n_a = 1;
    m.a = upload_f32(e.primary, t.data, (size_t) n);
}

void rms(Engine& e, float* x, const Mat& w, int rows, int cols) {
    seg_open(e, 'R');
    strata::kernels::rms_norm_weighted(x, (const float*) w.a, rows, cols, e.eps, e.sp);
    seg_close(e);
}

// Big-card path: one RMS+Q8_1 kernel, then GEMV reads that scratch.
// Split still norms and quantizes on each card, because the tail GEMV rebuilds Q8_1 from the float row.
void project(Engine& e, float* x, const Mat& norm, const Mat* ms, float* const* ys, int nmat) {
    if (e.split > 0.0f) {
        rms(e, x, norm, 1, kEmb);
        apply_mms(e, ms, ys, nmat, x, false);
    } else {
        rms_quant_row(e, x, norm);
        apply_mms(e, ms, ys, nmat, x, true);
    }
}

void ffn_split(Engine& e, const Layer& L) {
    seg_open(e, 'Y');
    CK(cudaMemcpyAsync(e.res, e.x, (size_t) kEmb * 4, cudaMemcpyDeviceToDevice, e.sp));
    seg_close(e);
    float* ys[2] = {e.up, e.gate};
    const Mat ms[2] = {L.up, L.gate};
    project(e, e.x, L.post_norm, ms, ys, 2);
    seg_open(e, 'W');
    swiglu_kernel<<<(kFF + 255) / 256, 256, 0, e.sp>>>(e.gate, e.up, kFF);
    ck_launch("swiglu");
    seg_close(e);
    apply_mm(e, L.down, e.gate, e.o);
    seg_open(e, 'A');
    strata::kernels::add_inplace(e.o, e.res, kEmb, e.sp);
    seg_close(e);
    seg_open(e, 'Y');
    CK(cudaMemcpyAsync(e.x, e.o, (size_t) kEmb * 4, cudaMemcpyDeviceToDevice, e.sp));
    seg_close(e);
}

// Big-card path: the residual add is fused with the row's sum-of-squares partials (the next norm reads them),
// swiglu writes its Q8_1 output directly, and no residual copy exists - the float row stays pre-norm.
void ffn(Engine& e, const Layer& L) {
    if (e.split > 0.0f) {
        ffn_split(e, L);
        return;
    }
    if (L.cat_upgate) {
        rms_quant_row(e, e.x, L.post_norm);
        apply_mm(e, L.upgate, e.x, e.ug_o, true);
    } else {
        float* ys[2] = {e.up, e.gate};
        const Mat ms[2] = {L.up, L.gate};
        project(e, e.x, L.post_norm, ms, ys, 2);
    }
    const float* up_v = L.cat_upgate ? e.ug_o : e.up;
    const float* gate_v = L.cat_upgate ? e.ug_o + kFF : e.gate;
    seg_open(e, 'W');
    swiglu_quant_kernel<<<kFF / 256, 256, 0, e.sp>>>(up_v, gate_v, static_cast<Q81Row*>(e.scratch_dn8), kFF);
    ck_launch("swiglu_quant");
    seg_close(e);
    if (old_gdn) {
        apply_mm_q8(e, L.down, e.scratch_dn8, e.o);
        seg_open(e, 'A');
        add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x, e.o, e.partials, kEmb);
        ck_launch("add_partials_1024 ffn");
        seg_close(e);
    } else {
        seg_open(e, 'A');
        mm_q_res(L.down.type, L.down.a, e.scratch_dn8, e.x, e.partials, L.down.n_in, L.down.n_out, e.sp);
        ck_launch("mm_q_res down");
        seg_close(e);
    }
}

void forward_gdn_split(Engine& e, Layer& L) {
    CK(cudaMemcpyAsync(e.res, e.x, (size_t) kEmb * 4, cudaMemcpyDeviceToDevice, e.sp));
    float* ys[4] = {e.qkv, e.z, e.beta, e.alpha};
    const Mat ms[4] = {L.qkv, L.z_w, L.beta_w, L.alpha_w};
    project(e, e.x, L.attn_norm, ms, ys, 4);
    seg_open(e, 'D');
    strata::kernels::gdn_beta_gate(e.beta, kHv, e.sp);
    strata::kernels::gdn_gate(e.alpha, L.ssm_dt, L.ssm_a, e.ggate, 1, kHv, e.sp);
    strata::kernels::gdn_conv_step(L.conv_state, e.qkv, L.conv_w, e.qkv, kQkv, 4, e.sp);
    strata::kernels::silu_inplace(e.qkv, kQkv, e.sp);
    float* q = e.qkv;
    float* k = e.qkv + kHk * kS;
    float* v = e.qkv + 2 * kHk * kS;
    strata::kernels::gdn_l2_norm(q, kHk, kS, e.eps, e.sp);
    strata::kernels::gdn_l2_norm(k, kHk, kS, e.eps, e.sp);
    strata::kernels::scale_inplace(q, kHk * kS, kInvSqrtS, e.sp);
    strata::kernels::GdnShapes sh;
    sh.S = kS;
    sh.h_k = kHk;
    sh.h_v = kHv;
    strata::kernels::gdn_step(L.gdn_state, q, k, v, e.ggate, e.beta, e.o, sh, e.sp);
    strata::kernels::gdn_out_norm_silu(e.o, e.z, L.ssm_norm, e.up, kHv, kS, e.eps, e.sp);
    seg_close(e);
    apply_mm(e, L.ssm_out, e.up, e.x);
    strata::kernels::add_inplace(e.x, e.res, kEmb, e.sp);
    ffn(e, L);
}

// Big-card path: the whole gated-delta recurrence runs as three launches - beta/gate epilogues, conv+silu+l2+q
// scale, then the fused step+out-norm whose Q8_1 output feeds ssm_out directly.
void forward_gdn(Engine& e, Layer& L) {
    static int g_gdn_layer_call = -1;
    if (g_vtrace) {
        if (g_vtrace_layer < 0) g_vtrace_layer = 0;  // trace only the first layer call after reset
    }
    if (e.split > 0.0f) {
        forward_gdn_split(e, L);
        return;
    }
    if (L.cat_qkvz) {
        rms_quant_row(e, e.x, L.attn_norm);
        apply_mm(e, L.qkvz, e.x, e.qkvz_o, true);
    } else {
        float* ys[4] = {e.qkv, e.z, e.beta, e.alpha};
        const Mat ms[4] = {L.qkv, L.z_w, L.beta_w, L.alpha_w};
        project(e, e.x, L.attn_norm, ms, ys, 4);
    }
    float* qkv_v = L.cat_qkvz ? e.qkvz_o : e.qkv;
    float* z_v = L.cat_qkvz ? e.qkvz_o + kQkv : e.z;
    if (L.cat_ab) {
        apply_mm(e, L.ab, e.x, e.ab_o, true);
    } else {
        float* ys2[2] = {e.beta, e.alpha};
        const Mat ms2[2] = {L.beta_w, L.alpha_w};
        apply_mms(e, ms2, ys2, 2, e.x, true);
    }
    float* beta_v = L.cat_ab ? e.ab_o : e.beta;
    float* alpha_v = beta_v + kHv;
    seg_open(e, 'D');
    if (old_gdn) {
        strata::kernels::gdn_beta_gate(beta_v, kHv, e.sp);
        strata::kernels::gdn_gate(alpha_v, L.ssm_dt, L.ssm_a, e.ggate, 1, kHv, e.sp);
        strata::kernels::gdn_conv_step(L.conv_state, qkv_v, L.conv_w, qkv_v, kQkv, 4, e.sp);
        strata::kernels::silu_inplace(qkv_v, kQkv, e.sp);
        strata::kernels::gdn_l2_norm(qkv_v, kHk, kS, e.eps, e.sp);
        strata::kernels::gdn_l2_norm(qkv_v + kHk * kS, kHk, kS, e.eps, e.sp);
        strata::kernels::scale_inplace(qkv_v, kHk * kS, kInvSqrtS, e.sp);
        strata::kernels::GdnShapes sh;
        sh.S = kS;
        sh.h_k = kHk;
        sh.h_v = kHv;
        strata::kernels::gdn_step(L.gdn_state, qkv_v, qkv_v + kHk * kS, qkv_v + 2 * kHk * kS, e.ggate, beta_v, e.o,
                                  sh, e.sp);
        strata::kernels::gdn_out_norm_silu(e.o, z_v, L.ssm_norm, e.up, kHv, kS, e.eps, e.sp);
    } else {
        beta_gate_kernel<<<1, 64, 0, e.sp>>>(beta_v, alpha_v, L.ssm_dt, L.ssm_a, e.ggate, kHv);
        ck_launch("beta_gate");
        // No q scale here: the fused step-norm multiplies its output by rsqrtf(S), which is the same 1/sqrt(128)
        // the old chain applied to q - but applied AFTER the state read-out, so the out-norm's eps sees the same
        // magnitudes it did in the old chain.
        strata::kernels::fused_gdn_conv_l2_qk(L.conv_state, qkv_v, L.conv_w, qkv_v, kQkv, 2 * kHk, 0, 0.0f,
                                              e.eps, e.sp);
        strata::kernels::fused_gdn_step_norm_silu(L.gdn_state, qkv_v, qkv_v + kHk * kS, qkv_v + 2 * kHk * kS, e.ggate,
                                                  beta_v, z_v, L.ssm_norm, e.eps, e.up, e.scratch_up8, kHk, kHv, e.sp);
    }
    vtrace(2, e.qkvz_o);   // after GEMVs (single)
    vtrace(3, e.ggate);
    vtrace(4, e.up);
    seg_close(e);
    if (old_gdn) {
        apply_mm(e, L.ssm_out, e.up, e.o);
        seg_open(e, 'A');
        add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x, e.o, e.partials, kEmb);
        ck_launch("add_partials_1024");
        seg_close(e);
    } else {
        seg_open(e, 'A');
        mm_q_res(L.ssm_out.type, L.ssm_out.a, e.scratch_up8, e.x, e.partials, L.ssm_out.n_in, L.ssm_out.n_out, e.sp);
        ck_launch("mm_q_res ssm_out");
        seg_close(e);
    }
    vtrace(5, e.x);
    ffn(e, L);
    vtrace(6, e.x);
    g_vtrace_layer = -1;  // only the first forward_gdn of a traced pass
}

void forward_attn_split(Engine& e, Layer& L) {
    CK(cudaMemcpyAsync(e.res, e.x, (size_t) kEmb * 4, cudaMemcpyDeviceToDevice, e.sp));
    float* ys[3] = {e.qkv, e.k, e.v};
    const Mat ms[3] = {L.wq, L.wk, L.wv};
    project(e, e.x, L.attn_norm, ms, ys, 3);
    seg_open(e, 'H');
    split_qg_kernel<<<kHeads, kHd, 0, e.sp>>>(e.qkv, e.o, e.z, kHeads, kHd);
    ck_launch("split_qg");
    seg_close(e);
    rms(e, e.o, L.qn, kHeads, kHd);
    rms(e, e.k, L.kn, kKv, kHd);
    seg_open(e, 'H');
    fill_pos_kernel<<<1, 32, 0, e.sp>>>(e.dpos, kHeads, e.dmeta);
    ck_launch("fill_pos");
    strata::kernels::native_rope_apply(e.o, e.o, kHeads, kHd, kRot, e.rope, e.dpos, e.sp);
    strata::kernels::native_rope_apply(e.k, e.k, kKv, kHd, kRot, e.rope, e.dpos, e.sp);
    constexpr int kv_n = kKv * kHd;
    store_kv_kernel<<<(kv_n + 255) / 256, 256, 0, e.sp>>>(L.kcache, e.k, e.dmeta, kv_n);
    store_kv_kernel<<<(kv_n + 255) / 256, 256, 0, e.sp>>>(L.vcache, e.v, e.dmeta, kv_n);
    ck_launch("store_kv");
    gqa_kernel<<<1, 32, 0, e.sp>>>(
        e.o, L.kcache, L.vcache, e.up, e.ascores, kHeads, kKv, kHd, e.dmeta, kAttnScale);
    ck_launch("gqa");
    sig_mul_kernel<<<(kEmb + 255) / 256, 256, 0, e.sp>>>(e.up, e.z, kEmb);
    ck_launch("sig_mul");
    seg_close(e);
    apply_mm(e, L.wo, e.up, e.x);
    strata::kernels::add_inplace(e.x, e.res, kEmb, e.sp);
    ffn(e, L);
}

// Big-card path: wq/wk/wv ride one GEMV, and the wo input is quantized inside the sigmoid-gate kernel.
void forward_attn(Engine& e, Layer& L) {
    if (e.split > 0.0f) {
        forward_attn_split(e, L);
        return;
    }
    if (L.cat_qkv3) {
        rms_quant_row(e, e.x, L.attn_norm);
        apply_mm(e, L.qkv3, e.x, e.q3_o, true);
    } else {
        float* ys[3] = {e.qkv, e.k, e.v};
        const Mat ms[3] = {L.wq, L.wk, L.wv};
        project(e, e.x, L.attn_norm, ms, ys, 3);
    }
    float* qg = L.cat_qkv3 ? e.q3_o : e.qkv;
    // wq emits 2*heads*hd (q and the output gate, per-head interleaved); k and v follow it.
    float* k_v = L.cat_qkv3 ? e.q3_o + 2 * kHeads * kHd : e.k;
    float* v_v = k_v + kKv * kHd;
    seg_open(e, 'H');
    split_qg_kernel<<<kHeads, kHd, 0, e.sp>>>(qg, e.o, e.z, kHeads, kHd);
    ck_launch("split_qg");
    seg_close(e);
    rms(e, e.o, L.qn, kHeads, kHd);
    rms(e, k_v, L.kn, kKv, kHd);
    seg_open(e, 'H');
    // dpos is filled once per token (forward_token); rope and the cache stores share it.
    strata::kernels::native_rope_apply(e.o, e.o, kHeads, kHd, kRot, e.rope, e.dpos, e.sp);
    strata::kernels::native_rope_apply(k_v, k_v, kKv, kHd, kRot, e.rope, e.dpos, e.sp);
    constexpr int kv_n = kKv * kHd;
    store_kv2_kernel<<<(2 * kv_n + 255) / 256, 256, 0, e.sp>>>(L.kcache, L.vcache, k_v, v_v, e.dmeta, kv_n);
    ck_launch("store_kv2");
    gqa_head_kernel<<<kHeads, 256, 0, e.sp>>>(
        e.o, L.kcache, L.vcache, e.up, e.ascores, kHeads, kKv, kHd, e.dmeta, kAttnScale);
    ck_launch("gqa_head");
    sig_mul_quant_kernel<<<kEmb / 256, 256, 0, e.sp>>>(e.up, e.z, static_cast<Q81Row*>(e.scratch_up8), kEmb);
    ck_launch("sig_mul_quant");
    seg_close(e);
    seg_open(e, 'A');
    mm_q_res(L.wo.type, L.wo.a, e.scratch_up8, e.x, e.partials, L.wo.n_in, L.wo.n_out, e.sp);
    ck_launch("mm_q_res wo");
    seg_close(e);
    ffn(e, L);
}

void launch_token_gpu(Engine& e) {
    const int nmax = e.vstop >= 0 ? e.vstop : kLayers;
    for (int il = 0; il < nmax; ++il) {
        if (e.layers[il].recr) forward_gdn(e, e.layers[il]);
        else forward_attn(e, e.layers[il]);
    }
    project(e, e.x, e.out_n, &e.out_w, &e.logits, 1);
    // the MTP draft consumes the FINAL-NORMED hidden (llama.cpp res->t_h_nextn = norm(h, output_norm))
    rms_norm_out_kernel<<<8, 256, 0, e.sp>>>(e.x, (const float*) e.out_n.a, e.partials, e.h_normed, kEmb, e.eps);
    row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.h_normed, e.hn_partials, kEmb);
}

// ============================ MTP speculative decode (blk.32 draft + 2-token verify) ============================

// GEMV over two Q8_1 columns: xq8 holds [col0 | col1] blocks, y receives [col0 | col1] outputs.
// QWYTHOS_GEMV1COLS bisects the multi-column kernel out: two single-column calls on the same bytes.
void mm_q2(int type, const void* w, const void* xq8, float* y, int n_in, int n_out, cudaStream_t st) {
    if (std::getenv("QWYTHOS_GEMV1COLS")) {
        const size_t col_bytes = (size_t) (n_in / 32) * 36;
        if (type == 14) {
            strata::kernels::native_q6_k_mmvq(w, xq8, y, n_in, n_out, 1, st);
            strata::kernels::native_q6_k_mmvq(w, (const char*) xq8 + col_bytes, y + n_out, n_in, n_out, 1, st);
        } else {
            strata::kernels::native_q8_0_mmvq(w, xq8, y, n_in, n_out, 1, st);
            strata::kernels::native_q8_0_mmvq(w, (const char*) xq8 + col_bytes, y + n_out, n_in, n_out, 1, st);
        }
        return;
    }
    if (type == 14) strata::kernels::native_q6_k_mmvq(w, xq8, y, n_in, n_out, 2, st);
    else if (type == 8) strata::kernels::native_q8_0_mmvq(w, xq8, y, n_in, n_out, 2, st);
    else {
        std::fprintf(stderr, "bad gemv2 type %d\n", type);
        std::exit(1);
    }
}

// One draft step: predict the token AFTER next_tok (the token at pos) from the trunk hidden h_prev (hidden
// at pos, with its sum-of-squares partials h_partials) and the embedding of next_tok.  Runs blk.32 - an
// attention-type layer with its own KV cache - at position pos+1 and returns the drafted token id (greedy
// over the shared lm_head).  A rejected draft leaves no state to roll back: its KV slots are simply
// overwritten by the next draft at the same position.
int draft_predict(Engine& e, int next_tok, int pos, const float* h_prev, const float* h_partials, bool with_head) {
    Layer& D = e.d32;
    e.hmeta[0] = pos;
    CK(cudaMemcpyAsync(e.dmeta, e.hmeta, sizeof(int), cudaMemcpyHostToDevice, e.sp));
    const uint8_t* src = e.emb_host + (size_t) next_tok * e.emb_stride;
    for (int b = 0; b < kEmb / 256; ++b)
        strata::dequantize_q6_K(src + (size_t) b * 210, e.emb_row + b * 256);
    CK(cudaSetDevice(e.primary));
    CK(cudaMemcpyAsync(e.x2, e.emb_row, (size_t) kEmb * 4, cudaMemcpyHostToDevice, e.sp));
    row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2, e.partials2, kEmb);
    ck_launch("draft row_sum");
    // eh_cat = [ enorm(emb(next_tok)) | hnorm(h_prev) ] - llama.cpp qwen35 graph_mtp concats
    // (e_norm, h_norm) in THAT order; the reverse feeds eh_proj scrambled inputs.
    rms_norm_out_kernel<<<8, 256, 0, e.sp>>>(e.x2, (const float*) e.enorm.a, e.partials2, e.eh_cat, kEmb, e.eps);
    ck_launch("draft enorm");
    rms_norm_out_kernel<<<8, 256, 0, e.sp>>>(h_prev, (const float*) e.hnorm.a, h_partials, e.eh_cat + kEmb, kEmb, e.eps);
    ck_launch("draft hnorm");
    strata::kernels::native_quantize_q8_1(e.eh_cat, e.eh_cat_q8, 2 * kEmb, 1, e.sp);
    mm_q(e.eh_proj.type, e.eh_proj.a, e.eh_cat_q8, e.d_x, e.eh_proj.n_in, e.eh_proj.n_out, e.sp);

    // blk.32 layer body (attention layer, same shape as the trunk's attention layers)
    row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.d_x, e.partials2, kEmb);
    rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.d_x, (const float*) D.attn_norm.a, e.partials2,
                                                 static_cast<Q81Row*>(e.scratch_p), kEmb, e.eps);
    ck_launch("draft attn_norm");
    if (D.cat_qkv3) {
        apply_mm(e, D.qkv3, e.d_x, e.q3_o, true);
    } else {
        float* ys[3] = {e.qkv, e.k, e.v};
        const Mat ms[3] = {D.wq, D.wk, D.wv};
        apply_mms(e, ms, ys, 3, e.d_x, true);
    }
    float* qg = D.cat_qkv3 ? e.q3_o : e.qkv;
    float* k_v = D.cat_qkv3 ? e.q3_o + 2 * kHeads * kHd : e.k;
    float* v_v = k_v + kKv * kHd;
    split_qg_kernel<<<kHeads, kHd, 0, e.sp>>>(qg, e.o, e.z, kHeads, kHd);
    ck_launch("draft split_qg");
    rms(e, e.o, D.qn, kHeads, kHd);
    rms(e, k_v, D.kn, kKv, kHd);
    fill_pos_kernel<<<1, 32, 0, e.sp>>>(e.dpos, kHeads, e.dmeta);
    ck_launch("draft fill_pos");
    strata::kernels::native_rope_apply(e.o, e.o, kHeads, kHd, kRot, e.rope, e.dpos, e.sp);
    strata::kernels::native_rope_apply(k_v, k_v, kKv, kHd, kRot, e.rope, e.dpos, e.sp);
    constexpr int kv_n = kKv * kHd;
    store_kv2_kernel<<<(2 * kv_n + 255) / 256, 256, 0, e.sp>>>(D.kcache, D.vcache, k_v, v_v, e.dmeta, kv_n);
    ck_launch("draft store_kv");
    gqa_head_kernel<<<kHeads, 256, 0, e.sp>>>(e.o, D.kcache, D.vcache, e.up, e.ascores, kHeads, kKv, kHd, e.dmeta,
                                              kAttnScale);
    ck_launch("draft gqa");
    sig_mul_quant_kernel<<<kEmb / 256, 256, 0, e.sp>>>(e.up, e.z, static_cast<Q81Row*>(e.scratch_up8_2), kEmb);
    ck_launch("draft sig_mul");
    apply_mm_q8(e, D.wo, e.scratch_up8_2, e.o);
    seg_open(e, 'A');
    add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.d_x, e.o, e.partials2, kEmb);
    ck_launch("draft add");
    seg_close(e);
    // ffn
    rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.d_x, (const float*) D.post_norm.a, e.partials2,
                                                 static_cast<Q81Row*>(e.scratch_p), kEmb, e.eps);
    ck_launch("draft post_norm");
    if (D.cat_upgate) {
        apply_mm(e, D.upgate, e.d_x, e.ug_o, true);
    } else {
        float* ys[2] = {e.up, e.gate};
        const Mat ms[2] = {D.up, D.gate};
        apply_mms(e, ms, ys, 2, e.d_x, true);
    }
    const float* up_v = D.cat_upgate ? e.ug_o : e.up;
    const float* gate_v = D.cat_upgate ? e.ug_o + kFF : e.gate;
    swiglu_quant_kernel<<<kFF / 256, 256, 0, e.sp>>>(up_v, gate_v, static_cast<Q81Row*>(e.scratch_dn8), kFF);
    ck_launch("draft swiglu");
    apply_mm_q8(e, D.down, e.scratch_dn8, e.o);
    add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.d_x, e.o, e.partials2, kEmb);
    ck_launch("draft add ffn");
    // shared head norm + lm head
    rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.d_x, (const float*) e.shnorm.a, e.partials2,
                                                 static_cast<Q81Row*>(e.scratch_p), kEmb, e.eps);
    ck_launch("draft shnorm");
    if (!with_head) return -1;  // cache-fill step: the KV slot is what matters, not the prediction
    mm_q(e.out_w.type, e.out_w.a, e.scratch_p, e.d_logits, e.out_w.n_in, e.out_w.n_out, e.sp);
    CK(cudaMemcpyAsync(e.hlogits, e.d_logits, (size_t) kVocab * 4, cudaMemcpyDeviceToHost, e.sp));
    CK(cudaStreamSynchronize(e.sp));
    int best = 0;
    for (int i = 1; i < kVocab; ++i)
        if (e.hlogits[i] > e.hlogits[best]) best = i;
    return best;
}

// The 2-column batched verify pass: feeds tokA at posA and tokB at posA+1.  GEMVs run ncols=2 so the
// weights stream once for both tokens; per-token ops (GDN recurrence, attention, norms, residual adds) run
// per column in col0 -> col1 order, which keeps every column bitwise-equal to a single-token pass.  After
// each GDN layer's col0 step the state is snapshotted so a rejected draft can be rolled back.
void launch_token_verify(Engine& e, int tokA, int tokB, int posA) {
    const size_t q8e = (size_t) kEmb / 32 * 36;   // Q8_1 bytes of one 4096 row
    const size_t q8f = (size_t) kFF / 32 * 36;    // Q8_1 bytes of one 12288 row
    // dequantize both embedding rows into the double-wide pinned buffer, then ONE H2D copy.  Two async
    // copies from a single staging row race the second CPU dequantize against the first in-flight DMA
    // (that race silently corrupted column 0 and made every verify logits argmax wrong).
    for (int c = 0; c < 2; ++c) {
        const int tok = c ? tokB : tokA;
        const uint8_t* src = e.emb_host + (size_t) tok * e.emb_stride;
        for (int b = 0; b < kEmb / 256; ++b)
            strata::dequantize_q6_K(src + (size_t) b * 210, e.emb_row2 + (size_t) c * kEmb + b * 256);
    }
    CK(cudaMemcpyAsync(e.x2, e.emb_row2, (size_t) 2 * kEmb * 4, cudaMemcpyHostToDevice, e.sp));
    CK(cudaSetDevice(e.primary));
    for (int c = 0; c < 2; ++c)
        row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2 + (size_t) c * kEmb,
                                                          e.partials2 + (size_t) c * (kEmb / 4), kEmb);
    ck_launch("verify row_sum");
    e.hmeta[0] = posA;
    e.hmeta1[0] = posA + 1;
    CK(cudaMemcpyAsync(e.dmeta, e.hmeta, sizeof(int), cudaMemcpyHostToDevice, e.sp));
    CK(cudaMemcpyAsync(e.dmeta1, e.hmeta1, sizeof(int), cudaMemcpyHostToDevice, e.sp));

    const int vmax = e.vstop >= 0 ? e.vstop : kLayers;
    for (int il = 0; il < vmax; ++il) {
        Layer& L = e.layers[il];
        if (L.recr) {
            for (int c = 0; c < 2; ++c)
                rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.x2 + (size_t) c * kEmb, (const float*) L.attn_norm.a,
                                                             e.partials2 + (size_t) c * (kEmb / 4),
                                                             reinterpret_cast<Q81Row*>((char*) e.scratch_p2 + c * q8e),
                                                             kEmb, e.eps);
            ck_launch("verify rms");
            mm_q2(L.qkvz.type, L.qkvz.a, e.scratch_p2, e.qkvz_o2, L.qkvz.n_in, L.qkvz.n_out, e.sp);
            mm_q2(L.ab.type, L.ab.a, e.scratch_p2, e.ab_o2, L.ab.n_in, L.ab.n_out, e.sp);
            static const int vskip1 = std::getenv("QWYTHOS_VSKIP1") ? std::atoi(std::getenv("QWYTHOS_VSKIP1")) : 0;
            for (int c = 0; c < 2; ++c) {
                if (c == 1 && vskip1) break;
                float* beta_v = e.ab_o2 + (size_t) c * 2 * kHv;
                float* alpha_v = beta_v + kHv;
                // qkvz GEMV emits kQkv + kEmb floats per column (qkv block, then the z block)
                float* qkv_c = e.qkvz_o2 + (size_t) c * (kQkv + kEmb);
                float* z_c = qkv_c + kQkv;
                beta_gate_kernel<<<1, 64, 0, e.sp>>>(beta_v, alpha_v, L.ssm_dt, L.ssm_a,
                                                     e.ggate2 + (size_t) c * kHv, kHv);
                ck_launch("verify beta_gate");
                strata::kernels::fused_gdn_conv_l2_qk(L.conv_state, qkv_c, L.conv_w, qkv_c, kQkv, 2 * kHk, 0, 0.0f,
                                                      e.eps, e.sp);
                strata::kernels::fused_gdn_step_norm_silu(L.gdn_state, qkv_c, qkv_c + kHk * kS, qkv_c + 2 * kHk * kS,
                                                          e.ggate2 + (size_t) c * kHv, beta_v, z_c,
                                                          L.ssm_norm, e.eps, e.up2 + (size_t) c * kEmb,
                                                          (char*) e.scratch_up8_2 + c * q8e, kHk, kHv, e.sp);
                if (c == 0) {
                    // snapshot the state after col0's step: col1's step must be undoable on rejection
                    CK(cudaMemcpyAsync(e.gdn_snap + (size_t) il * (kS * kHv * kS), L.gdn_state,
                                       (size_t) kS * kHv * kS * 4, cudaMemcpyDeviceToDevice, e.sp));
                    CK(cudaMemcpyAsync(e.conv_snap + (size_t) il * (3 * kQkv), L.conv_state,
                                       (size_t) 3 * kQkv * 4, cudaMemcpyDeviceToDevice, e.sp));
                    if (il == 0 && std::getenv("QWYTHOS_CMPV")) {
                        CK(cudaStreamSynchronize(e.sp));
                        float fv = 0, fu = 0;
                        CK(cudaMemcpy(&fv, qkv_c + 8090, 4, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(&fu, e.up2 + 3994, 4, cudaMemcpyDeviceToHost));
                        std::printf("VSTAGE L0c0: conv %g  up %g\n", fv, fu);
                        std::fflush(stdout);
                    }
                }
            }
            ck_launch("verify gdn");
            mm_q2(L.ssm_out.type, L.ssm_out.a, e.scratch_up8_2, e.o2, L.ssm_out.n_in, L.ssm_out.n_out, e.sp);
            for (int c = 0; c < 2; ++c)
                add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2 + (size_t) c * kEmb,
                                                                   e.o2 + (size_t) c * kEmb,
                                                                   e.partials2 + (size_t) c * (kEmb / 4), kEmb);
            ck_launch("verify ssm_add");
            // ffn (the normed row is kEmb wide: q8e stride, which is what the upgate GEMV reads)
            for (int c = 0; c < 2; ++c)
                rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.x2 + (size_t) c * kEmb,
                                                             (const float*) L.post_norm.a,
                                                             e.partials2 + (size_t) c * (kEmb / 4),
                                                             reinterpret_cast<Q81Row*>((char*) e.scratch_p2 + c * q8e),
                                                             kEmb, e.eps);
            ck_launch("verify ffn rms");
            mm_q2(L.upgate.type, L.upgate.a, e.scratch_p2, e.ug_o2, L.upgate.n_in, L.upgate.n_out, e.sp);
            for (int c = 0; c < 2; ++c)
                swiglu_quant_kernel<<<kFF / 256, 256, 0, e.sp>>>(
                    e.ug_o2 + (size_t) c * 2 * kFF, e.ug_o2 + (size_t) c * 2 * kFF + kFF,
                    reinterpret_cast<Q81Row*>((char*) e.scratch_dn8_2 + c * q8f), kFF);
            ck_launch("verify swiglu");
            mm_q2(L.down.type, L.down.a, e.scratch_dn8_2, e.o2, L.down.n_in, L.down.n_out, e.sp);
            for (int c = 0; c < 2; ++c)
                add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2 + (size_t) c * kEmb,
                                                                   e.o2 + (size_t) c * kEmb,
                                                                   e.partials2 + (size_t) c * (kEmb / 4), kEmb);
            ck_launch("verify down_add");
        } else {
            for (int c = 0; c < 2; ++c)
                rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.x2 + (size_t) c * kEmb, (const float*) L.attn_norm.a,
                                                             e.partials2 + (size_t) c * (kEmb / 4),
                                                             reinterpret_cast<Q81Row*>((char*) e.scratch_p2 + c * q8e),
                                                             kEmb, e.eps);
            ck_launch("verify rms");
            mm_q2(L.qkv3.type, L.qkv3.a, e.scratch_p2, e.q3_o2, L.qkv3.n_in, L.qkv3.n_out, e.sp);
            for (int c = 0; c < 2; ++c) {
                const int* meta_c = c ? e.dmeta1 : e.dmeta;
                float* qg = e.q3_o2 + (size_t) c * (2 * kHeads * kHd + 2 * kKv * kHd);
                float* k_v = qg + 2 * kHeads * kHd;
                float* v_v = k_v + kKv * kHd;
                split_qg_kernel<<<kHeads, kHd, 0, e.sp>>>(qg, e.o2 + (size_t) c * kEmb,
                                                          e.z2 + (size_t) c * kEmb, kHeads, kHd);
                ck_launch("verify split_qg");
                rms(e, e.o2 + (size_t) c * kEmb, L.qn, kHeads, kHd);
                rms(e, k_v, L.kn, kKv, kHd);
                fill_pos_kernel<<<1, 32, 0, e.sp>>>(e.dpos, kHeads, meta_c);
                ck_launch("verify fill_pos");
                strata::kernels::native_rope_apply(e.o2 + (size_t) c * kEmb, e.o2 + (size_t) c * kEmb, kHeads, kHd,
                                                   kRot, e.rope, e.dpos, e.sp);
                strata::kernels::native_rope_apply(k_v, k_v, kKv, kHd, kRot, e.rope, e.dpos, e.sp);
                constexpr int kv_n = kKv * kHd;
                store_kv2_kernel<<<(2 * kv_n + 255) / 256, 256, 0, e.sp>>>(L.kcache, L.vcache, k_v, v_v, meta_c,
                                                                           kv_n);
                ck_launch("verify store_kv");
                gqa_head_kernel<<<kHeads, 256, 0, e.sp>>>(e.o2 + (size_t) c * kEmb, L.kcache, L.vcache,
                                                          e.up2 + (size_t) c * kEmb, e.ascores, kHeads, kKv, kHd,
                                                          meta_c, kAttnScale);
                ck_launch("verify gqa");
                sig_mul_quant_kernel<<<kEmb / 256, 256, 0, e.sp>>>(e.up2 + (size_t) c * kEmb,
                                                                   e.z2 + (size_t) c * kEmb,
                                                                   reinterpret_cast<Q81Row*>((char*) e.scratch_up8_2 + c * q8e),
                                                                   kEmb);
                ck_launch("verify sig_mul");
            }
            mm_q2(L.wo.type, L.wo.a, e.scratch_up8_2, e.qkvz_o2, L.wo.n_in, L.wo.n_out, e.sp);
            for (int c = 0; c < 2; ++c)
                add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2 + (size_t) c * kEmb,
                                                                   e.qkvz_o2 + (size_t) c * kEmb,
                                                                   e.partials2 + (size_t) c * (kEmb / 4), kEmb);
            ck_launch("verify wo_add");
            // ffn (the normed row is kEmb wide: q8e stride, which is what the upgate GEMV reads)
            for (int c = 0; c < 2; ++c)
                rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.x2 + (size_t) c * kEmb,
                                                             (const float*) L.post_norm.a,
                                                             e.partials2 + (size_t) c * (kEmb / 4),
                                                             reinterpret_cast<Q81Row*>((char*) e.scratch_p2 + c * q8e),
                                                             kEmb, e.eps);
            ck_launch("verify ffn rms");
            mm_q2(L.upgate.type, L.upgate.a, e.scratch_p2, e.ug_o2, L.upgate.n_in, L.upgate.n_out, e.sp);
            for (int c = 0; c < 2; ++c)
                swiglu_quant_kernel<<<kFF / 256, 256, 0, e.sp>>>(
                    e.ug_o2 + (size_t) c * 2 * kFF, e.ug_o2 + (size_t) c * 2 * kFF + kFF,
                    reinterpret_cast<Q81Row*>((char*) e.scratch_dn8_2 + c * q8f), kFF);
            ck_launch("verify swiglu");
            mm_q2(L.down.type, L.down.a, e.scratch_dn8_2, e.o2, L.down.n_in, L.down.n_out, e.sp);
            for (int c = 0; c < 2; ++c)
                add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2 + (size_t) c * kEmb,
                                                                   e.o2 + (size_t) c * kEmb,
                                                                   e.partials2 + (size_t) c * (kEmb / 4), kEmb);
            ck_launch("verify down_add");
        }
    }
    // final norm + lm head per column
    for (int c = 0; c < 2; ++c)
        rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.x2 + (size_t) c * kEmb, (const float*) e.out_n.a,
                                                     e.partials2 + (size_t) c * (kEmb / 4),
                                                     reinterpret_cast<Q81Row*>((char*) e.scratch_p2 + c * q8e), kEmb,
                                                     e.eps);
    ck_launch("verify out_n");
    mm_q2(e.out_w.type, e.out_w.a, e.scratch_p2, e.logits2, e.out_w.n_in, e.out_w.n_out, e.sp);
    // the draft consumes the final-normed hiddens of both columns
    for (int c = 0; c < 2; ++c) {
        rms_norm_out_kernel<<<8, 256, 0, e.sp>>>(e.x2 + (size_t) c * kEmb, (const float*) e.out_n.a,
                                                 e.partials2 + (size_t) c * (kEmb / 4),
                                                 e.h_save + (size_t) c * kEmb, kEmb, e.eps);
        row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.h_save + (size_t) c * kEmb,
                                                          e.h_partials2 + (size_t) c * (kEmb / 4), kEmb);
    }
    if (std::getenv("QWYTHOS_CMPV")) {
        CK(cudaStreamSynchronize(e.sp));
        float rng[10] = {};
        CK(cudaMemcpy(rng, e.x2 + 3990, 10 * 4, cudaMemcpyDeviceToHost));
        std::printf("VDBG: vmax %d x2[3990..3999]:", vmax);
        for (int i = 0; i < 10; ++i) std::printf(" %g", rng[i]);
        std::printf("\n");
        std::fflush(stdout);
    }
    CK(cudaMemcpyAsync(e.hlogits2, e.logits2, (size_t) 2 * kVocab * 4, cudaMemcpyDeviceToHost, e.sp));
    CK(cudaStreamSynchronize(e.sp));
}


void probe_gemv(Engine& e) {
    if (!std::getenv("QWYTHOS_PROBE")) return;
    CK(cudaSetDevice(e.primary));
    constexpr int kSeg = 640;
    cudaEvent_t sb[kSeg], se[kSeg];
    char st[kSeg];
    for (int i = 0; i < kSeg; ++i) {
        CK(cudaEventCreate(&sb[i]));
        CK(cudaEventCreate(&se[i]));
    }
    e.seg_b = sb;
    e.seg_e = se;
    e.seg_tag = st;
    e.seg_cap = kSeg;
    e.seg_n = 0;
    e.hmeta[0] = 0;
    CK(cudaMemcpy(e.dmeta, e.hmeta, sizeof(int), cudaMemcpyHostToDevice));
    launch_token_gpu(e);
    launch_token_gpu(e);
    CK(cudaStreamSynchronize(e.sp));
    e.seg_n = 0;
    CK(cudaEventRecord(e.ev_layer, e.sp));
    launch_token_gpu(e);
    CK(cudaEventRecord(e.ev_done, e.sp));
    CK(cudaStreamSynchronize(e.sp));
    float total_ms = 0.0f;
    CK(cudaEventElapsedTime(&total_ms, e.ev_layer, e.ev_done));
    double gemv_ms = 0.0;
    double by_tag[128] = {};
    int n_tag[128] = {};
    for (int i = 0; i < e.seg_n; ++i) {
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, e.seg_b[i], e.seg_e[i]));
        gemv_ms += ms;
        unsigned char tag = (unsigned char) e.seg_tag[i];
        if (tag < 128) {
            by_tag[tag] += ms;
            n_tag[tag]++;
        }
    }
    std::printf("probe steady_token total %.2f ms  tagged %.2f ms  gap %.2f ms  segs %d\n",
                total_ms, gemv_ms, total_ms - gemv_ms, e.seg_n);
    for (int t = 0; t < 128; ++t) {
        if (!n_tag[t]) continue;
        std::printf("probe tag %c %6.2f ms  n %d\n", t, by_tag[t], n_tag[t]);
    }
    std::fflush(stdout);
    e.seg_b = nullptr;
    e.seg_e = nullptr;
    auto bytes_of = [](const Mat& m) { return (double) row_bytes(m.type, m.n_in) * (double) m.n_out; };
    auto bench = [&](const char* name, auto&& fn, double bytes) {
        for (int i = 0; i < 3; ++i) fn();
        CK(cudaStreamSynchronize(e.sp));
        CK(cudaEventRecord(e.ev_layer, e.sp));
        constexpr int kRep = 16;
        for (int i = 0; i < kRep; ++i) fn();
        CK(cudaEventRecord(e.ev_done, e.sp));
        CK(cudaStreamSynchronize(e.sp));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, e.ev_layer, e.ev_done));
        double sec = (ms / 1000.0) / kRep;
        std::printf("probe %-10s %7.3f ms  %6.0f GB/s  bytes %.2f MB\n",
                    name, ms / kRep, bytes / sec / 1e9, bytes / 1e6);
        std::fflush(stdout);
    };
    bench("lm_head", [&] { apply_mm(e, e.out_w, e.x, e.logits); }, bytes_of(e.out_w));
    // raw DRAM streaming ceiling: 256 MB device-to-device copy = 512 MB of traffic
    {
        void *ca = nullptr, *cb = nullptr;
        CK(cudaMalloc(&ca, 256u << 20));
        CK(cudaMalloc(&cb, 256u << 20));
        CK(cudaMemset(ca, 1, 256u << 20));
        bench("copy256", [&] { CK(cudaMemcpyAsync(cb, ca, 256u << 20, cudaMemcpyDeviceToDevice, e.sp)); },
              512.0 * 1024.0 * 1024.0);
        CK(cudaFree(ca));
        CK(cudaFree(cb));
    }
    // streaming READ ceiling: 256 MB consumed by a float4 grid-stride loop, one word written per block
    {
        const int nfloat4 = (256u << 20) / 16;
        float4* ra = nullptr;
        float* rsink = nullptr;
        CK(cudaMalloc(&ra, (size_t) nfloat4 * 16));
        CK(cudaMemset(ra, 1, (size_t) nfloat4 * 16));
        CK(cudaMalloc(&rsink, 65536 * 4));
        bench("read256", [&] {
            stream_read_kernel<<<2048, 256, 0, e.sp>>>(ra, rsink, nfloat4);
            ck_launch("read_kernel");
        }, 256.0 * 1024.0 * 1024.0);
        CK(cudaFree(ra));
        CK(cudaFree(rsink));
    }
    Layer& G = e.layers[0];
    if (G.cat_qkvz) bench("qkvz", [&] { apply_mm(e, G.qkvz, e.x, e.qkvz_o); }, bytes_of(G.qkvz));
    else {
        bench("qkv", [&] { apply_mm(e, G.qkv, e.x, e.qkv); }, bytes_of(G.qkv));
        bench("z", [&] { apply_mm(e, G.z_w, e.x, e.z); }, bytes_of(G.z_w));
    }
    if (G.cat_ab) bench("ab", [&] { apply_mm(e, G.ab, e.x, e.ab_o); }, bytes_of(G.ab));
    if (G.cat_upgate) bench("upgate", [&] { apply_mm(e, G.upgate, e.x, e.ug_o); }, bytes_of(G.upgate));
    else bench("ffn_up", [&] { apply_mm(e, G.up, e.x, e.up); }, bytes_of(G.up));
    bench("ffn_down", [&] { apply_mm(e, G.down, G.cat_upgate ? e.ug_o + kFF : e.gate, e.o); }, bytes_of(G.down));
    if (G.cat_qkv3) bench("qkv3", [&] { apply_mm(e, e.layers[3].qkv3, e.x, e.q3_o); }, bytes_of(e.layers[3].qkv3));
    bench("ssm_out", [&] { apply_mm(e, G.ssm_out, e.up, e.o); }, bytes_of(G.ssm_out));
    double gbytes = bytes_of(G.qkv) + bytes_of(G.z_w) + bytes_of(G.alpha_w) + bytes_of(G.beta_w) +
                    bytes_of(G.ssm_out) + bytes_of(G.up) + bytes_of(G.gate) + bytes_of(G.down);
    bench("gdn_layer", [&] { forward_gdn(e, G); }, gbytes);
    Layer& A = e.layers[3];
    double abytes = bytes_of(A.wq) + bytes_of(A.wk) + bytes_of(A.wv) + bytes_of(A.wo) +
                    bytes_of(A.up) + bytes_of(A.gate) + bytes_of(A.down);
    bench("attn_layer", [&] {
        e.hmeta[0] = 0;
        CK(cudaMemcpyAsync(e.dmeta, e.hmeta, sizeof(int), cudaMemcpyHostToDevice, e.sp));
        forward_attn(e, A);
    }, abytes);
    strata::kernels::GdnShapes sh;
    sh.S = kS;
    sh.h_k = kHk;
    sh.h_v = kHv;
    float* q = e.qkv;
    float* k = e.qkv + kHk * kS;
    float* v = e.qkv + 2 * kHk * kS;
    bench("gdn_step", [&] {
        strata::kernels::gdn_step(G.gdn_state, q, k, v, e.ggate, e.beta, e.o, sh, e.sp);
    }, 2.0 * 1024.0 * 1024.0);
    bench("gdn_l2", [&] { strata::kernels::gdn_l2_norm(q, kHk, kS, e.eps, e.sp); }, 0);
    bench("out_norm", [&] {
        strata::kernels::gdn_out_norm_silu(e.o, e.z, G.ssm_norm, e.up, kHv, kS, e.eps, e.sp);
    }, 0);
    bench("conv", [&] {
        strata::kernels::gdn_conv_step(G.conv_state, e.qkv, G.conv_w, e.qkv, kQkv, 4, e.sp);
    }, 0);
    bench("rms", [&] { rms(e, e.x, G.attn_norm, 1, kEmb); }, 0);
    std::exit(0);
}

void capture_graph(Engine& e) {
    if (e.split > 0.0f) return;
    e.hmeta[0] = 0;
    CK(cudaSetDevice(e.primary));
    CK(cudaStreamBeginCapture(e.sp, cudaStreamCaptureModeThreadLocal));
    launch_token_gpu(e);
    cudaGraph_t captured = nullptr;
    CK(cudaStreamEndCapture(e.sp, &captured));
    CK(cudaGraphInstantiate(&e.graph, captured, 0ull));
    CK(cudaGraphDestroy(captured));
    if (!g_serve_mode) {
        std::printf("graph 1\n");
        std::fflush(stdout);
    }
}

int forward_token(Engine& e, int token, int pos) {
    if (token < 0 || token >= kVocab) {
        std::fprintf(stderr, "token %d out of range\n", token);
        std::exit(1);
    }
    if (pos < 0 || pos >= kMaxCtx) {
        std::fprintf(stderr, "pos %d exceeds %d\n", pos, kMaxCtx);
        std::exit(1);
    }
    const uint8_t* src = e.emb_host + (size_t) token * e.emb_stride;
    for (int b = 0; b < kEmb / 256; ++b)
        strata::dequantize_q6_K(src + (size_t) b * 210, e.hrow + b * 256);
    e.hmeta[0] = pos;
    CK(cudaSetDevice(e.primary));
    CK(cudaEventRecord(e.ev_layer, e.sp));
    CK(cudaMemcpyAsync(e.x, e.hrow, (size_t) kEmb * 4, cudaMemcpyHostToDevice, e.sp));
    CK(cudaMemcpyAsync(e.dmeta, e.hmeta, sizeof(int), cudaMemcpyHostToDevice, e.sp));
    // The first project of the token reads the fresh embedding row: seed its sum-of-squares partials, and
    // fill dpos once (every attention layer of this token reads the same position).
    row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x, e.partials, kEmb);
    ck_launch("row_sum_partials");
    fill_pos_kernel<<<1, 32, 0, e.sp>>>(e.dpos, kHeads, e.dmeta);
    ck_launch("fill_pos");
    if (e.graph) CK(cudaGraphLaunch(e.graph, e.sp));
    else launch_token_gpu(e);
    CK(cudaEventRecord(e.ev_done, e.sp));
    CK(cudaStreamSynchronize(e.sp));
    float gpu_ms = 0.0f;
    CK(cudaEventElapsedTime(&gpu_ms, e.ev_layer, e.ev_done));
    e.layer_ms += gpu_ms;
    CK(cudaMemcpy(e.hlogits, e.logits, (size_t) kVocab * 4, cudaMemcpyDeviceToHost));
    int best = 0;
    float top = e.hlogits[0];
    for (int i = 1; i < kVocab; ++i) {
        if (e.hlogits[i] > top) {
            top = e.hlogits[i];
            best = i;
        }
    }
    static const bool topk = std::getenv("QWYTHOS_TOPK") != nullptr;
    if (topk && pos == e.seed_pos) {
        for (int r = 0; r < 8; ++r) {
            int id = 0;
            for (int i = 1; i < kVocab; ++i)
                if (e.hlogits[i] > e.hlogits[id] && (r == 0 || e.hlogits[i] < e.thresh)) id = i;
            e.thresh = e.hlogits[id];
            std::printf("top%d id %d logit %.4f\n", r, id, e.hlogits[id]);
        }
        std::fflush(stdout);
    }
    return best;
}

double meta_num(const strata::GgufFile& file, const char* key, double fallback, bool required) {
    const strata::MetaValue* m = file.get(key);
    if (!m) {
        if (required) {
            std::fprintf(stderr, "missing key %s\n", key);
            std::exit(1);
        }
        return fallback;
    }
    return m->num();
}

void alloc_primary(Engine& e) {
    CK(cudaSetDevice(e.primary));
    CK(cudaStreamCreate(&e.sp));
    CK(cudaEventCreate(&e.ev_layer));
    CK(cudaEventCreate(&e.ev_done));
    CK(cudaEventCreateWithFlags(&e.ev_x, cudaEventDisableTiming));
    size_t q8 = strata::kernels::native_q8_1_bytes(kFF, 1);
    CK(cudaMalloc(&e.scratch_p, q8));
    CK(cudaMalloc(&e.qkvz_o, (size_t) (kQkv + kEmb) * 4));
    CK(cudaMalloc(&e.ab_o, (size_t) (2 * kHv) * 4));
    CK(cudaMalloc(&e.ug_o, (size_t) (2 * kFF) * 4));
    CK(cudaMalloc(&e.q3_o, (size_t) (2 * kHeads * kHd + 2 * kKv * kHd) * 4));
    CK(cudaMalloc(&e.partials, (size_t) kEmb / 4 * 4));
    CK(cudaMalloc(&e.scratch_up8, (size_t) kEmb / 32 * 36));
    CK(cudaMalloc(&e.scratch_dn8, (size_t) kFF / 32 * 36));
    CK(cudaMemset(e.partials, 0, (size_t) kEmb / 4 * 4));
    // MTP speculative decode buffers
    CK(cudaMalloc(&e.x2, (size_t) 2 * kEmb * 4));
    CK(cudaMalloc(&e.o2, (size_t) 2 * kEmb * 4));
    CK(cudaMalloc(&e.qkvz_o2, (size_t) 2 * (kQkv + kEmb) * 4));
    CK(cudaMalloc(&e.ab_o2, (size_t) 2 * 2 * kHv * 4));
    CK(cudaMalloc(&e.ug_o2, (size_t) 2 * 2 * kFF * 4));
    CK(cudaMalloc(&e.q3_o2, (size_t) 2 * (2 * kHeads * kHd + 2 * kKv * kHd) * 4));
    CK(cudaMalloc(&e.up2, (size_t) 2 * kEmb * 4));
    CK(cudaMalloc(&e.z2, (size_t) 2 * kEmb * 4));
    CK(cudaMalloc(&e.ggate2, (size_t) 2 * kHv * 4));
    CK(cudaMalloc(&e.partials2, (size_t) 2 * (kEmb / 4) * 4));
    CK(cudaMalloc(&e.scratch_p2, (size_t) 2 * q8));
    CK(cudaMalloc(&e.scratch_up8_2, (size_t) 2 * kEmb / 32 * 36));
    CK(cudaMalloc(&e.scratch_dn8_2, (size_t) 2 * kFF / 32 * 36));
    CK(cudaMalloc(&e.logits2, (size_t) 2 * kVocab * 4));
    CK(cudaMalloc(&e.gdn_snap, (size_t) kLayers * kS * kHv * kS * 4));
    CK(cudaMalloc(&e.gdn_pre, (size_t) kLayers * kS * kHv * kS * 4));
    CK(cudaMalloc(&e.conv_pre, (size_t) kLayers * 3 * kQkv * 4));
    CK(cudaMalloc(&e.conv_snap, (size_t) kLayers * 3 * kQkv * 4));
    CK(cudaMalloc(&e.h_save, (size_t) 2 * kEmb * 4));
    CK(cudaMalloc(&e.h_partials2, (size_t) 2 * (kEmb / 4) * 4));
    CK(cudaMalloc(&e.kcache32, (size_t) kMaxCtx * kKv * kHd * 4));
    CK(cudaMalloc(&e.vcache32, (size_t) kMaxCtx * kKv * kHd * 4));
    CK(cudaMemset(e.kcache32, 0, (size_t) kMaxCtx * kKv * kHd * 4));
    CK(cudaMemset(e.vcache32, 0, (size_t) kMaxCtx * kKv * kHd * 4));
    CK(cudaMallocHost(&e.emb_row, (size_t) kEmb * 4));
    CK(cudaMallocHost(&e.emb_row2, (size_t) 2 * kEmb * 4));
    CK(cudaMalloc(&e.h_zero, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.p_zero, (size_t) kEmb / 4 * 4));
    CK(cudaMalloc(&e.h_normed, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.hn_partials, (size_t) kEmb / 4 * 4));
    CK(cudaMemset(e.h_zero, 0, (size_t) kEmb * 4));
    CK(cudaMemset(e.p_zero, 0, (size_t) kEmb / 4 * 4));
    CK(cudaMalloc(&e.emb_dev, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.d_partials, (size_t) kEmb / 4 * 4));
    CK(cudaMalloc(&e.eh_cat, (size_t) 2 * kEmb * 4));
    CK(cudaMalloc(&e.eh_cat_q8, (size_t) 2 * kEmb / 32 * 36));
    CK(cudaMalloc(&e.d_x, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.d_up, (size_t) kFF * 4));
    CK(cudaMalloc(&e.d_gate, (size_t) kFF * 4));
    CK(cudaMalloc(&e.d_o, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.d_qkv, (size_t) kQkv * 4));
    CK(cudaMalloc(&e.d_z, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.d_k, (size_t) kKv * kHd * 4));
    CK(cudaMalloc(&e.d_v, (size_t) kKv * kHd * 4));
    CK(cudaMalloc(&e.d_up8, (size_t) kEmb / 32 * 36));
    CK(cudaMalloc(&e.d_ugq8, (size_t) kFF / 32 * 36));
    CK(cudaMalloc(&e.d_logits, (size_t) kVocab * 4));
    CK(cudaMalloc(&e.dmeta1, sizeof(int)));
    CK(cudaMallocHost(&e.hmeta1, sizeof(int)));
    CK(cudaMallocHost(&e.hlogits2, (size_t) 2 * kVocab * 4));
    CK(cudaMalloc(&e.x, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.res, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.o, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.qkv, (size_t) kQkv * 4));
    CK(cudaMalloc(&e.z, (size_t) kEmb * 4));
    CK(cudaMalloc(&e.up, (size_t) kFF * 4));
    CK(cudaMalloc(&e.gate, (size_t) kFF * 4));
    CK(cudaMalloc(&e.k, (size_t) kKv * kHd * 4));
    CK(cudaMalloc(&e.v, (size_t) kKv * kHd * 4));
    CK(cudaMalloc(&e.beta, (size_t) kHv * 4));
    CK(cudaMalloc(&e.alpha, (size_t) kHv * 4));
    CK(cudaMalloc(&e.ggate, (size_t) kHv * 4));
    CK(cudaMalloc(&e.logits, (size_t) kVocab * 4));
    CK(cudaMalloc(&e.ascores, (size_t) kHeads * kMaxCtx * 4));
    CK(cudaMalloc(&e.dpos, sizeof(int) * kHeads));
    CK(cudaMalloc(&e.dmeta, sizeof(int)));
    CK(cudaMallocHost(&e.hrow, (size_t) kEmb * 4));
    CK(cudaMallocHost(&e.hmeta, sizeof(int)));
    CK(cudaMallocHost(&e.hlogits, (size_t) kVocab * 4));
    if (e.split > 0.0f) {
        int can = 0;
        CK(hipDeviceCanAccessPeer(&can, e.primary, e.secondary));
        int can2 = 0;
        CK(hipDeviceCanAccessPeer(&can2, e.secondary, e.primary));
        if (!g_serve_mode) std::printf("peer access primary->secondary %d secondary->primary %d\n", can, can2);
        if (can && can2) {
            CK(cudaSetDevice(e.primary));
            hipError_t pe = hipDeviceEnablePeerAccess(e.secondary, 0);
            if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) CK(pe);
            CK(cudaSetDevice(e.secondary));
            pe = hipDeviceEnablePeerAccess(e.primary, 0);
            if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) CK(pe);
            e.peer = true;
        }
        CK(cudaSetDevice(e.secondary));
        CK(cudaStreamCreate(&e.ss));
        CK(cudaEventCreateWithFlags(&e.ev_s, cudaEventDisableTiming));
        CK(cudaMalloc(&e.x_s, (size_t) kFF * 4));
        CK(cudaMalloc(&e.scratch_s, q8));
        CK(cudaSetDevice(e.primary));
    }
}

void alloc_tail(Engine& e) {
    if (e.max_tail <= 0) return;
    CK(cudaSetDevice(e.secondary));
    CK(cudaMalloc(&e.y_s, (size_t) e.max_tail * 4));
    CK(cudaSetDevice(e.primary));
    if (!g_serve_mode) std::printf("split tail rows %d peer %d\n", e.max_tail, e.peer ? 1 : 0);
}

void load_model(Engine& e, const strata::GgufFile& file) {
    const strata::MetaValue* arch = file.get("general.architecture");
    if (!arch || arch->s != "qwen35") {
        std::fprintf(stderr, "not a qwen35 GGUF\n");
        std::exit(1);
    }
    int blocks = (int) meta_num(file, "qwen35.block_count", 0, true);
    int nextn = (int) meta_num(file, "qwen35.nextn_predict_layers", 0, true);
    if (blocks - nextn != kLayers) {
        std::fprintf(stderr, "trunk layers %d, expected %d\n", blocks - nextn, kLayers);
        std::exit(1);
    }
    e.eps = (float) meta_num(file, "qwen35.attention.layer_norm_rms_epsilon", 1e-6, true);
    double factor = meta_num(file, "qwen35.rope.scaling.factor", 1, true);
    e.rope.type = strata::kernels::RopeScalingType::YaRN;
    e.rope.freq_base = meta_num(file, "qwen35.rope.freq_base", 1e7, true);
    e.rope.factor = factor;
    e.rope.orig_ctx = meta_num(file, "qwen35.rope.scaling.original_context_length", 262144, true);
    e.rope.ext_factor = 1;
    e.rope.attn_factor = 1;
    e.rope.beta_fast = 32;
    e.rope.beta_slow = 1;
    if (!g_serve_mode) std::printf("yarn factor %.3f base %.3g orig %.0f eps %.3g\n",
                e.rope.factor, e.rope.freq_base, e.rope.orig_ctx, e.eps);

    Tensor emb = take(file, "token_embd.weight");
    if (emb.type != 14 || emb.n_in != kEmb || emb.n_out != kVocab) {
        std::fprintf(stderr, "unexpected embedding\n");
        std::exit(1);
    }
    e.emb_host = emb.data;
    e.emb_stride = row_bytes(14, kEmb);
    upload_mat(e, e.out_w, take(file, "output.weight"));
    expect_f32(e, e.out_n, take(file, "output_norm.weight"), kEmb);

    for (int il = 0; il < kLayers; ++il) {
        Layer& L = e.layers[il];
        L.recr = ((il + 1) % 4) != 0;
        char name[96];
        std::snprintf(name, sizeof name, "blk.%d.attn_norm.weight", il);
        expect_f32(e, L.attn_norm, take(file, name), kEmb);
        std::snprintf(name, sizeof name, "blk.%d.post_attention_norm.weight", il);
        expect_f32(e, L.post_norm, take(file, name), kEmb);
        std::snprintf(name, sizeof name, "blk.%d.ffn_up.weight", il);
        upload_mat(e, L.up, take(file, name));
        std::snprintf(name, sizeof name, "blk.%d.ffn_gate.weight", il);
        upload_mat(e, L.gate, take(file, name));
        std::snprintf(name, sizeof name, "blk.%d.ffn_down.weight", il);
        upload_mat(e, L.down, take(file, name));
        if (L.recr) {
            std::snprintf(name, sizeof name, "blk.%d.attn_qkv.weight", il);
            upload_mat(e, L.qkv, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.attn_gate.weight", il);
            upload_mat(e, L.z_w, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.ssm_alpha.weight", il);
            upload_mat(e, L.alpha_w, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.ssm_beta.weight", il);
            upload_mat(e, L.beta_w, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.ssm_out.weight", il);
            upload_mat(e, L.ssm_out, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.ssm_conv1d.weight", il);
            Tensor conv = take(file, name);
            if (conv.type != 0 || conv.n_in != 4 || conv.n_out != kQkv) {
                std::fprintf(stderr, "bad conv %s\n", name);
                std::exit(1);
            }
            L.conv_w = upload_f32(e.primary, conv.data, (size_t) 4 * kQkv);
            std::snprintf(name, sizeof name, "blk.%d.ssm_a", il);
            Tensor a = take(file, name);
            L.ssm_a = upload_f32(e.primary, a.data, kHv);
            std::snprintf(name, sizeof name, "blk.%d.ssm_dt.bias", il);
            Tensor dt = take(file, name);
            L.ssm_dt = upload_f32(e.primary, dt.data, kHv);
            std::snprintf(name, sizeof name, "blk.%d.ssm_norm.weight", il);
            Tensor sn = take(file, name);
            if (sn.type != 0 || sn.n_in != kS) {
                std::fprintf(stderr, "bad ssm_norm %s\n", name);
                std::exit(1);
            }
            L.ssm_norm = upload_f32(e.primary, sn.data, kS);
            L.conv_state = upload_f32(e.primary, nullptr, (size_t) 3 * kQkv);
            CK(cudaMemset(L.conv_state, 0, (size_t) 3 * kQkv * 4));
            L.gdn_state = upload_f32(e.primary, nullptr, (size_t) kS * kHv * kS);
            CK(cudaMemset(L.gdn_state, 0, (size_t) kS * kHv * kS * 4));
        } else {
            std::snprintf(name, sizeof name, "blk.%d.attn_q.weight", il);
            upload_mat(e, L.wq, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.attn_k.weight", il);
            upload_mat(e, L.wk, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.attn_v.weight", il);
            upload_mat(e, L.wv, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.attn_output.weight", il);
            upload_mat(e, L.wo, take(file, name));
            std::snprintf(name, sizeof name, "blk.%d.attn_q_norm.weight", il);
            expect_f32(e, L.qn, take(file, name), kHd);
            std::snprintf(name, sizeof name, "blk.%d.attn_k_norm.weight", il);
            expect_f32(e, L.kn, take(file, name), kHd);
            size_t cache = (size_t) kMaxCtx * kKv * kHd;
            L.kcache = upload_f32(e.primary, nullptr, cache);
            L.vcache = upload_f32(e.primary, nullptr, cache);
            CK(cudaMemset(L.kcache, 0, cache * 4));
            CK(cudaMemset(L.vcache, 0, cache * 4));
        }
        if (e.split <= 0.0f) {
            if (L.recr) L.cat_qkvz = try_cat2(e, L.qkvz, L.qkv, L.z_w);
            else L.cat_qkv3 = try_cat3(e, L.qkv3, L.wq, L.wk, L.wv);
            L.cat_ab = try_cat2(e, L.ab, L.beta_w, L.alpha_w);
            L.cat_upgate = try_cat2(e, L.upgate, L.up, L.gate);
        }
        if ((il % 8) == 7) if (!g_serve_mode) std::printf("uploaded through layer %d\n", il);
        std::fflush(stdout);
    }
    // blk.32: the MTP draft layer (attention type) plus the nextn glue
    char name[96];
    Layer& D = e.d32;
    D.recr = false;
    std::snprintf(name, sizeof name, "blk.%d.attn_norm.weight", kLayers);
    expect_f32(e, D.attn_norm, take(file, name), kEmb);
    std::snprintf(name, sizeof name, "blk.%d.post_attention_norm.weight", kLayers);
    expect_f32(e, D.post_norm, take(file, name), kEmb);
    std::snprintf(name, sizeof name, "blk.%d.attn_q.weight", kLayers);
    upload_mat(e, D.wq, take(file, name));
    std::snprintf(name, sizeof name, "blk.%d.attn_k.weight", kLayers);
    upload_mat(e, D.wk, take(file, name));
    std::snprintf(name, sizeof name, "blk.%d.attn_v.weight", kLayers);
    upload_mat(e, D.wv, take(file, name));
    std::snprintf(name, sizeof name, "blk.%d.attn_output.weight", kLayers);
    upload_mat(e, D.wo, take(file, name));
    std::snprintf(name, sizeof name, "blk.%d.attn_q_norm.weight", kLayers);
    expect_f32(e, D.qn, take(file, name), kHd);
    std::snprintf(name, sizeof name, "blk.%d.attn_k_norm.weight", kLayers);
    expect_f32(e, D.kn, take(file, name), kHd);
    std::snprintf(name, sizeof name, "blk.%d.ffn_up.weight", kLayers);
    upload_mat(e, D.up, take(file, name));
    std::snprintf(name, sizeof name, "blk.%d.ffn_gate.weight", kLayers);
    upload_mat(e, D.gate, take(file, name));
    std::snprintf(name, sizeof name, "blk.%d.ffn_down.weight", kLayers);
    upload_mat(e, D.down, take(file, name));
    D.kcache = upload_f32(e.primary, nullptr, (size_t) kMaxCtx * kKv * kHd);
    D.vcache = upload_f32(e.primary, nullptr, (size_t) kMaxCtx * kKv * kHd);
    CK(cudaMemset(D.kcache, 0, (size_t) kMaxCtx * kKv * kHd * 4));
    CK(cudaMemset(D.vcache, 0, (size_t) kMaxCtx * kKv * kHd * 4));
    upload_mat(e, e.eh_proj, take(file, "blk.32.nextn.eh_proj.weight"));
    expect_f32(e, e.hnorm, take(file, "blk.32.nextn.hnorm.weight"), kEmb);
    expect_f32(e, e.enorm, take(file, "blk.32.nextn.enorm.weight"), kEmb);
    expect_f32(e, e.shnorm, take(file, "blk.32.nextn.shared_head_norm.weight"), kEmb);
    if (e.split <= 0.0f) {
        D.cat_qkv3 = try_cat3(e, D.qkv3, D.wq, D.wk, D.wv);
        D.cat_upgate = try_cat2(e, D.upgate, D.up, D.gate);
    }
    CK(cudaSetDevice(e.primary));
    CK(cudaDeviceSynchronize());
    alloc_tail(e);
    CK(cudaSetDevice(e.primary));
    CK(cudaDeviceSynchronize());
}

}  // namespace


// ============================ --serve: stdin JSON-lines serving mode ============================
//
// One JSON object per line on stdin, one line out per response, on stdout:
//   {"op":"reset"}                              -> wipe GDN/conv/KV/draft state, pos=0
//   {"op":"run","tokens":[..],"gen":N,"mtp":B,"topk":K}  -> ingest tokens, generate N
//     emits one line per generated token: {"id":I,"top":[[id,logit],..K]}
//     and a final line: {"done":true,"pos":P}
// Greedy (mtp=true) reproduces the CLI MTP stream bit-for-bit. The server owns tokenization
// and sampling; the engine owns state and speed.

static void json_escape_out(const char* key, const std::vector<std::pair<int, float>>& v) {
    std::printf("\"%s\":[", key);
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf("%s[%d,%.5f]", i ? "," : "", v[i].first, v[i].second);
    }
    std::printf("]");
}

static void topk_out(const float* logits, int n, int k, int emit_id) {
    std::vector<std::pair<int, float>> top;
    top.reserve(k);
    for (int i = 0; i < n; ++i) {
        if ((int) top.size() < k) {
            top.push_back({i, logits[i]});
            if ((int) top.size() == k) std::sort(top.begin(), top.end(),
                                                 [](auto& a, auto& b) { return a.second > b.second; });
        } else if (logits[i] > top.back().second) {
            top.back() = {i, logits[i]};
            for (size_t j = top.size() - 1; j > 0 && top[j].second > top[j - 1].second; --j)
                std::swap(top[j], top[j - 1]);
        }
    }
    std::printf("{\"id\":%d,", emit_id);
    json_escape_out("top", top);
    std::printf("}\n");
    std::fflush(stdout);
}

static void serve_reset_state(Engine& e) {
    CK(cudaSetDevice(e.primary));
    for (int il = 0; il < kLayers; ++il) {
        if (e.layers[il].recr) {
            CK(cudaMemset(e.layers[il].gdn_state, 0, (size_t) kS * kHv * kS * 4));
            CK(cudaMemset(e.layers[il].conv_state, 0, (size_t) 3 * kQkv * 4));
        }
        if (!e.layers[il].recr) {  // only attention layers carry a KV cache
            CK(cudaMemset(e.layers[il].kcache, 0, (size_t) kMaxCtx * kKv * kHd * 4));
            CK(cudaMemset(e.layers[il].vcache, 0, (size_t) kMaxCtx * kKv * kHd * 4));
        }
    }
    CK(cudaMemset(e.kcache32, 0, (size_t) kMaxCtx * kKv * kHd * 4));
    CK(cudaMemset(e.vcache32, 0, (size_t) kMaxCtx * kKv * kHd * 4));
    CK(cudaMemset(e.partials, 0, (size_t) kEmb / 4 * 4));
    CK(cudaMemset(e.partials2, 0, (size_t) 2 * (kEmb / 4) * 4));
}

static int serve_argmax(const float* l) {
    int best = 0;
    for (int i = 1; i < kVocab; ++i)
        if (l[i] > l[best]) best = i;
    return best;
}

static void serve_loop(Engine& e) {
    g_serve_mode = true;
    capture_graph(e);
    int pos = 0;
    bool primed = false;      // seed pass + draft slot 0 done
    int first_tok = -1;
    std::string line;
    std::fprintf(stderr, "rapier serve: ready\n");
    std::fflush(stderr);
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        // naive parse (ints only; the server sends exactly these keys)
        auto num_in = [&](const char* key) -> long long {
            const std::string pat = "\"" + std::string(key) + "\":";
            auto at = line.find(pat);
            if (at == std::string::npos) return -1;
            return std::atoll(line.c_str() + at + pat.size());
        };
        const bool is_reset = line.find("\"op\":\"reset\"") != std::string::npos;
        if (is_reset) {
            serve_reset_state(e);
            pos = 0;
            primed = false;
            first_tok = -1;
            std::printf("{\"ok\":true,\"op\":\"reset\"}\n");
            std::fflush(stdout);
            continue;
        }
        const long long gen = num_in("gen");
        const bool mtp = line.find("\"mtp\":true") != std::string::npos;
        // tokens array
        std::vector<int> toks;
        {
            auto at = line.find("\"tokens\":[");
            if (at != std::string::npos) {
                const char* p = line.c_str() + at + 10;
                long long v = 0; bool in = false, neg = false;
                for (; *p && *p != ']'; ++p) {
                    if (*p == '-') { neg = true; continue; }
                    if (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); in = true; continue; }
                    if (in) { toks.push_back((int) (neg ? -v : v)); v = 0; in = false; neg = false; }
                }
                if (in) toks.push_back((int) (neg ? -v : v));
            }
        }
        if (toks.empty() || gen <= 0) {
            std::printf("{\"error\":\"need tokens and gen\"}\n");
            std::fflush(stdout);
            continue;
        }
        if (pos + toks.size() + (size_t) gen + 8 >= (size_t) kMaxCtx) {
            std::printf("{\"error\":\"context full; send op reset\"}\n");
            std::fflush(stdout);
            continue;
        }
        const bool use_mtp = mtp && !e.split;
        if (!primed) first_tok = toks[0];
        // ingest the request's tokens (each also refreshes e.hlogits / e.h_normed)
        for (int tok : toks) {
            forward_token(e, tok, pos);
            ++pos;
        }
        if (!primed) {
            // seed MTP state: normed hidden of the last ingested token + draft slot 0
            CK(cudaMemcpyAsync(e.h_save, e.h_normed, (size_t) kEmb * 4, cudaMemcpyDeviceToDevice, e.sp));
            CK(cudaStreamSynchronize(e.sp));
            if (use_mtp) draft_predict(e, first_tok, 0, e.h_zero, e.p_zero, false);
            primed = true;
        }
        const float* h_part = e.hn_partials;
        // the last ingested token's logits (e.hlogits) produced the first generated token
        int cur = serve_argmax(e.hlogits);
        auto emit_token = [&](int id) { topk_out(e.hlogits, kVocab, 40, id); };
        if (!use_mtp) {
            long long out = 0;
            while (out < gen) {
                emit_token(cur);
                ++out;
                if (cur == 248046) break;
                int id = forward_token(e, cur, pos);
                ++pos;
                cur = id;
            }
            std::printf("{\"done\":true,\"pos\":%d}\n", pos);
            std::fflush(stdout);
            continue;
        }
        // MTP greedy path: cur is already generated (emitted below); the draft proposes
        // pos+1 and each batched verify confirms it, exactly like the CLI loop.
        long long emitted = 1;
        emit_token(cur);
        if (cur != 248046) {
            int cand = draft_predict(e, cur, pos, e.h_save, h_part, true);
            while (emitted < gen) {
                launch_token_verify(e, cur, cand, pos);
                int c0 = serve_argmax(e.hlogits2);
                int c1 = serve_argmax(e.hlogits2 + kVocab);
                {
                    std::vector<std::pair<int, float>> top;
                    const float* l0 = e.hlogits2;
                    top.reserve(40);
                    for (int i = 0; i < kVocab; ++i) {
                        if ((int) top.size() < 40) {
                            top.push_back({i, l0[i]});
                            if ((int) top.size() == 40) std::sort(top.begin(), top.end(),
                                                                 [](auto& a, auto& b) { return a.second > b.second; });
                        } else if (l0[i] > top.back().second) {
                            top.back() = {i, l0[i]};
                            for (size_t j = top.size() - 1; j > 0 && top[j].second > top[j - 1].second; --j)
                                std::swap(top[j], top[j - 1]);
                        }
                    }
                    std::printf("{\"id\":%d,", c0);
                    json_escape_out("top", top);
                    std::printf("}\n");
                    std::fflush(stdout);
                }
                ++emitted;
                if (c0 == cand) {
                    if (emitted < gen) {
                        std::vector<std::pair<int, float>> top1;
                        const float* l1 = e.hlogits2 + kVocab;
                        top1.reserve(40);
                        for (int i = 0; i < kVocab; ++i) {
                            if ((int) top1.size() < 40) {
                                top1.push_back({i, l1[i]});
                                if ((int) top1.size() == 40) std::sort(top1.begin(), top1.end(),
                                                                       [](auto& a, auto& b) { return a.second > b.second; });
                            } else if (l1[i] > top1.back().second) {
                                top1.back() = {i, l1[i]};
                                for (size_t j = top1.size() - 1; j > 0 && top1[j].second > top1[j - 1].second; --j)
                                    std::swap(top1[j], top1[j - 1]);
                            }
                        }
                        std::printf("{\"id\":%d,", c1);
                        json_escape_out("top", top1);
                        std::printf("}\n");
                        std::fflush(stdout);
                        ++emitted;
                        cur = c1;
                        pos += 2;
                        draft_predict(e, c0, pos - 1, e.h_save, e.h_partials2, false);
                        cand = draft_predict(e, c1, pos, e.h_save + kEmb, e.h_partials2 + kEmb / 4, true);
                        h_part = e.h_partials2 + kEmb / 4;
                    } else {
                        cur = c0;
                        pos += 2;
                    }
                } else {
                    for (int il = 0; il < kLayers; ++il) {
                        if (e.layers[il].recr) {
                            CK(cudaMemcpyAsync(e.layers[il].gdn_state,
                                               e.gdn_snap + (size_t) il * (kS * kHv * kS),
                                               (size_t) kS * kHv * kS * 4, cudaMemcpyDeviceToDevice, e.sp));
                            CK(cudaMemcpyAsync(e.layers[il].conv_state,
                                               e.conv_snap + (size_t) il * (3 * kQkv), (size_t) 3 * kQkv * 4,
                                               cudaMemcpyDeviceToDevice, e.sp));
                        }
                    }
                    cur = c0;
                    pos += 1;
                    cand = draft_predict(e, c0, pos, e.h_save, e.h_partials2, true);
                    h_part = e.h_partials2;
                }
                if (cur == 248046) break;
            }
        }
        std::printf("{\"done\":true,\"pos\":%d}\n", pos);
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    std::string model = "/data/hermes/models/qwythos-9b-v2/Qwythos-9B-v2-MTP-Q6_K.gguf";
    std::vector<int> tokens;
    bool serve_mode = false;
    int ngen = 32;
    float split = 0.0f;
    int device = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--model") model = need("--model");
        else if (a == "--n") ngen = std::atoi(need("--n").c_str());
        else if (a == "--split") split = std::atof(need("--split").c_str());
        else if (a == "--device") device = std::atoi(need("--device").c_str());
        else if (a == "--serve") serve_mode = true;
        else if (a == "--tokens") {
            std::string s = need("--tokens");
            size_t p = 0;
            while (p < s.size()) {
                size_t c = s.find(',', p);
                if (c == std::string::npos) c = s.size();
                tokens.push_back(std::atoi(s.substr(p, c - p).c_str()));
                p = c + 1;
            }
        } else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            std::exit(2);
        }
    }
    if (serve_mode) { g_serve_mode = true; tokens.push_back(0); }  // the arg check below only wants non-empty
    if (!serve_mode && (tokens.empty() || ngen < 1)) {
        std::fprintf(stderr, "usage: qwythos --tokens id,id --n 32 [--device N] [--split 0.22] [--model path]\n");
        return 2;
    }
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (device < 0) {
        size_t best = 0;
        device = 0;
        for (int i = 0; i < ndev; ++i) {
            cudaDeviceProp cand{};
            CK(cudaGetDeviceProperties(&cand, i));
            if ((size_t) cand.totalGlobalMem >= best) {
                best = (size_t) cand.totalGlobalMem;
                device = i;
            }
        }
    }
    if (device >= ndev) {
        std::fprintf(stderr, "device %d out of %d\n", device, ndev);
        return 2;
    }
    Engine e;
    e.primary = device;
    e.secondary = device == 0 ? 1 : 0;
    e.split = split;
    if (split > 0.0f && ndev < 2) {
        std::fprintf(stderr, "split needs two devices\n");
        return 2;
    }
    if (split <= 0.0f) e.split = 0.0f;
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, e.primary));
    if (!g_serve_mode) std::printf("primary %d %s %.1f GiB\n", e.primary, prop.name,
                (double) prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    if (e.split > 0.0f) {
        CK(cudaGetDeviceProperties(&prop, e.secondary));
        if (!g_serve_mode) std::printf("secondary %d %s split %.3f\n", e.secondary, prop.name, e.split);
    }
    std::fflush(stdout);
    try {
        auto t0 = std::chrono::steady_clock::now();
        strata::GgufFile file(model);
        alloc_primary(e);
        load_model(e, file);
        auto t1 = std::chrono::steady_clock::now();
        if (!g_serve_mode) std::printf("load_s %.2f\n", std::chrono::duration<double>(t1 - t0).count());
        if (serve_mode) {
            serve_loop(e);
            return 0;
        }
        probe_gemv(e);
        capture_graph(e);
        std::vector<int> seq = tokens;
        int prompt = (int) tokens.size();
        for (int i = 0; i < prompt - 1; ++i) forward_token(e, seq[(size_t) i], i);
        auto t2 = std::chrono::steady_clock::now();
        int cur = seq.back();
        int pos = prompt - 1;
        int first = 0;
        float first_logit = 0.0f;
        e.seed_pos = pos;
        static const bool nomtp = std::getenv("QWYTHOS_NOMTP") != nullptr;
        int mtp_steps = 0, mtp_accepted = 0;
        // Seed pass: feeds the last prompt token at position pos, producing the first generated token
        // (next0, for position pos+1), the trunk hidden for the fed token (e.x -> h_save) and its
        // sum-of-squares partials (e.partials).  The draft consumes (h_save, emb(next0)).
        {
            int id = forward_token(e, cur, pos);
            first = id;
            first_logit = e.hlogits[id];
            seq.push_back(id);
            cur = id;
            ++pos;
            CK(cudaMemcpyAsync(e.h_save, e.h_normed, (size_t) kEmb * 4, cudaMemcpyDeviceToDevice, e.sp));
            CK(cudaStreamSynchronize(e.sp));
        }
        int emitted = 1;
        const float* h_part = e.hn_partials;
        // the draft layer's KV needs slot 0: it processes position 0 with the shifted-hidden
        // convention (zero h_{-1}, emb(prompt token 0)) - without it every draft attends a hole.
        if (!nomtp) draft_predict(e, seq[0], 0, e.h_zero, e.p_zero, false);
        // the draft's candidate: prediction for position pos+1, from (hidden at pos, emb(cur))
        int cand = nomtp ? -1 : draft_predict(e, cur, pos, e.h_save, h_part, true);
        if (std::getenv("QWYTHOS_CMPV")) {
            if (const char* vn = std::getenv("QWYTHOS_CMPV_N")) e.vstop = std::atoi(vn);
            // one-shot diagnostic: verify [cur @ pos, junk @ pos+1] vs a true single pass for cur,
            // from the same pre-verify state.  Prints logits argmax + max hidden diff.
            for (int il = 0; il < kLayers; ++il) {
                if (e.layers[il].recr) {
                    CK(cudaMemcpy(e.gdn_pre + (size_t) il * (kS * kHv * kS), e.layers[il].gdn_state,
                                  (size_t) kS * kHv * kS * 4, cudaMemcpyDeviceToDevice));
                    CK(cudaMemcpy(e.conv_pre + (size_t) il * (3 * kQkv), e.layers[il].conv_state,
                                  (size_t) 3 * kQkv * 4, cudaMemcpyDeviceToDevice));
                }
            }
            launch_token_verify(e, cur, 0, pos);
            std::vector<float> lv((size_t) kVocab), hv((size_t) kEmb);
            for (int i = 0; i < kVocab; ++i) lv[i] = e.hlogits2[i];
            CK(cudaMemcpy(hv.data(), e.x2, (size_t) kEmb * 4, cudaMemcpyDeviceToHost));
            // roll ALL state back to pre-verify, then the true single pass
            for (int il = 0; il < kLayers; ++il) {
                if (e.layers[il].recr) {
                    CK(cudaMemcpy(e.layers[il].gdn_state, e.gdn_pre + (size_t) il * (kS * kHv * kS),
                                  (size_t) kS * kHv * kS * 4, cudaMemcpyDeviceToDevice));
                    CK(cudaMemcpy(e.layers[il].conv_state, e.conv_pre + (size_t) il * (3 * kQkv),
                                  (size_t) 3 * kQkv * 4, cudaMemcpyDeviceToDevice));
                }
            }
            e.hmeta[0] = pos;
            CK(cudaMemcpyAsync(e.dmeta, e.hmeta, sizeof(int), cudaMemcpyHostToDevice, e.sp));
            // seed the single pass from the TRUE embedding row of cur (not from the verify's hidden!)
            {
                const uint8_t* src = e.emb_host + (size_t) cur * e.emb_stride;
                for (int b = 0; b < kEmb / 256; ++b)
                    strata::dequantize_q6_K(src + (size_t) b * 210, e.emb_row + b * 256);
            }
            CK(cudaMemcpyAsync(e.x, e.emb_row, (size_t) kEmb * 4, cudaMemcpyHostToDevice, e.sp));
            row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x, e.partials, kEmb);
            fill_pos_kernel<<<1, 32, 0, e.sp>>>(e.dpos, kHeads, e.dmeta);
            launch_token_gpu(e);
            CK(cudaStreamSynchronize(e.sp));
            CK(cudaMemcpy(e.hlogits, e.logits, (size_t) kVocab * 4, cudaMemcpyDeviceToHost));
            int id_single = 0;
            for (int i = 1; i < kVocab; ++i) if (e.hlogits[i] > e.hlogits[id_single]) id_single = i;
            int c0v = 0;
            for (int i = 1; i < kVocab; ++i) if (lv[i] > lv[c0v]) c0v = i;
            double maxd = 0;
            for (int i = 0; i < kEmb; ++i) {
                double d = std::fabs((double) hv[i] - (double) e.hlogits[0]);
                (void) d;
            }
            std::printf("CMPV: verify c0 %d   single %d   (cur %d pos %d)\n", c0v, id_single, cur, pos);
            // ---- layer-0 step-by-step (non-tautological): single FIRST from the true embedding,
            // then verify-style on x2 col0, dumping stage values around the blown index 3994 ----
            Layer& L0 = e.layers[0];
            const int probe = 3994;
            float vA = 0;
            // the single pass above advanced the layer-0 state: restore it so A starts where B will
            CK(cudaMemcpy(e.layers[0].gdn_state, e.gdn_pre, (size_t) kS * kHv * kS * 4, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(e.layers[0].conv_state, e.conv_pre, (size_t) 3 * kQkv * 4, cudaMemcpyDeviceToDevice));
            // variant A: single-path layer 0, seeded from the embedding
            {
                const uint8_t* srcb = e.emb_host + (size_t) cur * e.emb_stride;
                for (int b = 0; b < kEmb / 256; ++b)
                    strata::dequantize_q6_K(srcb + (size_t) b * 210, e.emb_row + b * 256);
            }
            CK(cudaMemcpyAsync(e.x, e.emb_row, (size_t) kEmb * 4, cudaMemcpyHostToDevice, e.sp));
            row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x, e.partials, kEmb);
            rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.x, (const float*) L0.attn_norm.a, e.partials,
                                                         static_cast<Q81Row*>(e.scratch_p), kEmb, e.eps);
            CK(cudaStreamSynchronize(e.sp));
            std::vector<uint8_t> q8a((size_t) kEmb / 32 * 36);
            CK(cudaMemcpy(q8a.data(), e.scratch_p, q8a.size(), cudaMemcpyDeviceToHost));
            float pa[2] = {0, 0};
            CK(cudaMemcpy(pa, e.partials, 8, cudaMemcpyDeviceToHost));
            mm_q(L0.qkvz.type, L0.qkvz.a, e.scratch_p, e.qkvz_o, L0.qkvz.n_in, L0.qkvz.n_out, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            CK(cudaMemcpy(&vA, e.qkvz_o + 8090, 4, cudaMemcpyDeviceToHost));
            apply_mm(e, L0.ab, e.x, e.ab_o, true);
            float* beta_a = e.ab_o;
            float* alpha_a = beta_a + kHv;
            beta_gate_kernel<<<1, 64, 0, e.sp>>>(beta_a, alpha_a, L0.ssm_dt, L0.ssm_a, e.ggate, kHv);
            CK(cudaStreamSynchronize(e.sp));
            float vinA = 0, csA = 0;
            CK(cudaMemcpy(&vinA, e.qkvz_o + 8090, 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&csA, L0.conv_state + 3 * 8090, 4, cudaMemcpyDeviceToHost));
            strata::kernels::fused_gdn_conv_l2_qk(L0.conv_state, e.qkvz_o, L0.conv_w, e.qkvz_o, kQkv, 2 * kHk, 0,
                                                  0.0f, e.eps, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            float convA = 0;
            CK(cudaMemcpy(&convA, e.qkvz_o + 8090, 4, cudaMemcpyDeviceToHost));
            strata::kernels::fused_gdn_step_norm_silu(L0.gdn_state, e.qkvz_o, e.qkvz_o + kHk * kS,
                                                      e.qkvz_o + 2 * kHk * kS, e.ggate, beta_a,
                                                      e.qkvz_o + kQkv, L0.ssm_norm, e.eps, e.up,
                                                      e.scratch_up8, kHk, kHv, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            float upA = 0;
            CK(cudaMemcpy(&upA, e.up + probe, 4, cudaMemcpyDeviceToHost));
            mm_q_res(L0.ssm_out.type, L0.ssm_out.a, e.scratch_up8, e.x, e.partials, L0.ssm_out.n_in,
                     L0.ssm_out.n_out, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            float xA = 0;
            CK(cudaMemcpy(&xA, e.x + probe, 4, cudaMemcpyDeviceToHost));
            ffn(e, L0);
            CK(cudaStreamSynchronize(e.sp));
            float xAf = 0;
            CK(cudaMemcpy(&xAf, e.x + probe, 4, cudaMemcpyDeviceToHost));
            // variant B: verify-style layer 0 col0 on x2 (reseeded from the embedding)
            CK(cudaMemcpy(e.layers[0].gdn_state, e.gdn_pre, (size_t) kS * kHv * kS * 4, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(e.layers[0].conv_state, e.conv_pre, (size_t) 3 * kQkv * 4, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpyAsync(e.x2, e.emb_row, (size_t) kEmb * 4, cudaMemcpyHostToDevice, e.sp));
            row_sum_partials_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2, e.partials2, kEmb);
            rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.x2, (const float*) L0.attn_norm.a, e.partials2,
                                                         reinterpret_cast<Q81Row*>((char*) e.scratch_p2), kEmb,
                                                         e.eps);
            CK(cudaStreamSynchronize(e.sp));
            std::vector<uint8_t> q8b((size_t) kEmb / 32 * 36);
            CK(cudaMemcpy(q8b.data(), e.scratch_p2, q8b.size(), cudaMemcpyDeviceToHost));
            float pb[2] = {0, 0};
            CK(cudaMemcpy(pb, e.partials2, 8, cudaMemcpyDeviceToHost));
            {
                int qd = 0;
                for (size_t i = 0; i < q8a.size(); ++i)
                    if (q8a[i] != q8b[i]) ++qd;
                std::printf("CMPV q8-at-rms: byte diff %d/%zu  partials A %g %g B %g %g\n", qd, q8a.size(), pa[0],
                            pa[1], pb[0], pb[1]);
                std::fflush(stdout);
            }
            // head-to-head on the byte-identical Q8 input: single-column kernel vs multi kernel
            mm_q(L0.qkvz.type, L0.qkvz.a, e.scratch_p2, e.q3_o, L0.qkvz.n_in, L0.qkvz.n_out, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            float v_single_on_p2 = 0;
            CK(cudaMemcpy(&v_single_on_p2, e.q3_o + 8090, 4, cudaMemcpyDeviceToHost));
            mm_q2(L0.qkvz.type, L0.qkvz.a, e.scratch_p2, e.qkvz_o2, L0.qkvz.n_in, L0.qkvz.n_out, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            float v_multi_on_p2 = 0;
            CK(cudaMemcpy(&v_multi_on_p2, e.qkvz_o2 + 8090, 4, cudaMemcpyDeviceToHost));
            std::printf("CMPV head2head: single-kernel(scratch_p2) v=%g  multi-kernel(scratch_p2) v=%g  A(single on scratch_p) v=%g\n",
                        v_single_on_p2, v_multi_on_p2, vA);
            std::fflush(stdout);
            mm_q2(L0.ab.type, L0.ab.a, e.scratch_p2, e.ab_o2, L0.ab.n_in, L0.ab.n_out, e.sp);
            float* beta_b = e.ab_o2;
            float* alpha_b = beta_b + kHv;
            beta_gate_kernel<<<1, 64, 0, e.sp>>>(beta_b, alpha_b, L0.ssm_dt, L0.ssm_a, e.ggate2, kHv);
            float* qkv_b = e.qkvz_o2;
            float* z_b = qkv_b + kQkv;
            CK(cudaStreamSynchronize(e.sp));
            float vinB = 0, csB = 0;
            CK(cudaMemcpy(&vinB, qkv_b + 8090, 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&csB, L0.conv_state + 3 * 8090, 4, cudaMemcpyDeviceToHost));
            strata::kernels::fused_gdn_conv_l2_qk(L0.conv_state, qkv_b, L0.conv_w, qkv_b, kQkv, 2 * kHk, 0,
                                                  0.0f, e.eps, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            float convB = 0;
            CK(cudaMemcpy(&convB, qkv_b + 8090, 4, cudaMemcpyDeviceToHost));
            strata::kernels::fused_gdn_step_norm_silu(L0.gdn_state, qkv_b, qkv_b + kHk * kS,
                                                      qkv_b + 2 * kHk * kS, e.ggate2, beta_b, z_b,
                                                      L0.ssm_norm, e.eps, e.up2,
                                                      (char*) e.scratch_up8_2, kHk, kHv, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            float upB = 0;
            CK(cudaMemcpy(&upB, e.up2 + probe, 4, cudaMemcpyDeviceToHost));
            mm_q2(L0.ssm_out.type, L0.ssm_out.a, e.scratch_up8_2, e.o2, L0.ssm_out.n_in, L0.ssm_out.n_out, e.sp);
            CK(cudaStreamSynchronize(e.sp));
            float ssmB = 0;
            CK(cudaMemcpy(&ssmB, e.o2 + probe, 4, cudaMemcpyDeviceToHost));
            add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2, e.o2, e.partials2, kEmb);
            rms_quant_apply_kernel<<<16, 256, 0, e.sp>>>(e.x2, (const float*) L0.post_norm.a, e.partials2,
                                                         reinterpret_cast<Q81Row*>((char*) e.scratch_p2), kEmb,
                                                         e.eps);
            mm_q2(L0.upgate.type, L0.upgate.a, e.scratch_p2, e.ug_o2, L0.upgate.n_in, L0.upgate.n_out, e.sp);
            swiglu_quant_kernel<<<kFF / 256, 256, 0, e.sp>>>(e.ug_o2, e.ug_o2 + kFF,
                                                             static_cast<Q81Row*>(e.scratch_dn8_2), kFF);
            mm_q2(L0.down.type, L0.down.a, e.scratch_dn8_2, e.o2, L0.down.n_in, L0.down.n_out, e.sp);
            add_partials_1024_kernel<<<kEmb / 4, 1, 0, e.sp>>>(e.x2, e.o2, e.partials2, kEmb);
            CK(cudaStreamSynchronize(e.sp));
            float xB = 0;
            CK(cudaMemcpy(&xB, e.x2 + probe, 4, cudaMemcpyDeviceToHost));
            std::printf("CMPV stages @%d: vin A %g B %g | cstate A %g B %g | conv A %g B %g | up A %g B %g | x-ssm A %g (o2B %g) | x-ffn A %g B %g\n",
                        probe, vinA, vinB, csA, csB, convA, convB, upA, upB, xA, ssmB, xAf, xB);
            {
                // v-head 31 elem 26: v index 8090, q index 3994, k index 6042, z index 3994
                float vA, vB, qA, qB, kA, kB, zA, zB, bA, bB, gA, gB;
                CK(cudaMemcpy(&vA, e.qkvz_o + 8090, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&vB, e.qkvz_o2 + 8090, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&qA, e.qkvz_o + 3994, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&qB, e.qkvz_o2 + 3994, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&kA, e.qkvz_o + 6042, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&kB, e.qkvz_o2 + 6042, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&zA, e.qkvz_o + kQkv + 3994, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&zB, e.qkvz_o2 + kQkv + 3994, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&bA, e.ab_o + 31, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&bB, e.ab_o2 + 31, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&gA, e.ggate + 31, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&gB, e.ggate2 + 31, 4, cudaMemcpyDeviceToHost));
                std::printf("CMPV head31: q %g/%g  k %g/%g  v %g/%g  z %g/%g  beta %g/%g  gate %g/%g\n",
                            qA, qB, kA, kB, vA, vB, zA, zB, bA, bB, gA, gB);
                {
                    // kernel-vs-input isolation: single-column kernel on the VERIFY's Q8 bytes
                    mm_q(L0.qkvz.type, L0.qkvz.a, e.scratch_p2, e.q3_o, L0.qkvz.n_in, L0.qkvz.n_out, e.sp);
                    CK(cudaStreamSynchronize(e.sp));
                    float vB1 = 0;
                    CK(cudaMemcpy(&vB1, e.q3_o + 8090, 4, cudaMemcpyDeviceToHost));
                    // and byte-compare the two Q8 inputs (both freshly written from the same emb row)
                    std::vector<uint8_t> qaa((size_t) kEmb / 32 * 36), qbb((size_t) kEmb / 32 * 36);
                    CK(cudaMemcpy(qaa.data(), e.scratch_p, qaa.size(), cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(qbb.data(), e.scratch_p2, qbb.size(), cudaMemcpyDeviceToHost));
                    int qd = 0;
                    for (size_t i = 0; i < qaa.size(); ++i)
                        if (qaa[i] != qbb[i]) ++qd;
                    std::printf("CMPV gemv: single-kernel-on-verify-q8 v=%g (A %g, multi %g)  q8 byte diff %d/%zu\n",
                                vB1, vA, vB, qd, qaa.size());
                    float xin[4], x2in[4], pin[4], p2in[4], erow[4];
                    CK(cudaMemcpy(xin, e.x, 16, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(x2in, e.x2, 16, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(pin, e.partials, 16, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(p2in, e.partials2, 16, cudaMemcpyDeviceToHost));
                    std::memcpy(erow, e.emb_row, 16);
                    std::printf("CMPV inputs: x %g %g | x2 %g %g | partials %g %g | partials2 %g %g | emb_row %g %g\n",
                                xin[0], xin[1], x2in[0], x2in[1], pin[0], pin[1], p2in[0], p2in[1], erow[0],
                                erow[1]);
                }
            }
            std::fflush(stdout);
            std::exit(0);
        }
        while (emitted < ngen) {
            if (nomtp || cand < 0 || emitted >= ngen - 1 || pos + 2 >= kMaxCtx) {
                int id = forward_token(e, cur, pos);
                seq.push_back(id);
                ++emitted;
                cur = id;
                ++pos;
                CK(cudaMemcpyAsync(e.h_save, e.h_normed, (size_t) kEmb * 4, cudaMemcpyDeviceToDevice, e.sp));
                CK(cudaStreamSynchronize(e.sp));
                h_part = e.hn_partials;
                if (!nomtp) cand = draft_predict(e, cur, pos, e.h_save, h_part, true);
                continue;
            }
            // one batched pass verifies the candidate: [cur @ pos, cand @ pos+1]
            launch_token_verify(e, cur, cand, pos);
            ++mtp_steps;
            static const bool mtptrace = std::getenv("QWYTHOS_MTPTRACE") != nullptr;
            int c0 = 0, c1 = 0;
            const float* l0 = e.hlogits2;
            const float* l1 = e.hlogits2 + kVocab;
            for (int i = 1; i < kVocab; ++i) {
                if (l0[i] > l0[c0]) c0 = i;
                if (l1[i] > l1[c1]) c1 = i;
            }
            if (mtptrace) std::printf("mtp step %d pos %d cur %d cand %d c0 %d c1 %d %s\n",
                                       mtp_steps, pos, cur, cand, c0, c1, c0 == cand ? "ACC" : "REJ");
            seq.push_back(c0);
            ++emitted;
            static const bool noaccept = std::getenv("QWYTHOS_NOACCEPT") != nullptr;
            if (c0 == cand && !noaccept) {
                // the draft was confirmed: c1 (sampled at column 1) is final - two tokens for one pass
                ++mtp_accepted;
                seq.push_back(c1);
                ++emitted;
                cur = c1;
                pos += 2;
                // keep the draft cache dense: a no-head draft step at pos+1 (col0 hidden, emb(c0))
                draft_predict(e, c0, pos - 1, e.h_save, e.h_partials2, false);
                // next candidate: prediction for pos+2 from (hidden at pos+1 = col1, emb(c1))
                cand = draft_predict(e, c1, pos, e.h_save + kEmb, e.h_partials2 + kEmb / 4, true);
                h_part = e.h_partials2 + kEmb / 4;
            } else {
                // rejected: roll the GDN states back to after column 0; col0's hidden/logits are valid
                for (int il = 0; il < kLayers; ++il) {
                    if (e.layers[il].recr) {
                        CK(cudaMemcpyAsync(e.layers[il].gdn_state, e.gdn_snap + (size_t) il * (kS * kHv * kS),
                                           (size_t) kS * kHv * kS * 4, cudaMemcpyDeviceToDevice, e.sp));
                        CK(cudaMemcpyAsync(e.layers[il].conv_state,
                                           e.conv_snap + (size_t) il * (3 * kQkv), (size_t) 3 * kQkv * 4,
                                           cudaMemcpyDeviceToDevice, e.sp));
                    }
                }
                cur = c0;
                pos += 1;
                // corrected draft step at pos+1's position: (col0 hidden, emb(c0)) -> candidate for pos+2
                cand = draft_predict(e, c0, pos, e.h_save, e.h_partials2, true);
                h_part = e.h_partials2;
            }
            CK(cudaStreamSynchronize(e.sp));
        }
        std::printf("mtp steps %d accepted %d\n", mtp_steps, mtp_accepted);
        auto t3 = std::chrono::steady_clock::now();
        double gen_s = std::chrono::duration<double>(t3 - t2).count();
        double pre_s = std::chrono::duration<double>(t2 - t1).count();
        std::printf("prefill_tokens %d prefill_s %.3f\n", prompt > 0 ? prompt - 1 : 0, pre_s);
        std::printf("gen_tokens %d gen_s %.3f tok_s %.3f\n", ngen, gen_s, ngen / gen_s);
        std::printf("gpu_ms %.2f per_tok\n", e.layer_ms / ngen);
        std::printf("ids");
        for (int id : seq) std::printf(" %d", id);
        std::printf("\nfirst_gen %d logit %.6f\n", first, first_logit);
        std::fflush(stdout);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "qwythos: %s\n", ex.what());
        return 1;
    }
    return 0;
}
