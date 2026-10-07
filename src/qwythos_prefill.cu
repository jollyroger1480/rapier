#include "strata/kernels/f16_bits.hpp"
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
