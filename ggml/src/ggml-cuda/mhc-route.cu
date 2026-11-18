#include <cstdlib>

#include "common.cuh"
#include "mhc-route.cuh"

// Fused mHC routing chain (SenseNova-Loop). The reference graph spends ~14 nodes and
// therefore ~14 kernel launches per routing:
//   concat, rms_norm, mul_mat (GEMV 2*n_embd -> 8), add(bias),
//   sigmoid x2 (+ mul/add for two streams), scale,
//   sigmoid x2 + scale x2 (post), Sinkhorn (1 fused node)
// With 2 routings per mHC layer execution and 48 executions per token that is ~1300
// graph nodes per decoded token.
//
// Numerics:
//  * rms_norm: same formula as ggml_rms_norm (eps = 1e-6, scale = 1/sqrt(mean(x^2)+eps)),
//    single rounding per element like the reference kernel.
//  * GEMV: the accumulation order is the block-reduction order of this kernel, not the
//    hipBLAS order of the reference path. Differences are O(1 ULP) on the logits; near-tie
//    flips of the greedy argmax downstream are possible and accepted (documented in the
//    report). The routing math itself (sigmoid, Sinkhorn) is the exact code of
//    mhc-sinkhorn.cu, only evaluated on 4 lanes, except for the softplus identity below
//    (default on, <= 1 ULP, LLAMA_MHC_ROUTE_EXACT=1 restores the two-softplus form).
//  * mul/add in the input injection use __fmul_rn/__fadd_rn so the result matches the
//    unfused chain, where every mul/add is its own kernel (no FMA contraction).
//  * Sinkhorn normalize uses the exact identity softplus(d) == d + softplus(-d):
//        -(softplus(-d), softplus(d)) == (-s, -d - s)   with s = softplus(-d)
//    i.e. ONE log1pf(expf(.)) per normalize instead of two, which halves the dependent
//    chain of the Sinkhorn loop. Both forms are mathematically identical; in floating
//    point they differ by the rounding of the sum d + s, i.e. up to 1 ULP of |d| (not of
//    the result: for d << 0 the sum cancels). Simulated against float64 over 20 iterations
//    the mix weights are off by <= 6e-7 (two-softplus form: <= 5e-7), and the measured
//    KL divergence against the unfused graph is the same for both forms (see README).
//    A sign-stable variant (softplus(-|d|) + select) and an early exit at the floating
//    point fixpoint were measured and rejected: +1.5 us/call resp. no gain, since the
//    real routing logits rarely reach a fixpoint within 20 iterations.
//    LLAMA_MHC_ROUTE_EXACT=1 restores the two-softplus form as the A/B reference; the
//    CPU fallback (ggml-cpu) is untouched and stays bit-exact.
static __device__ __forceinline__ float mhc_route_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

// normalize(x, y) = log(soft_max([x, y])) == -(softplus(-d), softplus(d)) with d = x - y
// Reference form, two softplus (LLAMA_MHC_ROUTE_EXACT=1).
static __device__ __forceinline__ void mhc_route_normalize_exact(float & x, float & y) {
    const float d = __fsub_rn(x, y);
    x = -mhc_route_softplus(-d);
    y = -mhc_route_softplus( d);
}

// Optimized form, one softplus: s = softplus(-d), x = -s, y = -d - s.
static __device__ __forceinline__ void mhc_route_normalize_fast(float & x, float & y) {
    const float d = __fsub_rn(x, y);
    const float s = mhc_route_softplus(-d);
    x = -s;
    y = __fsub_rn(-d, s);
}

template<bool EXACT>
static __device__ __forceinline__ void mhc_route_normalize(float & x, float & y) {
    if (EXACT) {
        mhc_route_normalize_exact(x, y);
    } else {
        mhc_route_normalize_fast(x, y);
    }
}

// One lane of a 2-element normalize: 'd' is the same x - y difference on both lanes of the
// pair, 'high' selects the lane that holds y (the second element).
template<bool EXACT>
static __device__ __forceinline__ float mhc_route_normalize_lane(float d, bool high) {
    if (EXACT) {
        return high ? -mhc_route_softplus(d) : -mhc_route_softplus(-d);
    }
    const float s = mhc_route_softplus(-d);
    return high ? __fsub_rn(-d, s) : -s;
}

// same form as ggml-cuda/unary.cu (GGML_UNARY_OP_SIGMOID)
static __device__ __forceinline__ float mhc_route_sigmoid(float x) {
    return 1.0f/(1.0f + expf(-x));
}

// One thread block per token. The block first reduces the rms_norm scale of the
// concatenated [a | b] vector, then the 8 router logits, then writes the [n_embd] input
// row; warp 0 finishes the post/mix rows (including the Sinkhorn loop).
//
// Performance notes (measured, see report):
//  * decode runs this with n_tok = 1, i.e. ONE block on one CU. The kernel is therefore
//    latency bound, not throughput bound: the serial 2x2 Sinkhorn chain dominates.
//  * a/b are loaded once (float4), kept in registers for the rms sum, the GEMV and the
//    output row -> a single pass instead of three. The rms scale is applied to the logits
//    instead of to the 2*n_embd inputs (identical value up to the reduction rounding).
//  * the 2x2 log-space Sinkhorn runs on 4 lanes, one softplus each, with one lane
//    exchange per half-iteration: 2 dependent softplus per iteration instead of 8.
//  * with the softplus identity each of those two softplus is the only transcendent call
//    of its normalize; the second element is derived arithmetically (-d - s).
//  * NTH*NPT*4 == n_embd is required for the fast path; any other shape uses the generic
//    scalar kernel below (also kept as the reference for A/B).
template<int NTH, int NPT, bool EXACT>
static __global__ void __launch_bounds__(NTH) mhc_route_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        const float * __restrict__ weight,
        const float * __restrict__ bias,
        float * __restrict__ dst,
        int64_t n_embd,
        int64_t s_a1,
        int64_t s_b1,
        int64_t s_w1,
        int64_t s_d1,
        int32_t n_iter,
        float   pre_scale) {
    const int64_t t    = blockIdx.x;
    const int     tid  = threadIdx.x;
    const int     lane = tid & 31;
    const int     warp = tid >> 5;
    constexpr int NW   = NTH/32;

    ggml_cuda_pdl_lc();

    __shared__ float s_ss[NW];
    __shared__ float s_red[8][NW];
    __shared__ float s_scale;
    __shared__ float s_logits[8];

    const float4 * pa  = (const float4 *) (a + t*s_a1);
    const float4 * pb  = (const float4 *) (b + t*s_b1);
    float4       * po  = (float4 *)       (dst + t*s_d1);
    float        * pot = dst + t*s_d1;
    const float4 * pw  = (const float4 *) weight;
    const int sw4  = (int) (s_w1 >> 2);      // weight row stride in float4
    const int nf4w = (int) (n_embd >> 2);    // start of the b half inside a weight row

    ggml_cuda_pdl_sync();

    // ---- one pass over a/b: rms sum of squares, kept for GEMV and output row ---------
    float4 va[NPT], vb[NPT];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        const int i = tid + j*NTH;
        va[j] = pa[i];
        vb[j] = pb[i];
        ss = fmaf(va[j].x, va[j].x, ss); ss = fmaf(va[j].y, va[j].y, ss);
        ss = fmaf(va[j].z, va[j].z, ss); ss = fmaf(va[j].w, va[j].w, ss);
        ss = fmaf(vb[j].x, vb[j].x, ss); ss = fmaf(vb[j].y, vb[j].y, ss);
        ss = fmaf(vb[j].z, vb[j].z, ss); ss = fmaf(vb[j].w, vb[j].w, ss);
    }
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down_sync(GGML_CUDA_FULL_THREAD_MASK, ss, off, 32);
    }
    if (lane == 0) {
        s_ss[warp] = ss;
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < NW; ++w) {
            s += s_ss[w];
        }
        s_scale = 1.0f/sqrtf(s/(2.0f*n_embd) + 1e-6f);
    }

    // ---- 8 router logits (GEMV over the concatenated vector) -------------------------
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        acc[k] = 0.0f;
    }
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        const int i = tid + j*NTH;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 wa = pw[k*sw4 + i];
            acc[k] = fmaf(va[j].x, wa.x, acc[k]); acc[k] = fmaf(va[j].y, wa.y, acc[k]);
            acc[k] = fmaf(va[j].z, wa.z, acc[k]); acc[k] = fmaf(va[j].w, wa.w, acc[k]);
            const float4 wb = pw[k*sw4 + nf4w + i];
            acc[k] = fmaf(vb[j].x, wb.x, acc[k]); acc[k] = fmaf(vb[j].y, wb.y, acc[k]);
            acc[k] = fmaf(vb[j].z, wb.z, acc[k]); acc[k] = fmaf(vb[j].w, wb.w, acc[k]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] += __shfl_down_sync(GGML_CUDA_FULL_THREAD_MASK, acc[k], off, 32);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            s_red[k][warp] = acc[k];
        }
    }
    __syncthreads();
    const float rscale = s_scale;
    if (warp == 0 && lane < 8) {
        float s = 0.0f;
        for (int w = 0; w < NW; ++w) {
            s += s_red[lane][w];
        }
        s_logits[lane] = fmaf(s, rscale, bias[lane]);
    }
    __syncthreads();

    // ---- result.input = (a*sigmoid(r0) + b*sigmoid(r1)) * pre_scale ------------------
    const float g0 = mhc_route_sigmoid(s_logits[0]);
    const float g1 = mhc_route_sigmoid(s_logits[1]);

#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        const int i = tid + j*NTH;
        float4 o;
        o.x = __fmul_rn(__fadd_rn(__fmul_rn(va[j].x, g0), __fmul_rn(vb[j].x, g1)), pre_scale);
        o.y = __fmul_rn(__fadd_rn(__fmul_rn(va[j].y, g0), __fmul_rn(vb[j].y, g1)), pre_scale);
        o.z = __fmul_rn(__fadd_rn(__fmul_rn(va[j].z, g0), __fmul_rn(vb[j].z, g1)), pre_scale);
        o.w = __fmul_rn(__fadd_rn(__fmul_rn(va[j].w, g0), __fmul_rn(vb[j].w, g1)), pre_scale);
        po[i] = o;
    }

    // ---- post / Sinkhorn mix rows ---------------------------------------------------
    // warp 0 only, overlapping the input row store of the other warps.
    // lanes 0..3 hold m0..m3 (lanes 4..31 mirror them, so all shuffles use the full mask).
    if (warp == 0) {
        const int role = lane & 3;
        float u = (role == 0) ? s_logits[4] : (role == 1) ? s_logits[5] :
                  (role == 2) ? s_logits[6] : s_logits[7];
        if (lane == 0) {
            pot[n_embd + 0] = __fmul_rn(mhc_route_sigmoid(s_logits[2]), 2.0f);
            pot[n_embd + 1] = __fmul_rn(mhc_route_sigmoid(s_logits[3]), 2.0f);
        }
        for (int32_t i = 0; i < n_iter; ++i) {
            // rows: (m0,m1) on lanes 0/1, (m2,m3) on lanes 2/3 -> one softplus per lane,
            // and (optimized form) only one softplus for the whole 2-element normalize
            float peer = __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, u, 1);
            const float d1 = ((role & 1) == 0) ? __fsub_rn(u, peer) : __fsub_rn(peer, u);
            u = mhc_route_normalize_lane<EXACT>(d1, (role & 1) != 0);
            // columns: (m0,m2) on lanes 0/2, (m1,m3) on lanes 1/3
            peer = __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, u, 2);
            const float d2 = ((role & 2) == 0) ? __fsub_rn(u, peer) : __fsub_rn(peer, u);
            u = mhc_route_normalize_lane<EXACT>(d2, (role & 2) != 0);
        }
        if (lane < 4) {
            pot[n_embd + 2 + lane] = expf(u);
        }
    }
}

// Generic scalar path: any n_embd/alignment, and the bit-exact reference for A/B.
template<bool EXACT>
static __global__ void mhc_route_f32_generic(
        const float * __restrict__ a,
        const float * __restrict__ b,
        const float * __restrict__ weight,
        const float * __restrict__ bias,
        float * __restrict__ dst,
        int64_t n_embd,
        int64_t s_a1,
        int64_t s_b1,
        int64_t s_w1,
        int64_t s_d1,
        int32_t n_iter,
        float   pre_scale) {
    const int64_t t    = blockIdx.x;
    const int     tid  = threadIdx.x;
    const int     nth  = blockDim.x;
    const int     lane = tid & 31;
    const int     warp = tid >> 5;
    const int     nwarp = nth >> 5;

    __shared__ float s_ss[32];
    __shared__ float s_red[8][32];
    __shared__ float s_scale;
    __shared__ float s_logits[8];

    const int64_t n_flat = 2*n_embd;

    const float * pa = a + t*s_a1;
    const float * pb = b + t*s_b1;
    float       * po = dst + t*s_d1;

    float ss = 0.0f;
    for (int64_t i = tid; i < n_embd; i += nth) {
        const float v = pa[i];
        ss += v*v;
    }
    for (int64_t i = tid; i < n_embd; i += nth) {
        const float v = pb[i];
        ss += v*v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down_sync(GGML_CUDA_FULL_THREAD_MASK, ss, off, 32);
    }
    if (lane == 0) {
        s_ss[warp] = ss;
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < nwarp; ++w) {
            s += s_ss[w];
        }
        s_scale = 1.0f/sqrtf(s/n_flat + 1e-6f);
    }
    __syncthreads();

    const float rscale = s_scale;

    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        acc[k] = 0.0f;
    }
    for (int64_t i = tid; i < n_embd; i += nth) {
        const float v = pa[i]*rscale;
        const float * w = weight + i;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] += v*w[k*s_w1];
        }
    }
    for (int64_t i = tid; i < n_embd; i += nth) {
        const float v = pb[i]*rscale;
        const float * w = weight + (i + n_embd);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] += v*w[k*s_w1];
        }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float v = acc[k];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(GGML_CUDA_FULL_THREAD_MASK, v, off, 32);
        }
        if (lane == 0) {
            s_red[k][warp] = v;
        }
    }
    __syncthreads();
    if (tid < 8) {
        float s = 0.0f;
        for (int w = 0; w < nwarp; ++w) {
            s += s_red[tid][w];
        }
        s_logits[tid] = s + bias[tid];
    }
    __syncthreads();

    const float g0 = mhc_route_sigmoid(s_logits[0]);
    const float g1 = mhc_route_sigmoid(s_logits[1]);

    for (int64_t i = tid; i < n_embd; i += nth) {
        const float v = __fadd_rn(__fmul_rn(pa[i], g0), __fmul_rn(pb[i], g1));
        po[i] = __fmul_rn(v, pre_scale);
    }

    if (tid == 0) {
        po[n_embd + 0] = __fmul_rn(mhc_route_sigmoid(s_logits[2]), 2.0f);
        po[n_embd + 1] = __fmul_rn(mhc_route_sigmoid(s_logits[3]), 2.0f);

        float mix0 = s_logits[4];
        float mix1 = s_logits[5];
        float mix2 = s_logits[6];
        float mix3 = s_logits[7];

        for (int32_t i = 0; i < n_iter; ++i) {
            mhc_route_normalize<EXACT>(mix0, mix1); // row 0
            mhc_route_normalize<EXACT>(mix2, mix3); // row 1
            mhc_route_normalize<EXACT>(mix0, mix2); // column 0
            mhc_route_normalize<EXACT>(mix1, mix3); // column 1
        }

        po[n_embd + 2] = expf(mix0);
        po[n_embd + 3] = expf(mix1);
        po[n_embd + 4] = expf(mix2);
        po[n_embd + 5] = expf(mix3);
    }
}

void ggml_cuda_op_mhc_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a      = dst->src[0];
    const ggml_tensor * b      = dst->src[1];
    const ggml_tensor * weight = dst->src[2];
    const ggml_tensor * bias   = dst->src[3];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1]);
    GGML_ASSERT(a->ne[2] == 1 && a->ne[3] == 1);
    GGML_ASSERT(weight->ne[0] == 2*a->ne[0] && weight->ne[1] == 8);
    GGML_ASSERT(bias->ne[0] == 8 && bias->ne[1] == 1);
    GGML_ASSERT(dst->ne[0] >= a->ne[0] + 6 && dst->ne[1] == a->ne[1]);
    GGML_ASSERT(a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float));

    const int64_t n_embd = a->ne[0];
    const int64_t n_tok  = a->ne[1];
    const int32_t n_iter = ggml_get_op_params_i32(dst, 0);
    const float pre_scale = ggml_get_op_params_f32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    const int64_t s_a1 = (int64_t) (a->nb[1]      / sizeof(float));
    const int64_t s_b1 = (int64_t) (b->nb[1]      / sizeof(float));
    const int64_t s_w1 = (int64_t) (weight->nb[1] / sizeof(float));
    const int64_t s_d1 = (int64_t) (dst->nb[1]    / sizeof(float));

    // vectorized path: one float4 per thread slice, 16B aligned, 512 threads x 2 float4
    const bool aligned =
        ((uintptr_t) a->data      % 16 == 0) &&
        ((uintptr_t) b->data      % 16 == 0) &&
        ((uintptr_t) weight->data % 16 == 0) &&
        ((uintptr_t) dst->data    % 16 == 0) &&
        (s_a1 % 4 == 0) && (s_b1 % 4 == 0) && (s_d1 % 4 == 0) && (s_w1 % 4 == 0);

    const dim3 grid_dims((unsigned) n_tok, 1, 1);

    // LLAMA_MHC_ROUTE_GENERIC=1 forces the scalar reference kernel (A/B of the two
    // implementations without a rebuild; both produce the same result up to the
    // documented O(1 ULP) reduction rounding).
    static const bool force_generic = [] {
        const char * e = getenv("LLAMA_MHC_ROUTE_GENERIC");
        return e != nullptr && atoi(e) != 0;
    }();

    // LLAMA_MHC_ROUTE_EXACT=1 selects the two-softplus Sinkhorn normalize (the previous
    // bit-exact form) instead of the default one-softplus identity (<= 1 ULP).
    static const bool exact_math = [] {
        const char * e = getenv("LLAMA_MHC_ROUTE_EXACT");
        return e != nullptr && atoi(e) != 0;
    }();

    if (n_embd == 4096 && aligned && !force_generic) {
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());
        if (exact_math) {
            ggml_cuda_kernel_launch(mhc_route_f32<512, 2, true>, launch_params,
                    (const float *) a->data,
                    (const float *) b->data,
                    (const float *) weight->data,
                    (const float *) bias->data,
                    (float *) dst->data,
                    n_embd, s_a1, s_b1, s_w1, s_d1, n_iter, pre_scale);
        } else {
            ggml_cuda_kernel_launch(mhc_route_f32<512, 2, false>, launch_params,
                    (const float *) a->data,
                    (const float *) b->data,
                    (const float *) weight->data,
                    (const float *) bias->data,
                    (float *) dst->data,
                    n_embd, s_a1, s_b1, s_w1, s_d1, n_iter, pre_scale);
        }
        return;
    }

    const dim3 block_dims(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());
    if (exact_math) {
        ggml_cuda_kernel_launch(mhc_route_f32_generic<true>, launch_params,
                (const float *) a->data,
                (const float *) b->data,
                (const float *) weight->data,
                (const float *) bias->data,
                (float *) dst->data,
                n_embd, s_a1, s_b1, s_w1, s_d1, n_iter, pre_scale);
    } else {
        ggml_cuda_kernel_launch(mhc_route_f32_generic<false>, launch_params,
                (const float *) a->data,
                (const float *) b->data,
                (const float *) weight->data,
                (const float *) bias->data,
                (float *) dst->data,
                n_embd, s_a1, s_b1, s_w1, s_d1, n_iter, pre_scale);
    }
}
