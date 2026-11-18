#include "common.cuh"
#include "mhc-merge.cuh"

// Fused mHC residual merge (SenseNova-Loop).
//
// Reference graph per merge (8 elementwise nodes, each a separate kernel launch):
//   next_a = add(mul(a, mix0), mul(b, mix1))
//   next_b = add(mul(a, mix2), mul(b, mix3))
//   a      = add(next_a, mul(branch, post0))
//   b      = add(next_b, mul(branch, post1))
// There are two merges per mHC layer execution, so the fusion removes 18 kernel
// launches per layer execution (9 per merge: 10 nodes -> 1) at n_tok=1.
//
// __fmul_rn/__fadd_rn force separate rounding, matching the unfused chain where every
// mul/add is its own kernel launch (no FMA contraction there). The remaining A/B
// differences are attributed to allocator-dependent kernel selection of the large
// matmuls (different graph size -> different buffer sizes -> possible different
// split-k), not to this kernel.
static __global__ void mhc_merge_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        const float * __restrict__ branch,
        const float * __restrict__ post0,
        const float * __restrict__ post1,
        const float * __restrict__ mix0,
        const float * __restrict__ mix1,
        const float * __restrict__ mix2,
        const float * __restrict__ mix3,
        float * __restrict__ dst,
        int64_t n_embd,
        int64_t n_tok,
        int64_t a_s1,
        int64_t b_s1,
        int64_t c_s1,
        int64_t p0_s1,
        int64_t p1_s1,
        int64_t m0_s1,
        int64_t m1_s1,
        int64_t m2_s1,
        int64_t m3_s1,
        int64_t d_s1) {
    ggml_cuda_pdl_lc();

    const int64_t t = blockIdx.x;
    const int64_t i = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();

    if (t >= n_tok || i >= n_embd) {
        return;
    }

    const float m0 = mix0[t*m0_s1];
    const float m1 = mix1[t*m1_s1];
    const float m2 = mix2[t*m2_s1];
    const float m3 = mix3[t*m3_s1];
    const float p0 = post0[t*p0_s1];
    const float p1 = post1[t*p1_s1];

    const float ai = a[t*a_s1 + i];
    const float bi = b[t*b_s1 + i];
    const float ci = branch[t*c_s1 + i];

    float * oa = dst + t*d_s1;
    float * ob = dst + (t + n_tok)*d_s1;

    oa[i] = __fadd_rn(__fadd_rn(__fmul_rn(ai, m0), __fmul_rn(bi, m1)), __fmul_rn(ci, p0));
    ob[i] = __fadd_rn(__fadd_rn(__fmul_rn(ai, m2), __fmul_rn(bi, m3)), __fmul_rn(ci, p1));
}

void ggml_cuda_op_mhc_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a      = dst->src[0];
    const ggml_tensor * b      = dst->src[1];
    const ggml_tensor * branch = dst->src[2];
    const ggml_tensor * post0  = dst->src[3];
    const ggml_tensor * post1  = dst->src[4];
    const ggml_tensor * mix0   = dst->src[5];
    const ggml_tensor * mix1   = dst->src[6];
    const ggml_tensor * mix2   = dst->src[7];
    const ggml_tensor * mix3   = dst->src[8];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && branch->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == a->ne[0]);
    GGML_ASSERT(dst->ne[1] == 2*a->ne[1]);
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1);

    const int64_t n_embd = a->ne[0];
    const int64_t n_tok  = a->ne[1];

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((unsigned) n_tok, (unsigned) ((n_embd + block_size - 1) / block_size), 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(mhc_merge_f32, launch_params,
            (const float *) a->data,      (const float *) b->data,      (const float *) branch->data,
            (const float *) post0->data,  (const float *) post1->data,
            (const float *) mix0->data,   (const float *) mix1->data,
            (const float *) mix2->data,   (const float *) mix3->data,
            (float *) dst->data,
            n_embd, n_tok,
            (int64_t) (a->nb[1]      / sizeof(float)),
            (int64_t) (b->nb[1]      / sizeof(float)),
            (int64_t) (branch->nb[1] / sizeof(float)),
            (int64_t) (post0->nb[1]  / sizeof(float)),
            (int64_t) (post1->nb[1]  / sizeof(float)),
            (int64_t) (mix0->nb[1]   / sizeof(float)),
            (int64_t) (mix1->nb[1]   / sizeof(float)),
            (int64_t) (mix2->nb[1]   / sizeof(float)),
            (int64_t) (mix3->nb[1]   / sizeof(float)),
            (int64_t) (dst->nb[1]    / sizeof(float)));
}
