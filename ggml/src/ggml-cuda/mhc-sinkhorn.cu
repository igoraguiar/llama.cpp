#include "common.cuh"
#include "mhc-sinkhorn.cuh"

// One thread per token: the routing matrix of a single token is a 2x2 matrix living in
// four registers, so the whole Sinkhorn loop costs a single kernel launch per routing
// instead of 6 graph nodes per iteration.
//
// The arithmetic is deliberately the same as the reference node chain
//   normalize(x, y): d = x - y; x = -softplus(-d); y = -softplus(d)
// with the same softplus form as the generic unary kernels (ggml-cpu/unary-ops.cpp and
// ggml-cuda/unary.cu), so the fused result is bit-identical to the unfused graph.
static __device__ __forceinline__ float mhc_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

static __device__ __forceinline__ void mhc_normalize(float & x, float & y) {
    const float d = x - y;
    x = -mhc_softplus(-d);
    y = -mhc_softplus( d);
}

static __global__ void mhc_sinkhorn_f32(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int64_t n_tokens,
        int64_t ss0,
        int64_t ss1,
        int64_t sd0,
        int64_t sd1,
        int32_t n_iter) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    // rows 4..7 of the router logits: mix[0..3] = M[r][c] with r = k/2, c = k%2
    float mix0 = src[4*ss0 + it*ss1];
    float mix1 = src[5*ss0 + it*ss1];
    float mix2 = src[6*ss0 + it*ss1];
    float mix3 = src[7*ss0 + it*ss1];

    for (int32_t i = 0; i < n_iter; ++i) {
        mhc_normalize(mix0, mix1); // row 0
        mhc_normalize(mix2, mix3); // row 1
        mhc_normalize(mix0, mix2); // column 0
        mhc_normalize(mix1, mix3); // column 1
    }

    dst[0*sd0 + it*sd1] = expf(mix0);
    dst[1*sd0 + it*sd1] = expf(mix1);
    dst[2*sd0 + it*sd1] = expf(mix2);
    dst[3*sd0 + it*sd1] = expf(mix3);
}

void ggml_cuda_op_mhc_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits = dst->src[0];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(logits->ne[0] == 8);
    GGML_ASSERT(logits->ne[2] == 1);
    GGML_ASSERT(logits->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == 4);
    GGML_ASSERT(dst->ne[1] == logits->ne[1]);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    GGML_TENSOR_LOCALS(size_t, nbs, logits, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,    nb);

    const int64_t n_tokens = logits->ne[1];
    const int32_t n_iter   = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(n_iter > 0);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(mhc_sinkhorn_f32, launch_params,
            (const float *) logits->data, (float *) dst->data,
            n_tokens,
            nbs0 / sizeof(float), nbs1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            n_iter);
}
