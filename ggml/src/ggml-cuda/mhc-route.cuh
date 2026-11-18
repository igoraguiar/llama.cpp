#include "common.cuh"
#include "ggml.h"

// Fused mHC routing for SenseNova-Loop, one ggml node replacing the whole reference chain
//   flat   = concat(a, b);  flat = rms_norm(flat, 1e-6)
//   logits = mul_mat(weight, flat) + bias                       // GEMV 2*n_embd -> 8
//   input  = (a*sigmoid(r0) + b*sigmoid(r1)) * pre_scale
//   post_k = 2*sigmoid(r_{2+k})
//   mix_k  = exp(mix_k) after n_iter 2x2 log-space Sinkhorn iterations on r4..r7
//
// src[0] = a, src[1] = b   [n_embd, n_tok] F32
// src[2] = weight          [2*n_embd, 8] F32 (row i = router row i, contiguous nb[1])
// src[3] = bias            [8] F32
// op_params[0] (i32) = n_iter, op_params[1] (f32) = pre_scale
//
// dst [n_row, n_tok] F32 with n_row = n_embd + 6 rounded up to a multiple of 4:
//   rows 0            .. n_embd-1 : result.input
//   row  n_embd                   : post0
//   row  n_embd+1                 : post1
//   rows n_embd+2     .. n_embd+5 : mix0..mix3 (exp already applied)
// The trailing pad rows (n_embd+6, n_embd+7) keep nb[1] 16-byte aligned so the
// downstream vectorized rms_norm/mul_mat kernels stay on their float4 paths.
void ggml_cuda_op_mhc_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
