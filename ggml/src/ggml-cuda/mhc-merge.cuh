#include "common.cuh"
#include "ggml.h"

// Fused mHC residual merge (SenseNova-Loop routing), one node instead of eight.
// src[0..2] = a, b, branch [n_embd, n_tok] F32,
// src[3..4] = post0, post1 and src[5..8] = mix0..mix3 [1, n_tok] F32.
// dst is [n_embd, 2*n_tok]: rows [0, n_tok) = a*mix0 + b*mix1 + branch*post0,
// rows [n_tok, 2*n_tok) = a*mix2 + b*mix3 + branch*post1.
void ggml_cuda_op_mhc_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
