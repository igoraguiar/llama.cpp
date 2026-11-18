#include "common.cuh"
#include "ggml.h"

// Fused 2x2 log-space Sinkhorn normalization used by the SenseNova-Loop mHC routing.
// src[0] is the router logits [8, n_tok] (only rows 4..7 are read), the iteration count is
// stored in op_params[0] and dst is [4, n_tok] holding exp(mix_k).
void ggml_cuda_op_mhc_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
