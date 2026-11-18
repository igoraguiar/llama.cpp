#include "models.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

void llama_model_sensenova_loop::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, true);
    llama_model_qwen35::load_arch_hparams(ml);
    ml.get_key(LLM_KV_LOOP_START, hparams.loop_start);
    ml.get_key(LLM_KV_LOOP_END,   hparams.loop_end);
    ml.get_key(LLM_KV_LOOP_COUNT, hparams.loop_count);
    ml.get_key(LLM_KV_MHC_START, hparams.mhc_start);
    ml.get_key(LLM_KV_MHC_END, hparams.mhc_end);
    ml.get_key(LLM_KV_MHC_STREAMS, hparams.mhc_streams);
    ml.get_key(LLM_KV_MHC_ITERATIONS, hparams.mhc_iterations);
    ml.get_key(LLM_KV_MHC_PRE_SCALE, hparams.mhc_pre_scale);
    ml.get_key(LLM_KV_MHC_FIXED_IDENTITY, hparams.mhc_fixed_identity);

    const uint32_t n_layer = hparams.n_layer();
    if (hparams.n_layer_nextn > 1 || hparams.loop_count == 0 || hparams.loop_start >= hparams.loop_end || hparams.loop_end > n_layer) {
        throw std::runtime_error("invalid SenseNova Loop layer range or count");
    }
    const uint64_t extra = uint64_t(hparams.loop_count - 1) * (hparams.loop_end - hparams.loop_start);
    if (extra > n_layer - hparams.loop_end) {
        throw std::runtime_error("SenseNova Loop execution plan exceeds the block count");
    }
    const uint32_t n_physical = n_layer - extra;
    if (hparams.n_layer_nextn && hparams.is_recr(n_layer)) {
        throw std::runtime_error("SenseNova Loop MTP requires a full-attention layer");
    }
    if (hparams.mhc_start >= hparams.mhc_end || hparams.mhc_end > n_physical || hparams.mhc_streams != 2 ||
            hparams.mhc_iterations == 0 || hparams.mhc_iterations > 20 || !std::isfinite(hparams.mhc_pre_scale) || hparams.mhc_pre_scale <= 0.0f) {
        throw std::runtime_error("unsupported SenseNova mHC configuration");
    }
    if (!((hparams.mhc_end <= hparams.loop_start) ||
          (hparams.mhc_start >= hparams.loop_start && hparams.mhc_end <= hparams.loop_end) ||
          (hparams.mhc_start >= hparams.loop_end))) {
        throw std::runtime_error("SenseNova mHC range crosses a Loop boundary");
    }

    layer_tensor_sources.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t physical = physical_layer(il);
        layer_tensor_sources[il] = physical < hparams.loop_end ? physical : il;
        if (hparams.is_recr(il) != hparams.is_recr(layer_tensor_sources[il])) {
            throw std::runtime_error("SenseNova Loop visits have different attention types");
        }
    }
    type = n_physical == 32 ? LLM_TYPE_9B : LLM_TYPE_UNKNOWN;
}

uint32_t llama_model_sensenova_loop::physical_layer(uint32_t il) const {
    const uint32_t width = hparams.loop_end - hparams.loop_start;
    const uint32_t extra = (hparams.loop_count - 1) * width;
    if (il < hparams.loop_start) {
        return il;
    }
    if (il < hparams.loop_end + extra) {
        return hparams.loop_start + (il - hparams.loop_start) % width;
    }
    return il - extra;
}

void llama_model_sensenova_loop::load_arch_tensors(llama_model_loader & ml) {
    // Per-tensor scales are loaded after the shared layer aliases.
    for (const auto & item : ml.weights_map) {
        const auto type = item.second.tensor->type;
        if (type == GGML_TYPE_F8_E4M3 || type == GGML_TYPE_NVFP4) {
            throw std::runtime_error("SenseNova Loop does not support preserved FP8 or NVFP4 weights");
        }
    }
    llama_model_qwen35::load_arch_tensors(ml);
    LLAMA_LOAD_LOCALS;
    const int flags = nextn_flags(ml).trunk;
    for (int il = 0; il < n_layer; ++il) {
        const uint32_t physical = physical_layer(il);
        if (physical < hparams.mhc_start || physical >= hparams.mhc_end) {
            continue;
        }
        if (layer_tensor_sources[il] != (uint32_t) il) {
            layers[il] = layers[layer_tensor_sources[il]];
            continue;
        }
        auto & layer = layers[il];
        const int64_t n_route = hparams.mhc_fixed_identity ? 4 : 8;
        layer.hc_attn_fn   = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", il), {2*n_embd, n_route}, flags);
        layer.hc_attn_fn_b = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "bias",   il), {n_route}, flags);
        layer.hc_ffn_fn    = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,  "weight", il), {2*n_embd, n_route}, flags);
        layer.hc_ffn_fn_b  = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,  "bias",   il), {n_route}, flags);
        for (const auto * router : {layer.hc_attn_fn, layer.hc_attn_fn_b, layer.hc_ffn_fn, layer.hc_ffn_fn_b}) {
            if (router && router->type != GGML_TYPE_F32) {
                throw std::runtime_error("SenseNova mHC router tensors must be F32");
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_sensenova_loop::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<llama_model_qwen35::graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

llama_model_sensenova_loop::graph::routing llama_model_sensenova_loop::graph::build_route(
        ggml_tensor * a, ggml_tensor * b, ggml_tensor * weight, ggml_tensor * bias, int il) {
    // FUSED PATH: one GGML_OP_MHC_ROUTE node replaces the ~14-node reference chain
    // (concat + rms_norm + mul_mat + add(bias) + two stream sigmoid/mul/add + scale +
    // two post sigmoid/scale + the fused Sinkhorn node). With 96 routings per decoded
    // token that removes ~1300 graph nodes. LLAMA_MHC_ROUTE_FUSED=0 restores the
    // reference chain below (A/B reference, kept fully intact).
    static const bool route_fused = [] {
        const char * e = getenv("LLAMA_MHC_ROUTE_FUSED");
        return e == nullptr || atoi(e) != 0;
    }();
    if (route_fused && !hparams.mhc_fixed_identity) {
        // same diagnostic clamp as the reference path (LLAMA_MHC_ITER_CLAMP)
        static const uint32_t route_max_iter = []() -> uint32_t {
            const char * e = getenv("LLAMA_MHC_ITER_CLAMP");
            if (e == nullptr) return UINT32_MAX;
            const long v = atol(e);
            return v > 0 ? (uint32_t) v : UINT32_MAX;
        }();
        const uint32_t n_iter = std::min<uint32_t>(hparams.mhc_iterations, route_max_iter);
        ggml_tensor * fused = ggml_mhc_route(ctx0, a, b, weight, bias, (int32_t) n_iter, hparams.mhc_pre_scale);
        const int64_t n_embd = a->ne[0];
        const int64_t n_tok  = a->ne[1];
        routing result{};
        result.input   = ggml_view_2d(ctx0, fused, n_embd, n_tok, fused->nb[1], 0);
        result.post[0] = ggml_view_2d(ctx0, fused, 1, n_tok, fused->nb[1], (n_embd + 0)*fused->nb[0]);
        result.post[1] = ggml_view_2d(ctx0, fused, 1, n_tok, fused->nb[1], (n_embd + 1)*fused->nb[0]);
        for (int i = 0; i < 4; ++i) {
            result.mix[i] = ggml_view_2d(ctx0, fused, 1, n_tok, fused->nb[1], (n_embd + 2 + i)*fused->nb[0]);
        }
        cb(result.input, "mhc_input", il);
        return result;
    }
    ggml_tensor * flat = ggml_concat(ctx0, a, b, 0);
    flat = ggml_rms_norm(ctx0, flat, 1e-6f);
    ggml_tensor * logits = build_lora_mm(weight, flat);
    if (logits->op == GGML_OP_MUL_MAT) {
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        // NOTE: do NOT request GGML_PREC_F32 for src[1] here. The HIP backend rejects
        // MUL_MAT with prec_src1 == F32 on non-NVIDIA devices (ggml-cuda.cu:
        // f32_src1_ok = CC_IS_NVIDIA(cc) && cc < AMPERE), while the buft probe in
        // weight_buft_supported() builds a plain F32 mul_mat without that flag and
        // therefore reports the ROCm buffer type as supported. The mismatch left the
        // weights in ROCm buffers but ran the op on the CPU (96 device->host->device
        // round trips per token). The router weights are F32, so no tensor-core
        // downcast of src1 is possible either way; HIP uses the FP32 mul_mat path.
    }
    logits = ggml_add(ctx0, logits, bias);
    auto row = [&](int i) {
        return ggml_view_2d(ctx0, logits, 1, a->ne[1], logits->nb[1], i*logits->nb[0]);
    };

    routing result{};
    result.input = ggml_add(ctx0, ggml_mul(ctx0, a, ggml_sigmoid(ctx0, row(0))), ggml_mul(ctx0, b, ggml_sigmoid(ctx0, row(1))));
    result.input = ggml_scale(ctx0, result.input, hparams.mhc_pre_scale);
    result.post[0] = ggml_scale(ctx0, ggml_sigmoid(ctx0, row(2)), 2.0f);
    result.post[1] = ggml_scale(ctx0, ggml_sigmoid(ctx0, row(3)), 2.0f);
    if (!hparams.mhc_fixed_identity) {
        // 2x2 Sinkhorn (row/col log-normalization of the routing matrix).
        //
        // Default path below: the entire loop is a single fused ggml node
        // (GGML_OP_MHC_SINKHORN, one node per routing). LLAMA_MHC_SINKHORN_LEGACY=1
        // restores the original scalar node loop (reference for A/B), and
        // LLAMA_MHC_SINKHORN_NOFUSED=1 the vectorized graph chain.
        //
        // The reference implementation unrolls each pairwise normalize() into
        // sub/neg/softplus/neg scalar nodes: 24 graph nodes per iteration, two routes
        // per mHC layer execution -> ~46k tiny kernel launches per decoded token at
        // mhc_iterations=20, which makes token generation pure launch overhead on GPU
        // backends (measured ~6 t/s on a RX 6900 XT, GPU 96-98% busy doing nothing).
        //
        // A two-element normalize(x, y) = [x, y] - lse(x, y) is exactly
        // log(soft_max([x, y])), so a full row+col iteration over the 2x2 matrix
        // M[r][c] = mix[2r + c] becomes:
        //   row norm: X = log(soft_max(X)) on the [c, t, r] layout (softmax over ne[0])
        //   col norm: same on the transposed [r, t, c] layout
        // i.e. 6 nodes per iteration (soft_max, log, cont each) instead of 24, with
        // identical results in exact arithmetic (log-softmax identity).
        // LLAMA_MHC_SINKHORN_LEGACY=1 restores the original scalar node loop.
        static const bool legacy = [] {
            const char * e = getenv("LLAMA_MHC_SINKHORN_LEGACY");
            return e != nullptr && atoi(e) != 0;
        }();
        if (legacy) {
            for (int i = 0; i < 4; ++i) {
                result.mix[i] = row(4 + i);
            }
            auto normalize = [&](ggml_tensor * & x, ggml_tensor * & y) {
                // Two-element log-softmax, without exponentiating the unnormalized logits.
                ggml_tensor * delta = ggml_sub(ctx0, x, y);
                x = ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, delta)));
                y = ggml_neg(ctx0, ggml_softplus(ctx0, delta));
            };
            for (uint32_t i = 0; i < hparams.mhc_iterations; ++i) {
                normalize(result.mix[0], result.mix[1]);
                normalize(result.mix[2], result.mix[3]);
                normalize(result.mix[0], result.mix[2]);
                normalize(result.mix[1], result.mix[3]);
            }
            for (auto & mix : result.mix) {
                mix = ggml_exp(ctx0, mix);
            }
        } else {
            const int64_t n_tok = a->ne[1];
            // DIAGNOSTIC/OPTIMIZATION: allow clamping the Sinkhorn iteration count.
            // The model was trained with hparams.mhc_iterations (20); values below that
            // change the numerics slightly and are only for A/B measurements.
            static const uint32_t sinkhorn_max_iter = []() -> uint32_t {
                const char * e = getenv("LLAMA_MHC_ITER_CLAMP");
                if (e == nullptr) return UINT32_MAX;
                const long v = atol(e);
                return v > 0 ? (uint32_t) v : UINT32_MAX;
            }();
            const uint32_t n_sinkhorn_iter = std::min<uint32_t>(hparams.mhc_iterations, sinkhorn_max_iter);
            // FUSED OP: the whole Sinkhorn loop runs inside one ggml node
            // (GGML_OP_MHC_SINKHORN) instead of 6 nodes per iteration, which removes
            // ~11.5k kernel launches per decoded token. LLAMA_MHC_SINKHORN_NOFUSED=1
            // falls back to the vectorized log-softmax chain below.
            static const bool no_fused = [] {
                const char * e = getenv("LLAMA_MHC_SINKHORN_NOFUSED");
                return e != nullptr && atoi(e) != 0;
            }();
            if (no_fused) {
                // X[c, t, r] = logits[4 + 2*r + c, t] (strided view of the mix rows, then a
                // contiguous copy: the CUDA/HIP soft_max kernel assumes flat contiguous rows)
                ggml_tensor * X = ggml_cont(ctx0, ggml_view_3d(ctx0, logits, 2, n_tok, 2,
                                                                logits->nb[1], 2*logits->nb[0], 4*logits->nb[0]));
                for (uint32_t it = 0; it < n_sinkhorn_iter; ++it) {
                    // row normalization: pairs (mix[0], mix[1]) and (mix[2], mix[3]) along ne[0]
                    X = ggml_log(ctx0, ggml_soft_max(ctx0, X));
                    // transpose dims 0<->2 for the column normalization
                    X = ggml_cont(ctx0, ggml_permute(ctx0, X, 2, 1, 0, 3));
                    // col normalization: pairs (mix[0], mix[2]) and (mix[1], mix[3]) along ne[0]
                    X = ggml_log(ctx0, ggml_soft_max(ctx0, X));
                    if (it + 1 < n_sinkhorn_iter) {
                        X = ggml_cont(ctx0, ggml_permute(ctx0, X, 2, 1, 0, 3));
                    }
                }
                X = ggml_exp(ctx0, X); // back to mixture weights; X is [r, t, c] contiguous
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 2; ++c) {
                        result.mix[2*r + c] = ggml_view_2d(ctx0, X, 1, n_tok, X->nb[1],
                                                           r*X->nb[2] + c*X->nb[0]);
                    }
                }
            } else {
                ggml_tensor * sink = ggml_mhc_sinkhorn(ctx0, logits, (int32_t) n_sinkhorn_iter);
                for (int i = 0; i < 4; ++i) {
                    // exp(mix_i) as a [1, n_tok] row view, same layout as the logits rows
                    result.mix[i] = ggml_view_2d(ctx0, sink, 1, n_tok, sink->nb[1], i*sink->nb[0]);
                }
            }
        }
    }
    cb(result.input, "mhc_input", il);
    return result;
}

void llama_model_sensenova_loop::graph::build_merge(
        ggml_tensor * & a, ggml_tensor * & b, ggml_tensor * branch, const routing & route) {
    // FUSED PATH: one ggml node (GGML_OP_MHC_MERGE) replaces the ten elementwise nodes
    // of the reference chain below (6 mul + 4 add: 2x2 mixture + two branch injections).
    // Two merges run per mHC layer execution (attn + ffn), so this removes 18 graph nodes
    // per layer execution at n_tok=1 (measured: 4233 -> 3369 nodes per decode graph).
    // LLAMA_MHC_MERGE_FUSED=0 restores the reference chain (A/B reference path).
    static const bool merge_fused = [] {
        const char * e = getenv("LLAMA_MHC_MERGE_FUSED");
        return e == nullptr || atoi(e) != 0;
    }();
    if (merge_fused && !hparams.mhc_fixed_identity) {
        const int64_t n_tok = a->ne[1];
        ggml_tensor * merged = ggml_mhc_merge(ctx0, a, b, branch,
                route.post[0], route.post[1],
                route.mix[0], route.mix[1], route.mix[2], route.mix[3]);
        // Two contiguous row blocks: a' = [0, n_tok), b' = [n_tok, 2*n_tok).
        a = ggml_view_2d(ctx0, merged, merged->ne[0], n_tok, merged->nb[1], 0);
        b = ggml_view_2d(ctx0, merged, merged->ne[0], n_tok, merged->nb[1], n_tok*merged->nb[1]);
        return;
    }
    ggml_tensor * next_a = a;
    ggml_tensor * next_b = b;
    if (!hparams.mhc_fixed_identity) {
        next_a = ggml_add(ctx0, ggml_mul(ctx0, a, route.mix[0]), ggml_mul(ctx0, b, route.mix[1]));
        next_b = ggml_add(ctx0, ggml_mul(ctx0, a, route.mix[2]), ggml_mul(ctx0, b, route.mix[3]));
    }
    a = ggml_add(ctx0, next_a, ggml_mul(ctx0, branch, route.post[0]));
    b = ggml_add(ctx0, next_b, ggml_mul(ctx0, branch, route.post[1]));
}

llama_model_sensenova_loop::graph::graph(const llm_graph_params & params) : graph_base(params) {
    const auto & loop = static_cast<const llama_model_sensenova_loop &>(model);
    auto * inp = build_inp_mem_hybrid();
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const uint32_t physical = loop.physical_layer(il);
        const bool mhc = physical >= hparams.mhc_start && physical < hparams.mhc_end;
        res->t_layer_inp[il] = cur;
        if (mhc && physical == hparams.mhc_start) {
            a = cur;
            b = cur;
        }
        routing attn_route{};
        ggml_tensor * residual = cur;
        if (mhc) {
            attn_route = build_route(a, b, layer.hc_attn_fn, layer.hc_attn_fn_b, il);
            cur = attn_route.input;
        }
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }
        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            if (mhc) {
                a = ggml_get_rows(ctx0, a, inp_out_ids);
                b = ggml_get_rows(ctx0, b, inp_out_ids);
                for (auto & post : attn_route.post) { post = ggml_get_rows(ctx0, post, inp_out_ids); }
                if (!hparams.mhc_fixed_identity) {
                    for (auto & mix : attn_route.mix) { mix = ggml_get_rows(ctx0, mix, inp_out_ids); }
                }
            }
        }
        if (mhc) {
            build_merge(a, b, cur, attn_route);
            const routing ffn_route = build_route(a, b, layer.hc_ffn_fn, layer.hc_ffn_fn_b, il);
            cur = build_norm(ffn_route.input, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cur = build_layer_ffn(cur, il);
            build_merge(a, b, cur, ffn_route);
            a = build_cvec(a, il);
            b = build_cvec(b, il);
            cur = ggml_scale(ctx0, ggml_add(ctx0, a, b), 0.5f);
        } else {
            residual = ggml_add(ctx0, cur, residual);
            cur = build_norm(residual, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cur = ggml_add(ctx0, build_layer_ffn(cur, il), residual);
            cur = build_cvec(cur, il);
        }
        cb(cur, "l_out", il);
    }
    build_output(cur, inp_out_ids);
}
