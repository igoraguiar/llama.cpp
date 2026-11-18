#include "models.h"

#include <cstring>

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);
    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const int64_t n_seqs = ubatch.n_seqs;

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    build_conv_state_store(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0);

    return conv_input;
}

// The in-place GATED_DELTA_NET writes into src[5]; the scheduler does not know about that
// write. If the node runs on another backend than the one owning the state buffer, the
// scheduler hands the kernel a temporary copy and the new state is silently lost. So the
// in-place path is only safe when the node is certain to run where the state lives:
//  * state in a CUDA/HIP device buffer: the layer is offloaded, its nodes run on that device
//    and that backend implements the in-place flag (other GPU backends ignore op param 1)
//  * state in a host buffer and op offloading disabled: the node stays on the CPU backend
// Measured without this check (RX 6900 XT, -ngl 20): mean KLD 0.32 vs 0.0013.
static bool gdn_state_buffer_inplace_safe(const ggml_tensor * states_all, bool op_offload) {
    const ggml_tensor * base = states_all->view_src ? states_all->view_src : states_all;
    ggml_backend_buffer_t buf = base->buffer;
    if (buf == nullptr) {
        return false;
    }
    if (ggml_backend_buffer_is_host(buf)) {
        return !op_offload;
    }
    const char * name = ggml_backend_buft_name(ggml_backend_buffer_get_type(buf));
    return name != nullptr && (strncmp(name, "CUDA", 4) == 0 || strncmp(name, "ROCm", 4) == 0);
}

bool llm_build_delta_net_base::gdn_state_can_inplace(const llm_graph_input_rs * inp, const ggml_tensor * ssm_states_all, int64_t n_seqs) const {
    // LLAMA_GDN_INPLACE=0 restores the GET_ROWS + CPY state round trip (A/B reference)
    static const bool enabled = []() {
        const char * e = getenv("LLAMA_GDN_INPLACE");
        return !(e && atoi(e) == 0);
    }();
    if (!enabled || n_seqs != 1) {
        return false;
    }
    const auto * mctx_cur = inp->mctx;
    if (mctx_cur == nullptr) {
        return false;
    }
    if (cparams.n_rs_seq > 0) {
        return false; // rollback snapshots use the keep path
    }
    if (mctx_cur->get_n_rs() != 1) {
        return false; // exactly one gathered state row
    }
    if (mctx_cur->get_rs_z() >= 0) {
        return false; // a zeroed state slot is still in use
    }
    if (!gdn_state_buffer_inplace_safe(ssm_states_all, cparams.op_offload)) {
        return false; // the node may run on a backend that does not own the state buffer
    }
    // the row that build_rs() would gather must be the slot the delta-net writes back to.
    // s_copy() is side-effect free here because n_rs_seq == 0 (it returns cells[head].src0).
    return mctx_cur->s_copy(0) == (int32_t) mctx_cur->get_head();
}

ggml_tensor * llm_build_delta_net_base::build_gdn_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        int64_t              S_v,
        int64_t              H_v,
        int64_t              n_seqs,
        int                  il) {
    if (gdn_state_can_inplace(inp, ssm_states_all, n_seqs)) {
        const uint32_t head = inp->mctx->get_head();
        // the graph now depends on s_copy(0) == head, see llm_graph_input_rs::can_reuse_inplace()
        inp->gdn_inplace = true;
        GGML_ASSERT(hparams.n_embd_s() == S_v * S_v * H_v);
        ggml_tensor * state = ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), 1, ssm_states_all->nb[1],
                (size_t) head * hparams.n_embd_s() * ggml_element_size(ssm_states_all));
        state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, 1);
        cb(state, "state_predelta_inplace", il);
        return state;
    }
    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, n_seqs);
    cb(state, "state_predelta", il);
    return state;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il,
        bool                 inplace_state) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (!keep) {
        if (inplace_state) {
            // s already is the cache slot: the kernel reads and writes it in place, no GET_ROWS/CPY
            GGML_ASSERT(s->ne[3] == 1);
            ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);
            ggml_gated_delta_net_set_inplace(result);
            ggml_tensor * output = ggml_view_4d(ctx0, result,
                    S_v, H_v, n_tokens, n_seqs,
                    ggml_row_size(result->type, S_v),
                    ggml_row_size(result->type, S_v * H_v),
                    ggml_row_size(result->type, S_v * H_v * n_tokens), 0);
            cb(output, "attn_output", il);
            return output;
        }
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);
    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
        D, n_seqs, n_written,
        ssm_states_all->nb[1],
        (size_t) mem_size * row_size,
        (size_t) kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

    return output;
}
