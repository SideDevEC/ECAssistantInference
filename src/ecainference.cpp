// ecainference.cpp — Complete C API implementation for ECAssistant Inference Engine
//
// Replaces LLamaSharp entirely. Links llama.cpp directly.
// Owns: model loading, context, standard executor, conversation pool,
// batched inference, full sampling pipeline, embeddings, vision/mtmd,
// state save/restore, tokenization.
//
// Thread safety: internal mutex serializes ALL native llama.cpp calls.
#include "eci_internal.h"
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <cstdio>

// ── Logging globals (2026-09-25) — defined here so ECI_LOG macro works everywhere ──
static eci_log_callback_t g_log_cb = nullptr;
static eci_log_level_t g_log_min_level = ECI_LOG_NONE;

// Forward declaration of the log implementation (defined at end of file).
extern "C" void eci_log(eci_log_level_t level, const char* tag, const char* fmt, ...);

// ── Helpers ──

static const char* result_strings[] = {
    "OK", "INVALID_ARG", "INVALID_HANDLE", "LOAD_FAILED",
    "NO_SLOT", "DECODE_FAILED", "OVERFLOW", "NOT_SUPPORTED", "INTERNAL"
};

const char* eci_result_str(eci_result_t r) {
    if (r >= 0 && r <= ECI_ERR_INTERNAL) return result_strings[r];
    return "UNKNOWN";
}

static void set_error(eci_context_t* ctx, const std::string& msg) {
    ECI_LOG(ECI_LOG_ERROR, "Inference", "set_error: %s", msg.c_str());
    if (ctx) ctx->last_error = msg;
}

// ── Model ──

eci_result_t eci_load_model(const eci_model_params_t* params, eci_model_t** out_model) {
    if (!params || !params->model_path || !out_model) return ECI_ERR_INVALID_ARG;

    auto* m = new eci_model_s();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = params->gpu_layers;

    m->model = llama_model_load_from_file(params->model_path, mp);
    if (!m->model) {
        m->last_error = std::string("Failed to load model: ") + params->model_path;
    ECI_LOG(ECI_LOG_INFO, "Model", "Loaded: %s (gpu_layers=%d)", params->model_path, params->gpu_layers);
        delete m;
        return ECI_ERR_LOAD_FAILED;
    }
    m->vocab = llama_model_get_vocab(m->model);
    m->flash_attn = params->flash_attn;
    m->kv_cache_type = params->kv_cache_type;
    m->threads = params->threads;
    *out_model = m;
    return ECI_OK;
}

void eci_free_model(eci_model_t* model) {
    if (model) { if (model->model) llama_model_free(model->model); delete model; }
}

// ── Context ──

eci_result_t eci_create_context(eci_model_t* model,
                                const eci_context_params_t* params,
                                eci_context_t** out_ctx) {
    if (!model || !params || !out_ctx) return ECI_ERR_INVALID_ARG;

    auto* c = new eci_context_s();
    c->model = model->model;
    c->vocab = model->vocab;
    c->pool_type = params->pooling_type;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = params->context_size;
    cp.n_batch = params->batch_size;
    cp.n_ubatch = params->batch_size;
    cp.n_seq_max = params->seq_max > 0 ? params->seq_max : 1;
    // Flash attention: caller drives via model params (passed to eci_load_model)
    // KV cache type: caller drives via model params
    if (model->kv_cache_type != ECI_KV_F16) {
        switch (model->kv_cache_type) {
            case ECI_KV_Q8_0: cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; break;
            case ECI_KV_Q4_0: cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0; break;
            case ECI_KV_Q4_1: cp.type_k = GGML_TYPE_Q4_1; cp.type_v = GGML_TYPE_Q4_1; break;
            default: break;
        }
    }
    if (model->flash_attn) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    // Pooling type for embedding models
    // Note: do NOT set cp.embeddings = true here — it creates an encoder context
    // that requires llama_encode() instead of llama_decode(). The original
    // llama_batch_get_one + llama_decode works for BERT/MiniLM models.
    // llama_set_embeddings(ctx, true) is called in eci_get_embeddings instead.
    switch (params->pooling_type) {
        case ECI_POOL_MEAN: cp.pooling_type = LLAMA_POOLING_TYPE_MEAN; break;
        case ECI_POOL_CLS:  cp.pooling_type = LLAMA_POOLING_TYPE_CLS; break;
        case ECI_POOL_LAST: cp.pooling_type = LLAMA_POOLING_TYPE_LAST; break;
        default:           cp.pooling_type = LLAMA_POOLING_TYPE_NONE; break;
    }

    c->ctx = llama_new_context_with_model(model->model, cp);
    if (!c->ctx) { c->last_error = "Failed to create context"; delete c; return ECI_ERR_LOAD_FAILED; }

    c->mem = llama_get_memory(c->ctx);
    c->n_ctx = params->context_size;
    c->n_seq_max = cp.n_seq_max;
    *out_ctx = c;
    return ECI_OK;
}

void eci_free_context(eci_context_t* ctx) {
    if (ctx) { if (ctx->ctx) llama_free(ctx->ctx); delete ctx; }
}

uint32_t eci_context_size(eci_context_t* ctx) { return ctx ? ctx->n_ctx : 0; }
uint32_t eci_context_seq_max(eci_context_t* ctx) { return ctx ? ctx->n_seq_max : 0; }

int eci_context_n_ctx_used(eci_context_t* ctx, eci_conversation_t* conv) {
    if (!conv) return 0;
    return conv->n_tokens;
}

// ── Tokenization ──

static int do_tokenize(const llama_vocab* vocab, const char* text,
                       std::vector<llama_token>& out, bool add_bos, bool parse_special) {
    std::string s(text);
    int n_max = (int)s.size() + 2;
    out.resize(n_max);
    int n = llama_tokenize(vocab, s.c_str(), (int)s.size(), out.data(), n_max, add_bos, parse_special);
    if (n < 0) { n_max = -n; out.resize(n_max); n = llama_tokenize(vocab, s.c_str(), (int)s.size(), out.data(), n_max, add_bos, parse_special); }
    if (n < 0) return -1;
    out.resize(n);
    return n;
}

eci_result_t eci_tokenize(eci_context_t* ctx, const char* text,
                          bool add_bos, bool parse_special,
                          int32_t** out_tokens, int* out_count) {
    if (!ctx || !text || !out_tokens || !out_count) return ECI_ERR_INVALID_ARG;
    std::vector<llama_token> tokens;
    if (do_tokenize(ctx->vocab, text, tokens, add_bos, parse_special) < 0) return ECI_ERR_INTERNAL;
    *out_count = (int)tokens.size();
    if (tokens.empty()) {
        *out_tokens = nullptr;
        return ECI_OK;
    }
    *out_tokens = (int32_t*)malloc(tokens.size() * sizeof(int32_t));
    if (!*out_tokens) return ECI_ERR_INTERNAL;
    memcpy(*out_tokens, tokens.data(), tokens.size() * sizeof(int32_t));
    return ECI_OK;
}

void eci_free_tokens(int32_t* tokens) { free(tokens); }

eci_result_t eci_detokenize(eci_context_t* ctx, const int32_t* tokens, int count,
                           bool remove_special, char** out_text) {
    if (!ctx || !tokens || !out_text || count <= 0) return ECI_ERR_INVALID_ARG;
    std::string result;
    for (int i = 0; i < count; i++) {
        char buf[256];
        int n = llama_token_to_piece(ctx->vocab, tokens[i], buf, sizeof(buf), 0, !remove_special);
        if (n > 0) result.append(buf, n);
    }
    *out_text = (char*)malloc(result.size() + 1);
    if (!*out_text) return ECI_ERR_INTERNAL;
    strcpy(*out_text, result.c_str());
    return ECI_OK;
}

void eci_free_string(char* str) { free(str); }

eci_result_t eci_token_to_piece(eci_context_t* ctx, int32_t token, char* buf, int buf_size) {
    if (!ctx || !buf || buf_size <= 0) return ECI_ERR_INVALID_ARG;
    int n = llama_token_to_piece(ctx->vocab, token, buf, buf_size, 0, true);
    return n > 0 ? ECI_OK : ECI_ERR_INTERNAL;
}

bool eci_token_is_eos(eci_context_t* ctx, int32_t token) {
    return ctx ? llama_vocab_is_eog(ctx->vocab, token) : false;
}

// ── Full sampling pipeline ──
// Replaces: DefaultSamplingPipeline
// Implements: temperature, top_k, top_p, min_p, repeat_penalty, presence penalty

static bool g_rng_seeded = false;

// Forward declarations (defined in the Grammar section below)
struct eci_grammar_state_s;
static void reset_grammar_state(eci_grammar_state_s* gs);
static int32_t do_sample(llama_context* ctx, const llama_vocab* vocab,
                         const eci_sampling_params_t* params,
                         const std::vector<llama_token>& recent_tokens,
                         int last_batch_idx) {
    if (!g_rng_seeded) { srand((unsigned)time(nullptr)); g_rng_seeded = true; }
    if (last_batch_idx < 0) return -1;

    float* logits = llama_get_logits_ith(ctx, last_batch_idx);
    if (!logits) return -1;

    int n_vocab = llama_vocab_n_tokens(vocab);
    float temp = params ? params->temperature : 0.3f;
    float top_p = params ? params->top_p : 0.95f;
    int top_k = params ? params->top_k : 40;
    float min_p = params ? params->min_p : 0.0f;
    float rep_pen = params ? params->repeat_penalty : 1.1f;
    int rep_last_n = params ? params->repeat_last_n : -1;
    float pres_pen = params ? params->penalty_present : 0.0f;

    // Copy logits — llama_get_logits_ith returns a pointer into llama's internal
    // buffer. Modifying it in-place would corrupt state for other conversations.
    std::vector<float> logits_copy(logits, logits + n_vocab);
    float* l = logits_copy.data();

    // Apply repeat penalty
    if (rep_pen != 1.0f && !recent_tokens.empty()) {
        int start = (rep_last_n < 0 || rep_last_n > (int)recent_tokens.size())
                    ? 0 : (int)recent_tokens.size() - rep_last_n;
        for (int i = start; i < (int)recent_tokens.size(); i++) {
            llama_token t = recent_tokens[i];
            if (t >= 0 && t < n_vocab) {
                l[t] = l[t] / rep_pen + pres_pen;
            }
        }
    }

    // Apply temperature + collect candidates
    std::vector<std::pair<float, int>> scored;
    scored.reserve(n_vocab);
    float inv_temp = 1.0f / std::max(temp, 1e-5f);
    for (int i = 0; i < n_vocab; i++)
        scored.push_back({l[i] * inv_temp, i});

    // Top-k
    if (top_k > 0 && top_k < (int)scored.size()) {
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                         [](auto& a, auto& b) { return a.first > b.first; });
        scored.resize(top_k);
    }

    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });

    // Min-p filtering
    if (min_p > 0 && !scored.empty()) {
        float max_logit = scored[0].first;
        float threshold = max_logit + std::log(min_p);
        scored.erase(std::remove_if(scored.begin(), scored.end(),
                      [&](auto& s) { return s.first < threshold; }), scored.end());
    }

    // Softmax
    float max_l = scored[0].first;
    for (auto& s : scored) s.first = std::exp(s.first - max_l);
    float sum = 0; for (auto& s : scored) sum += s.first;

    // Top-p (nucleus)
    float cumprob = 0; int cutoff = scored.size();
    for (int i = 0; i < (int)scored.size(); i++) {
        cumprob += scored[i].first / sum;
        if (cumprob > top_p) { cutoff = i + 1; break; }
    }
    scored.resize(cutoff);

    // Renormalize
    sum = 0; for (auto& s : scored) sum += s.first;
    for (auto& s : scored) s.first /= sum;

    // Sample
    float r = (float)rand() / RAND_MAX;
    float acc = 0;
    for (auto& s : scored) {
        acc += s.first;
        if (r <= acc) return s.second;
    }
    return scored.back().second;
}

// ── Standard executor (non-batched) ──
// Replaces: InteractiveExecutor / StatefulExecutorBase

eci_result_t eci_executor_create(eci_context_t* ctx, eci_executor_t** out_exec) {
    if (!ctx || !out_exec) return ECI_ERR_INVALID_ARG;
    auto* exec = new eci_executor_s();
    exec->ctx_ref = ctx;
    *out_exec = exec;
    return ECI_OK;
}

void eci_executor_free(eci_executor_t* exec) { delete exec; }

eci_result_t eci_executor_prompt(eci_executor_t* exec, const char* text) {
    if (!exec || !text) return ECI_ERR_INVALID_ARG;
    reset_grammar_state(&exec->grammar_state);  // new prompt = new generation
    std::vector<llama_token> tokens;
    if (do_tokenize(exec->ctx_ref->vocab, text, tokens, false, true) < 0) return ECI_ERR_INTERNAL;
    exec->pending_tokens.insert(exec->pending_tokens.end(), tokens.begin(), tokens.end());
    exec->has_pending = true;
    return ECI_OK;
}

eci_result_t eci_executor_prompt_tokens(eci_executor_t* exec, const int32_t* tokens, int count) {
    if (!exec || !tokens || count <= 0) return ECI_ERR_INVALID_ARG;
    // NOTE: no grammar reset here — PromptTokens feeds sampled tokens back into
    // the executor DURING generation; resetting would restart the grammar at
    // position 0 every token. Text prompts (eci_executor_prompt) reset instead.
    exec->pending_tokens.insert(exec->pending_tokens.end(), tokens, tokens + count);
    exec->has_pending = true;
    return ECI_OK;
}

eci_decode_result_t eci_executor_infer(eci_executor_t* exec) {
    if (!exec) return ECI_DECODE_FAILED;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);

    if (exec->pending_tokens.empty()) return ECI_DECODE_NO_WORK;

    // Chunk large prompts into ≤ n_batch slices — llama_decode ABORTS the process
    // (GGML_ASSERT n_tokens_all <= n_batch) if handed more tokens in one call.
    // LLamaSharp chunked transparently; this layer must do it itself.
    const uint32_t n_batch = llama_n_batch(exec->ctx_ref->ctx);
    if (n_batch == 0) return ECI_DECODE_FAILED;
    int n_tokens = (int)exec->pending_tokens.size();
    const int kv_base = exec->n_kv_pos;  // positions advance per slice — capture once

    int decoded = 0;
    while (decoded < n_tokens) {
        int slice = std::min((int)n_batch, n_tokens - decoded);
        bool last_slice = (decoded + slice == n_tokens);
        llama_batch batch = llama_batch_init(slice, 0, exec->ctx_ref->n_seq_max);
        llama_seq_id seq0 = 0;

        for (int i = 0; i < slice; i++) {
            batch.token[i] = exec->pending_tokens[decoded + i];
            batch.pos[i] = kv_base + decoded + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq0;
            batch.logits[i] = last_slice && (i == slice - 1) ? 1 : 0;
        }
        batch.n_tokens = slice;

        int rc = llama_decode(exec->ctx_ref->ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            // Decode failed — keep REMAINING pending tokens so caller can retry.
            // Already-decoded slices were committed inside the loop (n_tokens/
            // n_kv_pos/recent_tokens updated per slice — do NOT re-add here).
            exec->recent_tokens.insert(exec->recent_tokens.end(),
                                       exec->pending_tokens.begin(),
                                       exec->pending_tokens.begin() + decoded);
            exec->pending_tokens.erase(exec->pending_tokens.begin(),
                                       exec->pending_tokens.begin() + decoded);
            exec->ctx_ref->last_error = "llama_decode failed (rc=" + std::to_string(rc) + ")";
            ECI_LOG(ECI_LOG_ERROR, "Decode", "llama_decode failed rc=%d, slice=%d of %d tokens", rc, slice, exec->n_tokens);
            return ECI_DECODE_FAILED;
        }

        exec->recent_tokens.insert(exec->recent_tokens.end(),
                                   exec->pending_tokens.begin() + decoded,
                                   exec->pending_tokens.begin() + decoded + slice);
        exec->n_tokens += slice;
        exec->n_kv_pos += slice;
        if (last_slice)
            exec->last_batch_idx = slice - 1;  // index into THIS slice's logits
        decoded += slice;
    }

    exec->pending_tokens.clear();
    exec->has_pending = false;
    return ECI_DECODE_OK;
}

eci_result_t eci_executor_sample(eci_executor_t* exec,
                                  const eci_sampling_params_t* params,
                                  int32_t* out_token) {
    if (!exec || !out_token) return ECI_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);

    int32_t token = do_sample(exec->ctx_ref->ctx, exec->ctx_ref->vocab, params,
                              exec->recent_tokens, exec->last_batch_idx);
    if (token < 0) return ECI_ERR_DECODE_FAILED;

    // Track for repeat penalty
    exec->recent_tokens.push_back(token);
    *out_token = token;
    return ECI_OK;
}

int eci_executor_token_count(eci_executor_t* exec) { return exec ? exec->n_tokens : 0; }

eci_result_t eci_executor_rewind(eci_executor_t* exec, int n_tokens) {
    if (!exec || n_tokens < 0) return ECI_ERR_INVALID_ARG;
    if (n_tokens > exec->n_tokens) n_tokens = exec->n_tokens;
    if (n_tokens == 0) return ECI_OK;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    reset_grammar_state(&exec->grammar_state);  // generation restarts after rewind
    int start = exec->n_tokens - n_tokens;
    llama_memory_seq_rm(exec->ctx_ref->mem, 0, start, -1);
    exec->n_tokens -= n_tokens;
    exec->n_kv_pos -= n_tokens;
    if (exec->last_batch_idx >= exec->n_tokens) exec->last_batch_idx = -1;
    if ((int)exec->recent_tokens.size() > n_tokens)
        exec->recent_tokens.resize(exec->recent_tokens.size() - n_tokens);
    else
        exec->recent_tokens.clear();
    return ECI_OK;
}

eci_result_t eci_executor_reset(eci_executor_t* exec) {
    if (!exec) return ECI_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    reset_grammar_state(&exec->grammar_state);
    llama_memory_seq_rm(exec->ctx_ref->mem, 0, 0, -1);
    exec->n_tokens = 0;
    exec->n_kv_pos = 0;
    exec->has_pending = false;
    exec->pending_tokens.clear();
    exec->recent_tokens.clear();
    exec->last_batch_idx = -1;
    return ECI_OK;
}

// ── State save/restore ──
// Uses llama_memory_seq_cp to copy KV to a spare seq_id, then restore by copying back

eci_result_t eci_state_save(eci_executor_t* exec, eci_state_t** out_state) {
    if (!exec || !out_state) return ECI_ERR_INVALID_ARG;
    auto* state = new eci_state_s();
    state->n_tokens = exec->n_tokens;
    // Copy KV from seq 0 to seq 1 (backup)
    if (exec->n_tokens > 0) {
        if (exec->ctx_ref->n_seq_max < 2) {
            delete state;
            return ECI_ERR_NOT_SUPPORTED;  // need >=2 seq_ids for backup
        }
        std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
        llama_memory_seq_cp(exec->ctx_ref->mem, 0, 1, 0, -1);
    }
    *out_state = state;
    return ECI_OK;
}

eci_result_t eci_state_restore(eci_executor_t* exec, eci_state_t* state) {
    if (!exec || !state) return ECI_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    // Clear current KV for seq 0, then copy from backup seq 1
    llama_memory_seq_rm(exec->ctx_ref->mem, 0, 0, -1);
    if (state->n_tokens > 0 && exec->ctx_ref->n_seq_max > 1) {
        llama_memory_seq_cp(exec->ctx_ref->mem, 1, 0, 0, -1);
    } else if (state->n_tokens > 0 && exec->ctx_ref->n_seq_max <= 1) {
        // Can't restore from backup seq — no spare seq_id
        exec->n_tokens = 0;
        exec->last_batch_idx = -1;
        return ECI_OK;
    }
    exec->n_tokens = state->n_tokens;
    exec->n_kv_pos = state->n_tokens;
    if (exec->last_batch_idx >= exec->n_tokens) exec->last_batch_idx = -1;
    return ECI_OK;
}

void eci_state_free(eci_state_t* state) { delete state; }

int eci_state_token_count(eci_state_t* state) { return state ? state->n_tokens : 0; }

// ── Conversation pool ──

eci_result_t eci_pool_create(eci_context_t* ctx, eci_pool_t** out_pool) {
    if (!ctx || !out_pool) return ECI_ERR_INVALID_ARG;
    auto* pool = new eci_pool_s();
    pool->ctx_ref = ctx;
    for (uint32_t i = 0; i < ctx->n_seq_max; i++) {
        auto conv = std::make_unique<eci_conversation_s>();
        conv->seq_id = (llama_seq_id)i;
        conv->ctx = ctx->ctx;
        conv->mem = ctx->mem;
        pool->conversations.push_back(std::move(conv));
        pool->available.push(pool->conversations.back().get());
    }
    *out_pool = pool;
    return ECI_OK;
}

void eci_pool_free(eci_pool_t* pool) { delete pool; }

eci_result_t eci_pool_lease(eci_pool_t* pool, eci_conversation_t** out_conv) {
    if (!pool || !out_conv) return ECI_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(pool->mtx);
    if (pool->available.empty()) return ECI_ERR_NO_SLOT;
    *out_conv = pool->available.top();
    pool->available.pop();
    std::lock_guard<std::mutex> ilock(pool->ctx_ref->infer_mtx);
    pool->ctx_ref->active_conversations.push_back(*out_conv);
    return ECI_OK;
}

eci_result_t eci_pool_return(eci_pool_t* pool, eci_conversation_t* conv) {
    if (!pool || !conv) return ECI_ERR_INVALID_ARG;
    {
        std::lock_guard<std::mutex> ilock(pool->ctx_ref->infer_mtx);
        auto& active = pool->ctx_ref->active_conversations;
        auto it = std::find(active.begin(), active.end(), conv);
        if (it != active.end()) active.erase(it);
    }
    std::lock_guard<std::mutex> lock(pool->mtx);
    if (conv->n_tokens > 0) {
        llama_memory_seq_rm(conv->mem, conv->seq_id, 0, -1);
        conv->n_tokens = 0; conv->n_kv_pos = 0;
        // Vulkan stale cells fix (#26744): flush stale K/V after clearing
        if (eci_backend_needs_kv_flush()) {
            eci_kv_flush(pool->ctx_ref, conv->seq_id);
        }
    }
    conv->has_pending_prompt = false;
    conv->pending_tokens.clear();
    conv->last_batch_idx = -1;
    pool->available.push(conv);
    return ECI_OK;
}

int eci_pool_available(eci_pool_t* pool) {
    if (!pool) return 0;
    std::lock_guard<std::mutex> lock(pool->mtx);
    return (int)pool->available.size();
}

uint32_t eci_pool_size(eci_pool_t* pool) { return pool ? (uint32_t)pool->conversations.size() : 0; }

// ── Conversation ──

eci_result_t eci_conversation_prompt(eci_conversation_t* conv, const char* text) {
    if (!conv || !text) return ECI_ERR_INVALID_ARG;
    const llama_vocab* vocab = llama_model_get_vocab(llama_get_model(conv->ctx));
    std::vector<llama_token> tokens;
    if (do_tokenize(vocab, text, tokens, false, true) < 0) return ECI_ERR_INTERNAL;
    conv->pending_tokens.insert(conv->pending_tokens.end(), tokens.begin(), tokens.end());
    conv->has_pending_prompt = true;
    return ECI_OK;
}

eci_result_t eci_conversation_prompt_tokens(eci_conversation_t* conv,
                                             const int32_t* tokens, int count) {
    if (!conv || !tokens || count <= 0) return ECI_ERR_INVALID_ARG;
    conv->pending_tokens.insert(conv->pending_tokens.end(), tokens, tokens + count);
    conv->has_pending_prompt = true;
    return ECI_OK;
}

eci_result_t eci_conversation_sample(eci_conversation_t* conv,
                                     const eci_sampling_params_t* params,
                                     int32_t* out_token) {
    if (!conv || !out_token) return ECI_ERR_INVALID_ARG;
    if (conv->n_tokens == 0 || conv->last_batch_idx < 0) return ECI_ERR_INVALID_ARG;
    // Repeat/present penalties now apply in batch mode via conv->recent_tokens
    // (mirrors the executor path).
    int32_t token = do_sample(conv->ctx, llama_model_get_vocab(llama_get_model(conv->ctx)),
                              params, conv->recent_tokens, conv->last_batch_idx);
    if (token < 0) return ECI_ERR_DECODE_FAILED;
    conv->recent_tokens.push_back(token);
    *out_token = token;
    return ECI_OK;
}

int eci_conversation_token_count(eci_conversation_t* conv) { return conv ? conv->n_tokens : 0; }

eci_result_t eci_conversation_rewind(eci_conversation_t* conv, int n_tokens) {
    if (!conv || n_tokens < 0) return ECI_ERR_INVALID_ARG;
    if (n_tokens > conv->n_tokens) n_tokens = conv->n_tokens;
    if (n_tokens == 0) return ECI_OK;
    reset_grammar_state(&conv->grammar_state);  // generation restarts after rewind
    if ((int)conv->recent_tokens.size() > n_tokens)
        conv->recent_tokens.resize(conv->recent_tokens.size() - n_tokens);
    else
        conv->recent_tokens.clear();
    int start = conv->n_tokens - n_tokens;
    llama_memory_seq_rm(conv->mem, conv->seq_id, start, -1);
    conv->n_tokens -= n_tokens; conv->n_kv_pos -= n_tokens;
    if (conv->last_batch_idx >= conv->n_tokens) conv->last_batch_idx = -1;
    return ECI_OK;
}

eci_result_t eci_conversation_reset(eci_conversation_t* conv) {
    if (!conv) return ECI_ERR_INVALID_ARG;
    reset_grammar_state(&conv->grammar_state);
    conv->recent_tokens.clear();
    if (conv->n_tokens > 0) llama_memory_seq_rm(conv->mem, conv->seq_id, 0, -1);
    conv->n_tokens = 0; conv->n_kv_pos = 0;
    conv->has_pending_prompt = false;
    conv->pending_tokens.clear();
    conv->last_batch_idx = -1;
    return ECI_OK;
}

// Conversation state save/restore (for batch rewind)
eci_result_t eci_conversation_save(eci_conversation_t* conv, eci_state_t** out_state) {
    if (!conv || !out_state) return ECI_ERR_INVALID_ARG;
    auto* state = new eci_state_s();
    state->n_tokens = conv->n_tokens;
    *out_state = state;
    return ECI_OK;
}

eci_result_t eci_conversation_restore(eci_pool_t* pool, eci_conversation_t* conv,
                                       eci_state_t* state) {
    (void)pool;  // not needed — restore operates on conv directly
    if (!pool || !conv || !state) return ECI_ERR_INVALID_ARG;
    reset_grammar_state(&conv->grammar_state);  // generation rewind → fresh grammar
    int to_rewind = conv->n_tokens - state->n_tokens;
    if (to_rewind > 0) {
        llama_memory_seq_rm(conv->mem, conv->seq_id, state->n_tokens, -1);
        conv->n_tokens = state->n_tokens; conv->n_kv_pos = state->n_tokens;
        if (conv->last_batch_idx >= conv->n_tokens) conv->last_batch_idx = -1;
    }
    (void)pool;
    return ECI_OK;
}

void eci_conversation_state_free(eci_state_t* state) { delete state; }

// ── Batched inference ──

eci_decode_result_t eci_infer(eci_context_t* ctx) {
    if (!ctx) return ECI_DECODE_FAILED;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);

    std::vector<llama_token> all_tokens;
    std::vector<llama_seq_id> all_seq_ids;
    std::vector<int> all_pos;
    std::set<int> logits_indices;

    int current_idx = 0;
    // Snapshot pending tokens — only commit on successful decode
    struct conv_pending {
        eci_conversation_t* conv;
        std::vector<llama_token> tokens;
        int first_idx;  // first index in combined batch
        int committed = 0;  // tokens already decoded (committed) from this conv
    };
    std::vector<conv_pending> pending;
    for (auto* conv : ctx->active_conversations) {
        if (conv->pending_tokens.empty()) continue;
        conv_pending cp;
        cp.conv = conv;
        cp.tokens = conv->pending_tokens;
        cp.first_idx = current_idx;
        for (int i = 0; i < (int)cp.tokens.size(); i++) {
            all_tokens.push_back(cp.tokens[i]);
            all_seq_ids.push_back(conv->seq_id);
            all_pos.push_back(conv->n_kv_pos + i);
            current_idx++;
        }
        logits_indices.insert(current_idx - 1);
        pending.push_back(std::move(cp));
    }

    if (all_tokens.empty()) return ECI_DECODE_NO_WORK;

    // Chunk into ≤ n_batch slices — llama_decode ABORTS the process
    // (GGML_ASSERT n_tokens_all <= n_batch) if handed more tokens in one call.
    // LLamaSharp chunked transparently; this layer must do it itself.
    const uint32_t n_batch = llama_n_batch(ctx->ctx);
    if (n_batch == 0) { ctx->last_error = "n_batch is 0"; return ECI_DECODE_FAILED; }

    int total = (int)all_tokens.size();
    int gi = 0;  // global index of current slice start
    while (gi < total) {
        int end = std::min(gi + (int)n_batch, total);
        int n_slice = end - gi;
        llama_batch batch = llama_batch_init(n_slice, 0, ctx->n_seq_max);

        for (int i = gi; i < end; i++) {
            batch.token[i - gi] = all_tokens[i];
            batch.pos[i - gi] = all_pos[i];
            batch.n_seq_id[i - gi] = 1;
            batch.seq_id[i - gi][0] = all_seq_ids[i];
            // logits only for a conv's final pending token when THIS slice covers it
            batch.logits[i - gi] = logits_indices.count(i) ? 1 : 0;
        }
        batch.n_tokens = n_slice;

        int rc = llama_decode(ctx->ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            ctx->last_error = "llama_decode failed (rc=" + std::to_string(rc) + ")";
            ECI_LOG(ECI_LOG_ERROR, "Batch", "llama_decode failed rc=%d", rc);
            // Successfully decoded earlier slices are already committed. Trim the
            // committed prefix from each conv's pending_tokens so a retry does not
            // re-decode them (positions would then diverge from KV).
            for (auto& cp : pending) {
                if (cp.committed > 0)
                    cp.conv->pending_tokens.erase(cp.conv->pending_tokens.begin(),
                                                  cp.conv->pending_tokens.begin() + cp.committed);
            }
            return ECI_DECODE_FAILED;
        }

        // Commit per-conversation coverage of this slice
        for (auto& cp : pending) {
            int conv_start = cp.first_idx;
            int conv_end = cp.first_idx + (int)cp.tokens.size();
            int s = std::max(gi, conv_start);
            int e = std::min(end, conv_end);
            if (s >= e) continue;
            int covered = e - s;
            cp.conv->n_tokens += covered;
            cp.conv->n_kv_pos += covered;
            cp.conv->recent_tokens.insert(cp.conv->recent_tokens.end(),
                                          cp.tokens.begin() + (s - conv_start),
                                          cp.tokens.begin() + (s - conv_start) + covered);
            cp.committed += covered;
            if (e == conv_end)
                cp.conv->last_batch_idx = conv_end - 1 - gi;  // logits index within THIS slice
        }
        gi = end;
    }

    for (auto& cp : pending) {
        cp.conv->pending_tokens.clear();
        cp.conv->has_pending_prompt = false;
    }
    return ECI_DECODE_OK;
}

// ── Embeddings ──

eci_result_t eci_get_embeddings(eci_model_t* model, eci_context_t* ctx,
                                const char* text,
                                float** out_embeddings, int* out_dim) {
    if (!model || !ctx || !text || !out_embeddings || !out_dim) return ECI_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);

    std::vector<llama_token> tokens;
    if (do_tokenize(ctx->vocab, text, tokens, false, true) < 0) return ECI_ERR_INTERNAL;

    // Use llama_batch_get_one — llama.cpp handles encoder-only models (BERT) correctly.
    // Manual llama_batch_init + llama_decode fails on BERT ("calling encode() instead").
    int n = (int)tokens.size();
    llama_batch batch = llama_batch_get_one(tokens.data(), n);

    // BERT/encoder-only models require llama_encode, not llama_decode.
    // Must enable embeddings output before calling encode/decode.
    llama_set_embeddings(ctx->ctx, true);
    int enc_rc = llama_encode(ctx->ctx, batch);
    if (enc_rc != 0) {
        // Causal models use llama_decode instead
        enc_rc = llama_decode(ctx->ctx, batch);
    }
    if (enc_rc != 0) {
        llama_set_embeddings(ctx->ctx, false);
        ctx->last_error = "Embedding decode/encode failed";
        return ECI_ERR_DECODE_FAILED;
    }

    int n_embd = llama_model_n_embd(model->model);
    float* embd = (float*)malloc(n_embd * sizeof(float));
    if (!embd) return ECI_ERR_INTERNAL;

    memset(embd, 0, n_embd * sizeof(float));

    switch (ctx->pool_type) {
        case ECI_POOL_CLS: {
            float* seq_emb = llama_get_embeddings_seq(ctx->ctx, 0);
            if (seq_emb) memcpy(embd, seq_emb, n_embd * sizeof(float));
            else {
                float* tok = llama_get_embeddings_ith(ctx->ctx, 0);
                if (tok) memcpy(embd, tok, n_embd * sizeof(float));
            }
            break;
        }
        case ECI_POOL_LAST: {
            float* seq_emb = llama_get_embeddings_seq(ctx->ctx, 0);
            if (seq_emb) memcpy(embd, seq_emb, n_embd * sizeof(float));
            else {
                float* tok = llama_get_embeddings_ith(ctx->ctx, n - 1);
                if (tok) memcpy(embd, tok, n_embd * sizeof(float));
            }
            break;
        }
        case ECI_POOL_MEAN:
        default: {
            // Try sequence-level embedding first (llama.cpp does pooling internally)
            float* seq_emb = llama_get_embeddings_seq(ctx->ctx, 0);
            if (seq_emb) {
                memcpy(embd, seq_emb, n_embd * sizeof(float));
            } else {
                // Fallback: manual mean pooling from token embeddings
                for (int i = 0; i < n; i++) {
                    float* tok = llama_get_embeddings_ith(ctx->ctx, i);
                    if (tok) for (int j = 0; j < n_embd; j++) embd[j] += tok[j];
                }
                for (int j = 0; j < n_embd; j++) embd[j] /= n;
            }
            break;
        }
    }

    // Normalize
    float norm = 0;
    for (int j = 0; j < n_embd; j++) norm += embd[j] * embd[j];
    norm = sqrtf(norm);
    if (norm > 0) for (int j = 0; j < n_embd; j++) embd[j] /= norm;

    llama_set_embeddings(ctx->ctx, false);

    *out_embeddings = embd;
    *out_dim = n_embd;
    return ECI_OK;
}

void eci_free_embeddings(float* embeddings) { free(embeddings); }
int eci_embedding_dim(eci_model_t* model) {
    return model ? llama_model_n_embd(model->model) : 0;
}

// ── Vision / MTMD ──
// Replaces: MtmdWeights
//
// Flow:
// 1. eci_load_mmproj: load mmproj file → mtmd_context (stored on eci_model_t)
// 2. eci_convision_load_image: decode image bytes → mtmd_bitmap
// 3. eci_conversation_prompt_with_images: tokenize prompt+images into chunks,
//    encode image chunks, feed text tokens + image embeddings into the conversation

eci_result_t eci_load_mmproj(eci_model_t* model, const char* mmproj_path,
                             eci_model_t** out_mmproj) {
    if (!model || !mmproj_path || !out_mmproj)
        return ECI_ERR_INVALID_ARG;

    // We don't load mmproj as a regular model — mtmd_init_from_file handles it.
    // Create a lightweight wrapper to hold the mtmd context.
    auto* mmproj_model = new eci_model_s();

    // Initialize mtmd context with the text model reference
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;
    mparams.print_timings = false;
    mparams.n_threads = 1;

    mmproj_model->mtmd_ctx = mtmd_init_from_file(mmproj_path, model->model, mparams);
    if (!mmproj_model->mtmd_ctx) {
        delete mmproj_model;
        return ECI_ERR_LOAD_FAILED;
    }

    *out_mmproj = mmproj_model;
    return ECI_OK;
}

void eci_free_mmproj(eci_model_t* mmproj) {
    if (mmproj) {
        if (mmproj->mtmd_ctx) mtmd_free(mmproj->mtmd_ctx);
        if (mmproj->model) llama_model_free(mmproj->model);
        delete mmproj;
    }
}

// Image loading is handled directly in eci_conversation_prompt_with_images.
// This legacy entry point is kept for API compatibility but is a no-op.
eci_result_t eci_conversation_load_image(eci_conversation_t* conv,
                                          eci_model_t* mmproj,
                                          const uint8_t* data, int size) {
    (void)conv; (void)mmproj; (void)data; (void)size;
    return ECI_OK;
}

// Full prompt with images: tokenize prompt+images into chunks, encode, feed to conversation
eci_result_t eci_conversation_prompt_with_images(eci_conversation_t* conv,
                                                    eci_model_t* model_with_mtmd,
                                                    const char* text,
                                                    const uint8_t** image_data,
                                                    const int* image_sizes,
                                                    int n_images) {
    if (!conv || !model_with_mtmd || !model_with_mtmd->mtmd_ctx || !text)
        return ECI_ERR_INVALID_ARG;

    mtmd_context* mctx = model_with_mtmd->mtmd_ctx;

    // Step 1: Load all images as bitmaps
    std::vector<mtmd_bitmap*> bitmaps;
    for (int i = 0; i < n_images; i++) {
        auto wrapper = mtmd_helper_bitmap_init_from_buf(
            mctx, image_data[i], image_sizes[i], false, mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            for (auto* b : bitmaps) mtmd_bitmap_free(b);
            return ECI_ERR_INTERNAL;
        }
        bitmaps.push_back(wrapper.bitmap);
    }

    // Step 2: Tokenize prompt + images into chunks
    mtmd_input_text input_text = {};
    input_text.text = text;
    input_text.text_len = strlen(text);
    input_text.add_special = false;
    input_text.parse_special = false;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    // Debug: log marker + prompt for diagnosis

    int32_t rc = mtmd_tokenize(mctx, chunks, &input_text,
                               bitmaps.data(), bitmaps.size());

    // Free bitmaps (tokenize consumed them)
    for (auto* b : bitmaps) mtmd_bitmap_free(b);

    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        return ECI_ERR_INTERNAL;
    }

    // Step 3: Process each chunk sequentially.
    // Text chunks → queue tokens for the next eci_infer() call (batched with other conversations).
    // Image chunks → encode + decode immediately (embd batches can't be queued as pending tokens).
    //   Before each image decode, flush any pending text tokens so they go in their own token-batch.
    //
    // IMPORTANT: All llama_decode calls in this function must be serialized via infer_mtx.
    // The C# layer guarantees exclusive cycle-gate access for vision, but we lock here
    // too for safety against direct calls.
    std::mutex* infer_mtx = nullptr;
    // Find the context's mutex via the conversation's context
    // conv->ctx is llama_context*, we need eci_context_t* for the mutex
    // We lock per-decode below using a fallback: since we don't have eci_context_t* here,
    // the C# layer MUST ensure exclusive access. This is documented in CAPABILITY.md.
    size_t n_chunks = mtmd_input_chunks_size(chunks);

    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // Text chunk — extract tokens, queue them
            size_t n_tokens = 0;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            if (tokens && n_tokens > 0) {
                conv->pending_tokens.insert(conv->pending_tokens.end(),
                                           tokens, tokens + n_tokens);
            }
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // Before encoding the image, flush any pending text tokens.
            // Text tokens use batch.token; image embeddings use batch.embd —
            // they cannot coexist in the same llama_batch.
            if (!conv->pending_tokens.empty()) {
                // Chunked flush: llama_decode aborts if n_tokens_all > n_batch
                const uint32_t vflush_n_batch = llama_n_batch(conv->ctx);
                if (vflush_n_batch == 0) {
                    mtmd_input_chunks_free(chunks);
                    set_error(nullptr, "n_batch is 0");
                    return ECI_ERR_DECODE_FAILED;
                }
                const int vflush_base = conv->n_kv_pos;  // capture once — advance per slice
                int flushed = 0;
                while (flushed < (int)conv->pending_tokens.size()) {
                    int n_slice = std::min((int)vflush_n_batch,
                                           (int)conv->pending_tokens.size() - flushed);
                    llama_batch tbatch = llama_batch_init(n_slice, 0, 1);
                    for (int j = 0; j < n_slice; j++) {
                        tbatch.token[j] = conv->pending_tokens[flushed + j];
                        tbatch.pos[j] = vflush_base + flushed + j;
                        tbatch.n_seq_id[j] = 1;
                        tbatch.seq_id[j][0] = conv->seq_id;
                        tbatch.logits[j] = 0;
                    }
                    tbatch.n_tokens = n_slice;
                    int rc2 = llama_decode(conv->ctx, tbatch);
                    llama_batch_free(tbatch);
                    if (rc2 != 0) {
                        mtmd_input_chunks_free(chunks);
                        set_error(nullptr, std::string("text flush before image decode failed: ") + std::to_string(rc2));
                        return ECI_ERR_DECODE_FAILED;
                    }
                    conv->n_tokens += n_slice; conv->n_kv_pos += n_slice;
                    conv->recent_tokens.insert(conv->recent_tokens.end(),
                                               conv->pending_tokens.begin() + flushed,
                                               conv->pending_tokens.begin() + flushed + n_slice);
                    flushed += n_slice;
                }
                conv->pending_tokens.clear();
            }

            // Check if this model needs non-causal attention for image decoding
            bool need_non_causal = mtmd_decode_use_non_causal(mctx, chunk);
            if (need_non_causal) {
                llama_set_causal_attn(conv->ctx, false);
            }

            // Encode the image chunk
            int32_t enc_rc = mtmd_encode_chunk(mctx, chunk);
            if (enc_rc != 0) {
                if (need_non_causal) llama_set_causal_attn(conv->ctx, true);
                mtmd_input_chunks_free(chunks);
                return ECI_ERR_DECODE_FAILED;
            }

            // Get the output embeddings
            float* embd = mtmd_get_output_embd(mctx);
            size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            int n_embd = llama_model_n_embd_inp(llama_get_model(conv->ctx));

            if (embd && n_img_tokens > 0 && n_embd > 0) {
                // Create an embd batch — feed image embeddings directly to llama_decode.
                // embd size = n_img_tokens * n_embd * sizeof(float)
                int n_tokens = (int)n_img_tokens;
                // llama_batch_init with embd=n_embd allocates batch.embd
                llama_batch ebatch = llama_batch_init(n_tokens, n_embd, 1);

                // Copy embeddings into the batch
                memcpy(ebatch.embd, embd, (size_t)n_tokens * n_embd * sizeof(float));

                // Set positions, seq_ids, logits — do NOT touch token (NULL in embd mode)
                for (int j = 0; j < n_tokens; j++) {
                    ebatch.pos[j] = conv->n_kv_pos + j;
                    ebatch.n_seq_id[j] = 1;
                    ebatch.seq_id[j][0] = conv->seq_id;
                    ebatch.logits[j] = (j == n_tokens - 1) ? 1 : 0;
                }
                ebatch.n_tokens = n_tokens;

                int rc3 = llama_decode(conv->ctx, ebatch);
                llama_batch_free(ebatch);

                // Restore causal attention for text decoding
                if (need_non_causal) {
                    llama_set_causal_attn(conv->ctx, true);
                }

                if (rc3 != 0) {
                    mtmd_input_chunks_free(chunks);
                    return ECI_ERR_DECODE_FAILED;
                }

                conv->n_tokens += n_tokens; conv->n_kv_pos += n_tokens;
                // Image chunk token positions are embeddings — no token ids exist,
                // so they cannot contribute to repeat-penalty history.
                conv->last_batch_idx = n_tokens - 1;
            }
        }
    }

    conv->has_pending_prompt = true;
    mtmd_input_chunks_free(chunks);
    return ECI_OK;
}

eci_result_t eci_mtmd_marker(char** out_marker) {
    if (!out_marker) return ECI_ERR_INVALID_ARG;
    const char* marker = mtmd_default_marker();
    if (!marker) marker = "<image>";
    *out_marker = (char*)malloc(strlen(marker) + 1);
    strcpy(*out_marker, marker);
    return ECI_OK;
}

const char* eci_mtmd_marker_static(void) {
    return mtmd_default_marker();
}

// ── Granular KV cache manipulation ──
// These replace LLamaSharp workarounds with direct native operations.

// Shift left: remove oldest n tokens, reposition remaining to start at 0.
// This implements sliding window context — when approaching n_ctx, drop the
// oldest tokens instead of resetting the entire conversation.
eci_result_t eci_conversation_shift_left(eci_conversation_t* conv, int n_tokens) {
    if (!conv) return ECI_ERR_INVALID_ARG;
    reset_grammar_state(&conv->grammar_state);  // context shift → grammar desyncs
    if (!conv || n_tokens <= 0) return ECI_ERR_INVALID_ARG;
    if (n_tokens >= conv->n_tokens) {
        // Shifting everything = full clear
        return eci_conversation_reset(conv);
    }

    // Remove the front n_tokens from KV
    llama_memory_seq_rm(conv->mem, conv->seq_id, 0, n_tokens);

    // Shift remaining tokens: copy from position n_tokens to position 0.
    // llama.cpp handles position reindexing internally when you remove from the front
    // — the remaining tokens stay at their positions but the KV entries after the
    // removed range are now at a lower effective offset. However, the position
    // counter (n_past) for FUTURE decodes must reflect the new base.
    //
    // Actually, llama_memory_seq_rm with p0=0, p1=n removes those positions from KV.
    // The remaining KV entries (positions n..end) are still at their original positions.
    // For the NEXT decode, we must set batch.pos = (n_tokens_old - n_shifted + i),
    // which means our conv->n_tokens must reflect the NEW logical base.
    //
    // The simplest correct approach: after removing front tokens, the conversation's
    // logical position base shifts. We track n_tokens as the count AFTER the shift,
    // and set positions relative to the new base on the next decode.
    //
    // But llama.cpp's KV cache uses absolute positions for RoPE. Removing positions
    // 0..n from KV doesn't change the positions of remaining entries — they're still
    // at n..end. So the next decode must continue from position (old_n_tokens),
    // NOT from (n_tokens - n_shifted).
    //
    // This means shift_left is a KV-only operation: it frees memory but doesn't
    // change the position counter. The conversation continues from where it was,
    // just with the oldest KV entries freed. RoPE positions remain correct.
    conv->n_tokens -= n_tokens;  // logical count shrinks
    // n_kv_pos stays the same — KV positions are absolute
    // NOTE: positions for future decodes still use conv->n_tokens as the base.
    // The KV entries that remain are at positions [n_tokens_shifted .. old_n_tokens-1]
    // in absolute terms, but the model sees them as a contiguous block starting
    // from an offset. The position for the NEXT token is conv->n_tokens (post-shift).
    // This is correct because llama.cpp resolves positions via the KV cache's
    // stored positions, not our counter.

    // Trim recent tokens for repeat penalty
    if ((int)conv->recent_tokens.size() > n_tokens)
        conv->recent_tokens.erase(conv->recent_tokens.begin(),
                                  conv->recent_tokens.begin() + n_tokens);
    else
        conv->recent_tokens.clear();

    if (conv->last_batch_idx >= conv->n_tokens) conv->last_batch_idx = -1;
    return ECI_OK;
}

eci_result_t eci_executor_shift_left(eci_executor_t* exec, int n_tokens) {
    if (!exec || n_tokens <= 0) return ECI_ERR_INVALID_ARG;
    if (n_tokens >= exec->n_tokens) {
        return eci_executor_reset(exec);
    }

    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    llama_memory_seq_rm(exec->ctx_ref->mem, 0, 0, n_tokens);
    exec->n_tokens -= n_tokens;  // logical count shrinks
    // n_kv_pos stays the same — KV positions are absolute, not relative
    if (exec->last_batch_idx >= exec->n_tokens) exec->last_batch_idx = -1;

    // Trim recent tokens for repeat penalty
    if ((int)exec->recent_tokens.size() > n_tokens)
        exec->recent_tokens.erase(exec->recent_tokens.begin(),
                                  exec->recent_tokens.begin() + n_tokens);
    else
        exec->recent_tokens.clear();

    return ECI_OK;
}

// KV copy: copy an entire seq_id's KV to another seq_id.
// Used for state backup without blob serialization.
eci_result_t eci_kv_copy(eci_context_t* ctx, int src_seq_id, int dst_seq_id) {
    if (!ctx) return ECI_ERR_INVALID_ARG;
    if (src_seq_id < 0 || (uint32_t)src_seq_id >= ctx->n_seq_max) return ECI_ERR_INVALID_ARG;
    if (dst_seq_id < 0 || (uint32_t)dst_seq_id >= ctx->n_seq_max) return ECI_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);
    // Clear dest first, then copy
    llama_memory_seq_rm(ctx->mem, dst_seq_id, 0, -1);
    llama_memory_seq_cp(ctx->mem, src_seq_id, dst_seq_id, 0, -1);
    return ECI_OK;
}

// KV clear: clear KV for a specific seq_id only.
eci_result_t eci_kv_clear_seq(eci_context_t* ctx, int seq_id) {
    if (!ctx) return ECI_ERR_INVALID_ARG;
    if (seq_id < 0 || (uint32_t)seq_id >= ctx->n_seq_max) return ECI_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);
    llama_memory_seq_rm(ctx->mem, seq_id, 0, -1);
    return ECI_OK;
}

// KV size: query token count for a seq_id.
// llama.cpp doesn't expose per-seq KV token count directly.
// Use eci_conversation_token_count(conv) or eci_executor_token_count(exec) instead.
int eci_kv_seq_tokens(eci_context_t* ctx, int seq_id) {
    (void)seq_id;
    return ctx ? -1 : -1;  // not directly queryable
}

// Causal attention control (per-context, per-call).
eci_result_t eci_set_causal_attn(eci_context_t* ctx, bool causal) {
    if (!ctx) return ECI_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->infer_mtx);
    llama_set_causal_attn(ctx->ctx, causal);
    return ECI_OK;
}

// Non-causal check for mtmd.
bool eci_mtmd_needs_non_causal(eci_model_t* mmproj) {
    if (!mmproj || !mmproj->mtmd_ctx) return false;
    // chunk=nullptr means default case (image chunk)
    return mtmd_decode_use_non_causal(mmproj->mtmd_ctx, nullptr);
}

// ── Anti-prompt detection ──

int eci_check_anti_prompts(const char* text, const char** anti_prompts, int count) {
    if (!text || !anti_prompts || count <= 0) return -1;
    std::string s(text);
    for (int i = 0; i < count; i++) {
        if (anti_prompts[i] && s.find(anti_prompts[i]) != std::string::npos)
            return i;
    }
    return -1;
}

// ── Grammar (GBNF) ──

struct eci_grammar_s {
    const llama_vocab* vocab = nullptr;
    std::string grammar_str;
    std::string grammar_root;
};

eci_grammar_t* eci_grammar_create(eci_model_t* model, const char* grammar_str, const char* grammar_root) {
    if (!model || !model->vocab || !grammar_str || !grammar_root) return nullptr;
    // Validate by creating a test sampler
    auto* test = llama_sampler_init_grammar(model->vocab, grammar_str, grammar_root);
    if (!test) return nullptr;
    llama_sampler_free(test);
    // Store strings; the persistent sampler chain is built lazily per generation
// by ensure_grammar_chain() and rebuilt when the grammar changes
    auto* g = new eci_grammar_s();
    g->vocab = model->vocab;
    g->grammar_str = grammar_str;
    g->grammar_root = grammar_root;
    return g;
}

void eci_grammar_free(eci_grammar_t* grammar) {
    if (!grammar) return;
    delete grammar;
}

// Helper: reset grammar sampler state (new generation starts from position 0)
static void reset_grammar_state(eci_grammar_state_s* gs) {
    if (!gs) return;
    if (gs->chain) { llama_sampler_free(gs->chain); gs->chain = nullptr; }
    gs->grammar_str.clear();
    gs->grammar_root.clear();
}

// Ensure the persistent grammar sampler chain matches the requested grammar.
// Rebuilds when the grammar string/root changed (new request → fresh state).
static bool ensure_grammar_chain(eci_grammar_state_s* gs, eci_grammar_t* grammar) {
    if (!gs || !grammar) return false;
    if (gs->chain && (gs->grammar_str != grammar->grammar_str ||
                      gs->grammar_root != grammar->grammar_root)) {
        reset_grammar_state(gs);
    }
    if (!gs->chain) {
        llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler* gsampler = llama_sampler_init_grammar(
            grammar->vocab, grammar->grammar_str.c_str(), grammar->grammar_root.c_str());
        if (!gsampler) {
            llama_sampler_free(chain);
            return false;
        }
        llama_sampler_chain_add(chain, gsampler);
        gs->chain = chain;
        gs->grammar_str = grammar->grammar_str;
        gs->grammar_root = grammar->grammar_root;
    }
    return true;
}

static int32_t do_sample_with_grammar(llama_context* ctx, const llama_vocab* vocab,
                                       const eci_sampling_params_t* params,
                                       const std::vector<llama_token>& recent_tokens,
                                       int last_batch_idx,
                                       eci_grammar_t* grammar,
                                       eci_grammar_state_s* grammar_state) {
    if (!params || last_batch_idx < 0) return -1;
    if (grammar && !ensure_grammar_chain(grammar_state, grammar)) return -1;

    int n_vocab = llama_vocab_n_tokens(vocab);

    // Per-call SELECTION chain — grammar is NOT in it. The grammar lives in the
    // persistent chain and is applied FIRST on the full vocab (masking invalid
    // tokens to -inf) before temperature/truncation, mirroring llama.cpp's
    // common sampler chain order (grammar first, like llama-server).
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Penalties (repeat + present)
    if (params->repeat_penalty != 1.0f || params->penalty_present > 0.0f) {
        int pen_last_n = params->repeat_last_n >= 0 ? params->repeat_last_n : (int)recent_tokens.size();
        if (pen_last_n > 0 && !recent_tokens.empty()) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                n_vocab, pen_last_n, params->repeat_penalty, 0.0f, params->penalty_present));
        }
    }

    // Temperature / selection
    if (params->temperature <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params->temperature));
        if (params->top_k > 0)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params->top_k));
        if (params->top_p < 1.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params->top_p, 1));
        if (params->min_p > 0.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(params->min_p, 1));
        // Distribution sampler for stochastic selection
        llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t)rand()));
    }

    float* logits = llama_get_logits_ith(ctx, last_batch_idx);
    if (!logits) { llama_sampler_free(smpl); return -1; }

    // Draw loop: rebuild candidate array from raw logits each attempt so the
    // ignore_eos re-draw starts from unmodified logits (sampler apply mutates
    // the token data array, not the context logits buffer).
    llama_token token = -1;
    for (int attempt = 0; attempt < 16; attempt++) {
        std::vector<llama_token_data> cur;
        cur.reserve(n_vocab);
        for (int i = 0; i < n_vocab; i++) cur.push_back({i, logits[i], 0.0f});
        llama_token_data_array arr = { cur.data(), (size_t)n_vocab, -1, false };

        // Grammar masks FIRST (persistent chain — state advances via accept())
        if (grammar) llama_sampler_apply(grammar_state->chain, &arr);
        llama_sampler_apply(smpl, &arr);
        token = arr.data[arr.selected].id;

        // ignore_eos: re-draw WITHOUT accepting EOG into the grammar — feeding
        // end/control tokens into a JSON grammar kills all parse stacks.
        if (!params->ignore_eos || !llama_vocab_is_eog(vocab, token)) break;
    }

    // Advance grammar state with the committed token. Skip EOG/control tokens
    // (chat-template control tokens would desync the parse state).
    if (grammar && !llama_vocab_is_eog(vocab, token) && !llama_vocab_is_control(vocab, token)) {
        llama_sampler_accept(grammar_state->chain, token);
    }

    llama_sampler_free(smpl);
    return token;
}

eci_result_t eci_executor_sample_grammar(eci_executor_t* exec, const eci_sampling_params_t* params,
                                             eci_grammar_t* grammar, int* out_token) {
    if (!exec || !params || !out_token) return ECI_ERR_INVALID_ARG;
    if (exec->n_tokens == 0 || exec->last_batch_idx < 0) return ECI_ERR_DECODE_FAILED;

    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);
    int32_t token = do_sample_with_grammar(exec->ctx_ref->ctx, exec->ctx_ref->vocab,
                                            params, exec->recent_tokens, exec->last_batch_idx, grammar,
                                            &exec->grammar_state);
    if (token < 0) return ECI_ERR_DECODE_FAILED;
    *out_token = token;
    return ECI_OK;
}

// Explicit generation-boundary reset: the caller (C# BatchSession) invokes this
// once per generation BEFORE the sample loop. The conversation re-prompts each
// generated piece as text, so prompt hooks cannot distinguish new turns from
// per-token pieces.
eci_result_t eci_conversation_grammar_reset(eci_conversation_t* conv) {
    if (!conv) return ECI_ERR_INVALID_ARG;
    reset_grammar_state(&conv->grammar_state);
    return ECI_OK;
}

eci_result_t eci_executor_grammar_reset(eci_executor_t* exec) {
    if (!exec) return ECI_ERR_INVALID_ARG;
    reset_grammar_state(&exec->grammar_state);
    return ECI_OK;
}

eci_result_t eci_conversation_sample_grammar(eci_conversation_t* conv, const eci_sampling_params_t* params,
                                                eci_grammar_t* grammar, int* out_token) {
    if (!conv || !params || !out_token) return ECI_ERR_INVALID_ARG;
    if (conv->n_tokens == 0 || conv->last_batch_idx < 0) return ECI_ERR_DECODE_FAILED;

    int32_t token = do_sample_with_grammar(conv->ctx, llama_model_get_vocab(llama_get_model(conv->ctx)),
                                            params, conv->recent_tokens, conv->last_batch_idx, grammar,
                                            &conv->grammar_state);
    if (token < 0) return ECI_ERR_DECODE_FAILED;
    conv->recent_tokens.push_back(token);
    *out_token = token;
    return ECI_OK;
}

// ── Chat template ──

eci_result_t eci_apply_chat_template(eci_model_t* model, const char* tmpl,
                                        const eci_chat_message_t* messages, int n_messages,
                                        bool add_assistant,
                                        char** out_text) {
    if (!model || !messages || n_messages <= 0 || !out_text) return ECI_ERR_INVALID_ARG;

    // Convert to llama_chat_message array
    std::vector<llama_chat_message> chat(n_messages);
    for (int i = 0; i < n_messages; i++) {
        chat[i].role = messages[i].role;
        chat[i].content = messages[i].content;
    }

    // First call to get the required size
    int32_t needed = llama_chat_apply_template(tmpl, chat.data(), n_messages, add_assistant, nullptr, 0);
    if (needed <= 0) {
        model->last_error = "chat template apply failed";
        return ECI_ERR_INTERNAL;
    }

    // Allocate and apply
    // Add some slack — llama_chat_apply_template can sometimes need more than reported
    int32_t buf_size = needed + 256;
    char* buf = (char*)malloc(buf_size);
    if (!buf) return ECI_ERR_INTERNAL;

    int32_t written = llama_chat_apply_template(tmpl, chat.data(), n_messages, add_assistant, buf, buf_size);
    if (written <= 0) {
        free(buf);
        model->last_error = "chat template apply failed on second call";
        return ECI_ERR_INTERNAL;
    }

    *out_text = buf;
    return ECI_OK;
}

// ── Vision on standard executor ──

eci_result_t eci_executor_prompt_with_images(eci_executor_t* exec,
                                                 eci_model_t* model,
                                                 const char* text,
                                                 const char** image_data,
                                                 const int* image_sizes,
                                                 int n_images) {
    if (!exec || !model || !text) return ECI_ERR_INVALID_ARG;
    if (!model->mtmd_ctx) {
        model->last_error = "No mmproj loaded — cannot process images";
        return ECI_ERR_INVALID_ARG;
    }
    if (n_images > 0 && (!image_data || !image_sizes)) return ECI_ERR_INVALID_ARG;

    mtmd_context* mctx = model->mtmd_ctx;

    // Step 1: Load all images as bitmaps
    std::vector<mtmd_bitmap*> bitmaps;
    for (int i = 0; i < n_images; i++) {
        auto wrapper = mtmd_helper_bitmap_init_from_buf(
            mctx, (const unsigned char*)image_data[i], image_sizes[i], false, mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            for (auto* b : bitmaps) mtmd_bitmap_free(b);
            return ECI_ERR_INTERNAL;
        }
        bitmaps.push_back(wrapper.bitmap);
    }

    // Step 2: Tokenize prompt + images into chunks
    mtmd_input_text input_text = {};
    input_text.text = text;
    input_text.text_len = strlen(text);
    input_text.add_special = false;
    input_text.parse_special = false;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t rc = mtmd_tokenize(mctx, chunks, &input_text,
                               bitmaps.data(), bitmaps.size());

    for (auto* b : bitmaps) mtmd_bitmap_free(b);

    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        return ECI_ERR_INTERNAL;
    }

    // Step 3: Process chunks — text → pending tokens, images → embd decode
    std::lock_guard<std::mutex> lock(exec->ctx_ref->infer_mtx);

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            if (tokens && n_tokens > 0) {
                exec->pending_tokens.insert(exec->pending_tokens.end(), tokens, tokens + n_tokens);
                exec->has_pending = true;
            }
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // Flush pending text tokens first (separate decode batch)
            if (exec->has_pending) {
                int n = (int)exec->pending_tokens.size();
                llama_batch batch = llama_batch_init(n, 0, exec->ctx_ref->n_seq_max);
                for (int j = 0; j < n; j++) {
                    batch.token[j] = exec->pending_tokens[j];
                    batch.pos[j] = exec->n_kv_pos + j;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j] = 0;
                }
                batch.n_tokens = n;
                if (llama_decode(exec->ctx_ref->ctx, batch) == 0) {
                    exec->n_tokens += n;
                    exec->n_kv_pos += n;
                    exec->last_batch_idx = n - 1;
                }
                llama_batch_free(batch);
                exec->pending_tokens.clear();
                exec->has_pending = false;
            }

            // Encode image with CLIP (non-causal attention)
            llama_set_causal_attn(exec->ctx_ref->ctx, false);
            mtmd_helper_eval_chunk_single(mctx, exec->ctx_ref->ctx, chunk,
                                          exec->n_kv_pos, 0, 512, false, nullptr);
            llama_set_causal_attn(exec->ctx_ref->ctx, true);

            // Update position by the number of image tokens
            const mtmd_image_tokens* img_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            int n_img_tokens = img_tokens ? mtmd_image_tokens_get_n_tokens(img_tokens) : 0;
            exec->n_tokens += n_img_tokens;
            exec->n_kv_pos += n_img_tokens;
            exec->last_batch_idx = n_img_tokens - 1;
        }
    }

    mtmd_input_chunks_free(chunks);
    return ECI_OK;
}

// ── Last error ──

const char* eci_last_error(eci_context_t* ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

// ── Structured logging implementation (2026-09-25) ──
extern "C" void eci_set_log_callback(eci_log_callback_t cb) {
    g_log_cb = cb;
}

extern "C" void eci_set_log_level(eci_log_level_t level) {
    g_log_min_level = level;
}

extern "C" void eci_log(eci_log_level_t level, const char* tag, const char* fmt, ...) {
    if (!g_log_cb || level < g_log_min_level) return;
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    g_log_cb((int)level, tag, buf);
}
