#ifndef ECAINFERENCE_H
#define ECAINFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ── Opaque handles ── */
typedef struct eci_model_s       eci_model_t;
typedef struct eci_context_s      eci_context_t;
typedef struct eci_pool_s         eci_pool_t;
typedef struct eci_conversation_s eci_conversation_t;
typedef struct eci_executor_s     eci_executor_t;     /* standard (non-batched) */
typedef struct eci_state_s        eci_state_t;        /* saved KV state */

/* ── Error codes ── */
typedef enum {
    ECI_OK = 0,
    ECI_ERR_INVALID_ARG = 1,
    ECI_ERR_INVALID_HANDLE = 2,
    ECI_ERR_LOAD_FAILED = 3,
    ECI_ERR_NO_SLOT = 4,
    ECI_ERR_DECODE_FAILED = 5,
    ECI_ERR_OVERFLOW = 6,
    ECI_ERR_NOT_SUPPORTED = 7,
    ECI_ERR_INTERNAL = 99,
} eci_result_t;

/* ── KV cache type ── */
typedef enum {
    ECI_KV_F16 = 0,
    ECI_KV_Q8_0 = 1,
    ECI_KV_Q4_0 = 2,
    ECI_KV_Q4_1 = 3,
} eci_kv_type_t;

/* ── Pooling type (embeddings) ── */
typedef enum {
    ECI_POOL_NONE = 0,
    ECI_POOL_MEAN = 1,
    ECI_POOL_CLS = 2,
    ECI_POOL_LAST = 3,
} eci_pooling_type_t;

/* ── Model params ── */
typedef struct {
    const char* model_path;
    int gpu_layers;          /* -1 = all, 0 = CPU only */
    int threads;             /* -1 = auto */
    bool flash_attn;
    eci_kv_type_t kv_cache_type;
} eci_model_params_t;

/* ── Context params ── */
typedef struct {
    uint32_t context_size;   /* n_ctx */
    uint32_t batch_size;     /* n_batch */
    uint32_t seq_max;        /* max sequences (batch mode) */
    eci_pooling_type_t pooling_type;  /* for embedding models */
    bool fused_gdn;          /* llama.cpp fused gated-delta-net kernels
                               (config-injected; replaces the former
                               ECI_DISABLE_GDN env var — behavior pinned,
                               never environment-dependent) */
} eci_context_params_t;

/* ── Sampling params (full pipeline) ── */
typedef struct {
    float temperature;
    float top_p;
    int   top_k;
    float min_p;
    float repeat_penalty;
    int   repeat_last_n;     /* -1 = full context */
    float penalty_present;  /* presence penalty */
    int   max_tokens;
    bool  ignore_eos;
} eci_sampling_params_t;

/* ── Decode result ── */
typedef enum {
    ECI_DECODE_OK = 0,
    ECI_DECODE_NO_WORK = 1,
    ECI_DECODE_FAILED = 2,
} eci_decode_result_t;

/* ════════════════════════════════════════════════════
 *  Model
 * ════════════════════════════════════════════════════ */

eci_result_t eci_load_model(const eci_model_params_t* params, eci_model_t** out_model);
void eci_free_model(eci_model_t* model);

/* ════════════════════════════════════════════════════
 *  Context (shared KV pool — used by both standard and batch paths)
 * ════════════════════════════════════════════════════ */

eci_result_t eci_create_context(eci_model_t* model,
                                const eci_context_params_t* params,
                                eci_context_t** out_ctx);
void eci_free_context(eci_context_t* ctx);

uint32_t eci_context_size(eci_context_t* ctx);
uint32_t eci_context_seq_max(eci_context_t* ctx);
int      eci_context_n_ctx_used(eci_context_t* ctx, eci_conversation_t* conv);

/* ════════════════════════════════════════════════════
 *  Tokenization
 * ════════════════════════════════════════════════════ */

eci_result_t eci_tokenize(eci_context_t* ctx, const char* text,
                          bool add_bos, bool parse_special,
                          int32_t** out_tokens, int* out_count);
void eci_free_tokens(int32_t* tokens);

eci_result_t eci_detokenize(eci_context_t* ctx, const int32_t* tokens, int count,
                            bool remove_special,
                            char** out_text);
void eci_free_string(char* str);

/* Token to piece (single token → string) */
eci_result_t eci_token_to_piece(eci_context_t* ctx, int32_t token, char* buf, int buf_size);

/* ════════════════════════════════════════════════════
 *  Standard (non-batched) executor — one session, one context
 *  Replaces: StatefulExecutorBase / InteractiveExecutor
 * ════════════════════════════════════════════════════ */

eci_result_t eci_executor_create(eci_context_t* ctx, eci_executor_t** out_exec);
void eci_executor_free(eci_executor_t* exec);

/* Prompt the executor (tokenizes + feeds to KV cache) */
eci_result_t eci_executor_prompt(eci_executor_t* exec, const char* text);

/* Prompt with raw tokens */
eci_result_t eci_executor_prompt_tokens(eci_executor_t* exec, const int32_t* tokens, int count);

/* Decode one step (processes pending tokens) */
eci_decode_result_t eci_executor_infer(eci_executor_t* exec);

/* Sample one token using full pipeline */
eci_result_t eci_executor_sample(eci_executor_t* exec,
                                  const eci_sampling_params_t* params,
                                  int32_t* out_token);

/* Check if token is end-of-generation */
bool eci_token_is_eos(eci_context_t* ctx, int32_t token);

/* Current token count in KV cache */
int eci_executor_token_count(eci_executor_t* exec);

/* Rewind (remove last n tokens from KV) */
eci_result_t eci_executor_rewind(eci_executor_t* exec, int n_tokens);

/* Reset (clear entire KV cache) */
eci_result_t eci_executor_reset(eci_executor_t* exec);

/* ════════════════════════════════════════════════════
 *  State save/restore (for rewind in standard path)
 *  Replaces: LLamaContext.State / StatefulExecutorBase.ExecutorBaseState
 * ════════════════════════════════════════════════════ */

eci_result_t eci_state_save(eci_executor_t* exec, eci_state_t** out_state);
eci_result_t eci_state_restore(eci_executor_t* exec, eci_state_t* state);
void eci_state_free(eci_state_t* state);

/* Get the token count saved in a state (for verification) */
int eci_state_token_count(eci_state_t* state);

/* ════════════════════════════════════════════════════
 *  Conversation pool (batched inference)
 *  Replaces: BatchedExecutor / Conversation
 * ════════════════════════════════════════════════════ */

eci_result_t eci_pool_create(eci_context_t* ctx, eci_pool_t** out_pool);
void eci_pool_free(eci_pool_t* pool);

eci_result_t eci_pool_lease(eci_pool_t* pool, eci_conversation_t** out_conv);
eci_result_t eci_pool_return(eci_pool_t* pool, eci_conversation_t* conv);
int eci_pool_available(eci_pool_t* pool);
uint32_t eci_pool_size(eci_pool_t* pool);

/* ════════════════════════════════════════════════════
 *  Conversation (leased from pool)
 * ════════════════════════════════════════════════════ */

eci_result_t eci_conversation_prompt(eci_conversation_t* conv, const char* text);
eci_result_t eci_conversation_prompt_tokens(eci_conversation_t* conv,
                                             const int32_t* tokens, int count);
eci_result_t eci_conversation_sample(eci_conversation_t* conv,
                                     const eci_sampling_params_t* params,
                                     int32_t* out_token);
int eci_conversation_token_count(eci_conversation_t* conv);
eci_result_t eci_conversation_rewind(eci_conversation_t* conv, int n_tokens);
eci_result_t eci_conversation_reset(eci_conversation_t* conv);
eci_result_t eci_conversation_grammar_reset(eci_conversation_t* conv);
eci_result_t eci_executor_grammar_reset(eci_executor_t* exec);

/* Save/restore conversation state (for batch rewind) */
eci_result_t eci_conversation_save(eci_conversation_t* conv, eci_state_t** out_state);
eci_result_t eci_conversation_restore(eci_pool_t* pool, eci_conversation_t* conv,
                                       eci_state_t* state);
void eci_conversation_state_free(eci_state_t* state);

/* ════════════════════════════════════════════════════
 *  Batched inference — ONE decode for ALL conversations
 * ════════════════════════════════════════════════════ */

eci_decode_result_t eci_infer(eci_context_t* ctx);

/* ════════════════════════════════════════════════════
 *  Embeddings
 *  Replaces: LLamaEmbedder
 * ════════════════════════════════════════════════════ */

eci_result_t eci_get_embeddings(eci_model_t* model, eci_context_t* ctx,
                                const char* text,
                                float** out_embeddings, int* out_dim);
void eci_free_embeddings(float* embeddings);

int eci_embedding_dim(eci_model_t* model);

/* ════════════════════════════════════════════════════
 *  Vision / MTMD
 *  Replaces: MtmdWeights
 * ════════════════════════════════════════════════════ */

eci_result_t eci_load_mmproj(eci_model_t* model, const char* mmproj_path,
                             eci_model_t** out_mmproj);
void eci_free_mmproj(eci_model_t* mmproj);

/* Load image bytes into a conversation's media queue */
eci_result_t eci_conversation_load_image(eci_conversation_t* conv,
                                          eci_model_t* mmproj,
                                          const uint8_t* data, int size);

/* Full prompt with images: tokenize prompt+images, encode, feed to conversation */
eci_result_t eci_conversation_prompt_with_images(eci_conversation_t* conv,
                                                    eci_model_t* model_with_mtmd,
                                                    const char* text,
                                                    const uint8_t** image_data,
                                                    const int* image_sizes,
                                                    int n_images);

/* Get the mtmd marker string (e.g. "<image>") */
eci_result_t eci_mtmd_marker(char** out_marker);
const char* eci_mtmd_marker_static(void);

/* ════════════════════════════════════════════════════
 *  Granular KV cache manipulation (impossible via LLamaSharp)
 * ════════════════════════════════════════════════════ */

/* Shift left: remove oldest n tokens, keep recent ones.
 * Sliding window context — no reset needed when approaching n_ctx.
 * Re-positions remaining tokens to start at position 0.
 * Returns ECI_ERR_NOT_SUPPORTED if the backend can't reposition.
 *
 * Implementation: llama_memory_seq_rm(seq, 0, n) removes the front n tokens,
 * then llama_memory_seq_cp shifts remaining positions. */
eci_result_t eci_conversation_shift_left(eci_conversation_t* conv, int n_tokens);
eci_result_t eci_executor_shift_left(eci_executor_t* exec, int n_tokens);

/* KV copy: copy an entire conversation's KV to another seq_id.
 * Used for state backup without blob serialization.
 * Source and dest must be on the same context. */
eci_result_t eci_kv_copy(eci_context_t* ctx, int src_seq_id, int dst_seq_id);

/* KV clear: clear KV for a specific seq_id only.
 * Does NOT affect other conversations. */
eci_result_t eci_kv_clear_seq(eci_context_t* ctx, int seq_id);

/* KV size: how many tokens are stored for a seq_id.
 * Returns -1 on error. */
int eci_kv_seq_tokens(eci_context_t* ctx, int seq_id);

/* Causal attention control (per-context, per-call).
 * Set false before decoding image embeddings, true for text.
 * Vision path handles this internally — exposed for advanced use. */
eci_result_t eci_set_causal_attn(eci_context_t* ctx, bool causal);

/* Non-causal check for mtmd: returns true if the current model
 * needs non-causal attention for image decode. */
bool eci_mtmd_needs_non_causal(eci_model_t* mmproj);

/* Check if generated text contains any anti-prompt string.
 * Returns index of matched anti-prompt (>=0) or -1 if none. */
int eci_check_anti_prompts(const char* text, const char** anti_prompts, int count);

/* ════════════════════════════════════════════════════
 *  Grammar (GBNF)
 * ════════════════════════════════════════════════════ */

/* Opaque grammar handle. */
typedef struct eci_grammar_s eci_grammar_t;

/* Create a GBNF grammar sampler. Returns null if parsing fails.
 * The grammar is applied during sampling — only tokens matching the
 * grammar are selected. Pass null to eci_*_sample_with_grammar to
 * sample without grammar constraint.
 * The grammar must be created from the same model's vocab. */
eci_grammar_t* eci_grammar_create(eci_model_t* model, const char* grammar_str, const char* grammar_root);

/* Free a grammar. */
void eci_grammar_free(eci_grammar_t* grammar);

/* Sample with grammar constraint. Like eci_executor_sample / eci_conversation_sample
 * but applies the grammar sampler before token selection. */
eci_result_t eci_executor_sample_grammar(eci_executor_t* exec, const eci_sampling_params_t* params,
                                             eci_grammar_t* grammar, int* out_token);
eci_result_t eci_conversation_sample_grammar(eci_conversation_t* conv, const eci_sampling_params_t* params,
                                                eci_grammar_t* grammar, int* out_token);

/* ════════════════════════════════════════════════════
 *  Chat template
 * ════════════════════════════════════════════════════ */

/* Apply the model's built-in chat template to a list of messages.
 * Uses llama_chat_apply_template (supports Qwen, Llama, ChatML, etc.).
 * Pass tmpl=null to use the model's default template.
 * messages: array of {role, content} pairs.
 * add_assistant: append the assistant turn marker.
 * out_text: receives the formatted prompt (caller frees with eci_free_string).
 * Returns ECI_OK or error. */
typedef struct { const char* role; const char* content; } eci_chat_message_t;

eci_result_t eci_apply_chat_template(eci_model_t* model, const char* tmpl,
                                        const eci_chat_message_t* messages, int n_messages,
                                        bool add_assistant,
                                        char** out_text);

/* ════════════════════════════════════════════════════
 *  Vision on standard executor
 * ════════════════════════════════════════════════════ */

/* Prompt the standard executor with text + images.
 * Like eci_executor_prompt but processes image markers via the mmproj projector.
 * The executor uses seq_id 0 — images are encoded and decoded with non-causal attention.
 * Must be called with no pending tokens (flush before vision prompt).
 * After this call, call eci_executor_infer as normal. */
eci_result_t eci_executor_prompt_with_images(eci_executor_t* exec,
                                                 eci_model_t* model,
                                                 const char* text,
                                                 const char** image_data_ptrs,
                                                 const int* image_sizes,
                                                 int n_images);

/* ════════════════════════════════════════════════════
 *  Logging / errors
 * ════════════════════════════════════════════════════ */

const char* eci_last_error(eci_context_t* ctx);
const char* eci_result_str(eci_result_t r);

/* ── Structured logging callback (2026-09-25) ──
 *
 * Zero-overhead when no callback is set: every log site is a single
 * `if (g_log_cb)` check — branch-predicted to skip. When the callback
 * is NULL (default), no string formatting, no argument marshalling.
 *
 * Levels: DEBUG < INFO < WARN < ERROR. Set the minimum level via
 * eci_set_log_level. Messages below the minimum are not dispatched.
 * The callback receives: level (int), a tag string (static), and the
 * formatted message. The callback owns the received strings for the
 * duration of the call only — copy if needed.
 */
typedef enum {
    ECI_LOG_DEBUG = 0,
    ECI_LOG_INFO  = 1,
    ECI_LOG_WARN  = 2,
    ECI_LOG_ERROR = 3,
    ECI_LOG_NONE  = 99
} eci_log_level_t;

typedef void (*eci_log_callback_t)(int level, const char* tag, const char* message);

/* Set the global log callback. Pass NULL to disable (zero-overhead). */
void eci_set_log_callback(eci_log_callback_t cb);

/* Set the minimum log level. Messages below this level are not dispatched. */
void eci_set_log_level(eci_log_level_t level);

/* Internal log helper — not for direct use by callers; use ECI_LOG macro. */
void eci_log(eci_log_level_t level, const char* tag, const char* fmt, ...);

/* Convenience macro: evaluates g_log_cb first (zero-cost skip). */
#define ECI_LOG(lvl, tag, ...) do { \
    if (g_log_cb && (lvl) >= g_log_min_level) { \
        eci_log((lvl), (tag), __VA_ARGS__); \
    } \
} while(0)

/* ════════════════════════════════════════════════════
 *  Backend safety layer
 * ════════════════════════════════════════════════════ */

/* Probe: validate the caller's config works on this backend.
 * Does a minimal decode. Does NOT change config -- just validates.
 * Returns ECI_OK if the config works, ECI_ERR_DECODE_FAILED if not. */
eci_result_t eci_probe_backend(eci_context_t* ctx);

/* Safe context creation with progressive fallback.
 * Tries the caller's requested config first (from ECAssistantLLM config).
 * If it fails, progressively reduces gpu_layers: N -> N/2 -> N/4 -> ... -> 0.
 * The caller's config (KV type, batch size, FA, seq_max) is RESPECTED --
 * only gpu_layers is reduced during fallback.
 * model may be reloaded with fewer gpu_layers (caller passes model**). */
eci_result_t eci_create_context_safe(eci_model_t** model,
                                      const eci_model_params_t* model_params,
                                      const eci_context_params_t* ctx_params,
                                      int requested_gpu_layers,
                                      eci_context_t** out_ctx);

/* VRAM guard: check if enough VRAM is available for an operation. */
bool eci_vram_available(size_t needed_bytes);

/* Backend state queries -- what the probe found about THIS config */
bool        eci_backend_probe_passed(void);
const char* eci_backend_name(void);
const char* eci_backend_probe_error(void);
bool        eci_backend_needs_kv_flush(void);
bool        eci_backend_gpu_layers_reduced(void);
int         eci_backend_effective_gpu_layers(void);
int         eci_backend_original_gpu_layers(void);

/* KV flush: flush stale K/V cells after clearing (Vulkan bug #26744).
 * Automatically called in eci_pool_return when eci_backend_needs_kv_flush() is true. */
eci_result_t eci_kv_flush(eci_context_t* ctx, int seq_id);

#ifdef __cplusplus
}
#endif

#endif /* ECAINFERENCE_H */
