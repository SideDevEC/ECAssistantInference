// eci_internal.h — Internal structs shared between ecainference.cpp and eci_backend.cpp
// NOT part of the public API (ecainference.h).

#ifndef ECI_INTERNAL_H
#define ECI_INTERNAL_H

#include "ecainference.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <string>
#include <vector>
#include <stack>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <set>

// ── Global destruction synchronization ──
//
// Metal (and potentially other backends) assert that all GPU resources
// (buffers, residency sets) are freed before the device is freed.
// eci_free_context frees llama_context (which owns Metal buffers), and
// eci_free_model frees llama_model (which owns Metal devices). If these
// run concurrently on different threads (e.g. .NET SafeHandle finalizers
// during process exit), the device destructor asserts because buffers
// haven't been freed yet.
//
// The destruction gate ensures:
// 1. eci_free_context acquires the gate, frees the context, releases the gate
// 2. eci_free_model acquires the gate, waits for all contexts to be freed,
//    then frees the model
// This is backend-agnostic — it protects ANY backend that requires
// ordered teardown (Metal, CUDA, Vulkan all have this constraint).
//
// The context count is per-model: eci_free_model blocks until its
// model's context count reaches zero. This preserves parallelism —
// contexts from DIFFERENT models can be freed concurrently.

struct eci_model_s {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    mtmd_context* mtmd_ctx = nullptr;
    std::string last_error;
    // Config from eci_model_params_t, read by eci_create_context
    bool flash_attn = true;
    eci_kv_type_t kv_cache_type = ECI_KV_F16;
    int threads = -1;  // -1 = auto

    // Destruction synchronization: track how many contexts are alive for this model.
    // eci_free_model waits on this CV until the count hits zero.
    int active_context_count = 0;
    std::mutex destruction_mtx;
    std::condition_variable destruction_cv;
};

struct eci_context_s {
    llama_context* ctx = nullptr;
    llama_model* model = nullptr;       // raw llama_model pointer (non-owning)
    eci_model_t* owning_model = nullptr; // back-reference for destruction accounting
    const llama_vocab* vocab = nullptr;
    llama_memory_t mem = nullptr;
    uint32_t n_ctx = 0;
    uint32_t n_seq_max = 0;
    eci_pooling_type_t pool_type = ECI_POOL_NONE;
    std::string last_error;

    std::vector<eci_conversation_t*> active_conversations;
    std::mutex infer_mtx;
};

// Per-executor/conversation grammar sampler holder. The chain MUST persist
// across sample calls within one generation (grammar is stateful — restarting
// from a fresh sampler per call makes the grammar expect position 0 forever).
// Rebuilt automatically when a different grammar string/root is passed.
struct eci_grammar_state_s {
    llama_sampler* chain = nullptr;
    std::string grammar_str;
    std::string grammar_root;
    ~eci_grammar_state_s() { if (chain) llama_sampler_free(chain); }
};

struct eci_conversation_s {
    llama_seq_id seq_id;
    llama_context* ctx;
    llama_memory_t mem;
    int n_tokens = 0;
    int n_kv_pos = 0;
    bool has_pending_prompt = false;
    std::vector<llama_token> pending_tokens;
    int last_batch_idx = -1;

    // Snapshot of the last committed token's logits row, copied at decode
    // time. llama_get_logits_ith() rows are only addressable while their
    // batch is the shared context's CURRENT batch — any later decode on the
    // same ctx (another conversation's slice or cycle in batched inference)
    // replaces it, and sampling a stale row GGML_ABORTs the process
    // (get_logits_ith: "batch.logits[i] != true"). Sampling reads this
    // private copy, immune to interleaved decodes.
    std::vector<float> last_logits;

    // Rolling window of committed tokens for repeat/present penalties
    // (mirrors eci_executor_s.recent_tokens)
    std::vector<llama_token> recent_tokens;

    // Persistent grammar sampler state (survives across sample calls within
    // one generation; rebuilt when a different grammar is passed)
    eci_grammar_state_s grammar_state;
};

struct eci_pool_s {
    eci_context_t* ctx_ref;
    std::vector<std::unique_ptr<eci_conversation_s>> conversations;
    std::stack<eci_conversation_t*> available;
    std::mutex mtx;
};

struct eci_executor_s {
    eci_context_t* ctx_ref;
    int n_tokens = 0;
    int n_kv_pos = 0;
    bool has_pending = false;
    std::vector<llama_token> pending_tokens;
    int last_batch_idx = -1;
    // Private logits-row snapshot — same rationale as
    // eci_conversation_s::last_logits (sampling must never read a batch row
    // that a later decode on the shared ctx may have invalidated).
    std::vector<float> last_logits;
    std::vector<llama_token> recent_tokens;
    eci_grammar_state_s grammar_state;
};

struct eci_state_s {
    int n_tokens;
};

#endif // ECI_INTERNAL_H
