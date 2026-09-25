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
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <set>

struct eci_model_s {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    mtmd_context* mtmd_ctx = nullptr;
    std::string last_error;
    // Config from eci_model_params_t, read by eci_create_context
    bool flash_attn = true;
    eci_kv_type_t kv_cache_type = ECI_KV_F16;
    int threads = -1;  // -1 = auto
};

struct eci_context_s {
    llama_context* ctx = nullptr;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_memory_t mem = nullptr;
    uint32_t n_ctx = 0;
    uint32_t n_seq_max = 0;
    eci_pooling_type_t pool_type = ECI_POOL_NONE;
    std::string last_error;

    std::vector<eci_conversation_t*> active_conversations;
    std::mutex infer_mtx;
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

    // Persistent grammar sampler chain
    llama_sampler* grammar_chain = nullptr;
    bool grammar_initialized = false;
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
    std::vector<llama_token> recent_tokens;
    llama_sampler* grammar_chain = nullptr;
};

struct eci_state_s {
    int n_tokens;
};

#endif // ECI_INTERNAL_H
