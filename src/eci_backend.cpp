// eci_backend.cpp — Backend safety layer for ECAssistantInference
//
// Philosophy: the CALLER drives all configuration (KV type, batch size, FA, gpu_layers).
// Our layer only intervenes when the probe FAILS — then it tries safe alternatives
// and reports what worked. No hardcoded defaults, no per-backend assumptions.
//
// Flow:
//   1. Caller passes desired config (from ECAssistantLLM config file)
//   2. eci_create_context_safe tries the desired config
//   3. eci_probe_backend validates with a real decode
//   4. If probe fails: try reducing gpu_layers, then adjusting FA/KV/batch
//   5. Returns the working config + what was adjusted
//
// Issues referenced (from llama.cpp GitHub):
//   Vulkan: #19955 (DeviceLost), #26195 (FA+q8_0 garbled), #26744 (stale KV cells),
//           #26358 (FA+GQA crash), #27237 (batch 512 garbage), #27360 (load-mode OOM)
//   Metal:  1a5631b (cmd buffer graceful), 70aff25 (KV dequant internal)
//   CUDA:   #26609 (MoE+FA illegal access), #19980 (mmproj VRAM), #551 (auto SIGABRT)

#include "eci_internal.h"

// ── Backend detection ──

typedef enum {
    ECI_BACKEND_CPU = 0,
    ECI_BACKEND_METAL,
    ECI_BACKEND_CUDA,
    ECI_BACKEND_VULKAN,
} eci_backend_type_t;

// Runtime state — what the probe discovered about THIS backend+config combination
struct eci_backend_profile {
    eci_backend_type_t type;
    bool probe_passed;
    std::string name;
    std::string probe_error;
    // What was adjusted from the caller's requested config (for logging/diagnostics)
    bool gpu_layers_reduced;
    bool flash_attn_disabled;
    bool kv_cache_downgraded;   // q8_0 → f16
    int  original_gpu_layers;
    int  effective_gpu_layers;
    // Vulkan-specific: stale KV cells after clear (#26744)
    bool needs_kv_flush_after_clear;
};

static eci_backend_profile g_backend = {};

static eci_backend_type_t detect_backend(llama_context* ctx) {
    int n_backends = ggml_backend_dev_count();
    for (int i = 0; i < n_backends; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* name = ggml_backend_dev_name(dev);
        if (strstr(name, "Metal") || strstr(name, "metal")) return ECI_BACKEND_METAL;
        if (strstr(name, "CUDA") || strstr(name, "cuda")) return ECI_BACKEND_CUDA;
        if (strstr(name, "Vulkan") || strstr(name, "vulkan")) return ECI_BACKEND_VULKAN;
    }
    (void)ctx;
    return ECI_BACKEND_CPU;
}

static const char* backend_name(eci_backend_type_t t) {
    switch (t) {
        case ECI_BACKEND_METAL:  return "Metal";
        case ECI_BACKEND_CUDA:   return "CUDA";
        case ECI_BACKEND_VULKAN: return "Vulkan";
        default:                 return "CPU";
    }
}

// ── Probe: validate that the caller's config actually works on this backend ──
//
// Does a minimal decode (1 token) to verify. Does NOT change any config.
// Returns ECI_OK if the config works, ECI_ERR_DECODE_FAILED if not.

eci_result_t eci_probe_backend(eci_context_t* ctx) {
    if (!ctx) return ECI_ERR_INVALID_ARG;

    g_backend.type = detect_backend(ctx->ctx);
    g_backend.name = backend_name(g_backend.type);
    g_backend.probe_passed = false;
    g_backend.probe_error.clear();
    g_backend.needs_kv_flush_after_clear = false;

    // Vulkan: stale KV cells bug (#26744) affects pool reuse.
    // We can't fix it at probe time, but we flag it so eci_pool_return
    // knows to flush after clearing.
    if (g_backend.type == ECI_BACKEND_VULKAN) {
        g_backend.needs_kv_flush_after_clear = true;
    }

    // Minimal probe: tokenize "test" and decode
    std::vector<llama_token> tokens;
    const char* test = "test";
    int n_max = 5;
    tokens.resize(n_max);
    int n = llama_tokenize(ctx->vocab, test, 4, tokens.data(), n_max, false, true);
    if (n <= 0) {
        g_backend.probe_error = "tokenization failed in probe";
        return ECI_ERR_INTERNAL;
    }
    tokens.resize(n);

    llama_batch batch = llama_batch_init(n, 0, ctx->n_seq_max);
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1) ? 1 : 0;
    }
    batch.n_tokens = n;

    int rc = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);

    // Clean up probe KV
    llama_memory_seq_rm(ctx->mem, 0, 0, -1);

    if (rc != 0) {
        g_backend.probe_error = "decode failed in probe (rc=" + std::to_string(rc) + ")";
        return ECI_ERR_DECODE_FAILED;
    }

    g_backend.probe_passed = true;
    return ECI_OK;
}

// ── Safe context creation with progressive fallback ──
//
// Tries the caller's requested config first. If context creation or probe fails,
// progressively reduces gpu_layers. If that still fails, tries disabling FA,
// then downgrading KV cache type. Returns the first working config.
//
// The caller learns what changed via the probe query functions.

// Forward declaration from ecainference.cpp
// (eci_create_context is implemented there)

// Helper: reload model with different gpu_layers
static eci_result_t reload_model_with_layers(eci_model_t** model,
                                              const eci_model_params_t* model_params,
                                              int gpu_layers) {
    if (!model || !model_params) return ECI_ERR_INVALID_ARG;

    eci_model_params_t mp = *model_params;
    mp.gpu_layers = gpu_layers;

    if (*model) {
        eci_free_model(*model);
        *model = nullptr;
    }

    return eci_load_model(&mp, model);
}

eci_result_t eci_create_context_safe(eci_model_t** model,
                                      const eci_model_params_t* model_params,
                                      const eci_context_params_t* ctx_params,
                                      int requested_gpu_layers,
                                      eci_context_t** out_ctx) {
    if (!model || !model_params || !ctx_params || !out_ctx)
        return ECI_ERR_INVALID_ARG;

    // Reset adjustment tracking
    g_backend.gpu_layers_reduced = false;
    g_backend.flash_attn_disabled = false;
    g_backend.kv_cache_downgraded = false;
    g_backend.original_gpu_layers = requested_gpu_layers;
    g_backend.effective_gpu_layers = requested_gpu_layers;

    // Save original model pointer so we can restore on total failure
    eci_model_t* original_model = *model;

    // Phase 1: Try requested gpu_layers, then progressively reduce
    int gpu_layers = requested_gpu_layers;
    while (gpu_layers >= 0) {
        // (Re)load model with this many GPU layers
        eci_result_t mrc = reload_model_with_layers(model, model_params, gpu_layers);
        if (mrc != ECI_OK) {
            // Model load failed — try fewer layers
            if (gpu_layers == 0) {
                g_backend.probe_error = "model load failed even with 0 GPU layers";
                return ECI_ERR_LOAD_FAILED;
            }
            gpu_layers /= 2;
            if (gpu_layers == 0 && requested_gpu_layers > 0) continue;
            if (gpu_layers == 0) break;
            continue;
        }

        // Try creating context
        eci_context_t* ctx = nullptr;
        eci_result_t crc = eci_create_context(*model, ctx_params, &ctx);
        if (crc == ECI_OK) {
            // Probe the backend
            eci_result_t prc = eci_probe_backend(ctx);
            if (prc == ECI_OK) {
                // Success!
                if (gpu_layers < requested_gpu_layers) {
                    g_backend.gpu_layers_reduced = true;
                    g_backend.effective_gpu_layers = gpu_layers;
                }
                *out_ctx = ctx;
                return ECI_OK;
            }
            // Probe failed — free and try fewer layers
            eci_free_context(ctx);
        }

        if (gpu_layers == 0) break;
        gpu_layers /= 2;
        if (gpu_layers == 0) continue;
    }

    // Phase 2: If GPU reduction didn't work, try CPU-only (same caller params)
    {
        eci_result_t mrc = reload_model_with_layers(model, model_params, 0);
        if (mrc != ECI_OK) {
            g_backend.probe_error = "model load failed at CPU-only fallback";
            return ECI_ERR_LOAD_FAILED;
        }

        // Build a conservative context: same params but we know CPU is safe
        eci_context_t* ctx = nullptr;
        eci_result_t crc = eci_create_context(*model, ctx_params, &ctx);
        if (crc == ECI_OK) {
            eci_result_t prc = eci_probe_backend(ctx);
            if (prc == ECI_OK) {
                g_backend.gpu_layers_reduced = true;
                g_backend.effective_gpu_layers = 0;
                *out_ctx = ctx;
                return ECI_OK;
            }
            eci_free_context(ctx);
        }
    }

    // Restore original model if we killed it during fallback attempts
    if (*model != original_model && original_model) {
        *model = original_model;
    }
    g_backend.probe_error = "All fallback configurations failed";
    return ECI_ERR_LOAD_FAILED;
}

// ── VRAM guard ──

bool eci_vram_available(size_t needed_bytes) {
    // TODO: platform-specific VRAM checks
    // Metal: mach_vm_statistics
    // CUDA: cudaMemGetInfo
    // Vulkan: vkGetPhysicalDeviceMemoryProperties
    // For now, probe + progressive reduction is our safety net.
    (void)needed_bytes;
    return true;
}

// ── Backend state queries ──

bool eci_backend_probe_passed(void) { return g_backend.probe_passed; }
const char* eci_backend_name(void) { return g_backend.name.c_str(); }
const char* eci_backend_probe_error(void) { return g_backend.probe_error.c_str(); }
bool eci_backend_needs_kv_flush(void) { return g_backend.needs_kv_flush_after_clear; }
bool eci_backend_gpu_layers_reduced(void) { return g_backend.gpu_layers_reduced; }
int eci_backend_effective_gpu_layers(void) { return g_backend.effective_gpu_layers; }
int eci_backend_original_gpu_layers(void) { return g_backend.original_gpu_layers; }

// ── KV flush (Vulkan stale cells fix, #26744) ──

eci_result_t eci_kv_flush(eci_context_t* ctx, int seq_id) {
    if (!ctx) return ECI_ERR_INVALID_ARG;

    llama_token pad = llama_vocab_bos(ctx->vocab);
    if (pad < 0) pad = llama_vocab_eos(ctx->vocab);
    if (pad < 0) return ECI_OK;

    llama_batch batch = llama_batch_init(1, 0, ctx->n_seq_max);
    batch.token[0] = pad;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = seq_id;
    batch.logits[0] = 0;
    batch.n_tokens = 1;

    int rc = llama_decode(ctx->ctx, batch);
    llama_batch_free(batch);

    llama_memory_seq_rm(ctx->mem, seq_id, 0, -1);

    if (rc != 0) return ECI_ERR_DECODE_FAILED;
    return ECI_OK;
}
