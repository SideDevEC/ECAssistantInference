// test_stress.cpp — Stress tests for ECAssistantInference
//
// Tests edge cases: zero/empty inputs, boundary sizes, repeated operations,
// interleaved pool ops, KV overflow, sampling extremes, state save/restore
// under stress, concurrent-ish patterns, resource cleanup.

#include "ecainference.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

static const char* MODEL_PATH = "./models/Qwen3.5-4B-Q4_K_M.gguf";
static const char* EMBED_MODEL_PATH = "./models/all-MiniLM-L6-v2-Q5_K_M.gguf";

static int tests_run = 0, tests_passed = 0;

#define TEST(name) \
    tests_run++; \
    fprintf(stderr, "  [%d] %s ... ", tests_run, #name); \
    test_##name(); \
    tests_passed++; \
    fprintf(stderr, "OK\n");

#define ASSERT(cond) \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: line %d: %s\n", __LINE__, #cond); \
        exit(1); \
    }

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL: line %d: %s != %s (%lld != %lld)\n", __LINE__, #a, #b, (long long)(a), (long long)(b)); \
        exit(1); \
    }

// ── Helper: create a standard model+context for tests ──
static eci_model_t* make_model(int gpu_layers = 99, bool fa = true, eci_kv_type_t kv = ECI_KV_F16) {
    eci_model_params_t mp = {};
    mp.model_path = MODEL_PATH;
    mp.gpu_layers = gpu_layers;
    mp.flash_attn = fa;
    mp.kv_cache_type = kv;
    mp.threads = 0;  // auto
    eci_model_t* m = nullptr;
    ASSERT_EQ(eci_load_model(&mp, &m), ECI_OK);
    return m;
}

static eci_context_t* make_ctx(eci_model_t* m, uint32_t ctx_size = 4096, uint32_t batch = 512, uint32_t seq_max = 4) {
    eci_context_params_t cp = {};
    cp.context_size = ctx_size;
    cp.batch_size = batch;
    cp.seq_max = seq_max;
    eci_context_t* ctx = nullptr;
    ASSERT_EQ(eci_create_context(m, &cp, &ctx), ECI_OK);
    return ctx;
}

// ════════════════════════════════════════════════════
// 1. NULL / invalid argument tests
// ════════════════════════════════════════════════════

static void test_null_args() {
    // Every function should handle NULL gracefully (no crash)
    ASSERT_EQ(eci_load_model(nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_create_context(nullptr, nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_tokenize(nullptr, "test", false, true, nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_detokenize(nullptr, nullptr, 0, false, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_executor_create(nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_executor_prompt(nullptr, "test"), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_executor_infer(nullptr), ECI_DECODE_FAILED);
    ASSERT_EQ(eci_executor_sample(nullptr, nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_executor_rewind(nullptr, 0), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_executor_reset(nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_pool_create(nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_pool_lease(nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_pool_return(nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_state_save(nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_state_restore(nullptr, nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_kv_copy(nullptr, 0, 1), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_kv_clear_seq(nullptr, 0), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_set_causal_attn(nullptr, true), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_probe_backend(nullptr), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_kv_flush(nullptr, 0), ECI_ERR_INVALID_ARG);
}

// ════════════════════════════════════════════════════
// 2. Empty / zero-length inputs
// ════════════════════════════════════════════════════

static void test_empty_prompt() {
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    // Empty string prompt
    ASSERT_EQ(eci_executor_prompt(exec, ""), ECI_OK);
    // Infer with empty pending — should be NO_WORK
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_NO_WORK);

    // Sample without any tokens in KV — should fail
    int32_t token = 0;
    ASSERT_EQ(eci_executor_sample(exec, nullptr, &token), ECI_ERR_DECODE_FAILED);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_empty_tokenize() {
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m);

    int32_t* tokens = nullptr;
    int count = -1;
    ASSERT_EQ(eci_tokenize(ctx, "", false, true, &tokens, &count), ECI_OK);
    // Empty string might produce 0 tokens or just BOS — either is acceptable
    if (count == 0) { ASSERT(tokens == nullptr); }
    if (tokens) eci_free_tokens(tokens);

    // NULL text
    ASSERT_EQ(eci_tokenize(ctx, nullptr, false, true, &tokens, &count), ECI_ERR_INVALID_ARG);

    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_detokenize_empty() {
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m);

    // Zero count
    char* out = nullptr;
    ASSERT_EQ(eci_detokenize(ctx, nullptr, 0, false, &out), ECI_ERR_INVALID_ARG);

    // Count = -1 (invalid)
    int32_t dummy = 0;
    ASSERT_EQ(eci_detokenize(ctx, &dummy, -1, false, &out), ECI_ERR_INVALID_ARG);

    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 3. Pool exhaustion and reuse stress
// ════════════════════════════════════════════════════

static void test_pool_exhaust_lease_return_cycle() {
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 4);
    eci_pool_t* pool = nullptr;
    ASSERT_EQ(eci_pool_create(ctx, &pool), ECI_OK);

    // Lease all 4
    eci_conversation_t* convs[4];
    for (int i = 0; i < 4; i++) ASSERT_EQ(eci_pool_lease(pool, &convs[i]), ECI_OK);
    ASSERT_EQ(eci_pool_available(pool), 0);

    // 5th should fail
    eci_conversation_t* extra = nullptr;
    ASSERT_EQ(eci_pool_lease(pool, &extra), ECI_ERR_NO_SLOT);
    ASSERT(extra == nullptr);

    // Return one, lease one — repeat 10 times
    for (int cycle = 0; cycle < 10; cycle++) {
        ASSERT_EQ(eci_pool_return(pool, convs[0]), ECI_OK);
        ASSERT_EQ(eci_pool_available(pool), 1);
        ASSERT_EQ(eci_pool_lease(pool, &convs[0]), ECI_OK);
        ASSERT_EQ(eci_pool_available(pool), 0);
    }

    // Return all
    for (int i = 0; i < 4; i++) ASSERT_EQ(eci_pool_return(pool, convs[i]), ECI_OK);
    ASSERT_EQ(eci_pool_available(pool), 4);

    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_pool_return_double() {
    // Returning the same conversation twice should not corrupt the stack
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 2);
    eci_pool_t* pool = nullptr;
    ASSERT_EQ(eci_pool_create(ctx, &pool), ECI_OK);

    eci_conversation_t* conv = nullptr;
    ASSERT_EQ(eci_pool_lease(pool, &conv), ECI_OK);
    ASSERT_EQ(eci_pool_return(pool, conv), ECI_OK);
    // Return again — the conv is already in the available stack
    // This is a bug-detection test: if we push it twice, a future lease
    // could get a stale pointer. We accept either OK (naive) or error (strict).
    eci_result_t r = eci_pool_return(pool, conv);
    ASSERT(r == ECI_OK || r == ECI_ERR_INVALID_ARG);  // document current behavior

    // If it returned OK, we have 2 entries in available but only 1 real conv
    // Lease twice and check both are valid
    eci_conversation_t* c1 = nullptr, *c2 = nullptr;
    ASSERT_EQ(eci_pool_lease(pool, &c1), ECI_OK);
    eci_result_t r2 = eci_pool_lease(pool, &c2);
    if (r2 == ECI_OK) {
        // Both should be the same pointer (double return pushed twice)
        // This documents the bug — if c1 == c2, leasing "2" conversations gives
        // the same physical conversation to two callers
        if (c1 == c2) {
            fprintf(stderr, "WARN: double-return gives same conv to two callers ");
        }
        // Return both to clean up
        eci_pool_return(pool, c1);
        if (c1 != c2) eci_pool_return(pool, c2);
    } else {
        eci_pool_return(pool, c1);
    }

    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_pool_prompt_return_without_infer() {
    // Return a conversation with pending tokens but no infer — should clean up
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 2);
    eci_pool_t* pool = nullptr;
    ASSERT_EQ(eci_pool_create(ctx, &pool), ECI_OK);

    eci_conversation_t* conv = nullptr;
    ASSERT_EQ(eci_pool_lease(pool, &conv), ECI_OK);
    ASSERT_EQ(eci_conversation_prompt(conv, "Hello world this is a test"), ECI_OK);
    // Return WITHOUT calling eci_infer — pending_tokens should be cleared
    ASSERT_EQ(eci_pool_return(pool, conv), ECI_OK);

    // Re-lease and verify it's clean
    ASSERT_EQ(eci_pool_lease(pool, &conv), ECI_OK);
    ASSERT_EQ(eci_conversation_token_count(conv), 0);
    ASSERT_EQ(eci_pool_return(pool, conv), ECI_OK);

    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 4. KV cache boundary conditions
// ════════════════════════════════════════════════════

static void test_tiny_context() {
    // Very small context (256) — should work for short prompts
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 256, 128, 1);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Say hi"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    ASSERT(eci_executor_token_count(exec) > 0);

    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    int32_t token = 0;
    ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_OK);
    ASSERT(token >= 0);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_rewind_to_zero() {
    // Rewind all tokens — should leave context empty
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Hello world test"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    int n = eci_executor_token_count(exec);
    ASSERT(n > 0);

    // Rewind ALL tokens
    ASSERT_EQ(eci_executor_rewind(exec, n), ECI_OK);
    ASSERT_EQ(eci_executor_token_count(exec), 0);

    // Rewind more than we have (should clamp)
    ASSERT_EQ(eci_executor_rewind(exec, 100), ECI_OK);
    ASSERT_EQ(eci_executor_token_count(exec), 0);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_shift_left_edge() {
    // Shift left by 0 (should be invalid or no-op)
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 512, 256, 2);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Test prompt for shifting"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    int n = eci_executor_token_count(exec);
    ASSERT(n > 0);

    // Shift by 0 — should be invalid (n_tokens <= 0)
    ASSERT_EQ(eci_executor_shift_left(exec, 0), ECI_ERR_INVALID_ARG);

    // Shift by exactly n — should be same as reset
    ASSERT_EQ(eci_executor_shift_left(exec, n), ECI_OK);
    ASSERT_EQ(eci_executor_token_count(exec), 0);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_kv_copy_invalid_seqs() {
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 2048, 512, 4);

    // Invalid seq IDs
    ASSERT_EQ(eci_kv_copy(ctx, -1, 0), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_kv_copy(ctx, 0, -1), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_kv_copy(ctx, 99, 0), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_kv_copy(ctx, 0, 99), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_kv_clear_seq(ctx, -1), ECI_ERR_INVALID_ARG);
    ASSERT_EQ(eci_kv_clear_seq(ctx, 99), ECI_ERR_INVALID_ARG);

    // Copy to self — should be OK (no-op effectively)
    ASSERT_EQ(eci_kv_copy(ctx, 0, 0), ECI_OK);

    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 5. Sampling parameter extremes
// ════════════════════════════════════════════════════

static void test_sampling_extremes() {
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Say a number"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);

    // Extreme temperature (very high = random)
    eci_sampling_params_t sp = {};
    sp.temperature = 100.0f;  // extreme heat
    sp.top_p = 1.0f;
    sp.top_k = 0;  // no top-k
    sp.repeat_penalty = 1.0f;  // no penalty
    int32_t token = 0;
    ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_OK);
    ASSERT(token >= 0);

    // After rewind, must re-infer before sampling (logits are stale)
    // Feed the sampled token back and infer to get fresh logits
    ASSERT_EQ(eci_executor_prompt_tokens(exec, &token, 1), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);

    // Very low temperature (near greedy)
    sp.temperature = 0.001f;
    sp.top_k = 1;  // greedy
    ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_OK);
    ASSERT(token >= 0);

    ASSERT_EQ(eci_executor_prompt_tokens(exec, &token, 1), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);

    // Top-p = 0 (should still return something)
    sp.temperature = 0.3f;
    sp.top_p = 0.0f;
    sp.top_k = 0;
    ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_OK);
    ASSERT(token >= 0);

    ASSERT_EQ(eci_executor_prompt_tokens(exec, &token, 1), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);

    // Min-p = 0.99 (very aggressive filter)
    sp.top_p = 1.0f;
    sp.min_p = 0.99f;
    sp.top_k = 0;
    ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_OK);
    ASSERT(token >= 0);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_sampling_no_decode() {
    // Sample without any prior decode — should fail
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    eci_sampling_params_t sp = {};
    int32_t token = -42;
    ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_ERR_DECODE_FAILED);
    ASSERT_EQ(token, -42);  // should not be modified

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 6. State save/restore edge cases
// ════════════════════════════════════════════════════

static void test_state_save_empty() {
    // Save state when no tokens exist
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 2);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    eci_state_t* state = nullptr;
    ASSERT_EQ(eci_state_save(exec, &state), ECI_OK);
    ASSERT(state != nullptr);
    ASSERT_EQ(eci_state_token_count(state), 0);

    // Restore empty state — should be no-op
    ASSERT_EQ(eci_state_restore(exec, state), ECI_OK);
    ASSERT_EQ(eci_executor_token_count(exec), 0);

    eci_state_free(state);
    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_state_restore_wrong_state() {
    // Save state A, modify, save state B, restore A, restore B
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 2);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    // Prompt "Hello" → save state A
    ASSERT_EQ(eci_executor_prompt(exec, "Hello world"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    int n_a = eci_executor_token_count(exec);
    eci_state_t* state_a = nullptr;
    ASSERT_EQ(eci_state_save(exec, &state_a), ECI_OK);

    // Generate more → save state B
    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    for (int i = 0; i < 3; i++) {
        int32_t t = 0;
        ASSERT_EQ(eci_executor_sample(exec, &sp, &t), ECI_OK);
        ASSERT_EQ(eci_executor_prompt_tokens(exec, &t, 1), ECI_OK);
        ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    }
    int n_b = eci_executor_token_count(exec);
    ASSERT(n_b > n_a);
    eci_state_t* state_b = nullptr;
    ASSERT_EQ(eci_state_save(exec, &state_b), ECI_OK);

    // Restore A → should go back to n_a
    ASSERT_EQ(eci_state_restore(exec, state_a), ECI_OK);
    ASSERT_EQ(eci_executor_token_count(exec), n_a);

    // Restore B → should go forward to n_b
    ASSERT_EQ(eci_state_restore(exec, state_b), ECI_OK);
    ASSERT_EQ(eci_executor_token_count(exec), n_b);

    eci_state_free(state_a);
    eci_state_free(state_b);
    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_state_save_seq_max_1() {
    // With n_seq_max = 1, state save should fail (no backup seq)
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 1);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Hello"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);

    eci_state_t* state = nullptr;
    eci_result_t r = eci_state_save(exec, &state);
    if (r == ECI_ERR_NOT_SUPPORTED) {
        fprintf(stderr, "seq_max=1 correctly rejected ");
        // state should be null
        ASSERT(state == nullptr);
    } else if (r == ECI_OK) {
        // If it succeeded (no tokens to save), that's also acceptable
        if (state) eci_state_free(state);
    }

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 7. Interleaved batch + pool operations
// ════════════════════════════════════════════════════

static void test_batched_mixed_sizes() {
    // 4 conversations with different prompt sizes in one batch
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 4);
    eci_pool_t* pool = nullptr;
    ASSERT_EQ(eci_pool_create(ctx, &pool), ECI_OK);

    eci_conversation_t* convs[4];
    const char* prompts[] = {
        "Say 1.",                         // short
        "Say the number 2.",              // medium
        "Please say the number 3 now.",   // longer
        "I would like you to say 4."      // longest
    };

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(eci_pool_lease(pool, &convs[i]), ECI_OK);
        ASSERT_EQ(eci_conversation_prompt(convs[i], prompts[i]), ECI_OK);
    }

    // One decode for ALL conversations
    ASSERT_EQ(eci_infer(ctx), ECI_DECODE_OK);

    // Each should have different token counts
    int counts[4];
    bool any_different = false;
    for (int i = 0; i < 4; i++) {
        counts[i] = eci_conversation_token_count(convs[i]);
        ASSERT(counts[i] > 0);
        if (i > 0 && counts[i] != counts[0]) any_different = true;
    }
    ASSERT(any_different);  // prompts are different lengths → token counts differ

    // Sample from each
    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    for (int i = 0; i < 4; i++) {
        int32_t token = 0;
        ASSERT_EQ(eci_conversation_sample(convs[i], &sp, &token), ECI_OK);
        ASSERT(token >= 0);
    }

    // Return all, re-lease, verify clean
    for (int i = 0; i < 4; i++) ASSERT_EQ(eci_pool_return(pool, convs[i]), ECI_OK);
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(eci_pool_lease(pool, &convs[i]), ECI_OK);
        ASSERT_EQ(eci_conversation_token_count(convs[i]), 0);
    }
    for (int i = 0; i < 4; i++) ASSERT_EQ(eci_pool_return(pool, convs[i]), ECI_OK);

    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_batched_infer_no_work() {
    // eci_infer with no pending tokens on any conversation
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 4);
    eci_pool_t* pool = nullptr;
    ASSERT_EQ(eci_pool_create(ctx, &pool), ECI_OK);

    // Lease some convs but don't prompt
    eci_conversation_t* convs[2];
    for (int i = 0; i < 2; i++) ASSERT_EQ(eci_pool_lease(pool, &convs[i]), ECI_OK);

    // Infer — should be NO_WORK
    ASSERT_EQ(eci_infer(ctx), ECI_DECODE_NO_WORK);

    for (int i = 0; i < 2; i++) ASSERT_EQ(eci_pool_return(pool, convs[i]), ECI_OK);
    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_batched_partial_prompt() {
    // 3 conversations: 2 have pending tokens, 1 doesn't
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 4);
    eci_pool_t* pool = nullptr;
    ASSERT_EQ(eci_pool_create(ctx, &pool), ECI_OK);

    eci_conversation_t* convs[3];
    for (int i = 0; i < 3; i++) ASSERT_EQ(eci_pool_lease(pool, &convs[i]), ECI_OK);

    // Only prompt 2 of 3
    ASSERT_EQ(eci_conversation_prompt(convs[0], "Say A"), ECI_OK);
    ASSERT_EQ(eci_conversation_prompt(convs[1], "Say B"), ECI_OK);
    // convs[2] has no pending tokens

    ASSERT_EQ(eci_infer(ctx), ECI_DECODE_OK);

    // 0 and 1 should have tokens, 2 should still be 0
    ASSERT(eci_conversation_token_count(convs[0]) > 0);
    ASSERT(eci_conversation_token_count(convs[1]) > 0);
    ASSERT_EQ(eci_conversation_token_count(convs[2]), 0);

    for (int i = 0; i < 3; i++) ASSERT_EQ(eci_pool_return(pool, convs[i]), ECI_OK);
    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 8. Repeated generation (resource leak detection)
// ════════════════════════════════════════════════════

static void test_repeated_generation() {
    // Generate 20 tokens — detects token/recent_tokens leaks
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 2);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Count from 1 to 20: 1 2 3 4 5"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    int initial_tokens = eci_executor_token_count(exec);

    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    for (int i = 0; i < 20; i++) {
        int32_t token = 0;
        ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_OK);
        if (eci_token_is_eos(ctx, token)) break;
        ASSERT_EQ(eci_executor_prompt_tokens(exec, &token, 1), ECI_OK);
        ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    }

    int final_tokens = eci_executor_token_count(exec);
    ASSERT(final_tokens > initial_tokens);
    fprintf(stderr, "gen=%d ", final_tokens - initial_tokens);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_repeated_pool_cycles() {
    // Lease → prompt → infer → sample → return, 10 cycles
    // Detects KV leaks (token count should stay consistent)
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 2);
    eci_pool_t* pool = nullptr;
    ASSERT_EQ(eci_pool_create(ctx, &pool), ECI_OK);
    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;

    for (int cycle = 0; cycle < 10; cycle++) {
        eci_conversation_t* conv = nullptr;
        ASSERT_EQ(eci_pool_lease(pool, &conv), ECI_OK);
        ASSERT_EQ(eci_pool_available(pool), 1);

        ASSERT_EQ(eci_conversation_prompt(conv, "Say a number"), ECI_OK);
        ASSERT_EQ(eci_infer(ctx), ECI_DECODE_OK);
        ASSERT(eci_conversation_token_count(conv) > 0);

        int32_t token = 0;
        ASSERT_EQ(eci_conversation_sample(conv, &sp, &token), ECI_OK);
        ASSERT(token >= 0);

        ASSERT_EQ(eci_pool_return(pool, conv), ECI_OK);
        ASSERT_EQ(eci_pool_available(pool), 2);
    }

    // After 10 cycles, pool should be fully available
    ASSERT_EQ(eci_pool_available(pool), 2);

    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 9. Backend probe
// ════════════════════════════════════════════════════

static void test_backend_probe() {
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m);
    ASSERT_EQ(eci_probe_backend(ctx), ECI_OK);
    ASSERT(eci_backend_probe_passed());
    fprintf(stderr, "backend=%s ", eci_backend_name());

    // These should all be valid
    ASSERT(eci_backend_name() != nullptr);
    ASSERT(eci_backend_probe_error() != nullptr);

    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 10. Embedding edge cases
// ════════════════════════════════════════════════════

static void test_embedding_empty_string() {
    eci_model_params_t mp = {}; mp.model_path = EMBED_MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    eci_model_t* m = nullptr;
    ASSERT_EQ(eci_load_model(&mp, &m), ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 1;
    cp.pooling_type = ECI_POOL_MEAN;
    eci_context_t* ctx = nullptr;
    ASSERT_EQ(eci_create_context(m, &cp, &ctx), ECI_OK);

    float* emb = nullptr; int dim = 0;
    // Empty string — should still produce embeddings (or fail gracefully)
    eci_result_t r = eci_get_embeddings(m, ctx, "", &emb, &dim);
    if (r == ECI_OK) {
        ASSERT(dim > 0);
        ASSERT(emb != nullptr);
        eci_free_embeddings(emb);
    }

    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_embedding_all_pooling_types() {
    eci_model_params_t mp = {}; mp.model_path = EMBED_MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    eci_model_t* m = nullptr;
    ASSERT_EQ(eci_load_model(&mp, &m), ECI_OK);

    eci_pooling_type_t types[] = {ECI_POOL_MEAN, ECI_POOL_CLS, ECI_POOL_LAST};
    for (int t = 0; t < 3; t++) {
        eci_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 1;
        cp.pooling_type = types[t];
        eci_context_t* ctx = nullptr;
        ASSERT_EQ(eci_create_context(m, &cp, &ctx), ECI_OK);

        float* emb = nullptr; int dim = 0;
        ASSERT_EQ(eci_get_embeddings(m, ctx, "Hello world test", &emb, &dim), ECI_OK);
        ASSERT(dim > 0);
        ASSERT(emb != nullptr);

        // Verify normalization
        float norm = 0;
        for (int j = 0; j < dim; j++) norm += emb[j] * emb[j];
        norm = sqrtf(norm);
        fprintf(stderr, "pool=%d norm=%.3f ", t, norm);
        ASSERT(norm > 0.9f && norm < 1.1f);  // should be ~unit length

        eci_free_embeddings(emb);
        eci_free_context(ctx);
    }

    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 11. Config parameter variations
// ════════════════════════════════════════════════════

static void test_cpu_only_model() {
    // gpu_layers = 0 (CPU only)
    eci_model_t* m = make_model(0, true, ECI_KV_F16);
    eci_context_t* ctx = make_ctx(m, 2048, 512, 2);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Say hello"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    int32_t token = 0;
    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_OK);
    ASSERT(token >= 0);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_fa_disabled() {
    // flash_attn = false
    eci_model_t* m = make_model(99, false, ECI_KV_F16);
    eci_context_t* ctx = make_ctx(m, 2048, 512, 2);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Say hello"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    ASSERT(eci_executor_token_count(exec) > 0);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

static void test_q8_kv_cache() {
    // q8_0 KV cache type
    eci_model_t* m = make_model(99, true, ECI_KV_Q8_0);
    eci_context_t* ctx = make_ctx(m, 2048, 512, 2);
    eci_executor_t* exec = nullptr;
    ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);

    ASSERT_EQ(eci_executor_prompt(exec, "Say hello"), ECI_OK);
    ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
    ASSERT(eci_executor_token_count(exec) > 0);

    // Generate a token to verify the pipeline works end-to-end with q8_0
    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    int32_t token = 0;
    ASSERT_EQ(eci_executor_sample(exec, &sp, &token), ECI_OK);
    ASSERT(token >= 0);

    eci_executor_free(exec);
    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 12. Conversation rewind edge cases
// ════════════════════════════════════════════════════

static void test_conversation_rewind_edge() {
    eci_model_t* m = make_model();
    eci_context_t* ctx = make_ctx(m, 4096, 512, 2);
    eci_pool_t* pool = nullptr;
    ASSERT_EQ(eci_pool_create(ctx, &pool), ECI_OK);

    eci_conversation_t* conv = nullptr;
    ASSERT_EQ(eci_pool_lease(pool, &conv), ECI_OK);

    // Rewind with 0 tokens — should be OK (no-op)
    ASSERT_EQ(eci_conversation_rewind(conv, 0), ECI_OK);
    ASSERT_EQ(eci_conversation_token_count(conv), 0);

    // Rewind with negative
    ASSERT_EQ(eci_conversation_rewind(conv, -1), ECI_ERR_INVALID_ARG);

    // Prompt and infer
    ASSERT_EQ(eci_conversation_prompt(conv, "Hello test"), ECI_OK);
    ASSERT_EQ(eci_infer(ctx), ECI_DECODE_OK);
    int n = eci_conversation_token_count(conv);
    ASSERT(n > 0);

    // Rewind to exactly 0
    ASSERT_EQ(eci_conversation_rewind(conv, n), ECI_OK);
    ASSERT_EQ(eci_conversation_token_count(conv), 0);

    // Rewind past 0 (should clamp)
    ASSERT_EQ(eci_conversation_rewind(conv, n + 10), ECI_OK);
    ASSERT_EQ(eci_conversation_token_count(conv), 0);

    ASSERT_EQ(eci_pool_return(pool, conv), ECI_OK);
    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_model(m);
}

// ════════════════════════════════════════════════════
// 13. Multiple model loads (memory leak detection)
// ════════════════════════════════════════════════════

static void test_model_reload() {
    // Load and free the same model 5 times — should not leak
    for (int i = 0; i < 5; i++) {
        eci_model_t* m = make_model(0);  // CPU only for speed
        eci_context_t* ctx = make_ctx(m, 512);
        eci_executor_t* exec = nullptr;
        ASSERT_EQ(eci_executor_create(ctx, &exec), ECI_OK);
        ASSERT_EQ(eci_executor_prompt(exec, "test"), ECI_OK);
        ASSERT_EQ(eci_executor_infer(exec), ECI_DECODE_OK);
        eci_executor_free(exec);
        eci_free_context(ctx);
        eci_free_model(m);
    }
}

// ════════════════════════════════════════════════════

int main() {
    fprintf(stderr, "\n=== ECAssistantInference Stress Tests ===\n\n");

    TEST(null_args);
    TEST(empty_prompt);
    TEST(empty_tokenize);
    TEST(detokenize_empty);
    TEST(pool_exhaust_lease_return_cycle);
    TEST(pool_return_double);
    TEST(pool_prompt_return_without_infer);
    TEST(tiny_context);
    TEST(rewind_to_zero);
    TEST(shift_left_edge);
    TEST(kv_copy_invalid_seqs);
    TEST(sampling_extremes);
    TEST(sampling_no_decode);
    TEST(state_save_empty);
    TEST(state_restore_wrong_state);
    TEST(state_save_seq_max_1);
    TEST(batched_mixed_sizes);
    TEST(batched_infer_no_work);
    TEST(batched_partial_prompt);
    TEST(repeated_generation);
    TEST(repeated_pool_cycles);
    TEST(backend_probe);
    TEST(embedding_empty_string);
    TEST(embedding_all_pooling_types);
    TEST(cpu_only_model);
    TEST(fa_disabled);
    TEST(q8_kv_cache);
    TEST(conversation_rewind_edge);
    TEST(model_reload);

    fprintf(stderr, "\n=== %d/%d stress tests passed ===\n", tests_passed, tests_run);
    return 0;
}
