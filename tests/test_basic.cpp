// test_basic.cpp — Full test suite for ECAssistantInference C API

#include "ecainference.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>

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

// ── Tests ──

static void test_load_model() {
    eci_model_params_t p = {}; p.model_path = MODEL_PATH; p.gpu_layers = 99; p.flash_attn = true;
    eci_model_t* m = nullptr;
    ASSERT(eci_load_model(&p, &m) == ECI_OK);
    eci_free_model(m);
}

static void test_create_context() {
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 4;
    eci_context_t* ctx = nullptr;
    ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);
    ASSERT(eci_context_size(ctx) == 4096);
    ASSERT(eci_context_seq_max(ctx) == 4);
    eci_free_context(ctx); eci_free_model(m);
}

static void test_tokenize() {
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 2;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    int32_t* tokens = nullptr; int count = 0;
    ASSERT(eci_tokenize(ctx, "Hello world", false, true, &tokens, &count) == ECI_OK);
    ASSERT(count > 0);

    char* text = nullptr;
    ASSERT(eci_detokenize(ctx, tokens, count, false, &text) == ECI_OK);
    ASSERT(strstr(text, "Hello") != nullptr);
    eci_free_string(text); eci_free_tokens(tokens);
    eci_free_context(ctx); eci_free_model(m);
}

static void test_pool() {
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 4;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    eci_pool_t* pool = nullptr; ASSERT(eci_pool_create(ctx, &pool) == ECI_OK);
    ASSERT(eci_pool_size(pool) == 4);
    ASSERT(eci_pool_available(pool) == 4);

    eci_conversation_t* convs[4];
    for (int i = 0; i < 4; i++) ASSERT(eci_pool_lease(pool, &convs[i]) == ECI_OK);
    ASSERT(eci_pool_available(pool) == 0);

    eci_conversation_t* extra = nullptr;
    ASSERT(eci_pool_lease(pool, &extra) == ECI_ERR_NO_SLOT);

    for (int i = 0; i < 4; i++) ASSERT(eci_pool_return(pool, convs[i]) == ECI_OK);
    ASSERT(eci_pool_available(pool) == 4);

    eci_pool_free(pool); eci_free_context(ctx); eci_free_model(m);
}

static void test_single_inference() {
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 2;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    eci_executor_t* exec = nullptr;
    ASSERT(eci_executor_create(ctx, &exec) == ECI_OK);

    ASSERT(eci_executor_prompt(exec, "Say the number 7 and nothing else.") == ECI_OK);
    ASSERT(eci_executor_infer(exec) == ECI_DECODE_OK);
    ASSERT(eci_executor_token_count(exec) > 0);

    eci_sampling_params_t sp = {};
    sp.temperature = 0.3f; sp.top_p = 0.95f; sp.top_k = 40;
    sp.repeat_penalty = 1.1f; sp.repeat_last_n = -1; sp.max_tokens = 32;

    std::string output;
    for (int i = 0; i < 32; i++) {
        int32_t token = 0;
        ASSERT(eci_executor_sample(exec, &sp, &token) == ECI_OK);
        if (eci_token_is_eos(ctx, token)) break;

        char buf[256];
        if (eci_token_to_piece(ctx, token, buf, sizeof(buf)) == ECI_OK) output += buf;

        ASSERT(eci_executor_prompt_tokens(exec, &token, 1) == ECI_OK);
        ASSERT(eci_executor_infer(exec) == ECI_DECODE_OK);
    }
    fprintf(stderr, "out=\"%s\" ", output.c_str());
    ASSERT(!output.empty());

    eci_executor_free(exec); eci_free_context(ctx); eci_free_model(m);
}

static void test_batched_inference() {
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 4;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    eci_pool_t* pool = nullptr; ASSERT(eci_pool_create(ctx, &pool) == ECI_OK);
    eci_conversation_t* convs[4];
    const char* prompts[] = {"Say 1.","Say 2.","Say 3.","Say 4."};

    for (int i = 0; i < 4; i++) {
        ASSERT(eci_pool_lease(pool, &convs[i]) == ECI_OK);
        ASSERT(eci_conversation_prompt(convs[i], prompts[i]) == ECI_OK);
    }
    ASSERT(eci_infer(ctx) == ECI_DECODE_OK);

    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_p = 0.95f; sp.top_k = 40;
    for (int i = 0; i < 4; i++) {
        int32_t token = 0;
        ASSERT(eci_conversation_sample(convs[i], &sp, &token) == ECI_OK);
    }
    for (int i = 0; i < 4; i++) ASSERT(eci_pool_return(pool, convs[i]) == ECI_OK);

    eci_pool_free(pool); eci_free_context(ctx); eci_free_model(m);
}

static void test_state_save_restore() {
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 2;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    eci_executor_t* exec = nullptr; ASSERT(eci_executor_create(ctx, &exec) == ECI_OK);

    ASSERT(eci_executor_prompt(exec, "Hello") == ECI_OK);
    ASSERT(eci_executor_infer(exec) == ECI_DECODE_OK);
    int tokens_after_prompt = eci_executor_token_count(exec);
    ASSERT(tokens_after_prompt > 0);

    eci_state_t* state = nullptr;
    ASSERT(eci_state_save(exec, &state) == ECI_OK);

    // Generate a few more tokens
    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40;
    for (int i = 0; i < 5; i++) {
        int32_t token = 0;
        ASSERT(eci_executor_sample(exec, &sp, &token) == ECI_OK);
        ASSERT(eci_executor_prompt_tokens(exec, &token, 1) == ECI_OK);
        ASSERT(eci_executor_infer(exec) == ECI_DECODE_OK);
    }
    ASSERT(eci_executor_token_count(exec) > tokens_after_prompt);

    // Restore
    ASSERT(eci_state_restore(exec, state) == ECI_OK);
    ASSERT(eci_executor_token_count(exec) == tokens_after_prompt);

    eci_state_free(state);
    eci_executor_free(exec); eci_free_context(ctx); eci_free_model(m);
}

static void test_embeddings() {
    eci_model_params_t mp = {}; mp.model_path = EMBED_MODEL_PATH; mp.gpu_layers = 0; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 1;
    cp.pooling_type = ECI_POOL_MEAN;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    float* emb = nullptr; int dim = 0;
    ASSERT(eci_get_embeddings(m, ctx, "Hello world", &emb, &dim) == ECI_OK);
    ASSERT(dim > 0);
    // Verify embeddings are non-zero (not all-null from failed compute)
    float norm = 0; for (int i = 0; i < dim; i++) norm += emb[i]*emb[i];
    norm = sqrtf(norm);
    fprintf(stderr, "dim=%d norm=%.3f ", dim, norm);
    ASSERT(norm > 0.5f);
    eci_free_embeddings(emb);
    eci_free_context(ctx); eci_free_model(m);
}

static void test_anti_prompts() {
    const char* text = "Hello world\n### User: What is 2+2?";
    const char* anti[] = {"### User:", "### Assistant:"};
    int idx = eci_check_anti_prompts(text, anti, 2);
    ASSERT(idx == 0);

    const char* text2 = "Just normal text";
    int idx2 = eci_check_anti_prompts(text2, anti, 2);
    ASSERT(idx2 == -1);
}

static void test_pool_reuse() {
    // Verify pool reuse works — lease, prompt, return, re-lease gets same conv back clean
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 4096; cp.batch_size = 512; cp.seq_max = 2;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    eci_pool_t* pool = nullptr; ASSERT(eci_pool_create(ctx, &pool) == ECI_OK);
    eci_conversation_t* conv1 = nullptr;
    ASSERT(eci_pool_lease(pool, &conv1) == ECI_OK);
    ASSERT(eci_conversation_prompt(conv1, "Hello world") == ECI_OK);
    ASSERT(eci_infer(ctx) == ECI_DECODE_OK);
    ASSERT(eci_conversation_token_count(conv1) > 0);

    // Return — should clear KV
    ASSERT(eci_pool_return(pool, conv1) == ECI_OK);
    ASSERT(eci_pool_available(pool) == 2);

    // Re-lease — should get a clean conversation
    eci_conversation_t* conv2 = nullptr;
    ASSERT(eci_pool_lease(pool, &conv2) == ECI_OK);
    ASSERT(eci_conversation_token_count(conv2) == 0);

    eci_pool_return(pool, conv2);
    eci_pool_free(pool); eci_free_context(ctx); eci_free_model(m);
}

static void test_vision() {
    // Test mmproj loading + image prompt with embeddings
    eci_model_params_t mp = {};
    mp.model_path = MODEL_PATH;
    mp.gpu_layers = 99;
    mp.flash_attn = true;

    eci_model_t* model = nullptr;
    ASSERT(eci_load_model(&mp, &model) == ECI_OK);

    // Load mmproj
    eci_model_t* mmproj = nullptr;
    const char* mmproj_path = "./models/mmproj-Qwen3.5-4B-BF16.gguf";
    ASSERT(eci_load_mmproj(model, mmproj_path, &mmproj) == ECI_OK);
    ASSERT(mmproj != nullptr);

    // Get mtmd marker
    const char* marker = eci_mtmd_marker_static();
    ASSERT(marker != nullptr);
    fprintf(stderr, "marker=\"%s\" ", marker);

    // Create context + pool
    eci_context_params_t cp = {};
    cp.context_size = 4096;
    cp.batch_size = 512;
    cp.seq_max = 2;

    eci_context_t* ctx = nullptr;
    ASSERT(eci_create_context(model, &cp, &ctx) == ECI_OK);

    eci_pool_t* pool = nullptr;
    ASSERT(eci_pool_create(ctx, &pool) == ECI_OK);

    eci_conversation_t* conv = nullptr;
    ASSERT(eci_pool_lease(pool, &conv) == ECI_OK);

    // Read test image
    FILE* f = fopen("./test_image.png", "rb");
    ASSERT(f != nullptr);
    fseek(f, 0, SEEK_END);
    long img_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* img_data = (uint8_t*)malloc(img_size);
    fread(img_data, 1, img_size, f);
    fclose(f);

    // Prompt with image — use the real marker from mtmd
    char* marker_str = nullptr;
    ASSERT(eci_mtmd_marker(&marker_str) == ECI_OK);
    std::string prompt = std::string("What is in this image? ") + marker_str;
    eci_free_string(marker_str);

    const uint8_t* imgs[] = {img_data};
    int sizes[] = {(int)img_size};
    eci_result_t r = eci_conversation_prompt_with_images(
        conv, mmproj, prompt.c_str(), imgs, sizes, 1);
    fprintf(stderr, "vision_prompt=%d ", r);
    // Should be OK (image encoded + text tokens queued)
    ASSERT(r == ECI_OK);

    // After image prompt, we should have tokens in the conversation
    int n_tokens = eci_conversation_token_count(conv);
    fprintf(stderr, "n_tokens=%d ", n_tokens);
    ASSERT(n_tokens > 0);

    // Run one infer to process remaining text tokens
    eci_infer(ctx);

    // Sample — should produce some output
    eci_sampling_params_t sp = {};
    sp.temperature = 0.3f;
    sp.top_p = 0.95f;
    sp.top_k = 40;
    int32_t token = 0;
    ASSERT(eci_conversation_sample(conv, &sp, &token) == ECI_OK);
    ASSERT(token >= 0);

    free(img_data);
    ASSERT(eci_pool_return(pool, conv) == ECI_OK);
    eci_pool_free(pool);
    eci_free_context(ctx);
    eci_free_mmproj(mmproj);
    eci_free_model(model);
}

static void test_shift_left() {
    // Sliding window: fill context, shift left, keep generating
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 512; cp.batch_size = 256; cp.seq_max = 2;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    eci_executor_t* exec = nullptr; ASSERT(eci_executor_create(ctx, &exec) == ECI_OK);

    // Prompt enough to fill most of the 512 context
    ASSERT(eci_executor_prompt(exec, "Hello world, this is a test of the sliding window context shift. "
                                      "We need enough tokens to approach the context limit. "
                                      "The quick brown fox jumps over the lazy dog repeatedly.") == ECI_OK);
    ASSERT(eci_executor_infer(exec) == ECI_DECODE_OK);
    int tokens_before = eci_executor_token_count(exec);
    ASSERT(tokens_before > 10);

    // Shift left by 10 — remove oldest 10 tokens
    ASSERT(eci_executor_shift_left(exec, 10) == ECI_OK);
    int tokens_after = eci_executor_token_count(exec);
    ASSERT(tokens_after == tokens_before - 10);

    // Continue generating — must prompt+infer first (shift invalidated last_batch_idx)
    eci_sampling_params_t sp = {}; sp.temperature = 0.3f; sp.top_k = 40; sp.max_tokens = 5;
    ASSERT(eci_executor_prompt(exec, "Continue: ") == ECI_OK);
    ASSERT(eci_executor_infer(exec) == ECI_DECODE_OK);
    int32_t token = 0;
    ASSERT(eci_executor_sample(exec, &sp, &token) == ECI_OK);
    ASSERT(eci_executor_prompt_tokens(exec, &token, 1) == ECI_OK);
    ASSERT(eci_executor_infer(exec) == ECI_DECODE_OK);
    ASSERT(eci_executor_token_count(exec) > tokens_after);

    eci_executor_free(exec); eci_free_context(ctx); eci_free_model(m);
}

static void test_kv_copy() {
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 4;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    // Clear and copy
    ASSERT(eci_kv_clear_seq(ctx, 0) == ECI_OK);
    ASSERT(eci_kv_clear_seq(ctx, 1) == ECI_OK);
    ASSERT(eci_kv_copy(ctx, 0, 1) == ECI_OK);
    ASSERT(eci_kv_clear_seq(ctx, 3) == ECI_OK);

    // Invalid seq_id
    ASSERT(eci_kv_clear_seq(ctx, 99) == ECI_ERR_INVALID_ARG);
    ASSERT(eci_kv_copy(ctx, 0, 99) == ECI_ERR_INVALID_ARG);

    eci_free_context(ctx); eci_free_model(m);
}

static void test_causal_attn() {
    eci_model_params_t mp = {}; mp.model_path = MODEL_PATH; mp.gpu_layers = 99; mp.flash_attn = true;
    eci_model_t* m = nullptr; ASSERT(eci_load_model(&mp, &m) == ECI_OK);
    eci_context_params_t cp = {}; cp.context_size = 2048; cp.batch_size = 512; cp.seq_max = 2;
    eci_context_t* ctx = nullptr; ASSERT(eci_create_context(m, &cp, &ctx) == ECI_OK);

    // Toggle causal attention
    ASSERT(eci_set_causal_attn(ctx, false) == ECI_OK);
    ASSERT(eci_set_causal_attn(ctx, true) == ECI_OK);

    eci_free_context(ctx); eci_free_model(m);
}

int main() {
    fprintf(stderr, "\n=== ECAssistantInference Tests ===\n\n");
    TEST(load_model);
    TEST(create_context);
    TEST(tokenize);
    TEST(pool);
    TEST(single_inference);
    TEST(batched_inference);
    TEST(state_save_restore);
    TEST(embeddings);
    TEST(anti_prompts);
    TEST(pool_reuse);
    TEST(vision);
    TEST(shift_left);
    TEST(kv_copy);
    TEST(causal_attn);
    fprintf(stderr, "\n=== %d/%d tests passed ===\n", tests_passed, tests_run);
    return 0;
}
