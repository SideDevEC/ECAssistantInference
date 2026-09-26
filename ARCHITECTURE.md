# ECAssistantInference — Architecture

**Summary:** C/C++ inference engine linking llama.cpp directly, with C# P/Invoke bindings. Replaces LLamaSharp.

**Addendum 2026-09-26 pm5 (grammar tier 2.5 — rejection-tail rescue):** The benchmark's grammar multi-turn case exposed a tail pathology in the pm2 rejection sampling: tier-2's `kept` array is top_p-TRIMMED after top_k, so with sharp distributions it holds a handful of tokens — when the model fixates on a grammar-invalid token (e.g. '+' in "C++" under a letters-only grammar) the entire set is invalid → tier-3 (~12ms full-vocab mask) runs per sample. Frequency is stochastic (correlated sampling paths made a whole run average 31.7 tok/s; a luckier run did 82). **Fix — tier 2.5:** before tier 3, `std::nth_element` the thread-local candidates by raw logit (≈0.5ms over 152k), grammar-mask the top-**256** slice, return the highest-logit survivor. Measured: tier-3 hits 198→39 across the full grammar matrix (156 rescued), multi-turn grammar stable at 80–86 tok/s; remaining tier-3s are mostly the strict-JSON cases where grammar-first IS the correct path. Tier diagnostics: `g_tier{1,2,2b,3}_hits` counters, `[grammar-tiers]` stderr line on each tier-3. Smoke 51/51.

**Addendum 2026-09-26 pm4 (executor KV leak — batched decode abort fix):** `eci_executor_free` was a bare `delete exec` and left seq 0's KV cells populated (executors always use seq 0; only `eci_executor_reset` cleared them). The next consumer on seq 0 — a fresh executor or pooled conversation leased with seq_id 0 — decodes position 0 into stale cells → llama.cpp position conflict → `llama_decode` ABORTS ("failed to initialize batch", rc=-1). Found via the benchmark's new Batched x4 case (pool convs pop seq 0 after executor cases ran). Fix: `eci_executor_free` now clears seq 0 KV under `infer_mtx`, mirroring `eci_pool_return`. C# smoke 51/51. **Batched benchmark finding:** 4 concurrent conversations aggregate 89 tok/s vs LLamaSharp BatchedExecutor's 107 — the gap is ECI per-sample overhead ×N per step (selection-chain rebuild + per-conv logits snapshot per sample call; LLamaSharp pools its chain once and samples in C#). Chain pooling keyed by `params_fingerprint` (already gated as next step) is the fix; projected to bring batched ECI to parity or better.

**Addendum 2026-09-26 pm3 (⚠ CRITICAL — build dirs: Debug vs Release):** `build-metal` is configured `CMAKE_BUILD_TYPE=Debug` (development build). **`build-metal-rel` is the Release build** (`-DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DBUILD_SHARED_LIBS=OFF`). NEVER benchmark, ship, or package the Debug dylib: Debug runs -O0 on the CPU-side graph construction/scheduler/samplers (~3ms/token, ~1338-node graphs per decode) → ~70 vs **~89 tok/s** on Qwen3-8B Q4_K_M/Metal (and grammar cases 7 → 87 tok/s after the pm2 rejection-sampling fix). FA on beats FA off at short contexts (89 vs 66 tok/s) — keep C# default `ModelConfig.FlashAttention = true`. **PACKAGING RULE:** refresh `csharp/runtimes/osx-arm64/native/libecainference.dylib` from `build-metal-rel/` before building the NuGet package (the runtimes copy is currently the Release dylib, verified 51/51 smoke tests, 2026-09-26). The benchmark project's sync target defaults to `build-metal-rel`.

**Addendum 2026-09-26 pm2 (grammar rejection sampling — 10x grammar speedup):** `do_sample_with_grammar` no longer applies the grammar chain FIRST over the full vocab (~152k candidates × token_to_piece + decode_utf8 heap-alloc per candidate ≈ 130 ms/token — the 10x grammar slowdown measured by the benchmark). Now mirrors llama-server (`common_sampler_sample`, grammar_first=false) and LLamaSharp 0.27 (`DefaultSamplingPipeline.GrammarOptimizationMode.Extended`): **Tier 1** — run the selection chain WITHOUT grammar (ignore_eos redraw loop unchanged), validate the chosen token with a 1-element grammar check (O(1), the common pass for loose grammars); **Tier 2** — grammar-mask the already top_k-truncated candidate array (sorted desc, size ≤ top_k) and take the highest-logit survivor (O(k)); **Tier 3** — full grammar-first mask over the whole vocab (old path, strict grammars like JSON). `grammar_accept` helper preserves the old invariants: apply is stateless (only `accept()` advances the automaton/partial_utf8), EOG/control tokens never grammar-accepted, ignore_eos never returns EOG through fallback tiers. Accepted semantic delta (same one llama-server accepts by default): sampled distribution is truncate-then-validate, not mask-then-truncate — output ALWAYS conforms (every returned token passed a grammar check) but is not bit-identical to grammar-first sampling. Measured (benchmark, Qwen3-8B Q4_K_M, Metal): WithGrammar 6.7→57.6 tok/s, Stateful+Grammar 7.1→54.5 tok/s, JsonGrammar (strict) 39.0 tok/s, baselines unchanged ~70. No grammar-first flag (explicitly declined). Tests: C# smoke 51/51 with new dylib.

**Addendum 2026-09-26 pm (sampling overhaul — unified chain-based pipeline):** Two commits (`97ae2af`, `ddda300`). (1) `do_sample_with_grammar` now uses a thread-local candidate scratch (`t_candidates`, buffer sized once to `n_vocab`, ids rewritten per attempt/re-draw) instead of per-attempt `std::vector<llama_token_data>` push_backs, and a per-thread `std::mt19937` (`t_rng`) replaces the global thread-unsafe `srand/rand`. (2) `params_fingerprint()` (FNV-1a over temp/top_k/top_p/min_p/ignore_eos) added for a future pooled-chain decision, intentionally unused until gated on measurement. (3) The hand-rolled no-grammar sampler (`do_sample`, ~8.6 ms/token measured: 150k-entry scored vector per call) was DELETED — it now delegates to `do_sample_with_grammar(grammar=null)`. One sampling pipeline for both paths; grammar masking stays a persistent conversation/executor-owned chain applied FIRST on full vocab, the selection chain is per-call and grammar-free. Accepted semantic deltas (documented in docs/sampling-overhaul-decisions.md): llama-style penalties (negative logits multiplied, presence per-unique-token), chain order top_k→top_p→min_p, `ignore_eos` now honored on no-grammar calls (16-attempt re-draw), outputs not bit-identical. Measured (Qwen3.5-4B, Metal): sampling overhead 8.6→4.1 ms/sample, generate-128 50.9→67.7 t/s; residual gap vs LLamaSharp is decode-side. Gated next step: pool the per-thread selection chain keyed by `params_fingerprint` (chain re-init is part of the residual ~4 ms). Invariants: grammar mask first on full vocab from the persistent chain; ignore_eos re-draws rebuild from the raw logits snapshot (sampler apply mutates the candidate array, not the context logits buffer); EOG/control tokens never grammar-accepted; dist-last. Tests: basic 14/14, stress 31/31.

**Addendum 2026-09-26 (stale logits rows — batched sampling crash fix):** PRIMARY crash behind the 2026-09-26 01:04 J7b batch-server death (and reproduced live 08:53). `llama_get_logits_ith` rows are only addressable while their batch is the ctx's CURRENT batch; any later decode on the shared context replaces it. `eci_conversation_sample` read `conv->last_batch_idx` (recorded at decode time) directly — in chunked batched inference (combined pending > n_batch, multi-conversation slices) or after any interleaved cycle, the recorded row is stale → `GGML_ABORT` (get_logits_ith: `batch.logits[i] != true`) → server dies. The earlier Metal rsets abort was only the SECONDARY crash during exit; this was the primary trigger. **Fix:** private per-conversation/executor logits snapshot (`last_logits`, `snapshot_logits_row()`) copied at decode time right after the slice that contains the conversation's final token (batched `eci_infer`, executor chunked decode, vision embd path). Sampling (`do_sample`, `do_sample_with_grammar`, all four sample entry points) reads the private snapshot — immune to interleaved decodes on the shared ctx, backend-agnostic, zero parallelism loss. Snapshots invalidated where committed tokens are removed (rewind/reset/restore/return/shift_left). New regression test: `batched_multi_conv_chunked_sample` (3 conversations chunked across slices + interleaved-decode sampling — old code hard-aborts). Tests: stress 31/31, basic 14/14.

**Addendum 2026-09-26 (destruction ordering — Metal rsets abort fix):** `eci_free_model` and `eci_free_context` ran without synchronization — .NET SafeHandle finalizers can fire concurrently during process exit, causing `llama_model_free` (which destroys Metal devices) to race with `llama_free` (which destroys Metal buffers/residency sets). The Metal backend asserts `GGML_ASSERT([rsets->data count] == 0)` in `ggml_metal_rsets_free` — if any context's buffers haven't been freed yet, the process aborts. **Fix:** per-model context counter + condition_variable. `eci_create_context` increments `model->active_context_count` under `model->destruction_mtx`. `eci_free_context` frees the llama_context, then decrements the count and notifies the CV. `eci_free_model` waits on the CV until `active_context_count == 0` before freeing the model. This is backend-agnostic (protects Metal, CUDA, Vulkan alike) and preserves full parallelism — contexts from different models can be freed concurrently, and context destruction doesn't serialize against anything except model destruction. C++ tests: stress 30/30, basic 14/14. C# tests: 51/51.

**Addendum 2026-09-25 (chunked decode — n_batch safety):** `llama_decode` ABORTS the process (GGML_ASSERT `n_tokens_all <= n_batch`) when handed more tokens than one batch. LLamaSharp chunked transparently; this layer must do it itself. All three decode sites now chunk pending tokens into ≤ `n_batch` slices with positions advancing per slice: (1) `eci_executor_infer` (standard path) — logits only on the final token of the final slice; failed slices keep the REMAINING pending tokens so retry doesn't re-decode committed ones. (2) batched `eci_infer` — per-conversation coverage committed per slice (`committed` tracking; on failure the committed prefix is trimmed so positions don't diverge from KV). (3) `eci_conversation_prompt_with_images` text flush before image decode. New stress test: `batched_large_prompt_small_nbatch` (prompt ≫ n_batch, sample right after chunked prefill). C++ tests: stress 30/30, basic 14/14.

## Layers

```
┌─────────────────────────────────────────────┐
│ ECAssistantLLM (C# — HTTP server)            │
│  OpenAI API, sessions, SemaphoreSlim gates   │
├─────────────────────────────────────────────┤
│ ECAssistantInference C# (this repo)          │
│  P/Invoke → SafeHandles → OOP wrappers       │
├─────────────────────────────────────────────┤
│ libecainference (C++ — this repo)            │
│  Model, context, pool, batched decode,       │
│  sampling, embeddings, vision, KV ops,       │
│  backend safety                              │
├─────────────────────────────────────────────┤
│ llama.cpp (submodule)                        │
│  Native inference, Metal/CUDA/Vulkan/CPU     │
└─────────────────────────────────────────────┘
```

## C API (ecainference.h)

Flat C API — 70+ functions. Opaque handles for model, context, pool, conversation, executor, state, grammar.

### Key types
- `eci_model_t` — loaded GGUF model
- `eci_context_t` — inference context (KV cache, n_seq_max)
- `eci_pool_t` — conversation pool (pre-allocated SeqMax conversations)
- `eci_conversation_t` — leased conversation (seq_id + KV slice)
- `eci_executor_t` — standard (non-batched) executor
- `eci_state_t` — saved KV state for rewind
- `eci_grammar_t` — GBNF grammar sampler (llama_sampler_init_grammar)

### Grammar (GBNF)
- `eci_grammar_create(model, grammar_str, grammar_root)` — validates + stores grammar strings
- `eci_executor_sample_grammar` / `eci_conversation_sample_grammar` — sample with grammar constraint
- C# exposes `IGrammar` + `NativeGrammar` + `SampleWithGrammar()` on both executor and conversation
- **Implementation:** persistent grammar chain on executor/conversation; grammar is FIRST in sampler chain; sampled token accepted into grammar state once at commit time (EOG/control tokens skipped). Prompt tokens are NOT grammar-accepted — a per-token reset/accept hook on the prompt path desynced the stateless feedback loop (grammar reset every generated token → `[[[[` repetition); the hook was removed and grammar state is reset explicitly at generation start (`eci_conversation_grammar_reset` / `eci_executor_grammar_reset`).
- **Known issue:** llama.cpp `llama_grammar_accept` crashes on multi-character tokens that span grammar rules with optional whitespace (e.g. Qwen token `[{` spans `"[" ws toolcall`). Try/catch fallback produces unconstrained output. Upstream bug in `src/llama-grammar.cpp:1028` — empty stacks dropped in `llama_grammar_accept_chr`. Affects grammar-constrained tool call generation only; early-stop JSON parsing handles termination as fallback. Callers should ALSO bound array repetition in their grammars (ECAssistantLLM uses `{0,N-1}` tool-call bounds) so the model cannot loop objects until max_tokens truncation.
- **Batch repeat penalties:** conversations track `recent_tokens` (rolling window): committed on prompt flush (incl. vision pre-image text flush), pushed on each sample, trimmed on rewind/shift_left, cleared on reset. Passed to `do_sample`/`do_sample_with_grammar` so repeat/present penalties behave identically on batch and executor paths.

### Chat template
- `eci_apply_chat_template(model, tmpl, messages, n_msg, add_assistant, out_text)` — wraps `llama_chat_apply_template`
- Pass tmpl=null for model's default template (Qwen, Llama, ChatML, etc.)
- C# exposes `IInferenceModel.ApplyChatTemplate()`

### Vision on standard executor
- `eci_executor_prompt_with_images(exec, model, text, images, sizes, n_images)` — uses `mtmd_tokenize` + `mtmd_helper_eval_chunk_single`
- C# exposes `IStandardExecutor.PromptWithImages()`

### Thread safety
- Internal `std::mutex` on every `llama_*` call (serialized) — `eci_context_s::infer_mtx`
- **Destruction ordering:** per-model `active_context_count` + condition_variable. `eci_free_model` blocks until all contexts created from that model are freed first. Prevents Metal `ggml_metal_rsets_free` abort when device is freed before its buffers. Backend-agnostic — protects any GPU backend that requires ordered teardown. Contexts from different models can be freed concurrently (no global lock).
- C# layer adds `SemaphoreSlim` for higher-level coordination
- Vision requires exclusive cycle-gate access (documented in CAPABILITY.md)

### Structured logging (2026-09-25)
- **Zero-overhead when disabled:** `g_log_cb` is NULL by default. Every `ECI_LOG(lvl, tag, fmt, ...)` macro checks the null pointer first — branch-predicted skip, no string formatting.
- **C API:** `eci_set_log_callback(cb)` / `eci_set_log_level(level)` — caller registers a callback + minimum level
- **Levels:** `ECI_LOG_DEBUG=0`, `INFO=1`, `WARN=2`, `ERROR=3`, `NONE=99` (disable all)
- **Log sites:** `set_error()` (all error paths), chunked/batched `llama_decode` failures, model load success
- **C# bridge:** `NativeLogBridge.Enable(Action<int,string,string>)` — pins the delegate + calls `EciNative.SetLogCallback`. Encapsulated in Inference layer, no dependency on LLM/Core loggers.
- **Caller responsibility:** keep the `Action` alive (GC would collect it). Pass null to disable.

## C# Bindings (csharp/)

### Namespace structure
| Namespace | Purpose |
|-----------|---------|
| `ECAssistantInference.Interop` | P/Invoke declarations, blittable structs, enums |
| `ECAssistantInference.SafeHandles` | SafeHandle for each native resource (auto-free on GC) |
| `ECAssistantInference.Abstractions` | Interfaces (IInferenceModel, IConversationPool, etc.) |
| `ECAssistantInference.Implementation` | Native wrappers implementing interfaces |
| `ECAssistantInference.Models` | User-facing config records (ModelConfig, ContextConfig, etc.) |
| `ECAssistantInference.Exceptions` | InferenceException + EciResult extension |
| `ECAssistantInference.Logging` | NativeLogBridge — bridges C-level log callback to managed Action<int,string,string> |

### Design rules
- One type per file, file named after type
- Constructor injection (SafeHandle → wrapper)
- No statics except `EciNative` (sanctioned interop exception) + `NativeLogBridge._pinned` (GC-pinned delegate, process-lifetime)
- All native resources wrapped in SafeHandle
- Factory: `NativeInferenceModel.Load(config)` creates model
- `internal` on P/Invoke + structs; `public` on interfaces + wrappers + configs

## Backend Safety (eci_backend.cpp)

Caller drives ALL config. Safety layer only intervenes on probe failure:

1. **Probe** (`eci_probe_backend`) — minimal 1-token decode validates config
2. **Progressive fallback** (`eci_create_context_safe`) — gpu_layers N → N/2 → 0
3. **Vulkan stale-KV flush** (`eci_kv_flush`) — after pool return, flush stale cells (#26744)

### Issues addressed
- Vulkan #19955: DeviceLost on long contexts → probe catches
- Vulkan #26195: FA + quantized V = garbled output → probe validates
- Vulkan #26744: Stale K/V in freed cells → eci_kv_flush after pool return
- Vulkan #27237: Batch 512 garbage on some models → caller-driven batch size
- Metal 1a5631b: Command buffer failures handled gracefully (March 2026)
- CUDA #26609: MoE + FA + partial offload crash → probe detects

## Test Coverage

### C++ (43 tests)
- 14 basic: model, context, tokenize, pool, single/batched inference, state, embeddings, vision, shift_left, kv_copy, causal_attn, grammar, chat template
- 29 stress: NULL args, empty inputs, pool exhaustion, double-return, KV boundaries, sampling extremes, state save/restore edge cases, repeated generation, model reload

### C# (51 tests)
- Model lifecycle (6), context (2), tokenization (4), executor (6), sampling extremes (4), pool (3), batched inference (3), shift_left (3), state (2), embeddings (5), vision (2), KV cache (4), backend (2), conversation ops (3), resource cleanup (2)

## CI Matrix (planned)
| Target | Backend | OS |
|--------|---------|----|
| osx-arm64 | Metal | macOS-14 |
| osx-arm64 | CPU | macOS-14 |
| linux-x64 | CPU | ubuntu-22.04 |
| linux-x64 | CUDA | ubuntu-22.04 (self-hosted) |
| linux-x64 | Vulkan | ubuntu-22.04 |
| win-x64 | CPU | windows-2022 |
| win-x64 | CUDA | windows-2022 (self-hosted) |
| win-x64 | Vulkan | windows-2022 |

## NuGet (prepared, not shipped)
- Package: `ECAssistantInference` (C# class library)
- Native binaries bundled as runtimes/{RID}/native/ content
- CI workflow builds native + packs NuGet, but does NOT publish/tag
- Ships in v15 train with ECAssistantLLM

## Files
- ✅ include/ecainference.h — public C API
- ✅ include/eci_internal.h — internal structs
- ✅ src/ecainference.cpp — core implementation
- ✅ src/eci_backend.cpp — backend safety layer
- ✅ tests/test_basic.cpp — 14 basic tests
- ✅ tests/test_stress.cpp — 29 stress tests
- ✅ docs/CAPABILITY.md — capability matrix
- ✅ csharp/ — C# bindings (45 files, 51 tests, LDC PASSED 23 types/10 edges)
- ✅ CMakeLists.txt — build system
- ✅ .github/workflows/ci.yml — CI (build + test, no publish)

**Status:** 43/43 C++ tests + 51/51 C# tests passing on macOS Metal. ECAssistantLLM: 310/313 (3 grammar-constrained tool call tests — upstream llama.cpp bug). LDC PASSED.
**Updated:** 2026-09-25 — added grammar, chat template, executor vision APIs
