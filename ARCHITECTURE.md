# ECAssistantInference — Architecture

**Summary:** C/C++ inference engine linking llama.cpp directly, with C# P/Invoke bindings. Replaces LLamaSharp.

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
- **Implementation:** persistent grammar chain on executor/conversation; grammar is FIRST in sampler chain; prompt tokens accepted into grammar before first sample (initializes state)
- **Known issue:** llama.cpp `llama_grammar_accept` crashes on multi-character tokens that span grammar rules with optional whitespace (e.g. Qwen token `[{` spans `"[" ws toolcall`). Try/catch fallback produces unconstrained output. Upstream bug in `src/llama-grammar.cpp:1028` — empty stacks dropped in `llama_grammar_accept_chr`. Affects grammar-constrained tool call generation only; early-stop JSON parsing handles termination as fallback.

### Chat template
- `eci_apply_chat_template(model, tmpl, messages, n_msg, add_assistant, out_text)` — wraps `llama_chat_apply_template`
- Pass tmpl=null for model's default template (Qwen, Llama, ChatML, etc.)
- C# exposes `IInferenceModel.ApplyChatTemplate()`

### Vision on standard executor
- `eci_executor_prompt_with_images(exec, model, text, images, sizes, n_images)` — uses `mtmd_tokenize` + `mtmd_helper_eval_chunk_single`
- C# exposes `IStandardExecutor.PromptWithImages()`

### Thread safety
- Internal `std::mutex` on every `llama_*` call (serialized)
- C# layer adds `SemaphoreSlim` for higher-level coordination
- Vision requires exclusive cycle-gate access (documented in CAPABILITY.md)

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

### Design rules
- One type per file, file named after type
- Constructor injection (SafeHandle → wrapper)
- No statics except `EciNative` (sanctioned interop exception)
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
