# ECAssistantInference — Capability & Integration Guide

## What It Is

A C/C++ inference engine that links llama.cpp directly and exposes a flat C API for P/Invoke from C#. Replaces LLamaSharp entirely. No wrapper, no managed/native split — we own the entire inference path.

## Current State (2026-09-25)

- **14/14 tests passing** on macOS Metal (M3 Ultra)
- Library size: 11.3MB (statically links llama.cpp + ggml + mtmd)
- llama.cpp: latest master (e351231c4)
- Build: cmake with Metal/CPU/CUDA/Vulkan backend flags

---

## Capabilities

### Model Management
- `eci_load_model` — load GGUF model with GPU layers, threads, flash attention, KV cache type
- `eci_free_model` — free model
- `eci_load_mmproj` — load vision projector (mmproj GGUF) → mtmd context
- `eci_free_mmproj` — free vision projector

### Context
- `eci_create_context` — create llama_context with n_ctx, n_batch, n_seq_max, pooling type
- `eci_free_context` — free context
- `eci_context_size` / `eci_context_seq_max` — query context params

### Tokenization
- `eci_tokenize` — text → tokens (with add_bos, parse_special flags)
- `eci_detokenize` — tokens → text (with remove_special flag)
- `eci_token_to_piece` — single token → string
- `eci_token_is_eos` — check if token is end-of-generation

### Standard Executor (non-batched, one session per context)
- `eci_executor_create` / `eci_executor_free`
- `eci_executor_prompt` — tokenize + queue text tokens
- `eci_executor_prompt_tokens` — queue raw tokens
- `eci_executor_infer` — decode pending tokens (one batch)
- `eci_executor_sample` — sample one token (full pipeline: temp, top_k, top_p, min_p, repeat_penalty, presence_penalty)
- `eci_executor_token_count` — current KV token count
- `eci_executor_rewind` — remove last N tokens from KV
- `eci_executor_reset` — clear entire KV (in-place, no context recreation)

### Conversation Pool (batched inference)
- `eci_pool_create` — pre-allocate SeqMax conversations (each owns a permanent seq_id)
- `eci_pool_lease` / `eci_pool_return` — lease/return conversations (thread-safe)
- `eci_pool_available` / `eci_pool_size` — query pool state
- **Key fix:** No `executor.Create()` ever called on request threads → no seq-id leak, no thread-safety crash

### Batched Inference
- `eci_conversation_prompt` / `eci_conversation_prompt_tokens` — queue tokens per conversation
- `eci_infer` — ONE decode for ALL conversations with pending tokens (true multi-seq batching)
- `eci_conversation_sample` — sample from a conversation's last token
- `eci_conversation_token_count` — per-conversation token count
- `eci_conversation_rewind` / `eci_conversation_reset` — per-conversation KV manipulation

### Full Sampling Pipeline
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling
- Min-p filtering
- Repeat penalty (with configurable window)
- Presence penalty
- EOS detection
- **Logits are copied before modification** — no corruption of shared internal state

### Embeddings
- `eci_get_embeddings` — text → normalized embedding vector
- Pooling modes: MEAN, CLS, LAST (configurable per context)
- L2 normalization
- `eci_embedding_dim` — query embedding dimensions

### Vision / MTMD
- `eci_load_mmproj` — load mmproj file → mtmd_context
- `eci_conversation_prompt_with_images` — full pipeline:
  1. Decode image bytes (PNG/JPEG) via `mtmd_helper_bitmap_init_from_buf`
  2. Tokenize prompt + images into chunks via `mtmd_tokenize`
  3. Text chunks → queue as pending tokens
  4. Image chunks → encode via `mtmd_encode_chunk` → get float embeddings
  5. Flush pending text tokens (causal decode)
  6. Set non-causal attention → decode image embeddings via embd batch
  7. Restore causal attention
- `eci_mtmd_marker` / `eci_mtmd_marker_static` — get the media marker string
- `eci_mtmd_needs_non_causal` — check if model requires non-causal for images
- `eci_set_causal_attn` — per-call causal attention control

### Granular KV Cache Manipulation (impossible via LLamaSharp)
- `eci_conversation_shift_left` / `eci_executor_shift_left` — **sliding window context**: remove oldest N tokens, keep recent ones, continue generating without reset. RoPE positions stay correct via separate `n_kv_pos` tracking.
- `eci_kv_copy` — copy KV from one seq_id to another (state backup without blob serialization)
- `eci_kv_clear_seq` — clear one seq_id's KV without affecting other conversations
- `eci_set_causal_attn` — toggle causal/non-causal attention per-call

### State Save/Restore
- `eci_state_save` — snapshot KV to backup seq_id (via `llama_memory_seq_cp`)
- `eci_state_restore` — restore from backup seq_id
- `eci_conversation_save` / `eci_conversation_restore` — per-conversation state
- **Key improvement:** No blob serialization, no managed/native split, one atomic operation

### Anti-Prompt Detection
- `eci_check_anti_prompts` — check if generated text contains any anti-prompt string

### Thread Safety
- Internal `std::mutex infer_mtx` on every context serializes ALL native llama.cpp calls
- Pool has its own mutex for lease/return
- C# layer handles per-session SemaphoreSlim gates and cycle coordination

---

## What Needs Careful Handling at the C# Layer

### 1. Vision Processing (Single, Non-Batchable)

**The constraint:** Image embeddings require non-causal attention (`llama_set_causal_attn(ctx, false)`). This is a **context-wide flag** — it affects ALL conversations on that context, not just the one with the image.

**What this means for the C# layer:**
- `eci_conversation_prompt_with_images` must be called **inside the cycle gate** (the `_cycleGate` SemaphoreSlim in `BatchInferenceCoordinator`)
- No other conversation can be decoding on the same context while image embeddings are being processed
- The C++ layer handles the causal/non-causal toggle internally, but the C# layer must ensure **exclusive access** during the call

**Flow:**
```
C# acquires _cycleGate
  → C# calls eci_conversation_prompt_with_images(conv, mmproj, text, images)
    → C++ flushes text tokens (causal decode)
    → C++ sets non-causal
    → C++ encodes image (CLIP)
    → C++ decodes image embeddings (non-causal)
    → C++ restores causal
  → C# releases _cycleGate
→ Other conversations can now batch normally
```

**Cannot batch multiple conversations' image decodes in one llama_batch** because:
- Text tokens need causal attention
- Image embeddings need non-causal
- You can't mix them in one `llama_decode` call
- Multiple images COULD be batched (both non-causal), but the CLIP encoding is the bottleneck and `mtmd_batch_encode` already handles that

### 2. Shift Left (Sliding Window) — Position Tracking

**The constraint:** After `shift_left(n)`, the logical token count (`n_tokens`) decreases but the absolute KV position (`n_kv_pos`) stays the same. RoPE positions in the remaining KV entries are absolute and must not be reindexed.

**What the C# layer must track:**
- `n_tokens` (logical) — for headroom checks, overflow detection, max_tokens clamping
- The C++ layer handles `n_kv_pos` internally — C# never needs to know it
- After shift_left, the C# layer must re-evaluate headroom: `headroom = context_size - n_tokens - safety_margin`
- The C# layer should call `shift_left` BEFORE the context is full, not after — e.g., when `n_tokens > context_size * 0.8`, shift left by 10-20% of context size

**When to use shift_left vs reset:**
- **shift_left:** Conversation is long but still relevant. Remove oldest tokens, keep recent context. No re-prompt needed.
- **reset:** Conversation is done or corrupted. Clear everything. Must re-prefill.

### 3. Repeat Penalty in Batch Mode

**The constraint:** `eci_conversation_sample` does NOT apply repeat penalty in batch mode (passes empty `recent_tokens`). The C++ layer doesn't track per-conversation token history for batch conversations.

**What the C# layer must do:**
- Track recent generated tokens per `BatchSession` in C# (already done via `_exactTokenCount` and the session's prompt history)
- If repeat penalty is needed in batch mode, either:
  - Option A: Pass recent tokens from C# to a new `eci_conversation_sample_with_history` function (TODO if needed)
  - Option B: Apply repeat penalty at the C# level by post-processing the sampled token (not ideal — can't modify logits after sampling)
  - Option C: Accept no repeat penalty in batch mode (current approach — acceptable for most use cases)

### 4. Pool Exhaustion

**The constraint:** The pool has exactly `SeqMax` conversations. If all are leased and a new request comes in, `eci_pool_lease` returns `ECI_ERR_NO_SLOT`.

**What the C# layer must do:**
- Set `SeqMax = max_sessions` in the config (already done)
- When `ECI_ERR_NO_SLOT` is returned, either:
  - Queue the request and retry when a conversation is returned (preferred)
  - Return HTTP 503 (service unavailable) to the client
- Do NOT create new conversations on the fly — that would exhaust seq_ids (the original bug)

### 5. State Save/Restore with n_seq_max == 1

**The constraint:** `eci_state_save` copies KV to backup seq_id 1. If `n_seq_max == 1`, there's no backup seq_id.

**What the C# layer must do:**
- For the standard executor (non-batched), set `seq_max >= 2` so state save/restore works
- For batch mode, `seq_max = max_sessions` (always >= 2), so this is a non-issue
- If `n_seq_max == 1` and state restore is called, the C++ layer returns OK with `n_tokens = 0` — the C# layer must re-prompt the conversation

### 6. Thread Safety Model

**The C++ layer guarantees:**
- `infer_mtx` serializes all native llama.cpp calls within a context
- Pool mutex serializes lease/return

**The C# layer must guarantee:**
- Per-session `SemaphoreSlim(1,1)` request gate — only one request per session at a time
- `BatchInferenceCoordinator._cycleGate` — serializes the drain+infer cycle
- Vision prompt calls are made from the cycle coordinator thread (under the cycle gate)
- No concurrent calls to `eci_conversation_prompt_with_images` and `eci_infer` on the same context

### 7. Error Recovery

**On decode failure (`ECI_DECODE_FAILED`):**
- The C++ layer does NOT modify `n_tokens` or `n_kv_pos` on failure
- The C# layer should:
  - Log the error
  - Call `eci_conversation_reset` to clear the KV (safe — just `llama_memory_seq_rm`)
  - Re-prompt the conversation from scratch
  - Or call `eci_conversation_rewind` to try removing the last prompt

**On pool exhaustion (`ECI_ERR_NO_SLOT`):**
- Queue the request, don't crash
- Return 503 to the client with a Retry-After header

**On model load failure (`ECI_ERR_LOAD_FAILED`):**
- Check `eci_last_error` for the native error message
- Common causes: wrong path, corrupted GGUF, insufficient VRAM for requested GPU layers

---

## Build Matrix

| Platform | Backend | CMake Flag | Tested? |
|----------|---------|-----------|---------|
| macOS arm64 | Metal | `ECI_BACKEND_METAL=ON` | ✅ 14/14 |
| macOS arm64 | CPU | (default) | ✅ (implied) |
| Linux x64 | CPU | (default) | ⏳ CI |
| Linux x64 | CUDA | `ECI_BACKEND_CUDA=ON` | ⏳ CI |
| Linux x64 | Vulkan | `ECI_BACKEND_VULKAN=ON` | ⏳ CI |
| Windows x64 | CPU | (default) | ⏳ CI |
| Windows x64 | CUDA | `ECI_BACKEND_CUDA=ON` | ⏳ CI |
| Windows x64 | Vulkan | `ECI_BACKEND_VULKAN=ON` | ⏳ CI |

---

## Backend Safety Layer

Automatic detection, probing, and fallback for all GPU backends. No hardcoded assumptions — everything is verified at runtime.

### Backend Probe (`eci_probe_backend`)
- After context creation, does a minimal 1-token decode to verify the GPU backend works
- If it fails → caller retries with fewer GPU layers (`eci_create_context_safe`)
- Detects: Metal, CUDA, Vulkan, CPU via `ggml_backend_dev` list

### Progressive GPU Layer Reduction (`eci_create_context_safe`)
- Try `requested_gpu_layers` → if fail, try N/2 → N/4 → ... → 0 (CPU)
- Finds the working threshold automatically — no manual tuning

### Per-Backend Feature Profiles

| Feature | Metal | CUDA | Vulkan | CPU |
|---------|-------|------|--------|-----|
| Flash attention | ✅ Safe | ✅ Safe | ⚠️ Probe | ✅ Safe |
| Quantized KV (q8_0) | ✅ Safe (dequant internally) | ✅ Safe | ❌ Default unsafe | ✅ Safe |
| Multi-seq decode | ✅ Works | ✅ Works | ✅ Works (risky with FA+GQA) | ✅ Works |
| Pool reuse | ✅ Safe | ✅ Safe | ⚠️ Needs flush | ✅ Safe |
| Min batch size | 512 | 512 | 1024 | 512 |
| Stale KV cells after clear | No | No | Yes (#26744) | No |

### Vulkan Stale KV Fix (`eci_kv_flush`)
- After `llama_memory_seq_rm` (pool return), Vulkan may leave stale K/V data in freed cells (#26744)
- `eci_kv_flush` does a no-op decode + clear to flush stale state
- Automatically called in `eci_pool_return` when `eci_backend_needs_kv_flush()` returns true
- On safe backends (Metal/CUDA/CPU), this is skipped (no-op)

### Known Issues Addressed

**Vulkan:**
- `vk::DeviceLostError` on long contexts (#19955) → probe catches this
- FA + quantized V cache = garbled output (#26195) → quantized_kv_safe defaults to false
- Stale K/V in freed cells (#26744) → eci_kv_flush after pool return
- FA + GQA packing + multi-token crash (#26358) → multi_seq_safe flag
- Batch 512 garbage on some models (#27237) → min_batch_size = 1024
- load-mode auto OOM (#27360) → progressive layer reduction

**Metal:**
- Command buffer failures (1a5631b) → handled gracefully since March 2026 (we have this)
- Mixed quantized KV when FA unavailable (#21450) → Metal dequantizes internally (70aff25)

**CUDA:**
- OOM without warning → progressive layer reduction + probe
- MoE + FA + partial offload illegal memory access (#26609) → probe detects, can disable FA
- mmproj VRAM not in auto-fit (#19980) → manual VRAM guard

### VRAM Guard (`eci_vram_available`)
- Pre-operation VRAM check (platform-specific implementation TODO)
- Currently relies on probe + progressive reduction as safety net
- TODO: Metal (mach_vm_statistics), CUDA (cudaMemGetInfo), Vulkan (vkGetPhysicalDeviceMemoryProperties)

---

## LLamaSharp Workarounds Eliminated

| Workaround in C# | C++ Replacement | Status |
|-----------------|----------------|--------|
| Double state snapshot (ExecutorBaseState + KV blob) | `eci_state_save` (one atomic `llama_memory_seq_cp`) | ✅ Eliminated |
| Reset = destroy + recreate context | `eci_executor_reset` (in-place `llama_memory_seq_rm`) | ✅ Eliminated |
| Rewind = full blob restore (O(n_ctx) copy) | `eci_conversation_rewind` (O(rewind_count) removal) | ✅ Eliminated |
| Overflow = ThrowException + catch + reset | `eci_conversation_shift_left` (sliding window) | ✅ Eliminated |
| Save/Load creating new conversations (seq-id leak) | `eci_conversation_restore` (in-place rewind) | ✅ Eliminated |
| Broken ShiftLeft | `eci_conversation_shift_left` (clean, with n_kv_pos tracking) | ✅ Eliminated |
| Context.Tokenize for exact counts | Tracked natively in C++ | ✅ Eliminated |

---

## TODO

- [ ] C# P/Invoke bindings project (`ECAssistant.Inference.dll`)
- [ ] Integrate into ECAssistantLLM (replace LLamaSharp calls)
- [ ] GitHub Actions CI matrix (8 build targets)
- [ ] NuGet package layout
- [ ] `eci_conversation_sample_with_history` (repeat penalty for batch mode, if needed)
- [ ] CLIP batch encoding (`mtmd_batch_encode` for multiple images)
- [ ] Remove LLamaSharp dependency from ECAssistantLLM csproj
