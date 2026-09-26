# ECAssistantInference

C/C++ inference engine + C# bindings for ECAssistant. Links [llama.cpp](https://github.com/ggml-org/llama.cpp) directly via a flat C API, P/Invoked from C#.

Replaces LLamaSharp entirely — full control over batching, KV cache, vision, and backend safety.

## Structure

```
ECAssistantInference/
├── include/              # Public C API header
│   ├── ecainference.h    # Flat C API — all 60+ functions
│   └── eci_internal.h    # Internal structs (not public)
├── src/                  # C++ implementation
│   ├── ecainference.cpp  # Core: model, context, pool, executor, sampling, vision, KV
│   └── eci_backend.cpp   # Backend safety: probe, progressive fallback, Vulkan stale-KV fix
├── tests/                # C++ tests (14 basic + 29 stress = 43 total)
│   ├── test_basic.cpp
│   └── test_stress.cpp
├── docs/
│   └── CAPABILITY.md     # Capability matrix + careful-handling areas for C# layer
├── csharp/               # C# P/Invoke bindings (.NET 8 class library)
│   ├── ECAssistantInference.csproj
│   ├── Interop/          # P/Invoke declarations, blittable structs, enums
│   ├── SafeHandles/      # SafeHandle for each native resource
│   ├── Abstractions/     # Interfaces (IInferenceModel, IConversationPool, etc.)
│   ├── Implementation/   # Native wrappers implementing interfaces
│   ├── Models/           # User-facing config records
│   ├── Exceptions/       # InferenceException + error extensions
│   ├── Tests/            # 51 xUnit tests
│   └── ARCHITECTURE.md
├── CMakeLists.txt        # Build: Metal/CUDA/Vulkan/CPU × macOS/Linux/Windows
└── .github/workflows/    # CI: build native + C#, run tests, NuGet pack (no publish)
```

## Features

- **Model loading** — GGUF, configurable GPU layers, flash attention, KV cache type (F16/Q8_0/Q4_0/Q4_1)
- **Standard executor** — prompt → infer → sample loop with full sampling pipeline
- **Batched inference** — 4+ conversations in ONE decode (true `n_seqs > 1`)
- **Conversation pool** — pre-allocated conversations, lease/return with KV rewind (no seq-id leak)
- **Vision (MTMD)** — image bytes → CLIP encode → embd batch decode, non-causal attention handling
- **Embeddings** — mean/CLS/last pooling, normalized, encoder-only model support (BERT/MiniLM)
- **KV cache manipulation** — shift_left (sliding window), kv_copy, kv_clear_seq, causal_attn toggle
- **State save/restore** — KV snapshot via `llama_memory_seq_cp`
- **Backend safety layer** — runtime probe, progressive GPU layer reduction, Vulkan stale-KV flush (#26744)
- **Cross-platform** — Metal (macOS), CUDA (Linux/Windows), Vulkan (Linux/Windows), CPU fallback

## Building

### Native library (C++)

```bash
git submodule update --init --recursive

# macOS (Metal)
cmake -B build-metal -DECI_BACKEND_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal -j

# Run C++ tests
cd build-metal && DYLD_LIBRARY_PATH=. ./ecainference_tests      # 14 tests
cd build-metal && DYLD_LIBRARY_PATH=. ./ecainference_stress_tests # 29 tests
```

### C# bindings

```bash
cd csharp
dotnet build
DYLD_LIBRARY_PATH=../build-metal dotnet test Tests/  # 51 tests
```

## C# Quick Start

```csharp
using ECAssistantInference.Implementation;
using ECAssistantInference.Models;

var model = NativeInferenceModel.Load(new ModelConfig {
    Path = "model.gguf", GpuLayers = 99, FlashAttention = true
});

var ctx = model.CreateContext(new ContextConfig {
    ContextSize = 4096, BatchSize = 512, SeqMax = 4
});

// Standard inference
var exec = ctx.CreateExecutor();
exec.Prompt("Hello!");
exec.Infer();
var token = exec.Sample(new SamplingConfig { Temperature = 0.3f });

// Batched inference (4 conversations in one decode)
var pool = ctx.CreatePool();
var conv = pool.Lease();
conv.Prompt("Say a number");
ctx.InferAll();  // single decode for all active conversations
var t = conv.Sample();
pool.Return(conv);

// Embeddings
var emb = model.GetEmbeddings(ctx, "Hello world");

// Vision
var vision = model.LoadVisionEncoder("mmproj.gguf");
conv.PromptWithImages("Describe this", vision, imageBytes);
```

## Backend Safety

The caller drives all config (KV type, batch size, FA, gpu_layers). The safety layer only intervenes on failure:

1. **Probe** — minimal 1-token decode validates the config works
2. **Progressive fallback** — if probe fails: gpu_layers N → N/2 → N/4 → 0 (CPU)
3. **Vulkan stale-KV flush** — after pool return, flush stale cells (#26744)

## History Note

Commit `07ce77a` is a short-lived intermediate state: it added a `fused_gdn` context param
that depended on a local llama.cpp submodule patch (never pushed — the submodule points at
pristine upstream `ggml-org/llama.cpp`). **That commit does not build standalone.** The very next
commit (`1b4b67a`) reverted it entirely; `main` is the consistent state. Do not check out or
build from `07ce77a`.

Owner decision (2026-09-26): the llama.cpp submodule stays **pristine upstream** — no fork, no
local diff. Behavior is pinned by upstream defaults (fused gated-delta-net kernels are hardcoded
`true`). No environment variables influence behavior anywhere in product code; behavioral knobs
are config-injected only.

## License

MIT
