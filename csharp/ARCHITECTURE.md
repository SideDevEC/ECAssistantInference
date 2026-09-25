# ECAssistantInference — Architecture

P/Invoke bindings + object-oriented wrapper layer for the `ECAssistantInference` C API
(`libecainference.dylib` / `.so` / `.dll`). Replaces LLamaSharp as the .NET → llama.cpp bridge.

## Principles (from IDENTITY.md)
- Strict OOP: encapsulation, interfaces, constructor injection, one type per file.
- No static mutable state. The only static class is `Interop.EciNative` (pure P/Invoke
  declarations — the standard interop exception) plus factory methods on immutable types
  (`NativeInferenceModel.Load`).
- No public mutable state: wrappers expose read-only properties and operations.
- Folder structure mirrors namespaces: `Interop/`, `Abstractions/`, `SafeHandles/`,
  `Models/`, `Implementation/`, `Exceptions/`, `Tests/`.
- All native dependencies injectable (SafeHandle constructor injection) → testable by design.

## Layering
```
Caller (ECAssistantLLM)
   │
Abstractions/     IInferenceModel, IInferenceContext, IConversationPool, IConversation,
                  IStandardExecutor, IInferenceBackend, IInferenceState, IVisionEncoder
   │
Implementation/   NativeInferenceModel, NativeInferenceContext, NativeConversationPool,
                  NativeConversation, NativeStandardExecutor, NativeInferenceBackend,
                  NativeInferenceState, NativeVisionEncoder
   │
SafeHandles/      ModelSafeHandle, ContextSafeHandle, PoolSafeHandle, ConversationSafeHandle,
                  ExecutorSafeHandle, StateSafeHandle, MmprojSafeHandle
   │
Interop/          EciNative (P/Invoke), EciResult, EciDecodeResult, EciKvType,
                  EciPoolingType, EciModelParams, EciContextParams, EciSamplingParams
```

## Key decisions
1. **SafeHandle for all native resources** — each ReleaseHandle calls the matching
   `eci_free_*` (model, mmproj, context, pool, conversation, executor, state). Guaranteed
   cleanup even on exceptions/thread aborts.
2. **P/Invoke** — `DllImport("ecainference")`, `SetLastError` not used (the API returns
   typed results); `CharSet.Ansi` for `char*` strings; llm-style blittable structs declared
   with explicit `LayoutKind.Sequential` and packed to match the C ABI.
3. **String marshalling** — inputs: `MarshalAs(LPStr)`. Outputs (`eci_detokenize`,
   `eci_mtmd_marker`): `IntPtr` → `Marshal.PtrToStringAnsi` → `eci_free_string`.
4. **Error handling** — `ECI_ERR_*` results are converted to `InferenceException` in the
   Implementation layer; text is fetched from `eci_result_str`. `eci_decode_result_t`
   (`InferResult`) is a legitimate outcome, not an exception — returned as an enum.
5. **Wrapper classes** — hold a SafeHandle, implement the matching interface, take the
   SafeHandle (or owning context) in the constructor. Ownership: contexts belong to the
   model, pool/executor to the context, conversations to the pool; disposal order is
   enforced by disposing the outermost object first.
6. **Factory** — `NativeInferenceModel.Load(ModelConfig)` (and `LoadVisionEncoder` via
   `IVisionEncoder`) are factory methods returning new immutable wrapper instances.
7. **IDisposable** — every wrapper; use `using`.
8. **Thread safety** — the C++ core is internally mutex-protected. No C# locking here;
   ECAssistantLLM adds `SemaphoreSlim` coordination above this layer.

## Native library resolution
`DllImport("ecainference")` resolves via `DYLD_LIBRARY_PATH` /
`LD_LIBRARY_PATH` / the test output directory (default macOS: `~/Agent/ECAssistantInference/build-metal`).
