# csharp.API.md

Types: 22  |  LOC: 1823  |  ~1531 tokens

---

### Interface: IConversation
> Resets grammar sampler state — call once per generation before a grammar-constrained sample loop.
Implements: IDisposable
Properties:
  - int TokenCount { get; set; }
Methods:
  - void Prompt(string text)
  - void PromptTokens(int[] tokens)
  - int Sample(SamplingConfig? config = null)
  - int SampleWithGrammar(SamplingConfig? config, IGrammar grammar)
  - void Rewind(int tokenCount)
  - void Reset()
  - void ResetGrammarState()
  - void ShiftLeft(int tokenCount)
  - IConversationState SaveState()
  - void RestoreState(IConversationPool pool, IConversationState state)
  - void PromptWithImages(string text, IVisionEncoder vision, params byte[][] images)
Cross-package deps: ECAssistantInference.Models

### Interface: IConversationPool
Implements: IDisposable
Properties:
  - int AvailableCount { get; set; }
  - uint Size { get; set; }
Methods:
  - IConversation Lease()
  - void Return(IConversation conversation)

### Interface: IConversationState
Implements: IDisposable
Properties:
  - int TokenCount { get; set; }

### Interface: IGrammar
Implements: IDisposable

### Interface: IInferenceBackend
Methods:
  - BackendInfo Probe(IInferenceContext context)
  - bool VramAvailable(long neededBytes)
Cross-package deps: ECAssistantInference.Models

### Interface: IInferenceContext
Implements: IDisposable
Properties:
  - uint ContextSize { get; set; }
  - uint SeqMax { get; set; }
Methods:
  - IStandardExecutor CreateExecutor()
  - IConversationPool CreatePool()
  - InferResult InferAll()
  - void KvCopy(int srcSeqId, int dstSeqId)
  - void KvClearSeq(int seqId)
  - void SetCausalAttention(bool causal)
  - int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true)
  - string Detokenize(int[] tokens, bool removeSpecial = false)
  - string TokenToPiece(int token)
  - bool IsEos(int token)
Cross-package deps: ECAssistantInference.Models, ECAssistantInference.Abstractions

### Interface: IInferenceModel
Implements: IDisposable
Properties:
  - int EmbeddingDimension { get; set; }
Methods:
  - IInferenceContext CreateContext(ContextConfig config)
  - float[] GetEmbeddings(IInferenceContext context, string text)
  - IVisionEncoder LoadVisionEncoder(string mmprojPath)
  - IGrammar CreateGrammar(string grammarStr, string grammarRoot)
Cross-package deps: ECAssistantInference.Models

### Interface: IInferenceState
Implements: IDisposable
Properties:
  - int TokenCount { get; set; }

### Interface: IStandardExecutor
Implements: IDisposable
Properties:
  - int TokenCount { get; set; }
Methods:
  - void Prompt(string text)
  - void PromptTokens(int[] tokens)
  - void PromptWithImages(string text, IInferenceModel model, params byte[][] images)
  - InferResult Infer()
  - int Sample(SamplingConfig? config = null)
  - int SampleWithGrammar(SamplingConfig? config, IGrammar grammar)
  - void Rewind(int tokenCount)
  - void Reset()
  - void ShiftLeft(int tokenCount)
  - IInferenceState SaveState()
  - void RestoreState(IInferenceState state)
Cross-package deps: ECAssistantInference.Models

### Interface: IVisionEncoder
Implements: IDisposable
Properties:
  - string Marker { get; set; }
  - bool NeedsNonCausalAttention { get; set; }

### Class: InferenceException
> Thrown when a native ECAssistantInference call returns an error code.
Implements: Exception

### Class: NativeConversation
> Wraps a leased eci_conversation_t. Does NOT own the handle — the pool does.
Implements: IConversation
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Interop, ECAssistantInference.Models, ECAssistantInference.SafeHandles

### Class: NativeConversationPool
Implements: IConversationPool
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Interop, ECAssistantInference.SafeHandles

### Class: NativeConversationState
Implements: IConversationState
Cross-package deps: ECAssistantInference.Exceptions, ECAssistantInference.Abstractions, ECAssistantInference.Interop, ECAssistantInference.SafeHandles

### Class: NativeGrammar
Implements: IGrammar
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Interop, ECAssistantInference.SafeHandles

### Class: NativeInferenceBackend
> Wraps the backend safety layer (eci_probe_backend + queries).
Implements: IInferenceBackend
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Interop, ECAssistantInference.Models

### Class: NativeInferenceContext
> Wraps a native eci_context_t.
Implements: IInferenceContext
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Interop, ECAssistantInference.Models, ECAssistantInference.SafeHandles

### Class: NativeInferenceModel
> Wraps a native eci_model_t. Factory: Load().
Implements: IInferenceModel
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Interop, ECAssistantInference.Models, ECAssistantInference.SafeHandles

### Class: NativeInferenceState
> Wraps an executor state (eci_state_t from eci_state_save).
Implements: IInferenceState
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Interop, ECAssistantInference.SafeHandles

### Class: NativeStandardExecutor
Implements: IStandardExecutor
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Interop, ECAssistantInference.Models, ECAssistantInference.SafeHandles

### Class: NativeVisionEncoder
> Wraps an mmproj context for vision/image encoding.
Implements: IVisionEncoder
Cross-package deps: ECAssistantInference.Exceptions, ECAssistantInference.Abstractions, ECAssistantInference.Interop, ECAssistantInference.SafeHandles

### Class: SmokeTests
> Comprehensive test suite for ECAssistantInference C# bindings.
Cross-package deps: ECAssistantInference.Abstractions, ECAssistantInference.Exceptions, ECAssistantInference.Implementation, ECAssistantInference.Models, Xunit
