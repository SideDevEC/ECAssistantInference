using System.Runtime.InteropServices;
using System.Security;

namespace ECAssistantInference.Interop;

/// <summary>
/// Pure P/Invoke declarations for libecainference. This is the single static
/// class in the project (the sanctioned interop exception). EntryPoint matches
/// the C API snake_case names exactly.
/// </summary>
[SuppressUnmanagedCodeSecurity]
internal static class EciNative
{
    private const string Lib = "ecainference";

    // ── Result / errors ──

    [DllImport(Lib, EntryPoint = "eci_result_str", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr ResultStr(EciResult code);

    [DllImport(Lib, EntryPoint = "eci_free_string", CallingConvention = CallingConvention.Cdecl)]
    public static extern void FreeString(IntPtr str);

    [DllImport(Lib, EntryPoint = "eci_last_error", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr LastError(IntPtr context);

    // ── Model ──

    [DllImport(Lib, EntryPoint = "eci_load_model", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult LoadModel(in EciModelParams parameters, out IntPtr model);

    [DllImport(Lib, EntryPoint = "eci_free_model", CallingConvention = CallingConvention.Cdecl)]
    public static extern void FreeModel(IntPtr model);

    [DllImport(Lib, EntryPoint = "eci_load_mmproj", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult LoadMmproj(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string mmprojPath, out IntPtr mmproj);

    [DllImport(Lib, EntryPoint = "eci_free_mmproj", CallingConvention = CallingConvention.Cdecl)]
    public static extern void FreeMmproj(IntPtr mmproj);

    // ── Context ──

    [DllImport(Lib, EntryPoint = "eci_create_context", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult CreateContext(IntPtr model, in EciContextParams parameters, out IntPtr context);

    [DllImport(Lib, EntryPoint = "eci_free_context", CallingConvention = CallingConvention.Cdecl)]
    public static extern void FreeContext(IntPtr context);

    [DllImport(Lib, EntryPoint = "eci_context_size", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint ContextSize(IntPtr context);

    [DllImport(Lib, EntryPoint = "eci_context_seq_max", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint ContextSeqMax(IntPtr context);

    [DllImport(Lib, EntryPoint = "eci_context_n_ctx_used", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ContextNCtxUsed(IntPtr context, IntPtr conversation);

    // ── Tokenization ──

    [DllImport(Lib, EntryPoint = "eci_tokenize", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern EciResult Tokenize(IntPtr context, [MarshalAs(UnmanagedType.LPStr)] string text,
        [MarshalAs(UnmanagedType.U1)] bool addBos, [MarshalAs(UnmanagedType.U1)] bool parseSpecial,
        out IntPtr tokens, out int count);

    [DllImport(Lib, EntryPoint = "eci_free_tokens", CallingConvention = CallingConvention.Cdecl)]
    public static extern void FreeTokens(IntPtr tokens);

    [DllImport(Lib, EntryPoint = "eci_detokenize", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult Detokenize(IntPtr context, int[] tokens, int count,
        [MarshalAs(UnmanagedType.U1)] bool removeSpecial, out IntPtr outText);

    [DllImport(Lib, EntryPoint = "eci_token_to_piece", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult TokenToPiece(IntPtr context, int token, byte[] buffer, int bufferSize);

    [DllImport(Lib, EntryPoint = "eci_token_is_eos", CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool TokenIsEos(IntPtr context, int token);

    // ── Standard executor ──

    [DllImport(Lib, EntryPoint = "eci_executor_create", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ExecutorCreate(IntPtr context, out IntPtr executor);

    [DllImport(Lib, EntryPoint = "eci_executor_free", CallingConvention = CallingConvention.Cdecl)]
    public static extern void ExecutorFree(IntPtr executor);

    [DllImport(Lib, EntryPoint = "eci_executor_prompt", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern EciResult ExecutorPrompt(IntPtr executor, [MarshalAs(UnmanagedType.LPStr)] string text);

    [DllImport(Lib, EntryPoint = "eci_executor_prompt_tokens", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ExecutorPromptTokens(IntPtr executor, int[] tokens, int count);

    [DllImport(Lib, EntryPoint = "eci_executor_infer", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciDecodeResult ExecutorInfer(IntPtr executor);

    [DllImport(Lib, EntryPoint = "eci_executor_sample", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ExecutorSample(IntPtr executor, in EciSamplingParams parameters, out int token);

    [DllImport(Lib, EntryPoint = "eci_executor_token_count", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ExecutorTokenCount(IntPtr executor);

    [DllImport(Lib, EntryPoint = "eci_executor_rewind", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ExecutorRewind(IntPtr executor, int tokenCount);

    [DllImport(Lib, EntryPoint = "eci_executor_reset", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ExecutorReset(IntPtr executor);

    [DllImport(Lib, EntryPoint = "eci_executor_shift_left", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ExecutorShiftLeft(IntPtr executor, int tokenCount);

    // ── State save/restore ──

    [DllImport(Lib, EntryPoint = "eci_state_save", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult StateSave(IntPtr executor, out IntPtr state);

    [DllImport(Lib, EntryPoint = "eci_state_restore", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult StateRestore(IntPtr executor, IntPtr state);

    [DllImport(Lib, EntryPoint = "eci_state_free", CallingConvention = CallingConvention.Cdecl)]
    public static extern void StateFree(IntPtr state);

    [DllImport(Lib, EntryPoint = "eci_state_token_count", CallingConvention = CallingConvention.Cdecl)]
    public static extern int StateTokenCount(IntPtr state);

    // ── Conversation pool ──

    [DllImport(Lib, EntryPoint = "eci_pool_create", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult PoolCreate(IntPtr context, out IntPtr pool);

    [DllImport(Lib, EntryPoint = "eci_pool_free", CallingConvention = CallingConvention.Cdecl)]
    public static extern void PoolFree(IntPtr pool);

    [DllImport(Lib, EntryPoint = "eci_pool_lease", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult PoolLease(IntPtr pool, out IntPtr conversation);

    [DllImport(Lib, EntryPoint = "eci_pool_return", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult PoolReturn(IntPtr pool, IntPtr conversation);

    [DllImport(Lib, EntryPoint = "eci_pool_available", CallingConvention = CallingConvention.Cdecl)]
    public static extern int PoolAvailable(IntPtr pool);

    [DllImport(Lib, EntryPoint = "eci_pool_size", CallingConvention = CallingConvention.Cdecl)]
    public static extern uint PoolSize(IntPtr pool);

    // ── Conversation ──

    [DllImport(Lib, EntryPoint = "eci_conversation_prompt", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern EciResult ConversationPrompt(IntPtr conversation, [MarshalAs(UnmanagedType.LPStr)] string text);

    [DllImport(Lib, EntryPoint = "eci_conversation_prompt_tokens", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationPromptTokens(IntPtr conversation, int[] tokens, int count);

    [DllImport(Lib, EntryPoint = "eci_conversation_sample", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationSample(IntPtr conversation, in EciSamplingParams parameters, out int token);

    [DllImport(Lib, EntryPoint = "eci_conversation_token_count", CallingConvention = CallingConvention.Cdecl)]
    public static extern int ConversationTokenCount(IntPtr conversation);

    [DllImport(Lib, EntryPoint = "eci_conversation_rewind", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationRewind(IntPtr conversation, int tokenCount);

    [DllImport(Lib, EntryPoint = "eci_conversation_reset", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationReset(IntPtr conversation);

    [DllImport(Lib, EntryPoint = "eci_conversation_save", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationSave(IntPtr conversation, out IntPtr state);

    [DllImport(Lib, EntryPoint = "eci_conversation_restore", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationRestore(IntPtr pool, IntPtr conversation, IntPtr state);

    [DllImport(Lib, EntryPoint = "eci_conversation_state_free", CallingConvention = CallingConvention.Cdecl)]
    public static extern void ConversationStateFree(IntPtr state);

    [DllImport(Lib, EntryPoint = "eci_conversation_shift_left", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationShiftLeft(IntPtr conversation, int tokenCount);

    [DllImport(Lib, EntryPoint = "eci_conversation_load_image", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationLoadImage(IntPtr conversation, IntPtr mmproj, byte[] data, int size);

    [DllImport(Lib, EntryPoint = "eci_conversation_prompt_with_images", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern EciResult ConversationPromptWithImages(IntPtr conversation, IntPtr mmproj,
        [MarshalAs(UnmanagedType.LPStr)] string text,
        IntPtr[] imageData, int[] imageSizes, int nImages);

    // ── Batched inference ──

    [DllImport(Lib, EntryPoint = "eci_infer", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciDecodeResult Infer(IntPtr context);

    // ── Embeddings ──

    [DllImport(Lib, EntryPoint = "eci_get_embeddings", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern EciResult GetEmbeddings(IntPtr model, IntPtr context,
        [MarshalAs(UnmanagedType.LPStr)] string text, out IntPtr embeddings, out int dim);

    [DllImport(Lib, EntryPoint = "eci_free_embeddings", CallingConvention = CallingConvention.Cdecl)]
    public static extern void FreeEmbeddings(IntPtr embeddings);

    [DllImport(Lib, EntryPoint = "eci_embedding_dim", CallingConvention = CallingConvention.Cdecl)]
    public static extern int EmbeddingDim(IntPtr model);

    // ── Vision / MTMD ──

    [DllImport(Lib, EntryPoint = "eci_mtmd_marker", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult MtmdMarker(out IntPtr outMarker);

    [DllImport(Lib, EntryPoint = "eci_mtmd_marker_static", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr MtmdMarkerStatic();

    [DllImport(Lib, EntryPoint = "eci_mtmd_needs_non_causal", CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool MtmdNeedsNonCausal(IntPtr mmproj);

    // ── KV cache manipulation ──

    [DllImport(Lib, EntryPoint = "eci_kv_copy", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult KvCopy(IntPtr context, int srcSeqId, int dstSeqId);

    [DllImport(Lib, EntryPoint = "eci_kv_clear_seq", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult KvClearSeq(IntPtr context, int seqId);

    [DllImport(Lib, EntryPoint = "eci_kv_seq_tokens", CallingConvention = CallingConvention.Cdecl)]
    public static extern int KvSeqTokens(IntPtr context, int seqId);

    [DllImport(Lib, EntryPoint = "eci_set_causal_attn", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult SetCausalAttn(IntPtr context, [MarshalAs(UnmanagedType.U1)] bool causal);

    // ── Anti-prompt detection ──

    [DllImport(Lib, EntryPoint = "eci_check_anti_prompts", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int CheckAntiPrompts([MarshalAs(UnmanagedType.LPStr)] string text,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] antiPrompts, int count);

    // ── Backend safety ──

    [DllImport(Lib, EntryPoint = "eci_probe_backend", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ProbeBackend(IntPtr context);

    [DllImport(Lib, EntryPoint = "eci_create_context_safe", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult CreateContextSafe(ref IntPtr model, in EciModelParams modelParams,
        in EciContextParams ctxParams, int requestedGpuLayers, out IntPtr context);

    [DllImport(Lib, EntryPoint = "eci_vram_available", CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool VramAvailable(long neededBytes);

    [DllImport(Lib, EntryPoint = "eci_backend_probe_passed", CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool BackendProbePassed();

    [DllImport(Lib, EntryPoint = "eci_backend_name", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr BackendName();

    [DllImport(Lib, EntryPoint = "eci_backend_probe_error", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr BackendProbeError();

    [DllImport(Lib, EntryPoint = "eci_backend_needs_kv_flush", CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool BackendNeedsKvFlush();

    [DllImport(Lib, EntryPoint = "eci_backend_gpu_layers_reduced", CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool BackendGpuLayersReduced();

    [DllImport(Lib, EntryPoint = "eci_backend_effective_gpu_layers", CallingConvention = CallingConvention.Cdecl)]
    public static extern int BackendEffectiveGpuLayers();

    [DllImport(Lib, EntryPoint = "eci_backend_original_gpu_layers", CallingConvention = CallingConvention.Cdecl)]
    public static extern int BackendOriginalGpuLayers();

    [DllImport(Lib, EntryPoint = "eci_kv_flush", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult KvFlush(IntPtr context, int seqId);

    // ── Grammar (GBNF) ──

    [DllImport(Lib, EntryPoint = "eci_grammar_create", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern IntPtr GrammarCreate(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string grammarStr, [MarshalAs(UnmanagedType.LPStr)] string grammarRoot);

    [DllImport(Lib, EntryPoint = "eci_grammar_free", CallingConvention = CallingConvention.Cdecl)]
    public static extern void GrammarFree(IntPtr grammar);

    [DllImport(Lib, EntryPoint = "eci_executor_sample_grammar", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ExecutorSampleGrammar(IntPtr executor, in EciSamplingParams parameters, IntPtr grammar, out int token);

    [DllImport(Lib, EntryPoint = "eci_conversation_sample_grammar", CallingConvention = CallingConvention.Cdecl)]
    public static extern EciResult ConversationSampleGrammar(IntPtr conversation, in EciSamplingParams parameters, IntPtr grammar, out int token);

    // ── Chat template ──

    [DllImport(Lib, EntryPoint = "eci_apply_chat_template", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern EciResult ApplyChatTemplate(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string? tmpl,
        IntPtr messages, int nMessages, [MarshalAs(UnmanagedType.U1)] bool addAssistant, out IntPtr outText);

    // ── Vision on standard executor ──

    [DllImport(Lib, EntryPoint = "eci_executor_prompt_with_images", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern EciResult ExecutorPromptWithImages(IntPtr executor, IntPtr model,
        [MarshalAs(UnmanagedType.LPStr)] string text,
        IntPtr[] imageData, int[] imageSizes, int nImages);
}
