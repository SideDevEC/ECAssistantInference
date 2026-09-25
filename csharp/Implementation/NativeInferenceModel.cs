using System.Runtime.InteropServices;
using ECAssistantInference.Abstractions;
using ECAssistantInference.Exceptions;
using ECAssistantInference.Interop;
using ECAssistantInference.Models;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

/// <summary>Wraps a native eci_model_t. Factory: Load().</summary>
public sealed class NativeInferenceModel : IInferenceModel
{
    private readonly ModelSafeHandle _handle;
    private readonly ModelConfig _config;
    private bool _disposed;

    private NativeInferenceModel(ModelSafeHandle handle, ModelConfig config)
    {
        _handle = handle;
        _config = config;
    }

    /// <summary>Factory: loads a model from disk.</summary>
    public static NativeInferenceModel Load(ModelConfig config)
    {
        var pathPtr = Marshal.StringToHGlobalAnsi(config.Path);
        try
        {
            var nativeParams = new EciModelParams(pathPtr, config.GpuLayers, config.Threads,
                config.FlashAttention, (EciKvType)config.KvCacheType);
            EciNative.LoadModel(in nativeParams, out var raw).ThrowIfError();
            var handle = new ModelSafeHandle();
            handle.SetHandleSafe(raw);
            return new NativeInferenceModel(handle, config);
        }
        finally { Marshal.FreeHGlobal(pathPtr); }
    }

    public IInferenceContext CreateContext(ContextConfig config)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeParams = new EciContextParams(config.ContextSize, config.BatchSize,
            config.SeqMax, (EciPoolingType)config.PoolingType);
        EciNative.CreateContext(_handle.DangerousGetHandle(), in nativeParams, out var raw).ThrowIfError();
        var ctxHandle = new ContextSafeHandle();
        ctxHandle.SetHandleSafe(raw);
        return new NativeInferenceContext(ctxHandle, this);
    }

    public float[] GetEmbeddings(IInferenceContext context, string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var ctx = (NativeInferenceContext)context;
        EciNative.GetEmbeddings(_handle.DangerousGetHandle(), ctx.RawHandle, text,
            out var embPtr, out var dim).ThrowIfError();
        try
        {
            var result = new float[dim];
            Marshal.Copy(embPtr, result, 0, dim);
            return result;
        }
        finally { EciNative.FreeEmbeddings(embPtr); }
    }

    public int EmbeddingDimension
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return EciNative.EmbeddingDim(_handle.DangerousGetHandle());
        }
    }

    public IVisionEncoder LoadVisionEncoder(string mmprojPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.LoadMmproj(_handle.DangerousGetHandle(), mmprojPath, out var raw).ThrowIfError();
        var handle = new MmprojSafeHandle();
        handle.SetHandleSafe(raw);
        return new NativeVisionEncoder(handle);
    }

    internal IntPtr RawHandle => _handle.DangerousGetHandle();
    internal ModelConfig Config => _config;

    public string ApplyChatTemplate(string? template, IReadOnlyList<(string role, string content)> messages, bool addAssistant = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        // Marshal messages to eci_chat_message_t struct array
        // eci_chat_message_t is { const char* role; const char* content; } — blittable
        var nMsg = messages.Count;
        var msgSize = System.Runtime.InteropServices.Marshal.SizeOf<EciChatMessage>();
        var msgPtr = System.Runtime.InteropServices.Marshal.AllocHGlobal(nMsg * msgSize);
        try
        {
            var allocatedStrings = new List<IntPtr>();
            try
            {
                for (int i = 0; i < nMsg; i++)
                {
                    var (role, content) = messages[i];
                    var rolePtr = System.Runtime.InteropServices.Marshal.StringToHGlobalAnsi(role);
                    var contentPtr = System.Runtime.InteropServices.Marshal.StringToHGlobalAnsi(content);
                    allocatedStrings.Add(rolePtr);
                    allocatedStrings.Add(contentPtr);
                    var msg = new EciChatMessage { Role = rolePtr, Content = contentPtr };
                    System.Runtime.InteropServices.Marshal.StructureToPtr(msg, msgPtr + i * msgSize, false);
                }

                EciNative.ApplyChatTemplate(_handle.DangerousGetHandle(), template,
                    msgPtr, nMsg, addAssistant, out var textPtr).ThrowIfError();
                try
                {
                    return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(textPtr) ?? string.Empty;
                }
                finally { EciNative.FreeString(textPtr); }
            }
            finally
            {
                foreach (var p in allocatedStrings) System.Runtime.InteropServices.Marshal.FreeHGlobal(p);
            }
        }
        finally
        {
            System.Runtime.InteropServices.Marshal.FreeHGlobal(msgPtr);
        }
    }

    public IGrammar CreateGrammar(string grammarStr, string grammarRoot)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return NativeGrammar.Create(this, grammarStr, grammarRoot);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
