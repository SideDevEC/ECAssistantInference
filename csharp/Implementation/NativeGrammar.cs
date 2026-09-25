using ECAssistantInference.Abstractions;
using ECAssistantInference.Exceptions;
using ECAssistantInference.Interop;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

public sealed class NativeGrammar : IGrammar
{
    private readonly GrammarSafeHandle _handle;
    private bool _disposed;

    internal NativeGrammar(GrammarSafeHandle handle) { _handle = handle; }
    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    public static NativeGrammar Create(IInferenceModel model, string grammarStr, string grammarRoot)
    {
        var nativeModel = (NativeInferenceModel)model;
        var raw = EciNative.GrammarCreate(nativeModel.RawHandle, grammarStr, grammarRoot);
        if (raw == IntPtr.Zero)
            throw new InferenceException(1, "Failed to parse grammar");
        var handle = new GrammarSafeHandle();
        handle.SetHandleSafe(raw);
        return new NativeGrammar(handle);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
