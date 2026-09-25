using ECAssistantInference.Exceptions;
using ECAssistantInference.Abstractions;
using ECAssistantInference.Interop;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

public sealed class NativeConversationState : IConversationState
{
    private readonly IntPtr _handle;
    private bool _disposed;

    internal NativeConversationState(IntPtr handle) { _handle = handle; }
    internal IntPtr RawHandle => _handle;

    public int TokenCount => EciNative.StateTokenCount(_handle);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_handle != IntPtr.Zero) EciNative.ConversationStateFree(_handle);
    }
}
