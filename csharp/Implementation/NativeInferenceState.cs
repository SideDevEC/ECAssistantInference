using ECAssistantInference.Abstractions;
using ECAssistantInference.Exceptions;
using ECAssistantInference.Interop;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

/// <summary>Wraps an executor state (eci_state_t from eci_state_save).</summary>
public sealed class NativeInferenceState : IInferenceState
{
    private readonly StateSafeHandle _handle;
    private bool _disposed;

    internal NativeInferenceState(StateSafeHandle handle) { _handle = handle; }
    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    public int TokenCount => EciNative.StateTokenCount(_handle.DangerousGetHandle());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
