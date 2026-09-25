using ECAssistantInference.Exceptions;
using System.Runtime.InteropServices;
using ECAssistantInference.Abstractions;
using ECAssistantInference.Interop;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

/// <summary>Wraps an mmproj context for vision/image encoding.</summary>
public sealed class NativeVisionEncoder : IVisionEncoder
{
    private readonly MmprojSafeHandle _handle;
    private bool _disposed;

    internal NativeVisionEncoder(MmprojSafeHandle handle) { _handle = handle; }
    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    public string Marker
    {
        get
        {
            var ptr = EciNative.MtmdMarkerStatic();
            return ptr != IntPtr.Zero ? Marshal.PtrToStringAnsi(ptr) ?? "<image>" : "<image>";
        }
    }

    public bool NeedsNonCausalAttention => EciNative.MtmdNeedsNonCausal(_handle.DangerousGetHandle());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
