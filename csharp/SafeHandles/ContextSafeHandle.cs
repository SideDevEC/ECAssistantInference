using System.Runtime.InteropServices;
using ECAssistantInference.Interop;

namespace ECAssistantInference.SafeHandles;

internal sealed class ContextSafeHandle : SafeHandle
{
    public ContextSafeHandle() : base(IntPtr.Zero, true) { }
    public override bool IsInvalid => handle == IntPtr.Zero;
    internal void SetHandleSafe(IntPtr value) => SetHandle(value);
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero) EciNative.FreeContext(handle);
        return true;
    }
}
