using System.Runtime.InteropServices;
using ECAssistantInference.Interop;

namespace ECAssistantInference.SafeHandles;

internal sealed class StateSafeHandle : SafeHandle
{
    public StateSafeHandle() : base(IntPtr.Zero, true) { }
    public override bool IsInvalid => handle == IntPtr.Zero;
    internal void SetHandleSafe(IntPtr value) => SetHandle(value);
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero) EciNative.StateFree(handle);
        return true;
    }
}
