using ECAssistantInference.Interop;
using System.Runtime.InteropServices;

namespace ECAssistantInference.Exceptions;

/// <summary>Extension methods for EciResult error handling.</summary>
internal static class EciResultExtensions
{
    internal static void ThrowIfError(this EciResult result, IntPtr contextHandle = default)
    {
        if (result == EciResult.Ok) return;
        var strPtr = EciNative.ResultStr(result);
        var baseMsg = strPtr != IntPtr.Zero ? Marshal.PtrToStringAnsi(strPtr) ?? "unknown" : "unknown";
        if (contextHandle != default)
        {
            var errPtr = EciNative.LastError(contextHandle);
            if (errPtr != IntPtr.Zero)
            {
                var detail = Marshal.PtrToStringAnsi(errPtr);
                if (!string.IsNullOrEmpty(detail))
                    baseMsg = $"{baseMsg}: {detail}";
            }
        }
        throw new InferenceException((int)result, baseMsg);
    }
}
