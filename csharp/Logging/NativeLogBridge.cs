using System;
using System.Runtime.InteropServices;
using ECAssistantInference.Interop;

namespace ECAssistantInference.Logging;

/// <summary>
/// Bridges native C-level logging (eci_set_log_callback) to a managed callback.
/// Encapsulated in the Inference layer — no dependency on LLM or Core loggers.
/// The caller provides an Action&lt;int,string,string&gt; that forwards to whatever
/// logger they have (LLM's ServerLogger, Core's Logger, console, etc.).
/// </summary>
public static class NativeLogBridge
{
    // Keep the delegate alive for the process lifetime — GC would collect it otherwise.
    private static EciNative.LogDelegate? _pinned;

     /// <summary>Enable native logging. Pass null to disable (zero-overhead).</summary>
    public static void Enable(Action<int, string, string> handler)
     {
        if (handler == null)
         {
             EciNative.SetLogCallback(null);
            _pinned = null;
            return;
         }

        // Marshal the managed delegate to a native function pointer.
        var nativePtr = Marshal.GetFunctionPointerForDelegate(handler);
        _pinned = (EciNative.LogDelegate)Marshal.GetDelegateForFunctionPointer(nativePtr, typeof(EciNative.LogDelegate));

        EciNative.SetLogCallback(_pinned);
     }

     /// <summary>Set the minimum log level. 0=Debug, 1=Info, 2=Warn, 3=Error, 99=None.</summary>
    public static void SetLevel(int level) => EciNative.SetLogLevel(level);
}