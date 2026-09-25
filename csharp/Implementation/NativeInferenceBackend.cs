using System.Runtime.InteropServices;
using ECAssistantInference.Abstractions;
using ECAssistantInference.Exceptions;
using ECAssistantInference.Interop;
using ECAssistantInference.Models;

namespace ECAssistantInference.Implementation;

/// <summary>Wraps the backend safety layer (eci_probe_backend + queries).</summary>
public sealed class NativeInferenceBackend : IInferenceBackend
{
    public BackendInfo Probe(IInferenceContext context)
    {
        var ctx = (NativeInferenceContext)context;
        EciNative.ProbeBackend(ctx.RawHandle).ThrowIfError();

        var namePtr = EciNative.BackendName();
        var name = namePtr != IntPtr.Zero ? Marshal.PtrToStringAnsi(namePtr) ?? "Unknown" : "Unknown";

        var errPtr = EciNative.BackendProbeError();
        var error = errPtr != IntPtr.Zero ? Marshal.PtrToStringAnsi(errPtr) : null;

        return new BackendInfo
        {
            Name = name,
            ProbePassed = EciNative.BackendProbePassed(),
            ProbeError = error,
            NeedsKvFlush = EciNative.BackendNeedsKvFlush(),
            GpuLayersReduced = EciNative.BackendGpuLayersReduced(),
            EffectiveGpuLayers = EciNative.BackendEffectiveGpuLayers(),
            OriginalGpuLayers = EciNative.BackendOriginalGpuLayers(),
        };
    }

    public bool VramAvailable(long neededBytes) => EciNative.VramAvailable(neededBytes);
}
