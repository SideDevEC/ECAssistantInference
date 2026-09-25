using System.Runtime.InteropServices;

namespace ECAssistantInference.Interop;

/// <summary>Blittable mirror of eci_context_params_t.</summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct EciContextParams
{
    public readonly uint ContextSize;
    public readonly uint BatchSize;
    public readonly uint SeqMax;
    public readonly EciPoolingType PoolingType;

    public EciContextParams(uint contextSize, uint batchSize, uint seqMax, EciPoolingType poolingType)
    {
        ContextSize = contextSize;
        BatchSize = batchSize;
        SeqMax = seqMax;
        PoolingType = poolingType;
    }
}
