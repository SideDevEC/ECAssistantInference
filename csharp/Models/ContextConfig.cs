using ECAssistantInference.Interop;

namespace ECAssistantInference.Models;

/// <summary>User-facing context configuration.</summary>
public sealed record ContextConfig
{
    public uint ContextSize { get; init; } = 4096;
    public uint BatchSize { get; init; } = 512;
    public uint SeqMax { get; init; } = 4;
    public PoolingType PoolingType { get; init; } = PoolingType.None;

    /// <summary>
    /// llama.cpp fused gated-delta-net kernels (default true). Config-injected —
    /// replaces the former ECI_DISABLE_GDN environment variable; behavior is
    /// pinned and injected, never environment-dependent.
    /// </summary>
    public bool FusedGdn { get; init; } = true;
}

/// <summary>Managed enum mirroring EciPoolingType.</summary>
public enum PoolingType { None = 0, Mean = 1, Cls = 2, Last = 3 }
