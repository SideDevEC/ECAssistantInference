namespace ECAssistantInference.Models;

/// <summary>Snapshot of backend probe results.</summary>
public sealed record BackendInfo
{
    public required string Name { get; init; }
    public bool ProbePassed { get; init; }
    public string? ProbeError { get; init; }
    public bool NeedsKvFlush { get; init; }
    public bool GpuLayersReduced { get; init; }
    public int EffectiveGpuLayers { get; init; }
    public int OriginalGpuLayers { get; init; }
}
