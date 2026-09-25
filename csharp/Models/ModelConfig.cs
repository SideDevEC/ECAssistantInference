using ECAssistantInference.Interop;

namespace ECAssistantInference.Models;

/// <summary>User-facing model configuration (not blittable — converted to EciModelParams internally).</summary>
public sealed record ModelConfig
{
    public required string Path { get; init; }
    public int GpuLayers { get; init; } = -1;      // -1 = all
    public int Threads { get; init; } = 0;          // 0 = auto
    public bool FlashAttention { get; init; } = true;
    public KvType KvCacheType { get; init; } = KvType.F16;
}

/// <summary>Managed enum mirroring EciKvType.</summary>
public enum KvType { F16 = 0, Q8_0 = 1, Q4_0 = 2, Q4_1 = 3 }
