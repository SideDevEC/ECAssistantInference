namespace ECAssistantInference.Models;

/// <summary>User-facing sampling configuration.</summary>
public sealed record SamplingConfig
{
    public float Temperature { get; init; } = 0.3f;
    public float TopP { get; init; } = 0.95f;
    public int TopK { get; init; } = 40;
    public float MinP { get; init; } = 0.0f;
    public float RepeatPenalty { get; init; } = 1.1f;
    public int RepeatLastN { get; init; } = -1;
    public float PenaltyPresent { get; init; } = 0.0f;
    public int MaxTokens { get; init; } = 256;
    public bool IgnoreEos { get; init; } = false;
}
