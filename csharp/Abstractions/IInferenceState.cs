namespace ECAssistantInference.Abstractions;

public interface IInferenceState : IDisposable
{
    int TokenCount { get; }
}
