using ECAssistantInference.Models;

namespace ECAssistantInference.Abstractions;

public interface IStandardExecutor : IDisposable
{
    int TokenCount { get; }
    void Prompt(string text);
    void PromptTokens(int[] tokens);
    InferResult Infer();
    int Sample(SamplingConfig? config = null);
    void Rewind(int tokenCount);
    void Reset();
    void ShiftLeft(int tokenCount);
    IInferenceState SaveState();
    void RestoreState(IInferenceState state);
}
