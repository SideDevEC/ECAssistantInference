using ECAssistantInference.Models;

namespace ECAssistantInference.Abstractions;

public interface IStandardExecutor : IDisposable
{
    int TokenCount { get; }
    void Prompt(string text);
    void PromptTokens(int[] tokens);
    void PromptWithImages(string text, IInferenceModel model, params byte[][] images);
    InferResult Infer();
    int Sample(SamplingConfig? config = null);
    int SampleWithGrammar(SamplingConfig? config, IGrammar grammar);
    void Rewind(int tokenCount);
    void Reset();
    void ShiftLeft(int tokenCount);
    IInferenceState SaveState();
    void RestoreState(IInferenceState state);
}
