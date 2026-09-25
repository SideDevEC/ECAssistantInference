using ECAssistantInference.Models;

namespace ECAssistantInference.Abstractions;

public interface IConversation : IDisposable
{
    int TokenCount { get; }
    void Prompt(string text);
    void PromptTokens(int[] tokens);
    int Sample(SamplingConfig? config = null);
    int SampleWithGrammar(SamplingConfig? config, IGrammar grammar);
    void Rewind(int tokenCount);
    void Reset();
    /// <summary>Resets grammar sampler state — call once per generation before a grammar-constrained sample loop.</summary>
    void ResetGrammarState();
    void ShiftLeft(int tokenCount);
    IConversationState SaveState();
    void RestoreState(IConversationPool pool, IConversationState state);
    void PromptWithImages(string text, IVisionEncoder vision, params byte[][] images);
}
