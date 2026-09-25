using ECAssistantInference.Models;

namespace ECAssistantInference.Abstractions;

public interface IInferenceModel : IDisposable
{
    IInferenceContext CreateContext(ContextConfig config);
    float[] GetEmbeddings(IInferenceContext context, string text);
    int EmbeddingDimension { get; }
    IVisionEncoder LoadVisionEncoder(string mmprojPath);
    string ApplyChatTemplate(string? template, IReadOnlyList<(string role, string content)> messages, bool addAssistant = true);
    IGrammar CreateGrammar(string grammarStr, string grammarRoot);
}
