namespace ECAssistantInference.Abstractions;

public interface IConversationState : IDisposable
{
    int TokenCount { get; }
}
