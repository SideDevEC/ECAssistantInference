namespace ECAssistantInference.Abstractions;

public interface IConversationPool : IDisposable
{
    IConversation Lease();
    void Return(IConversation conversation);
    int AvailableCount { get; }
    uint Size { get; }
}
