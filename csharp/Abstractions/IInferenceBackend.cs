using ECAssistantInference.Models;

namespace ECAssistantInference.Abstractions;

public interface IInferenceBackend
{
    BackendInfo Probe(IInferenceContext context);
    bool VramAvailable(long neededBytes);
}
