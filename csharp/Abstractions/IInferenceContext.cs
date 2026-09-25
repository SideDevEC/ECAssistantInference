using ECAssistantInference.Models;
using ECAssistantInference.Abstractions;

namespace ECAssistantInference.Abstractions;

public interface IInferenceContext : IDisposable
{
    uint ContextSize { get; }
    uint SeqMax { get; }
    IStandardExecutor CreateExecutor();
    IConversationPool CreatePool();
    InferResult InferAll();
    void KvCopy(int srcSeqId, int dstSeqId);
    void KvClearSeq(int seqId);
    void SetCausalAttention(bool causal);
    int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true);
    string Detokenize(int[] tokens, bool removeSpecial = false);
    string TokenToPiece(int token);
    bool IsEos(int token);
}
