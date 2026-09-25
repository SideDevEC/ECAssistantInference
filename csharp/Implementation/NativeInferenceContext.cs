using System.Runtime.InteropServices;
using ECAssistantInference.Abstractions;
using ECAssistantInference.Exceptions;
using ECAssistantInference.Interop;
using ECAssistantInference.Models;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

/// <summary>Wraps a native eci_context_t.</summary>
public sealed class NativeInferenceContext : IInferenceContext
{
    private readonly ContextSafeHandle _handle;
    private readonly NativeInferenceModel _model;
    private bool _disposed;

    internal NativeInferenceContext(ContextSafeHandle handle, NativeInferenceModel model)
    {
        _handle = handle;
        _model = model;
    }

    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    public uint ContextSize => EciNative.ContextSize(_handle.DangerousGetHandle());
    public uint SeqMax => EciNative.ContextSeqMax(_handle.DangerousGetHandle());

    public IStandardExecutor CreateExecutor()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ExecutorCreate(_handle.DangerousGetHandle(), out var raw).ThrowIfError();
        var execHandle = new ExecutorSafeHandle();
        execHandle.SetHandleSafe(raw);
        return new NativeStandardExecutor(execHandle, this);
    }

    public IConversationPool CreatePool()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.PoolCreate(_handle.DangerousGetHandle(), out var raw).ThrowIfError();
        var poolHandle = new PoolSafeHandle();
        poolHandle.SetHandleSafe(raw);
        return new NativeConversationPool(poolHandle, this);
    }

    public InferResult InferAll()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = EciNative.Infer(_handle.DangerousGetHandle());
        return result switch
        {
            EciDecodeResult.Ok => InferResult.Ok,
            EciDecodeResult.NoWork => InferResult.NoWork,
            _ => InferResult.Failed,
        };
    }

    public void KvCopy(int srcSeqId, int dstSeqId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.KvCopy(_handle.DangerousGetHandle(), srcSeqId, dstSeqId).ThrowIfError();
    }

    public void KvClearSeq(int seqId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.KvClearSeq(_handle.DangerousGetHandle(), seqId).ThrowIfError();
    }

    public void SetCausalAttention(bool causal)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.SetCausalAttn(_handle.DangerousGetHandle(), causal).ThrowIfError();
    }

    public int[] Tokenize(string text, bool addBos = false, bool parseSpecial = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.Tokenize(_handle.DangerousGetHandle(), text, addBos, parseSpecial,
            out var tokensPtr, out var count).ThrowIfError();
        try
        {
            if (count == 0) return Array.Empty<int>();
            var result = new int[count];
            Marshal.Copy(tokensPtr, result, 0, count);
            return result;
        }
        finally { EciNative.FreeTokens(tokensPtr); }
    }

    public string Detokenize(int[] tokens, bool removeSpecial = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.Detokenize(_handle.DangerousGetHandle(), tokens, tokens.Length,
            removeSpecial, out var textPtr).ThrowIfError();
        try
        {
            return Marshal.PtrToStringAnsi(textPtr) ?? string.Empty;
        }
        finally { EciNative.FreeString(textPtr); }
    }

    public string TokenToPiece(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var buffer = new byte[256];
        EciNative.TokenToPiece(_handle.DangerousGetHandle(), token, buffer, buffer.Length).ThrowIfError();
        var end = Array.IndexOf(buffer, (byte)0);
        if (end < 0) end = buffer.Length;
        return System.Text.Encoding.ASCII.GetString(buffer, 0, end);
    }

    public bool IsEos(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return EciNative.TokenIsEos(_handle.DangerousGetHandle(), token);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
