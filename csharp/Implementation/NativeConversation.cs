using System.Runtime.InteropServices;
using ECAssistantInference.Abstractions;
using ECAssistantInference.Exceptions;
using ECAssistantInference.Interop;
using ECAssistantInference.Models;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

/// <summary>
/// Wraps a leased eci_conversation_t. Does NOT own the handle — the pool does.
/// Dispose() returns it to the pool if not already returned.
/// </summary>
public sealed class NativeConversation : IConversation
{
    private readonly IntPtr _handle;
    private readonly NativeConversationPool _pool;
    private bool _returned;
    private bool _disposed;

    internal NativeConversation(IntPtr handle, NativeConversationPool pool)
    {
        _handle = handle;
        _pool = pool;
    }

    internal IntPtr RawHandle => _handle;

    internal void MarkReturned() => _returned = true;

    public int TokenCount => EciNative.ConversationTokenCount(_handle);

    public void Prompt(string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ConversationPrompt(_handle, text).ThrowIfError();
    }

    public void PromptTokens(int[] tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ConversationPromptTokens(_handle, tokens, tokens.Length).ThrowIfError();
    }

    public int Sample(SamplingConfig? config = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cfg = config ?? new SamplingConfig();
        var nativeParams = NativeStandardExecutor.ToNativeSampling(cfg);
        EciNative.ConversationSample(_handle, in nativeParams, out var token).ThrowIfError();
        return token;
    }

    public int SampleWithGrammar(SamplingConfig? config, IGrammar grammar)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cfg = config ?? new SamplingConfig();
        var nativeParams = NativeStandardExecutor.ToNativeSampling(cfg);
        var nativeGrammar = (NativeGrammar)grammar;
        EciNative.ConversationSampleGrammar(_handle, in nativeParams, nativeGrammar.RawHandle, out var token).ThrowIfError();
        return token;
    }

    public void Rewind(int tokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ConversationRewind(_handle, tokenCount).ThrowIfError();
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ConversationReset(_handle).ThrowIfError();
    }

    public void ResetGrammarState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ConversationGrammarReset(_handle).ThrowIfError();
    }

    public void ShiftLeft(int tokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ConversationShiftLeft(_handle, tokenCount).ThrowIfError();
    }

    public IConversationState SaveState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ConversationSave(_handle, out var raw).ThrowIfError();
        return new NativeConversationState(raw);
    }

    public void RestoreState(IConversationPool pool, IConversationState state)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeState = (NativeConversationState)state;
        var nativePool = (NativeConversationPool)pool;
        EciNative.ConversationRestore(nativePool.RawHandle, _handle, nativeState.RawHandle).ThrowIfError();
    }

    public void PromptWithImages(string text, IVisionEncoder vision, params byte[][] images)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeVision = (NativeVisionEncoder)vision;
        var nImages = images.Length;
        var ptrs = new IntPtr[nImages];
        var sizes = new int[nImages];
        try
        {
            for (int i = 0; i < nImages; i++)
            {
                sizes[i] = images[i].Length;
                ptrs[i] = Marshal.AllocHGlobal(images[i].Length);
                Marshal.Copy(images[i], 0, ptrs[i], images[i].Length);
            }
            EciNative.ConversationPromptWithImages(_handle, nativeVision.RawHandle,
                text, ptrs, sizes, nImages).ThrowIfError();
        }
        finally
        {
            foreach (var p in ptrs) if (p != IntPtr.Zero) Marshal.FreeHGlobal(p);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (!_returned) _pool.Return(this);
    }
}
