using ECAssistantInference.Abstractions;
using ECAssistantInference.Exceptions;
using ECAssistantInference.Interop;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

public sealed class NativeConversationPool : IConversationPool
{
    private readonly PoolSafeHandle _handle;
    private readonly NativeInferenceContext _context;
    private bool _disposed;

    internal NativeConversationPool(PoolSafeHandle handle, NativeInferenceContext context)
    {
        _handle = handle;
        _context = context;
    }

    internal IntPtr RawHandle => _handle.DangerousGetHandle();
    internal NativeInferenceContext Context => _context;

    public IConversation Lease()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.PoolLease(_handle.DangerousGetHandle(), out var raw).ThrowIfError();
        return new NativeConversation(raw, this);
    }

    public void Return(IConversation conversation)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var conv = (NativeConversation)conversation;
        EciNative.PoolReturn(_handle.DangerousGetHandle(), conv.RawHandle).ThrowIfError();
        conv.MarkReturned();
    }

    public int AvailableCount => EciNative.PoolAvailable(_handle.DangerousGetHandle());
    public uint Size => EciNative.PoolSize(_handle.DangerousGetHandle());

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
