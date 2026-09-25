using ECAssistantInference.Abstractions;
using ECAssistantInference.Exceptions;
using ECAssistantInference.Interop;
using ECAssistantInference.Models;
using ECAssistantInference.SafeHandles;

namespace ECAssistantInference.Implementation;

public sealed class NativeStandardExecutor : IStandardExecutor
{
    private readonly ExecutorSafeHandle _handle;
    private readonly NativeInferenceContext _context;
    private bool _disposed;

    internal NativeStandardExecutor(ExecutorSafeHandle handle, NativeInferenceContext context)
    {
        _handle = handle;
        _context = context;
    }

    internal IntPtr RawHandle => _handle.DangerousGetHandle();

    public int TokenCount => EciNative.ExecutorTokenCount(_handle.DangerousGetHandle());

    public void Prompt(string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ExecutorPrompt(_handle.DangerousGetHandle(), text).ThrowIfError();
    }

    public void PromptTokens(int[] tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ExecutorPromptTokens(_handle.DangerousGetHandle(), tokens, tokens.Length).ThrowIfError();
    }

    public void PromptWithImages(string text, IInferenceModel model, params byte[][] images)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeModel = (NativeInferenceModel)model;
        var nImages = images.Length;
        var ptrs = new IntPtr[nImages];
        var sizes = new int[nImages];
        try
        {
            for (int i = 0; i < nImages; i++)
            {
                sizes[i] = images[i].Length;
                ptrs[i] = System.Runtime.InteropServices.Marshal.AllocHGlobal(images[i].Length);
                System.Runtime.InteropServices.Marshal.Copy(images[i], 0, ptrs[i], images[i].Length);
            }
            EciNative.ExecutorPromptWithImages(_handle.DangerousGetHandle(), nativeModel.RawHandle,
                text, ptrs, sizes, nImages).ThrowIfError();
        }
        finally
        {
            foreach (var p in ptrs) if (p != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(p);
        }
    }

    public int SampleWithGrammar(SamplingConfig? config, IGrammar grammar)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cfg = config ?? new SamplingConfig();
        var nativeParams = ToNativeSampling(cfg);
        var nativeGrammar = (NativeGrammar)grammar;
        EciNative.ExecutorSampleGrammar(_handle.DangerousGetHandle(), in nativeParams, nativeGrammar.RawHandle, out var token).ThrowIfError();
        return token;
    }

    public InferResult Infer()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = EciNative.ExecutorInfer(_handle.DangerousGetHandle());
        return result switch
        {
            EciDecodeResult.Ok => InferResult.Ok,
            EciDecodeResult.NoWork => InferResult.NoWork,
            _ => InferResult.Failed,
        };
    }

    public int Sample(SamplingConfig? config = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cfg = config ?? new SamplingConfig();
        var nativeParams = ToNativeSampling(cfg);
        EciNative.ExecutorSample(_handle.DangerousGetHandle(), in nativeParams, out var token).ThrowIfError();
        return token;
    }

    public void Rewind(int tokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ExecutorRewind(_handle.DangerousGetHandle(), tokenCount).ThrowIfError();
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ExecutorReset(_handle.DangerousGetHandle()).ThrowIfError();
    }

    public void ShiftLeft(int tokenCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.ExecutorShiftLeft(_handle.DangerousGetHandle(), tokenCount).ThrowIfError();
    }

    public IInferenceState SaveState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EciNative.StateSave(_handle.DangerousGetHandle(), out var raw).ThrowIfError();
        var stateHandle = new StateSafeHandle();
        stateHandle.SetHandleSafe(raw);
        return new NativeInferenceState(stateHandle);
    }

    public void RestoreState(IInferenceState state)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var nativeState = (NativeInferenceState)state;
        EciNative.StateRestore(_handle.DangerousGetHandle(), nativeState.RawHandle).ThrowIfError();
    }

    internal static EciSamplingParams ToNativeSampling(SamplingConfig cfg)
    {
        return new EciSamplingParams(cfg.Temperature, cfg.TopP, cfg.TopK, cfg.MinP,
            cfg.RepeatPenalty, cfg.RepeatLastN, cfg.PenaltyPresent,
            cfg.MaxTokens, cfg.IgnoreEos);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }
}
