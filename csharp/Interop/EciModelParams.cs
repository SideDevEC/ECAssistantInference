using System.Runtime.InteropServices;

namespace ECAssistantInference.Interop;

/// <summary>
/// Blittable mirror of eci_model_params_t. The C string is pinned by the caller
/// as an HGlobal Ansi pointer; ownership of the pointer stays with the caller.
/// bool is marshalled as a 1-byte field to match C99 stdbool.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct EciModelParams
{
    public readonly IntPtr ModelPath;
    public readonly int GpuLayers;
    public readonly int Threads;
    public readonly byte FlashAttn;
    public readonly EciKvType KvCacheType;

    public EciModelParams(IntPtr modelPath, int gpuLayers, int threads, bool flashAttn, EciKvType kvCacheType)
    {
        ModelPath = modelPath;
        GpuLayers = gpuLayers;
        Threads = threads;
        FlashAttn = flashAttn ? (byte)1 : (byte)0;
        KvCacheType = kvCacheType;
    }
}
