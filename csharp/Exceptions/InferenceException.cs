using System.Runtime.InteropServices;

namespace ECAssistantInference.Exceptions;

/// <summary>Thrown when a native ECAssistantInference call returns an error code.</summary>
public sealed class InferenceException : Exception
{
    public int ErrorCode { get; }

    internal InferenceException(int code, string message) : base(message)
    {
        ErrorCode = code;
    }
}
