namespace ECAssistantInference.Interop;

/// <summary>Maps eci_result_t. C enums are int-sized; underlying type is explicit for clarity.</summary>
internal enum EciResult : int
{
    Ok = 0,
    ErrInvalidArg = 1,
    ErrInvalidHandle = 2,
    ErrLoadFailed = 3,
    ErrNoSlot = 4,
    ErrDecodeFailed = 5,
    ErrOverflow = 6,
    ErrNotSupported = 7,
    ErrInternal = 99,
}
