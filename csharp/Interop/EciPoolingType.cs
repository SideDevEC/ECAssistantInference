namespace ECAssistantInference.Interop;

/// <summary>Maps eci_pooling_type_t.</summary>
internal enum EciPoolingType : int
{
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
}
