using System.Runtime.InteropServices;

namespace ECAssistantInference.Interop;

/// <summary>Blittable struct for eci_chat_message_t { const char* role; const char* content; }</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct EciChatMessage
{
    public IntPtr Role;
    public IntPtr Content;
}
