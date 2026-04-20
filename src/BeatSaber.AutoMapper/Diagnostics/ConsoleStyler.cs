using System.Runtime.InteropServices;

namespace BeatSaber.AutoMapper.Diagnostics;

public static class ConsoleStyler
{
    private const int StdOutputHandle = -11;
    private const uint EnableVirtualTerminalProcessing = 0x0004;

    public static bool Enabled { get; private set; } = !Console.IsOutputRedirected;

    public static void Initialize(bool enabled)
    {
        if (!enabled || Console.IsOutputRedirected)
        {
            Enabled = false;
            return;
        }

        if (!OperatingSystem.IsWindows())
        {
            Enabled = true;
            return;
        }

        Enabled = TryEnableVirtualTerminalProcessing();
    }

    public static string Colorize(string text, ConsoleColor color)
    {
        if (!Enabled)
            return text;

        string code = color switch
        {
            ConsoleColor.Red => "31",
            ConsoleColor.Green => "32",
            ConsoleColor.Yellow => "33",
            ConsoleColor.Blue => "34",
            ConsoleColor.Magenta => "35",
            ConsoleColor.Cyan => "36",
            ConsoleColor.White => "37",
            ConsoleColor.DarkYellow => "33",
            ConsoleColor.DarkRed => "31",
            ConsoleColor.DarkGreen => "32",
            ConsoleColor.DarkCyan => "36",
            _ => "37"
        };

        return $"\u001b[{code}m{text}\u001b[0m";
    }

    private static bool TryEnableVirtualTerminalProcessing()
    {
        IntPtr handle = GetStdHandle(StdOutputHandle);
        if (handle == IntPtr.Zero || handle == new IntPtr(-1))
            return false;

        if (!GetConsoleMode(handle, out uint mode))
            return false;

        if ((mode & EnableVirtualTerminalProcessing) != 0)
            return true;

        return SetConsoleMode(handle, mode | EnableVirtualTerminalProcessing);
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetStdHandle(int nStdHandle);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
}
