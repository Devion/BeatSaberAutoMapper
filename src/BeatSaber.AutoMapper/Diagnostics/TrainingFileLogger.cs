using System.Text;

namespace BeatSaber.AutoMapper.Diagnostics;

/// <summary>
/// Thread-safe training event log writer.
/// Appends timestamped records to <c>training_log.txt</c> in the artifacts directory
/// so every dashboard event survives a session and can be inspected offline.
/// </summary>
public sealed class TrainingFileLogger : IDisposable
{
    private readonly StreamWriter _writer;
    private readonly object _lock = new();
    private bool _disposed;

    public TrainingFileLogger(string artifactsPath)
    {
        Directory.CreateDirectory(artifactsPath);
        string path = Path.Combine(artifactsPath, "training_log.txt");
        _writer = new StreamWriter(
            path,
            append: true,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false))
        {
            AutoFlush = true,
        };
        WriteLine("SESSION", $"=== Session started {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===");
    }

    /// <summary>Writes a single tagged, timestamped log line.</summary>
    public void Log(string tag, string message)
    {
        if (_disposed) return;
        WriteLine(tag, message);
    }

    private void WriteLine(string tag, string message)
    {
        string line = $"[{DateTime.Now:HH:mm:ss}] [{tag,-10}] {message}";
        lock (_lock)
        {
            if (!_disposed)
                _writer.WriteLine(line);
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            try
            {
                _writer.WriteLine(
                    $"[{DateTime.Now:HH:mm:ss}] [SESSION   ] === Session ended ===");
                _writer.Dispose();
            }
            catch { }
        }
    }
}
