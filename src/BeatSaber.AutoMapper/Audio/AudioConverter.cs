using System.Diagnostics;

namespace BeatSaber.AutoMapper.Audio;

public sealed class AudioConverter
{
    public string FfmpegPath { get; set; } = "ffmpeg";

    /// <summary>Convert audio to OGG Vorbis format using ffmpeg.</summary>
    public string ConvertToOgg(string inputPath, string outputPath, int quality = 6)
    {
        Guard.NotNullOrEmpty(inputPath, nameof(inputPath));
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));
        RunFfmpeg($"-i \"{inputPath}\" -c:a libvorbis -q:a {quality} -y \"{outputPath}\"");
        return outputPath;
    }

    /// <summary>Convert audio to WAV for internal processing using ffmpeg.</summary>
    public string ConvertToWav(string inputPath, string outputPath, int sampleRate = 44100)
    {
        Guard.NotNullOrEmpty(inputPath, nameof(inputPath));
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));
        RunFfmpeg($"-i \"{inputPath}\" -ar {sampleRate} -ac 1 -f wav -y \"{outputPath}\"");
        return outputPath;
    }

    /// <summary>Returns true if ffmpeg is available on the system path or at the configured path.</summary>
    public bool IsFfmpegAvailable()
    {
        try
        {
            using var proc = new Process
            {
                StartInfo = new ProcessStartInfo(FfmpegPath, "-version")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };
            proc.Start();
            proc.WaitForExit(3000);
            return proc.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private void RunFfmpeg(string args)
    {
        using var proc = new Process
        {
            StartInfo = new ProcessStartInfo(FfmpegPath, args)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };
        proc.Start();
        proc.WaitForExit();
        if (proc.ExitCode != 0)
        {
            string err = proc.StandardError.ReadToEnd();
            throw new InvalidOperationException($"ffmpeg exited with code {proc.ExitCode}: {err}");
        }
    }
}
