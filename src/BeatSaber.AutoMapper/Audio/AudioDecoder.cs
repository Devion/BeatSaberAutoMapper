using NAudio.Wave;
using NVorbis;

namespace BeatSaber.AutoMapper.Audio;

public sealed class AudioDecoder
{
    /// <summary>
    /// Decode an audio file to mono PCM float samples.
    /// Supports MP3, WAV (via NAudio) and OGG Vorbis (via NVorbis).
    /// </summary>
    public static (float[] Samples, int SampleRate, double Duration) Decode(
        string filePath,
        int targetSampleRate = 22050,
        bool mono = true)
    {
        Guard.NotNullOrEmpty(filePath, nameof(filePath));
        string ext = Path.GetExtension(filePath).ToLowerInvariant();

        return ext switch
        {
            ".ogg" or ".egg" => DecodeOggWithFallback(filePath, targetSampleRate, mono),
            _ => DecodeNAudio(filePath, targetSampleRate, mono)
        };
    }

    public static async Task<(float[] Samples, int SampleRate, double Duration)> DecodeAsync(
        string filePath, int targetSampleRate = 22050, bool mono = true)
    {
        // Offload to thread-pool since audio decoding is CPU-bound I/O
        return await Task.Run(() => Decode(filePath, targetSampleRate, mono)).ConfigureAwait(false);
    }

    // -----------------------------------------------------------------------

    private static (float[] Samples, int SampleRate, double Duration) DecodeNAudio(
        string filePath, int targetSampleRate, bool mono)
    {
        using var reader = CreateReader(filePath);
        var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(targetSampleRate, mono ? 1 : reader.WaveFormat.Channels);
        using var resampler = new MediaFoundationResampler(reader, waveFormat);
        resampler.ResamplerQuality = 60;

        var samples = new List<float>();
        var buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = resampler.Read(buffer, 0, buffer.Length)) > 0)
        {
            int floatCount = bytesRead / 4;
            for (int i = 0; i < floatCount; i++)
                samples.Add(BitConverter.ToSingle(buffer, i * 4));
        }

        float[] arr = samples.ToArray();
        double duration = arr.Length / (double)(targetSampleRate * (mono ? 1 : reader.WaveFormat.Channels));
        return (arr, targetSampleRate, duration);
    }

    private static WaveStream CreateReader(string filePath)
    {
        string ext = Path.GetExtension(filePath).ToLowerInvariant();
        return ext switch
        {
            ".mp3" => new Mp3FileReader(filePath),
            ".wav" => new WaveFileReader(filePath),
            ".aiff" or ".aif" => new AiffFileReader(filePath),
            _ => new AudioFileReader(filePath)
        };
    }

    private static (float[] Samples, int SampleRate, double Duration) DecodeOggWithFallback(
        string filePath, int targetSampleRate, bool mono)
    {
        try
        {
            return DecodeOgg(filePath, targetSampleRate, mono);
        }
        catch (Exception ex)
        {
            var converter = new AudioConverter();
            if (!converter.IsFfmpegAvailable())
                throw new InvalidOperationException(
                    $"Failed to decode OGG/EGG directly ({ex.Message}). " +
                    "This file may contain a non-Vorbis or mixed Ogg bitstream, and ffmpeg is not available for fallback decoding.",
                    ex);

            return DecodeWithFfmpegFallback(filePath, targetSampleRate, mono, ex.Message);
        }
    }

    private static (float[] Samples, int SampleRate, double Duration) DecodeOgg(
        string filePath, int targetSampleRate, bool mono)
    {
        using var reader = new VorbisReader(filePath);
        int srcChannels = reader.Channels;
        int srcRate = reader.SampleRate;
        double duration = reader.TotalTime.TotalSeconds;

        var rawSamples = new List<float>();
        var buffer = new float[4096];
        int samplesRead;
        while ((samplesRead = reader.ReadSamples(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < samplesRead; i++)
                rawSamples.Add(buffer[i]);
        }

        float[] srcArr = rawSamples.ToArray();

        // Down-mix to mono if needed
        if (mono && srcChannels > 1)
            srcArr = DownMixToMono(srcArr, srcChannels);

        // Simple linear resampling if rates differ
        if (srcRate != targetSampleRate)
            srcArr = Resample(srcArr, srcRate, targetSampleRate);

        return (srcArr, targetSampleRate, duration);
    }

    private static (float[] Samples, int SampleRate, double Duration) DecodeWithFfmpegFallback(
        string filePath,
        int targetSampleRate,
        bool mono,
        string originalError)
    {
        string tempWav = Path.Combine(Path.GetTempPath(), $"bsam_decode_{Guid.NewGuid():N}.wav");
        try
        {
            var converter = new AudioConverter();
            converter.ConvertToWav(filePath, tempWav, targetSampleRate);
            return DecodeNAudio(tempWav, targetSampleRate, mono);
        }
        catch (Exception ffmpegEx)
        {
            throw new InvalidOperationException(
                $"Failed to decode OGG/EGG directly ({originalError}) and ffmpeg fallback also failed ({ffmpegEx.Message}).",
                ffmpegEx);
        }
        finally
        {
            try
            {
                if (File.Exists(tempWav))
                    File.Delete(tempWav);
            }
            catch
            {
            }
        }
    }

    private static float[] DownMixToMono(float[] interleaved, int channels)
    {
        int frames = interleaved.Length / channels;
        var mono = new float[frames];
        for (int i = 0; i < frames; i++)
        {
            float sum = 0f;
            for (int c = 0; c < channels; c++)
                sum += interleaved[i * channels + c];
            mono[i] = sum / channels;
        }
        return mono;
    }

    private static float[] Resample(float[] src, int srcRate, int dstRate)
    {
        double ratio = (double)dstRate / srcRate;
        int dstLen = (int)(src.Length * ratio);
        var dst = new float[dstLen];
        for (int i = 0; i < dstLen; i++)
        {
            double srcIdx = i / ratio;
            int lo = (int)srcIdx;
            int hi = Math.Min(lo + 1, src.Length - 1);
            double t = srcIdx - lo;
            dst[i] = (float)(src[lo] * (1 - t) + src[hi] * t);
        }
        return dst;
    }
}
