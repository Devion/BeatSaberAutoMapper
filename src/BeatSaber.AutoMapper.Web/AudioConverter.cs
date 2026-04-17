using NAudio.Wave;
using OggVorbisEncoder;

namespace BeatSaber.AutoMapper.Web;

/// <summary>
/// Converts uploaded audio (MP3/WAV/etc.) to OGG Vorbis bytes suitable for a
/// Beat Saber .egg audio file. If the source is already OGG/EGG it is returned
/// as-is without any re-encoding.
/// </summary>
public static class AudioConverter
{
    private const float OggQuality     = 0.5f;  // ~160 kbps VBR
    private const int   ChunkFrames    = 1024;

    /// <summary>
    /// Reads <paramref name="inputPath"/> and returns OGG Vorbis bytes.
    /// </summary>
    public static byte[] ToOgg(string inputPath)
    {
        string ext = Path.GetExtension(inputPath).ToLowerInvariant();

        // OGG/EGG are already Vorbis — copy directly.
        if (ext is ".ogg" or ".egg")
            return File.ReadAllBytes(inputPath);

        // Decode to interleaved float PCM via NAudio.
        float[] interleaved;
        int sampleRate, channels;

        using (var reader = new AudioFileReader(inputPath))
        {
            sampleRate = reader.WaveFormat.SampleRate;
            channels   = reader.WaveFormat.Channels;

            var buf  = new float[4096 * channels];
            var list = new List<float>();
            int read;
            while ((read = reader.Read(buf, 0, buf.Length)) > 0)
                list.AddRange(buf.AsSpan(0, read));

            interleaved = [.. list];
        }

        // De-interleave into [channel][frame].
        int frameCount = interleaved.Length / channels;
        var pcm = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            pcm[c] = new float[frameCount];
            for (int f = 0; f < frameCount; f++)
                pcm[c][f] = interleaved[f * channels + c];
        }

        return EncodeOgg(pcm, channels, sampleRate, frameCount);
    }

    private static byte[] EncodeOgg(float[][] pcm, int channels, int sampleRate, int frameCount)
    {
        var info     = VorbisInfo.InitVariableBitRate(channels, sampleRate, OggQuality);
        var comments = new Comments();
        comments.AddTag("ENCODER", "BeatSaberAutoMapper");

        var output    = new MemoryStream();
        var oggStream = new OggStream(Random.Shared.Next());
        var state     = ProcessingState.Create(info);

        // Write the three Vorbis header packets.
        oggStream.PacketIn(HeaderPacketBuilder.BuildInfoPacket(info));
        oggStream.PacketIn(HeaderPacketBuilder.BuildCommentsPacket(comments));
        oggStream.PacketIn(HeaderPacketBuilder.BuildBooksPacket(info));
        FlushPages(oggStream, output, force: true);

        // Feed audio in chunks, then flush end-of-stream.
        int pos = 0;
        while (true)
        {
            bool eos = pos >= frameCount;
            if (eos)
            {
                state.WriteEndOfStream();
            }
            else
            {
                int count = Math.Min(ChunkFrames, frameCount - pos);
                var chunk = new float[channels][];
                for (int c = 0; c < channels; c++)
                {
                    chunk[c] = new float[ChunkFrames];
                    Array.Copy(pcm[c], pos, chunk[c], 0, count);
                }
                state.WriteData(chunk, count);
                pos += count;
            }

            while (state.PacketOut(out var packet))
            {
                oggStream.PacketIn(packet);
                FlushPages(oggStream, output, force: false);
            }

            if (eos) break;
        }

        FlushPages(oggStream, output, force: true);
        return output.ToArray();
    }

    private static void FlushPages(OggStream stream, Stream output, bool force)
    {
        while (stream.PageOut(out var page, force))
        {
            output.Write(page.Header, 0, page.Header.Length);
            output.Write(page.Body,   0, page.Body.Length);
        }
    }
}
