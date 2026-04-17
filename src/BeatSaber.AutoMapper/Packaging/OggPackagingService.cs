namespace BeatSaber.AutoMapper.Packaging;

public sealed class OggPackagingService
{
    private readonly Audio.AudioConverter _converter;

    public OggPackagingService(string ffmpegPath = "ffmpeg")
    {
        _converter = new Audio.AudioConverter { FfmpegPath = ffmpegPath };
    }

    /// <summary>
    /// Prepare the audio file as an OGG Vorbis file for a Beat Saber map.
    /// Returns the path to the OGG file.
    /// </summary>
    public string PrepareAudio(string inputAudioPath, string outputFolder, string outputName = "song")
    {
        Guard.NotNullOrEmpty(inputAudioPath, nameof(inputAudioPath));
        Guard.NotNullOrEmpty(outputFolder, nameof(outputFolder));

        Directory.CreateDirectory(outputFolder);
        string ext = Path.GetExtension(inputAudioPath).ToLowerInvariant();

        if (ext == ".ogg")
        {
            // Already OGG: just copy
            string dest = Path.Combine(outputFolder, $"{outputName}.ogg");
            File.Copy(inputAudioPath, dest, overwrite: true);
            return dest;
        }

        string outputPath = Path.Combine(outputFolder, $"{outputName}.ogg");
        if (_converter.IsFfmpegAvailable())
            return _converter.ConvertToOgg(inputAudioPath, outputPath);

        throw new InvalidOperationException(
            "ffmpeg is required to convert audio to OGG format but was not found. " +
            "Install ffmpeg or use an OGG input file directly.");
    }
}
