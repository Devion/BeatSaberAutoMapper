using System.Security.Cryptography;
using BeatSaber.AutoMapper.Audio;

namespace BeatSaber.AutoMapper.Training;

/// <summary>
/// Persists audio analysis results to disk so validation songs are not
/// re-analysed on every training run restart. Cache entries are invalidated
/// automatically when the source audio file's last-write time changes,
/// or when the schema version is bumped (e.g. after adding new feature arrays).
/// </summary>
internal sealed class ValidationAudioCache
{
    // Bump this string whenever the AudioAnalysisResult schema changes
    // (new arrays added, serialization format changed, etc.).
    // Any cache file whose Version != CacheVersion is automatically discarded.
    private const string CacheVersion = "v3";

    private readonly string _cacheDir;

    public ValidationAudioCache(string artifactsPath)
    {
        _cacheDir = Path.Combine(artifactsPath, "validation_audio_cache");
        Directory.CreateDirectory(_cacheDir);
    }

    /// <summary>Returns a cached result, or <c>null</c> if the entry is missing or stale.</summary>
    public AudioAnalysisResult? TryLoad(string audioFilePath)
    {
        string cachePath = CacheFilePath(audioFilePath);
        if (!File.Exists(cachePath)) return null;

        try
        {
            var envelope = JsonSerializer.Deserialize<CacheEnvelope>(
                File.ReadAllText(cachePath), SerializerOptions);
            if (envelope?.Result is null) return null;
            if (envelope.Version != CacheVersion) return null;

            long lastWrite = File.GetLastWriteTimeUtc(audioFilePath).Ticks;
            if (envelope.SourceLastWriteTicks != lastWrite) return null;

            return envelope.Result;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Persists an analysis result. Write failures are non-fatal.</summary>
    public void Store(string audioFilePath, AudioAnalysisResult result)
    {
        try
        {
            var envelope = new CacheEnvelope
            {
                Version              = CacheVersion,
                SourceLastWriteTicks = File.GetLastWriteTimeUtc(audioFilePath).Ticks,
                Result               = result,
            };
            File.WriteAllText(CacheFilePath(audioFilePath),
                JsonSerializer.Serialize(envelope, SerializerOptions));
        }
        catch { /* non-fatal */ }
    }

    // -----------------------------------------------------------------------

    private string CacheFilePath(string audioFilePath)
    {
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(Path.GetFullPath(audioFilePath)));
        return Path.Combine(_cacheDir, Convert.ToHexString(hash)[..16] + ".json");
    }

    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented        = false,
        Converters           = { new JsonStringEnumConverter() },
        IncludeFields        = false,
    };

    private sealed class CacheEnvelope
    {
        public string?            Version              { get; set; }
        public long               SourceLastWriteTicks { get; set; }
        public AudioAnalysisResult? Result             { get; set; }
    }
}
