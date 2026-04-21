using System.Security.Cryptography;
using System.Text;
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
    private readonly string[] _fallbackCacheDirs;

    public ValidationAudioCache(
        string cachePath,
        bool exactDirectory = false,
        params string[] fallbackCacheDirs)
    {
        _cacheDir = exactDirectory
            ? cachePath
            : Path.Combine(cachePath, "validation_audio_cache");
        _fallbackCacheDirs = fallbackCacheDirs
            .Where(d => !string.IsNullOrWhiteSpace(d))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        Directory.CreateDirectory(_cacheDir);
    }

    /// <summary>Returns a cached result, or <c>null</c> if the entry is missing or stale.</summary>
    public AudioAnalysisResult? TryLoad(string audioFilePath)
    {
        string stableCachePath = StableCacheFilePath(audioFilePath, _cacheDir);
        foreach (var candidate in EnumerateCandidateCachePaths(audioFilePath))
        {
            if (!File.Exists(candidate.Path)) continue;

            try
            {
                var envelope = JsonSerializer.Deserialize<CacheEnvelope>(
                    File.ReadAllText(candidate.Path), SerializerOptions);
                if (envelope?.Result is null) continue;
                if (envelope.Version != CacheVersion) continue;

                long lastWrite = File.GetLastWriteTimeUtc(audioFilePath).Ticks;
                if (envelope.SourceLastWriteTicks != lastWrite) continue;

                if (!candidate.IsStable)
                    PromoteLegacyEntry(stableCachePath, envelope);

                return envelope.Result;
            }
            catch
            {
                // Keep searching fallback cache locations.
            }
        }

        return null;
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
            File.WriteAllText(StableCacheFilePath(audioFilePath, _cacheDir),
                JsonSerializer.Serialize(envelope, SerializerOptions));
        }
        catch { /* non-fatal */ }
    }

    // -----------------------------------------------------------------------

    private IEnumerable<CacheCandidate> EnumerateCandidateCachePaths(string audioFilePath)
    {
        foreach (string cacheDir in EnumerateCacheDirs())
        {
            yield return new CacheCandidate(StableCacheFilePath(audioFilePath, cacheDir), IsStable: true);

            foreach (string legacyAudioPath in EnumerateLegacyAudioPathAliases(audioFilePath))
                yield return new CacheCandidate(LegacyPathCacheFilePath(legacyAudioPath, cacheDir), IsStable: false);
        }
    }

    private IEnumerable<string> EnumerateCacheDirs()
    {
        yield return _cacheDir;
        foreach (var dir in _fallbackCacheDirs)
            yield return dir;
    }

    private static string StableCacheFilePath(string audioFilePath, string cacheDir)
    {
        string cacheKey = ComputeStableCacheKey(audioFilePath);
        return Path.Combine(cacheDir, cacheKey + ".json");
    }

    private static string LegacyPathCacheFilePath(string audioFilePath, string cacheDir)
    {
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(Path.GetFullPath(audioFilePath)));
        return Path.Combine(cacheDir, Convert.ToHexString(hash)[..16] + ".json");
    }

    private static string ComputeStableCacheKey(string audioFilePath)
    {
        var fileInfo = new FileInfo(audioFilePath);
        using var stream = File.OpenRead(audioFilePath);
        using var hasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        Span<byte> head = stackalloc byte[64 * 1024];
        int headRead = stream.Read(head);
        hasher.AppendData(Encoding.UTF8.GetBytes(Path.GetFileName(audioFilePath)));
        hasher.AppendData(Encoding.UTF8.GetBytes("|"));
        hasher.AppendData(BitConverter.GetBytes(fileInfo.Length));
        hasher.AppendData(Encoding.UTF8.GetBytes("|"));
        hasher.AppendData(head[..headRead]);

        Span<byte> tail = stackalloc byte[64 * 1024];
        if (stream.Length > tail.Length)
        {
            stream.Seek(-tail.Length, SeekOrigin.End);
            int tailRead = stream.Read(tail);
            hasher.AppendData(Encoding.UTF8.GetBytes("|"));
            hasher.AppendData(tail[..tailRead]);
        }

        byte[] hash = hasher.GetHashAndReset();
        return Convert.ToHexString(hash)[..16];
    }

    private static IEnumerable<string> EnumerateLegacyAudioPathAliases(string audioFilePath)
    {
        yield return audioFilePath;

        string fullPath = Path.GetFullPath(audioFilePath);
        string? collapsedBsamPath = CollapseBsamSegment(fullPath);
        if (!string.Equals(collapsedBsamPath, fullPath, StringComparison.OrdinalIgnoreCase))
            yield return collapsedBsamPath;
    }

    private static string CollapseBsamSegment(string fullPath)
    {
        string root = Path.GetPathRoot(fullPath) ?? string.Empty;
        string relative = fullPath[root.Length..];
        const string LegacyParent = "BSAM\\";
        if (relative.StartsWith(LegacyParent, StringComparison.OrdinalIgnoreCase))
            return Path.Combine(root, relative[LegacyParent.Length..]);

        return fullPath;
    }

    private static void PromoteLegacyEntry(string stableCachePath, CacheEnvelope envelope)
    {
        try
        {
            if (File.Exists(stableCachePath))
                return;

            File.WriteAllText(
                stableCachePath,
                JsonSerializer.Serialize(envelope, SerializerOptions));
        }
        catch
        {
            // Promotion is best-effort only.
        }
    }

    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented        = false,
        Converters           = { new JsonStringEnumConverter() },
        IncludeFields        = false,
    };

    private sealed record CacheCandidate(string Path, bool IsStable);

    private sealed class CacheEnvelope
    {
        public string?            Version              { get; set; }
        public long               SourceLastWriteTicks { get; set; }
        public AudioAnalysisResult? Result             { get; set; }
    }
}
