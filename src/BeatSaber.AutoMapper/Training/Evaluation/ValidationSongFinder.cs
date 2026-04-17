using BeatSaber.AutoMapper.Beatmap;

namespace BeatSaber.AutoMapper.Training.Evaluation;

/// <summary>
/// Finds map folders in the library that have an accessible audio file,
/// returning a sample suitable for generation-quality validation during training.
/// </summary>
public sealed class ValidationSongFinder
{
    public sealed record ValidationPair(
        string MapFolder,
        string AudioPath,
        CanonicalBeatmap ReferenceMap);

    /// <summary>
    /// Find up to <paramref name="maxCount"/> random map folders that have audio + parseable notes.
    /// </summary>
    public IReadOnlyList<ValidationPair> Find(
        IReadOnlyList<string> mapFolders,
        int  maxCount = 5,
        long seed     = 42)
    {
        var result   = new List<ValidationPair>();
        var rng      = new Random((int)seed);
        var shuffled = mapFolders.OrderBy(_ => rng.Next()).ToList();

        foreach (string folder in shuffled)
        {
            if (result.Count >= maxCount) break;

            string? audioPath = FindAudio(folder);
            if (audioPath is null) continue;

            try
            {
                var maps = BeatmapImporter.Import(folder);
                // Prefer the highest difficulty that has enough notes
                var best = maps
                    .Where(m => m.Notes.Count >= 20)
                    .MaxBy(m => (int)m.Difficulty.Difficulty);

                if (best is null) continue;
                result.Add(new ValidationPair(folder, audioPath, best));
            }
            catch { /* skip un-parseable maps */ }
        }

        return result;
    }

    // -----------------------------------------------------------------------

    private static string? FindAudio(string folder)
    {
        // Check Info.dat / info.dat for the declared audio filename
        string? infoPath =
            File.Exists(Path.Combine(folder, "Info.dat")) ? Path.Combine(folder, "Info.dat") :
            File.Exists(Path.Combine(folder, "info.dat")) ? Path.Combine(folder, "info.dat") :
            null;

        if (infoPath is not null)
        {
            try
            {
                var root = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(infoPath));
                if (root.TryGetProperty("_songFilename", out var fn))
                {
                    string declared = fn.GetString() ?? "";
                    string candidate = Path.Combine(folder, declared);
                    if (File.Exists(candidate)) return candidate;
                }
            }
            catch { }
        }

        // Fallback: scan for any audio file (Beat Saber uses .ogg in the maps, .egg is a renamed .ogg)
        foreach (string pattern in new[] { "*.ogg", "*.egg", "*.mp3", "*.wav" })
        {
            var files = Directory.GetFiles(folder, pattern);
            if (files.Length > 0) return files[0];
        }

        return null;
    }
}
