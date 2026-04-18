using BeatSaber.AutoMapper.Beatmap;

namespace BeatSaber.AutoMapper.Training.Evaluation;

/// <summary>
/// Finds map folders in the library that have an accessible audio file,
/// returning validation pairs suitable for generation-quality validation during training.
/// Each pair carries one specific difficulty so generated maps are always compared
/// difficulty-to-difficulty (Hard→Hard, Expert→Expert, etc.).
/// </summary>
public sealed class ValidationSongFinder
{
    public sealed record ValidationPair(
        string MapFolder,
        string AudioPath,
        CanonicalBeatmap ReferenceMap);

    /// <summary>
    /// Find up to <paramref name="maxCount"/> validation pairs from a random sample of folders.
    /// Returns ONE pair per available difficulty per selected folder, so if a folder has Hard
    /// and Expert the evaluator will generate and compare both separately.
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

                // Add one pair for EACH difficulty that has enough notes,
                // so generated maps are compared at the matching difficulty level.
                var validDiffs = maps
                    .Where(m => m.Notes.Count >= 20)
                    .OrderByDescending(m => (int)m.Difficulty.Difficulty)
                    .ToList();

                foreach (var map in validDiffs)
                {
                    if (result.Count >= maxCount) break;
                    result.Add(new ValidationPair(folder, audioPath, map));
                }
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
