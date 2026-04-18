using BeatSaber.AutoMapper.Beatmap;

namespace BeatSaber.AutoMapper.Training.Evaluation;

/// <summary>
/// Finds map folders in the library that have an accessible audio file,
/// returning song groups suitable for generation-quality validation during training.
/// <para>
/// A <see cref="SongGroup"/> contains ONE unique audio file plus ALL of its valid
/// difficulties (≥ 20 notes). <paramref name="maxSongs"/> limits the number of unique
/// songs (== unique audio cache entries). The total validation pairs across the epoch
/// is typically 2–4× that, since most songs have multiple difficulties.
/// </para>
/// </summary>
public sealed class ValidationSongFinder
{
    public sealed record ValidationPair(
        string MapFolder,
        string AudioPath,
        CanonicalBeatmap ReferenceMap);

    /// <summary>Groups all valid difficulties for a single audio file.</summary>
    public sealed record SongGroup(
        string FolderPath,
        string AudioPath,
        IReadOnlyList<CanonicalBeatmap> Difficulties);

    /// <summary>
    /// Find up to <paramref name="maxSongs"/> song groups (unique audio files).
    /// All difficulties with ≥ 20 notes are included per folder, ordered hardest-first.
    /// <paramref name="maxSongs"/> == unique audio files cached/analysed.
    /// Total validation pairs may exceed <paramref name="maxSongs"/> proportionally.
    /// </summary>
    public IReadOnlyList<SongGroup> Find(
        IReadOnlyList<string> mapFolders,
        int  maxSongs = 5,
        long seed     = 42)
    {
        var result   = new List<SongGroup>();
        var rng      = new Random((int)seed);
        var shuffled = mapFolders.OrderBy(_ => rng.Next()).ToList();

        foreach (string folder in shuffled)
        {
            if (result.Count >= maxSongs) break;

            string? audioPath = FindAudio(folder);
            if (audioPath is null) continue;

            try
            {
                var maps = BeatmapImporter.Import(folder);

                // Collect ALL valid difficulties from this folder, ordered hardest-first.
                // maxSongs limits unique songs; total pairs may be 2-4× higher.
                var diffs = maps
                    .Where(m => m.Notes.Count >= 20)
                    .OrderByDescending(m => (int)m.Difficulty.Difficulty)
                    .ToList();

                if (diffs.Count > 0)
                    result.Add(new SongGroup(folder, audioPath, diffs));
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
