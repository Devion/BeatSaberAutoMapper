using BeatSaber.AutoMapper.Beatmap.Import;
using BeatSaber.AutoMapper.Beatmap.Raw;

namespace BeatSaber.AutoMapper.Beatmap;

public sealed class BeatmapImporter
{
    private static readonly JsonSerializerOptions _opts =
        new() { PropertyNameCaseInsensitive = true };

    public static IReadOnlyList<CanonicalBeatmap> Import(string mapFolderPath)
    {
        Guard.NotNullOrEmpty(mapFolderPath, nameof(mapFolderPath));
        var (meta, diffRefs) = InfoDatReader.Read(mapFolderPath);
        return diffRefs.Select(d =>
        {
            string filePath = Path.Combine(mapFolderPath, d.BeatmapFile);
            var desc = new DifficultyDescriptor(d.Difficulty, d.Characteristic,
                d.NoteJumpMovementSpeed, d.NoteJumpStartBeatOffset, null);
            return ImportFile(filePath, meta, desc, meta.BeatsPerMinute);
        }).ToList();
    }

    public static async Task<IReadOnlyList<CanonicalBeatmap>> ImportAsync(string mapFolderPath)
    {
        Guard.NotNullOrEmpty(mapFolderPath, nameof(mapFolderPath));
        var (meta, diffRefs) = await InfoDatReader.ReadAsync(mapFolderPath).ConfigureAwait(false);
        var tasks = diffRefs.Select(async d =>
        {
            string filePath = Path.Combine(mapFolderPath, d.BeatmapFile);
            var desc = new DifficultyDescriptor(d.Difficulty, d.Characteristic,
                d.NoteJumpMovementSpeed, d.NoteJumpStartBeatOffset, null);
            string json = await File.ReadAllTextAsync(filePath).ConfigureAwait(false);
            return ParseBeatmapJson(json, meta, desc);
        });
        return await Task.WhenAll(tasks).ConfigureAwait(false);
    }

    public static CanonicalBeatmap ImportFile(
        string beatmapFilePath,
        SongMetadata meta,
        DifficultyDescriptor difficulty,
        double bpm)
    {
        Guard.NotNullOrEmpty(beatmapFilePath, nameof(beatmapFilePath));
        Guard.NotNull(meta, nameof(meta));
        Guard.NotNull(difficulty, nameof(difficulty));

        string json = File.ReadAllText(beatmapFilePath);
        return ParseBeatmapJson(json, meta, difficulty);
    }

    private static CanonicalBeatmap ParseBeatmapJson(
        string json,
        SongMetadata meta,
        DifficultyDescriptor difficulty)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        // Detect version
        string version = "";
        if (root.TryGetProperty("version", out var vEl)) version = vEl.GetString() ?? "";
        else if (root.TryGetProperty("_version", out var vEl2)) version = vEl2.GetString() ?? "";

        if (version.StartsWith("4", StringComparison.Ordinal))
            return V4BeatmapAdapter.Convert("", meta, difficulty);

        if (version.StartsWith("3", StringComparison.Ordinal) || root.TryGetProperty("colorNotes", out _))
        {
            var v3 = JsonSerializer.Deserialize<V3Beatmap>(json, _opts)
                     ?? throw new InvalidDataException("Failed to parse v3 beatmap.");
            return V3BeatmapAdapter.Convert(v3, meta, difficulty);
        }

        // v1 ("1.x.x") and v2 share the same _notes/_obstacles field names.
        // Both are parsed via V2Beatmap + V2BeatmapAdapter.
        var v2 = JsonSerializer.Deserialize<V2Beatmap>(json, _opts)
                 ?? throw new InvalidDataException("Failed to parse beatmap.");
        return V2BeatmapAdapter.Convert(v2, meta, difficulty, meta.BeatsPerMinute);
    }
}
