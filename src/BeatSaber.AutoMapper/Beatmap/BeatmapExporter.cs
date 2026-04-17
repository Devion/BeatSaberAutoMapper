using BeatSaber.AutoMapper.Beatmap.Raw;

namespace BeatSaber.AutoMapper.Beatmap;

public sealed class BeatmapExporter
{
    // v3 beatmap files use explicit [JsonPropertyName] attrs – camelCase policy is harmless.
    private static readonly JsonSerializerOptions _writeOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    // Info.dat uses v2 underscore keys; explicit attrs already set them – no naming policy needed.
    private static readonly JsonSerializerOptions _infoOpts = new() { WriteIndented = true };

    public static void ExportV3(CanonicalBeatmap beatmap, string outputFilePath)
    {
        Guard.NotNull(beatmap, nameof(beatmap));
        Guard.NotNullOrEmpty(outputFilePath, nameof(outputFilePath));
        string json = SerialiseV3(beatmap);
        File.WriteAllText(outputFilePath, json, Encoding.UTF8);
    }

    public static async Task ExportV3Async(CanonicalBeatmap beatmap, string outputFilePath)
    {
        Guard.NotNull(beatmap, nameof(beatmap));
        Guard.NotNullOrEmpty(outputFilePath, nameof(outputFilePath));
        string json = SerialiseV3(beatmap);
        await File.WriteAllTextAsync(outputFilePath, json, Encoding.UTF8).ConfigureAwait(false);
    }

    public static void ExportInfoDat(
        SongMetadata meta,
        IReadOnlyList<(DifficultyDescriptor Desc, string Filename)> difficulties,
        string outputFolderPath)
    {
        Guard.NotNull(meta, nameof(meta));
        Guard.NotNull(difficulties, nameof(difficulties));
        Guard.NotNullOrEmpty(outputFolderPath, nameof(outputFolderPath));
        Directory.CreateDirectory(outputFolderPath);

        var sets = difficulties
            .GroupBy(d => d.Desc.Characteristic)
            .Select(g => new V2DifficultyBeatmapSet
            {
                BeatmapCharacteristicName = g.Key.ToString(),
                DifficultyBeatmaps = g.Select(d => new V2DifficultyBeatmap
                {
                    Difficulty = d.Desc.Difficulty.ToString(),
                    DifficultyRank = DifficultyToRank(d.Desc.Difficulty),
                    BeatmapFilename = d.Filename,
                    NoteJumpMovementSpeed = d.Desc.NoteJumpMovementSpeed ?? 16,
                    NoteJumpStartBeatOffset = d.Desc.NoteJumpStartBeatOffset ?? 0.0
                }).ToList()
            }).ToList();

        var info = new V2Info
        {
            SongName          = meta.Title,
            SongSubName       = meta.SubTitle ?? "",
            SongAuthorName    = meta.Artist,
            BeatsPerMinute    = meta.BeatsPerMinute,
            PreviewStartTime  = meta.PreviewStartTime,
            PreviewDuration   = meta.PreviewDuration,
            SongFilename      = meta.AudioPath ?? "song.egg",
            CoverImageFilename = meta.CoverImagePath ?? "cover.jpg",
            DifficultyBeatmapSets = sets
        };

        string path = Path.Combine(outputFolderPath, "Info.dat");
        string json = JsonSerializer.Serialize(info, _infoOpts);
        File.WriteAllText(path, json, Encoding.UTF8);
    }

    private static string SerialiseV3(CanonicalBeatmap beatmap)
    {
        var v3 = new V3Beatmap
        {
            ColorNotes = beatmap.Notes.Select(n => new V3ColorNote
            {
                Beat = n.Beat,
                Lane = n.Lane,
                Row = n.Row,
                Color = (int)n.Color,
                CutDirection = (int)n.CutDirection
            }).ToList(),
            BombNotes = beatmap.Bombs.Select(b => new V3BombNote
            {
                Beat = b.Beat,
                Lane = b.Lane,
                Row = b.Row
            }).ToList(),
            Obstacles = beatmap.Obstacles.Select(o => new V3Obstacle
            {
                Beat = o.Beat,
                Lane = o.Lane,
                Row = o.StartRow,
                Duration = o.Duration,
                Width = o.Width,
                Height = o.Height
            }).ToList(),
            BpmEvents = beatmap.TimingPoints
                .Where(tp => tp.Beat > 0)
                .Select(tp => new V3BpmChange { Beat = tp.Beat, Bpm = tp.BeatsPerMinute })
                .ToList()
        };

        return JsonSerializer.Serialize(v3, _writeOpts);
    }

    private static int DifficultyToRank(DifficultyLevel d) => d switch
    {
        DifficultyLevel.Easy => 1,
        DifficultyLevel.Normal => 3,
        DifficultyLevel.Hard => 5,
        DifficultyLevel.Expert => 7,
        DifficultyLevel.ExpertPlus => 9,
        _ => 1
    };
}
