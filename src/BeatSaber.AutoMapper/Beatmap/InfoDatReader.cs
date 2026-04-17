using BeatSaber.AutoMapper.Beatmap.Raw;

namespace BeatSaber.AutoMapper.Beatmap;

public sealed record DifficultyFileRef(
    BeatmapCharacteristic Characteristic,
    DifficultyLevel Difficulty,
    string BeatmapFile,
    double? NoteJumpMovementSpeed,
    double? NoteJumpStartBeatOffset
);

public sealed class InfoDatReader
{
    private static readonly JsonSerializerOptions _opts =
        new() { PropertyNameCaseInsensitive = true };

    public static (SongMetadata Metadata, IReadOnlyList<DifficultyFileRef> Difficulties) Read(
        string mapFolderPath)
    {
        Guard.NotNullOrEmpty(mapFolderPath, nameof(mapFolderPath));
        string infoPath = ResolveInfoPath(mapFolderPath);
        string json = File.ReadAllText(infoPath);
        return ParseJson(json);
    }

    public static async Task<(SongMetadata, IReadOnlyList<DifficultyFileRef>)> ReadAsync(
        string mapFolderPath)
    {
        Guard.NotNullOrEmpty(mapFolderPath, nameof(mapFolderPath));
        string infoPath = ResolveInfoPath(mapFolderPath);
        string json = await File.ReadAllTextAsync(infoPath).ConfigureAwait(false);
        return ParseJson(json);
    }

    private static string ResolveInfoPath(string folder)
    {
        // Try both capitalisation variants
        foreach (string name in new[] { "Info.dat", "info.dat", "info.json" })
        {
            string p = Path.Combine(folder, name);
            if (File.Exists(p)) return p;
        }
        throw new FileNotFoundException("Info.dat not found in map folder.", folder);
    }

    private static (SongMetadata, IReadOnlyList<DifficultyFileRef>) ParseJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        // v3: has a "song" sub-object
        if (root.TryGetProperty("song", out _))
            return ParseV3Info(json);

        // v1: flat difficultyLevels array, no underscore-prefixed keys
        if (root.TryGetProperty("difficultyLevels", out _))
            return ParseV1Info(json);

        // Default: v2
        return ParseV2Info(json);
    }

    private static (SongMetadata, IReadOnlyList<DifficultyFileRef>) ParseV1Info(string json)
    {
        var raw = JsonSerializer.Deserialize<V1Info>(json, _opts)
                  ?? throw new InvalidDataException("Failed to parse v1 info.json.");

        // V1 uses a single audio file across all difficulties (first one wins).
        string audioPath = raw.DifficultyLevels.FirstOrDefault()?.AudioPath ?? "song.ogg";

        var meta = new SongMetadata(
            Title: raw.SongName,
            Artist: raw.AuthorName,
            SubTitle: string.IsNullOrEmpty(raw.SongSubName) ? null : raw.SongSubName,
            BeatsPerMinute: raw.BeatsPerMinute,
            SongTimeOffset: 0,   // v1 offset is per-difficulty in ms; irrelevant for generation
            PreviewStartTime: raw.PreviewStartTime,
            PreviewDuration: raw.PreviewDuration,
            CoverImagePath: raw.CoverImagePath,
            AudioPath: audioPath
        );

        var diffs = raw.DifficultyLevels
            .Where(d => !string.IsNullOrEmpty(d.JsonPath))
            .Select(d => new DifficultyFileRef(
                Characteristic: BeatmapCharacteristic.Standard, // v1 only had Standard
                Difficulty: ParseDifficulty(d.Difficulty),
                BeatmapFile: d.JsonPath,
                NoteJumpMovementSpeed: d.NoteJumpSpeed > 0 ? d.NoteJumpSpeed : null,
                NoteJumpStartBeatOffset: null
            ))
            .ToList();

        return (meta, diffs);
    }

    private static (SongMetadata, IReadOnlyList<DifficultyFileRef>) ParseV2Info(string json)
    {
        var raw = JsonSerializer.Deserialize<V2Info>(json, _opts)
                  ?? throw new InvalidDataException("Failed to parse v2 Info.dat.");

        var meta = new SongMetadata(
            Title: raw.SongName,
            Artist: raw.SongAuthorName,
            SubTitle: string.IsNullOrEmpty(raw.SongSubName) ? null : raw.SongSubName,
            BeatsPerMinute: raw.BeatsPerMinute,
            SongTimeOffset: raw.SongTimeOffset,
            PreviewStartTime: raw.PreviewStartTime,
            PreviewDuration: raw.PreviewDuration,
            CoverImagePath: raw.CoverImageFilename,
            AudioPath: raw.SongFilename
        );

        var diffs = new List<DifficultyFileRef>();
        foreach (var set in raw.DifficultyBeatmapSets)
        {
            var characteristic = ParseCharacteristic(set.BeatmapCharacteristicName);
            foreach (var d in set.DifficultyBeatmaps)
            {
                diffs.Add(new DifficultyFileRef(
                    characteristic,
                    ParseDifficulty(d.Difficulty),
                    d.BeatmapFilename,
                    d.NoteJumpMovementSpeed,
                    d.NoteJumpStartBeatOffset
                ));
            }
        }
        return (meta, diffs);
    }

    private static (SongMetadata, IReadOnlyList<DifficultyFileRef>) ParseV3Info(string json)
    {
        var raw = JsonSerializer.Deserialize<V3Info>(json, _opts)
                  ?? throw new InvalidDataException("Failed to parse v3 Info.dat.");

        var meta = new SongMetadata(
            Title: raw.Song.Title,
            Artist: raw.Song.Author,
            SubTitle: string.IsNullOrEmpty(raw.Song.SubTitle) ? null : raw.Song.SubTitle,
            BeatsPerMinute: raw.Audio.Bpm,
            SongTimeOffset: 0,
            PreviewStartTime: raw.Audio.PreviewStartTime,
            PreviewDuration: raw.Audio.PreviewDuration,
            CoverImagePath: raw.CoverImageFilename,
            AudioPath: raw.Audio.SongFilename
        );

        var diffs = new List<DifficultyFileRef>();
        foreach (var set in raw.DifficultyBeatmapSets)
        {
            var characteristic = ParseCharacteristic(set.BeatmapCharacteristicName);
            foreach (var d in set.DifficultyBeatmaps)
            {
                diffs.Add(new DifficultyFileRef(
                    characteristic,
                    ParseDifficulty(d.Difficulty),
                    d.BeatmapFilename,
                    d.NoteJumpMovementSpeed,
                    d.NoteJumpStartBeatOffset
                ));
            }
        }
        return (meta, diffs);
    }

    private static BeatmapCharacteristic ParseCharacteristic(string name) => name switch
    {
        "Standard" => BeatmapCharacteristic.Standard,
        "OneSaber" => BeatmapCharacteristic.OneSaber,
        "NoArrows" => BeatmapCharacteristic.NoArrows,
        "90Degree" => BeatmapCharacteristic.NinetyDegree,
        "360Degree" => BeatmapCharacteristic.ThreeSixtyDegree,
        "Lightshow" => BeatmapCharacteristic.Lightshow,
        "Lawless" => BeatmapCharacteristic.Lawless,
        _ => BeatmapCharacteristic.Standard
    };

    private static DifficultyLevel ParseDifficulty(string name) => name switch
    {
        "Easy" => DifficultyLevel.Easy,
        "Normal" => DifficultyLevel.Normal,
        "Hard" => DifficultyLevel.Hard,
        "Expert" => DifficultyLevel.Expert,
        "ExpertPlus" => DifficultyLevel.ExpertPlus,
        _ => DifficultyLevel.Easy
    };
}
