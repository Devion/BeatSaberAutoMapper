using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Beatmap.Raw;

namespace BeatSaber.AutoMapper.Beatmap;

public sealed class BeatmapExporter
{
    private const double DefaultNoteJumpMovementSpeed = 12.0;

    // v3 beatmap files use explicit [JsonPropertyName] attrs – camelCase policy is harmless.
    private static readonly JsonSerializerOptions _writeOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    private sealed record BasicLightEvent(double Beat, int EventType, int Value, int FloatValue = 1);

    // Info.dat uses v2 underscore keys; explicit attrs already set them – no naming policy needed.
    private static readonly JsonSerializerOptions _infoOpts = new() { WriteIndented = true };

    // UTF-8 without BOM — BeatSaver and most JSON parsers reject the 0xEF BB BF preamble.
    private static readonly Encoding _utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

    public static void ExportV3(CanonicalBeatmap beatmap, string outputFilePath)
    {
        Guard.NotNull(beatmap, nameof(beatmap));
        Guard.NotNullOrEmpty(outputFilePath, nameof(outputFilePath));
        string json = SerialiseV3(beatmap);
        File.WriteAllText(outputFilePath, json, _utf8NoBom);
    }

    public static async Task ExportV3Async(CanonicalBeatmap beatmap, string outputFilePath)
    {
        Guard.NotNull(beatmap, nameof(beatmap));
        Guard.NotNullOrEmpty(outputFilePath, nameof(outputFilePath));
        string json = SerialiseV3(beatmap);
        await File.WriteAllTextAsync(outputFilePath, json, _utf8NoBom).ConfigureAwait(false);
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
                    NoteJumpMovementSpeed = d.Desc.NoteJumpMovementSpeed ?? DefaultNoteJumpMovementSpeed,
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
        File.WriteAllText(path, json, _utf8NoBom);
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
                .ToList(),
            BasicBeatmapEvents = GenerateBasicLightEvents(beatmap),
            UseNormalEventsAsCompatibleEvents = true
        };

        return JsonSerializer.Serialize(v3, _writeOpts);
    }

    private static List<JsonElement> GenerateBasicLightEvents(CanonicalBeatmap beatmap)
    {
        if (beatmap.Notes.Count == 0)
            return [];

        var events = new List<BasicLightEvent>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        static bool IsNearInteger(double beat) => Math.Abs(beat - Math.Round(beat)) < 0.01;
        static int ColorValueForBeat(double beat) => ((int)Math.Round(beat / 4.0) % 2 == 0) ? 1 : 5;

        void AddPulse(double beat, int eventType, int colorValue, double durationBeats)
        {
            string onKey = $"{Math.Round(beat, 3):F3}|{eventType}|{colorValue}";
            if (seen.Add(onKey))
                events.Add(new BasicLightEvent(Math.Round(beat, 3), eventType, colorValue));

            double offBeat = Math.Round(beat + durationBeats, 3);
            string offKey = $"{offBeat:F3}|{eventType}|0";
            if (seen.Add(offKey))
                events.Add(new BasicLightEvent(offBeat, eventType, 0));
        }

        foreach (var section in beatmap.Sections)
        {
            double beat = Math.Round(beatmap.SecondsToBeat(section.TimeSeconds));
            if (beat < 0)
                continue;

            int colorValue = ColorValueForBeat(beat);
            AddPulse(beat, eventType: 4, colorValue, durationBeats: 1.0);
            AddPulse(beat, eventType: 0, colorValue, durationBeats: 0.75);
        }

        double lastSideAccentBeat = double.NegativeInfinity;
        foreach (var group in beatmap.Notes
                     .GroupBy(n => Math.Round(n.Beat, 2))
                     .OrderBy(g => g.Key))
        {
            double beat = group.Key;
            int noteCount = group.Count();
            bool integerBeat = IsNearInteger(beat);
            bool downbeat = integerBeat && ((int)Math.Round(beat) % 4 == 0);
            bool phraseStart = integerBeat && ((int)Math.Round(beat) % 32 == 0);
            int colorValue = ColorValueForBeat(beat);

            if (phraseStart)
            {
                AddPulse(beat, eventType: 4, colorValue, durationBeats: 1.0);
                AddPulse(beat, eventType: 1, colorValue, durationBeats: 0.75);
            }
            else if (downbeat)
            {
                AddPulse(beat, eventType: 0, colorValue, durationBeats: 0.5);
                AddPulse(beat, eventType: 1, colorValue, durationBeats: 0.5);
            }
            else if (integerBeat && ((int)Math.Round(beat) % 2 == 0) && noteCount > 0)
            {
                AddPulse(beat, eventType: 0, colorValue, durationBeats: 0.35);
            }

            bool denseAccent = noteCount >= 2 || group.Any(n => n.CutDirection == CutDirection.Dot);
            if (denseAccent && beat - lastSideAccentBeat >= 1.0)
            {
                bool preferLeft = group.Count(n => n.Hand == NoteHand.Left) >= group.Count(n => n.Hand == NoteHand.Right);
                AddPulse(beat, eventType: preferLeft ? 2 : 3, colorValue, durationBeats: 0.35);
                lastSideAccentBeat = beat;
            }
        }

        return events
            .OrderBy(e => e.Beat)
            .ThenBy(e => e.EventType)
            .Select(e => JsonSerializer.SerializeToElement(new
            {
                b = e.Beat,
                et = e.EventType,
                i = e.Value,
                f = e.FloatValue
            }))
            .ToList();
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
