namespace BeatSaber.AutoMapper.Beatmap.Raw;

// ---------------------------------------------------------------------------
// Beat Saber v2 JSON raw models
// All property names are underscore-prefixed per the v2 format spec.
// ---------------------------------------------------------------------------

internal sealed class V2Info
{
    [JsonPropertyName("_version")] public string Version { get; set; } = "2.1.0";
    [JsonPropertyName("_songName")] public string SongName { get; set; } = "";
    [JsonPropertyName("_songSubName")] public string SongSubName { get; set; } = "";
    [JsonPropertyName("_songAuthorName")] public string SongAuthorName { get; set; } = "";
    [JsonPropertyName("_levelAuthorName")] public string LevelAuthorName { get; set; } = "AutoMapper";
    [JsonPropertyName("_beatsPerMinute")] public double BeatsPerMinute { get; set; }
    [JsonPropertyName("_songTimeOffset")] public double SongTimeOffset { get; set; }
    [JsonPropertyName("_shuffle")] public double Shuffle { get; set; } = 0;
    [JsonPropertyName("_shufflePeriod")] public double ShufflePeriod { get; set; } = 0.5;
    [JsonPropertyName("_previewStartTime")] public double PreviewStartTime { get; set; }
    [JsonPropertyName("_previewDuration")] public double PreviewDuration { get; set; }
    [JsonPropertyName("_songFilename")] public string SongFilename { get; set; } = "song.egg";
    [JsonPropertyName("_coverImageFilename")] public string CoverImageFilename { get; set; } = "cover.jpg";
    [JsonPropertyName("_environmentName")] public string EnvironmentName { get; set; } = "DefaultEnvironment";
    [JsonPropertyName("_allDirectionsEnvironmentName")] public string AllDirectionsEnvironmentName { get; set; } = "GlassDesertEnvironment";
    [JsonPropertyName("_difficultyBeatmapSets")] public List<V2DifficultyBeatmapSet> DifficultyBeatmapSets { get; set; } = [];
}

internal sealed class V2DifficultyBeatmapSet
{
    [JsonPropertyName("_beatmapCharacteristicName")] public string BeatmapCharacteristicName { get; set; } = "Standard";
    [JsonPropertyName("_difficultyBeatmaps")] public List<V2DifficultyBeatmap> DifficultyBeatmaps { get; set; } = [];
}

internal sealed class V2DifficultyBeatmap
{
    [JsonPropertyName("_difficulty")] public string Difficulty { get; set; } = "Easy";
    [JsonPropertyName("_difficultyRank")] public int DifficultyRank { get; set; }
    [JsonPropertyName("_beatmapFilename")] public string BeatmapFilename { get; set; } = "";
    [JsonPropertyName("_noteJumpMovementSpeed")] public double NoteJumpMovementSpeed { get; set; }
    [JsonPropertyName("_noteJumpStartBeatOffset")] public double NoteJumpStartBeatOffset { get; set; }
}

internal sealed class V2Beatmap
{
    [JsonPropertyName("_version")] public string Version { get; set; } = "2.0.0";
    [JsonPropertyName("_notes")] public List<V2Note> Notes { get; set; } = [];
    [JsonPropertyName("_obstacles")] public List<V2Obstacle> Obstacles { get; set; } = [];
    [JsonPropertyName("_events")] public List<JsonElement> Events { get; set; } = [];
}

internal sealed class V2Note
{
    [JsonPropertyName("_time")] public double Time { get; set; }
    [JsonPropertyName("_lineIndex")] public int LineIndex { get; set; }
    [JsonPropertyName("_lineLayer")] public int LineLayer { get; set; }
    /// <summary>0 = red, 1 = blue, 3 = bomb</summary>
    [JsonPropertyName("_type")] public int Type { get; set; }
    [JsonPropertyName("_cutDirection")] public int CutDirection { get; set; }
}

internal sealed class V2Obstacle
{
    [JsonPropertyName("_time")] public double Time { get; set; }
    [JsonPropertyName("_lineIndex")] public int LineIndex { get; set; }
    /// <summary>0 = full-height wall, 1 = crouch wall</summary>
    [JsonPropertyName("_type")] public int Type { get; set; }
    [JsonPropertyName("_duration")] public double Duration { get; set; }
    [JsonPropertyName("_width")] public int Width { get; set; }
}
