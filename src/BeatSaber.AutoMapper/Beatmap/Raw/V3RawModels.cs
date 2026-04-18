namespace BeatSaber.AutoMapper.Beatmap.Raw;

// ---------------------------------------------------------------------------
// Beat Saber v3 JSON raw models
// ---------------------------------------------------------------------------

internal sealed class V3Info
{
    [JsonPropertyName("version")] public string Version { get; set; } = "3.0.0";
    [JsonPropertyName("song")] public V3InfoSong Song { get; set; } = new();
    [JsonPropertyName("audio")] public V3InfoAudio Audio { get; set; } = new();
    [JsonPropertyName("coverImageFilename")] public string CoverImageFilename { get; set; } = "cover.jpg";
    [JsonPropertyName("difficultyBeatmapSets")] public List<V3DifficultyBeatmapSet> DifficultyBeatmapSets { get; set; } = [];
}

internal sealed class V3InfoSong
{
    [JsonPropertyName("title")] public string Title { get; set; } = "";
    [JsonPropertyName("subTitle")] public string SubTitle { get; set; } = "";
    [JsonPropertyName("author")] public string Author { get; set; } = "";
}

internal sealed class V3InfoAudio
{
    [JsonPropertyName("songFilename")] public string SongFilename { get; set; } = "song.ogg";
    [JsonPropertyName("songDuration")] public double SongDuration { get; set; }
    [JsonPropertyName("audioDataFilename")] public string AudioDataFilename { get; set; } = "AudioData.dat";
    [JsonPropertyName("bpm")] public double Bpm { get; set; }
    [JsonPropertyName("previewStartTime")] public double PreviewStartTime { get; set; }
    [JsonPropertyName("previewDuration")] public double PreviewDuration { get; set; }
}

internal sealed class V3DifficultyBeatmapSet
{
    [JsonPropertyName("beatmapCharacteristicName")] public string BeatmapCharacteristicName { get; set; } = "Standard";
    [JsonPropertyName("difficultyBeatmaps")] public List<V3DifficultyBeatmap> DifficultyBeatmaps { get; set; } = [];
}

internal sealed class V3DifficultyBeatmap
{
    [JsonPropertyName("difficulty")] public string Difficulty { get; set; } = "Easy";
    [JsonPropertyName("difficultyRank")] public int DifficultyRank { get; set; }
    [JsonPropertyName("beatmapFilename")] public string BeatmapFilename { get; set; } = "";
    [JsonPropertyName("noteJumpMovementSpeed")] public double NoteJumpMovementSpeed { get; set; }
    [JsonPropertyName("noteJumpStartBeatOffset")] public double NoteJumpStartBeatOffset { get; set; }
}

internal sealed class V3Beatmap
{
    [JsonPropertyName("version")] public string Version { get; set; } = "3.0.0";
    [JsonPropertyName("bpmEvents")] public List<V3BpmChange> BpmEvents { get; set; } = [];
    [JsonPropertyName("rotationEvents")] public List<JsonElement> RotationEvents { get; set; } = [];
    [JsonPropertyName("colorNotes")] public List<V3ColorNote> ColorNotes { get; set; } = [];
    [JsonPropertyName("bombNotes")] public List<V3BombNote> BombNotes { get; set; } = [];
    [JsonPropertyName("obstacles")] public List<V3Obstacle> Obstacles { get; set; } = [];
    [JsonPropertyName("sliders")] public List<JsonElement> Sliders { get; set; } = [];
    [JsonPropertyName("burstSliders")] public List<JsonElement> BurstSliders { get; set; } = [];
    [JsonPropertyName("waypoints")] public List<JsonElement> Waypoints { get; set; } = [];
    [JsonPropertyName("basicBeatmapEvents")] public List<JsonElement> BasicBeatmapEvents { get; set; } = [];
    [JsonPropertyName("colorBoostBeatmapEvents")] public List<JsonElement> ColorBoostBeatmapEvents { get; set; } = [];
    [JsonPropertyName("lightColorEventBoxGroups")] public List<JsonElement> LightColorEventBoxGroups { get; set; } = [];
    [JsonPropertyName("lightRotationEventBoxGroups")] public List<JsonElement> LightRotationEventBoxGroups { get; set; } = [];
    [JsonPropertyName("basicEventTypesWithKeywords")] public V3BasicEventTypesWithKeywords BasicEventTypesWithKeywords { get; set; } = new();
    [JsonPropertyName("useNormalEventsAsCompatibleEvents")] public bool UseNormalEventsAsCompatibleEvents { get; set; } = false;
}

internal sealed class V3BasicEventTypesWithKeywords
{
    [JsonPropertyName("d")] public List<JsonElement> D { get; set; } = [];
}

internal sealed class V3ColorNote
{
    [JsonPropertyName("b")] public double Beat { get; set; }
    [JsonPropertyName("x")] public int Lane { get; set; }
    [JsonPropertyName("y")] public int Row { get; set; }
    /// <summary>0 = red, 1 = blue</summary>
    [JsonPropertyName("c")] public int Color { get; set; }
    [JsonPropertyName("d")] public int CutDirection { get; set; }
    [JsonPropertyName("a")] public int AngleOffset { get; set; }
}

internal sealed class V3BombNote
{
    [JsonPropertyName("b")] public double Beat { get; set; }
    [JsonPropertyName("x")] public int Lane { get; set; }
    [JsonPropertyName("y")] public int Row { get; set; }
}

internal sealed class V3Obstacle
{
    [JsonPropertyName("b")] public double Beat { get; set; }
    [JsonPropertyName("x")] public int Lane { get; set; }
    [JsonPropertyName("y")] public int Row { get; set; }
    [JsonPropertyName("d")] public double Duration { get; set; }
    [JsonPropertyName("w")] public int Width { get; set; }
    [JsonPropertyName("h")] public int Height { get; set; }
}

internal sealed class V3BpmChange
{
    [JsonPropertyName("b")] public double Beat { get; set; }
    [JsonPropertyName("m")] public double Bpm { get; set; }
}
