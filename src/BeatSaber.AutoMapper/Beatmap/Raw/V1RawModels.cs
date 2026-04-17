namespace BeatSaber.AutoMapper.Beatmap.Raw;

// ---------------------------------------------------------------------------
// Beat Saber v1 JSON raw models
// V1 uses camelCase field names (no underscore prefix) in info.json and
// stores difficulty file references as a flat "difficultyLevels" array.
// The actual difficulty files (e.g. ExpertPlus.json) use the same
// underscore-prefixed _notes/_obstacles schema as v2 — they are deserialized
// via V2Beatmap and converted with V2BeatmapAdapter.
// ---------------------------------------------------------------------------

internal sealed class V1Info
{
    [JsonPropertyName("songName")]         public string SongName         { get; set; } = "";
    [JsonPropertyName("songSubName")]      public string SongSubName      { get; set; } = "";
    [JsonPropertyName("authorName")]       public string AuthorName       { get; set; } = "";
    [JsonPropertyName("beatsPerMinute")]   public double BeatsPerMinute   { get; set; }
    [JsonPropertyName("previewStartTime")] public double PreviewStartTime { get; set; }
    [JsonPropertyName("previewDuration")]  public double PreviewDuration  { get; set; }
    [JsonPropertyName("coverImagePath")]   public string CoverImagePath   { get; set; } = "cover.jpg";
    [JsonPropertyName("environmentName")]  public string EnvironmentName  { get; set; } = "DefaultEnvironment";
    [JsonPropertyName("difficultyLevels")] public List<V1DifficultyLevel> DifficultyLevels { get; set; } = [];
}

internal sealed class V1DifficultyLevel
{
    [JsonPropertyName("difficulty")]     public string Difficulty     { get; set; } = "Easy";
    [JsonPropertyName("difficultyRank")] public int    DifficultyRank { get; set; }
    /// <summary>Path to the audio file (OGG) — may differ per difficulty in very old maps.</summary>
    [JsonPropertyName("audioPath")]      public string AudioPath      { get; set; } = "";
    /// <summary>Path to the difficulty JSON file (e.g. "ExpertPlus.json").</summary>
    [JsonPropertyName("jsonPath")]       public string JsonPath        { get; set; } = "";
    /// <summary>Song start offset in milliseconds.</summary>
    [JsonPropertyName("offset")]         public double Offset          { get; set; }
    [JsonPropertyName("noteJumpSpeed")]  public double NoteJumpSpeed   { get; set; }
}
