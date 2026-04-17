namespace BeatSaber.AutoMapper.Persistence;

public sealed class SongEntity
{
    public long Id { get; set; }
    public string Title { get; set; } = "";
    public string Artist { get; set; } = "";
    public string? SubTitle { get; set; }
    public double BeatsPerMinute { get; set; }
    public string? AudioPath { get; set; }
    public string? CoverImagePath { get; set; }
    public string? MapFolderPath { get; set; }
    public string? MapHash { get; set; }
    public DateTime ImportedAt { get; set; } = DateTime.UtcNow;

    public List<BeatmapSetEntity> BeatmapSets { get; set; } = [];
    public AudioAnalysisEntity? AudioAnalysis { get; set; }
}
