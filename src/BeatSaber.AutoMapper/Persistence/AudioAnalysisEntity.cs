namespace BeatSaber.AutoMapper.Persistence;

public sealed class AudioAnalysisEntity
{
    public long Id { get; set; }
    public long SongId { get; set; }
    public double DurationSeconds { get; set; }
    public int SampleRate { get; set; }
    public double EstimatedBpm { get; set; }
    public double BpmConfidence { get; set; }
    public string? BeatTimesJson { get; set; }
    public string? OnsetTimesJson { get; set; }
    public string? SectionsJson { get; set; }
    public DateTime AnalysedAt { get; set; } = DateTime.UtcNow;

    public SongEntity? Song { get; set; }
}
