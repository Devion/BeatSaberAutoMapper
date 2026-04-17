namespace BeatSaber.AutoMapper.Persistence;

public sealed class GenerationRunEntity
{
    public long Id { get; set; }
    public long? SongId { get; set; }
    public string TargetDifficulty { get; set; } = "Hard";
    public string? ArtifactsPath { get; set; }
    public string? OutputPath { get; set; }
    public int NoteCount { get; set; }
    public double EstimatedBpm { get; set; }
    public string Status { get; set; } = "Pending";
    public DateTime StartedAt { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }

    public SongEntity? Song { get; set; }
    public List<ValidationRunEntity> ValidationRuns { get; set; } = [];
}
