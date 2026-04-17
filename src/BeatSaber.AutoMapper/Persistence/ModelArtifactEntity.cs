namespace BeatSaber.AutoMapper.Persistence;

public sealed class ModelArtifactEntity
{
    public long Id { get; set; }
    public long? TrainingRunId { get; set; }
    public string ArtifactType { get; set; } = "";
    public string Version { get; set; } = "1.0";
    public string FilePath { get; set; } = "";
    public double? Accuracy { get; set; }
    public DateTime RegisteredAt { get; set; } = DateTime.UtcNow;

    public TrainingRunEntity? TrainingRun { get; set; }
}
