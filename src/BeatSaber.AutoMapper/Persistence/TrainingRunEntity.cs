namespace BeatSaber.AutoMapper.Persistence;

public sealed class TrainingRunEntity
{
    public long Id { get; set; }
    public long? DatasetManifestId { get; set; }
    public string ProfileName { get; set; } = "baseline";
    public int Epochs { get; set; }
    public double TrainAccuracy { get; set; }
    public double ValidationAccuracy { get; set; }
    public string? ArtifactsPath { get; set; }
    public string Status { get; set; } = "Pending";
    public DateTime StartedAt { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }

    public DatasetManifestEntity? DatasetManifest { get; set; }
    public List<ModelArtifactEntity> ModelArtifacts { get; set; } = [];
}
