namespace BeatSaber.AutoMapper.Persistence;

public sealed class DatasetManifestEntity
{
    public long Id { get; set; }
    public string Name { get; set; } = "";
    public string Version { get; set; } = "1.0";
    public int TotalExamples { get; set; }
    public int TrainCount { get; set; }
    public int ValidationCount { get; set; }
    public int TestCount { get; set; }
    public string? ManifestPath { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
