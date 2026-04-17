namespace BeatSaber.AutoMapper.Persistence;

public sealed class CorpusDuplicateEntity
{
    public long Id { get; set; }
    public string MapHash { get; set; } = "";
    public string OriginalPath { get; set; } = "";
    public string DuplicatePath { get; set; } = "";
    public DateTime DetectedAt { get; set; } = DateTime.UtcNow;
}
