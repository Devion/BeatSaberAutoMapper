namespace BeatSaber.AutoMapper.Persistence;

public sealed class ValidationRunEntity
{
    public long Id { get; set; }
    public long? DifficultyId { get; set; }
    public long? GenerationRunId { get; set; }
    public string RuleSetVersion { get; set; } = "1.0";
    public double Score { get; set; }
    public int ErrorCount { get; set; }
    public int WarningCount { get; set; }
    public int InfoCount { get; set; }
    public DateTime RanAt { get; set; } = DateTime.UtcNow;

    public DifficultyEntity? Difficulty { get; set; }
    public GenerationRunEntity? GenerationRun { get; set; }
    public List<ValidationIssueEntity> Issues { get; set; } = [];
}
