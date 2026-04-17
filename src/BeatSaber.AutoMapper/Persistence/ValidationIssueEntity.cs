namespace BeatSaber.AutoMapper.Persistence;

public sealed class ValidationIssueEntity
{
    public long Id { get; set; }
    public long ValidationRunId { get; set; }
    public string RuleName { get; set; } = "";
    public string Severity { get; set; } = "Info";
    public double? Beat { get; set; }
    public double? TimeSeconds { get; set; }
    public string? Hand { get; set; }
    public string Description { get; set; } = "";
    public string? SuggestedFix { get; set; }

    public ValidationRunEntity? ValidationRun { get; set; }
}
