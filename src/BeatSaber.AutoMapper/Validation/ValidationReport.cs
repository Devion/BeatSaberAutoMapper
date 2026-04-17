namespace BeatSaber.AutoMapper.Validation;

public sealed class ValidationReport
{
    public IReadOnlyList<ValidationIssue> Issues { get; init; } = [];
    public string RuleSetVersion { get; init; } = "1.0";
    public double Score { get; init; }

    public int ErrorCount => Issues.Count(i => i.Severity == IssueSeverity.Error);
    public int WarningCount => Issues.Count(i => i.Severity == IssueSeverity.Warning);
    public int InfoCount => Issues.Count(i => i.Severity == IssueSeverity.Info);
    public bool IsValid => ErrorCount == 0;

    public static ValidationReport Create(
        IReadOnlyList<ValidationIssue> issues,
        string ruleSetVersion = "1.0")
    {
        int errors = issues.Count(i => i.Severity == IssueSeverity.Error);
        int warnings = issues.Count(i => i.Severity == IssueSeverity.Warning);
        int infos = issues.Count(i => i.Severity == IssueSeverity.Info);

        double score = 100.0 - (errors * 10.0 + warnings * 2.0 + infos * 0.5);
        score = Math.Max(0.0, Math.Min(100.0, score));

        return new ValidationReport
        {
            Issues = issues,
            RuleSetVersion = ruleSetVersion,
            Score = score
        };
    }
}
