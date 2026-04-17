using BeatSaber.AutoMapper.Validation;

namespace BeatSaber.AutoMapper.Diagnostics;

public static class ValidationReportWriter
{
    private static readonly JsonSerializerOptions _jsonOpts = new() { WriteIndented = true };

    public static void WriteText(ValidationReport report, string outputPath)
    {
        Guard.NotNull(report, nameof(report));
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));

        var sb = new StringBuilder();
        sb.AppendLine($"Validation Report — Rule Set v{report.RuleSetVersion}");
        sb.AppendLine(new string('-', 60));
        sb.AppendLine($"Score: {report.Score:F1}/100  " +
                      $"Errors: {report.ErrorCount}  Warnings: {report.WarningCount}  Info: {report.InfoCount}");
        sb.AppendLine();

        foreach (var issue in report.Issues.OrderBy(i => i.Beat))
        {
            string beat = issue.Beat.HasValue ? $"Beat {issue.Beat:F2}" : "Global";
            string hand = issue.Hand.HasValue ? $" [{issue.Hand}]" : "";
            sb.AppendLine($"[{issue.Severity}] {beat}{hand} — {issue.RuleName}: {issue.Description}");
            if (issue.SuggestedFix is not null)
                sb.AppendLine($"  Fix: {issue.SuggestedFix}");
        }

        File.WriteAllText(outputPath, sb.ToString(), Encoding.UTF8);
    }

    public static void WriteJson(ValidationReport report, string outputPath)
    {
        Guard.NotNull(report, nameof(report));
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));
        File.WriteAllText(outputPath, JsonSerializer.Serialize(report, _jsonOpts), Encoding.UTF8);
    }

    public static void WriteMarkdown(ValidationReport report, string outputPath)
    {
        Guard.NotNull(report, nameof(report));
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));

        var sb = new StringBuilder();
        sb.AppendLine($"# Validation Report");
        sb.AppendLine();
        sb.AppendLine($"**Rule Set Version:** {report.RuleSetVersion}  ");
        sb.AppendLine($"**Score:** {report.Score:F1}/100  ");
        sb.AppendLine($"**Errors:** {report.ErrorCount} | **Warnings:** {report.WarningCount} | **Info:** {report.InfoCount}  ");
        sb.AppendLine();
        sb.AppendLine("## Issues");
        sb.AppendLine();
        sb.AppendLine("| Severity | Beat | Hand | Rule | Description |");
        sb.AppendLine("|---|---|---|---|---|");

        foreach (var issue in report.Issues.OrderBy(i => i.Beat))
        {
            string beat = issue.Beat.HasValue ? $"{issue.Beat:F2}" : "-";
            string hand = issue.Hand.HasValue ? issue.Hand.ToString()! : "-";
            sb.AppendLine($"| {issue.Severity} | {beat} | {hand} | {issue.RuleName} | {issue.Description} |");
        }

        File.WriteAllText(outputPath, sb.ToString(), Encoding.UTF8);
    }
}
