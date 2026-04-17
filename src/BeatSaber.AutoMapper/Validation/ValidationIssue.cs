namespace BeatSaber.AutoMapper.Validation;

public sealed record ValidationIssue(
    string RuleName,
    IssueSeverity Severity,
    double? Beat,
    double? TimeSeconds,
    NoteHand? Hand,
    string Description,
    string? SuggestedFix
);
