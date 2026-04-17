namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class DifficultyEnvelopeRule : IValidationRule
{
    public string RuleName => "DifficultyEnvelope";

    private const int MinNotes = 50;
    private const double MaxSilentBeats = 16.0;

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        // Minimum note count
        if (beatmap.Notes.Count < MinNotes)
        {
            yield return new ValidationIssue(
                RuleName: RuleName,
                Severity: IssueSeverity.Warning,
                Beat: null,
                TimeSeconds: null,
                Hand: null,
                Description: $"Map has only {beatmap.Notes.Count} notes, below the recommended minimum of {MinNotes}.",
                SuggestedFix: "Add more notes to make the map playable."
            );
        }

        // Check for long empty sections
        var sorted = beatmap.Notes.OrderBy(n => n.Beat).ToList();
        for (int i = 1; i < sorted.Count; i++)
        {
            double gap = sorted[i].Beat - sorted[i - 1].Beat;
            if (gap > MaxSilentBeats)
            {
                yield return new ValidationIssue(
                    RuleName: RuleName,
                    Severity: IssueSeverity.Info,
                    Beat: sorted[i - 1].Beat,
                    TimeSeconds: beatmap.BeatToSeconds(sorted[i - 1].Beat),
                    Hand: null,
                    Description: $"Empty section of {gap:F1} beats starting at beat {sorted[i - 1].Beat:F1}.",
                    SuggestedFix: "Consider adding notes or obstacles to fill this gap."
                );
            }
        }
    }
}
