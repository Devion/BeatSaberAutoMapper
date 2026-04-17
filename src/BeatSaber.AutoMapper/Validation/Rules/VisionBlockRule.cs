namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class VisionBlockRule : IValidationRule
{
    public string RuleName => "VisionBlock";

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        // Group notes by beat (with tolerance)
        var byBeat = beatmap.Notes
            .GroupBy(n => Math.Round(n.Beat, 3))
            .Where(g => g.Count() > 1);

        foreach (var group in byBeat)
        {
            var notes = group.ToList();
            for (int i = 0; i < notes.Count; i++)
            {
                for (int j = i + 1; j < notes.Count; j++)
                {
                    var a = notes[i];
                    var b = notes[j];
                    if (IsVisionBlock(a, b))
                    {
                        yield return new ValidationIssue(
                            RuleName: RuleName,
                            Severity: IssueSeverity.Warning,
                            Beat: a.Beat,
                            TimeSeconds: beatmap.BeatToSeconds(a.Beat),
                            Hand: null,
                            Description: $"Potential vision block at beat {a.Beat:F2}: " +
                                         $"lane {a.Lane}/row {a.Row} and lane {b.Lane}/row {b.Row}.",
                            SuggestedFix: "Move one note away from centre lanes or adjust the row."
                        );
                    }
                }
            }
        }

        // Check notes behind obstacles
        foreach (var note in beatmap.Notes)
        {
            foreach (var obs in beatmap.Obstacles)
            {
                if (note.Beat >= obs.Beat && note.Beat <= obs.Beat + obs.Duration
                    && note.Lane >= obs.Lane && note.Lane < obs.Lane + obs.Width)
                {
                    yield return new ValidationIssue(
                        RuleName: RuleName,
                        Severity: IssueSeverity.Warning,
                        Beat: note.Beat,
                        TimeSeconds: beatmap.BeatToSeconds(note.Beat),
                        Hand: null,
                        Description: $"Note at beat {note.Beat:F2} lane {note.Lane} is inside an obstacle.",
                        SuggestedFix: "Move note outside the obstacle or shorten the obstacle."
                    );
                }
            }
        }
    }

    private static bool IsVisionBlock(CanonicalNote a, CanonicalNote b)
    {
        // Two stacked notes near centre with same lane but different rows
        bool sameLane = a.Lane == b.Lane;
        bool nearCentre = a.Lane is 1 or 2;
        bool differentRows = a.Row != b.Row;
        return sameLane && nearCentre && differentRows;
    }
}
