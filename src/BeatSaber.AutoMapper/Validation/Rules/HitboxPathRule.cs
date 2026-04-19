namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class HitboxPathRule : IValidationRule
{
    public string RuleName => "HitboxPath";

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        var notes = beatmap.Notes.OrderBy(n => n.Beat).ToList();
        for (int i = 1; i < notes.Count; i++)
        {
            if (!PlayabilityHeuristics.IsHitboxPath(notes[i - 1], notes[i]))
                continue;

            yield return new ValidationIssue(
                RuleName,
                IssueSeverity.Warning,
                notes[i].Beat,
                beatmap.BeatToSeconds(notes[i].Beat),
                notes[i].Hand,
                $"Potential hitbox/path conflict near beat {notes[i].Beat:F2}.",
                "Move one note to a different lane/row or widen the timing gap.");
        }
    }
}
