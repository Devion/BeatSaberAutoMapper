namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class ExcessiveDoubleRule : IValidationRule
{
    public string RuleName => "ExcessiveDouble";

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        foreach (var group in beatmap.Notes.GroupBy(n => Math.Round(n.Beat, 3)))
        {
            var notes = group.ToList();
            if (!PlayabilityHeuristics.IsExcessiveDouble(notes))
                continue;

            yield return new ValidationIssue(
                RuleName,
                IssueSeverity.Warning,
                notes[0].Beat,
                beatmap.BeatToSeconds(notes[0].Beat),
                null,
                $"Double/chord pattern at beat {notes[0].Beat:F2} with {notes.Count} simultaneous notes.",
                "Reduce doubles or keep chords for stronger accents only.");
        }
    }
}
