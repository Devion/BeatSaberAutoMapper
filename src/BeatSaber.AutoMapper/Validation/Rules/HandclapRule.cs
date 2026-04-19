namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class HandclapRule : IValidationRule
{
    public string RuleName => "Handclap";

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
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
                    if (!PlayabilityHeuristics.IsHandclap(notes[i], notes[j]))
                        continue;

                    yield return new ValidationIssue(
                        RuleName,
                        IssueSeverity.Warning,
                        notes[i].Beat,
                        beatmap.BeatToSeconds(notes[i].Beat),
                        null,
                        $"Handclap-style inward double at beat {notes[i].Beat:F2}.",
                        "Rotate one cut direction outward or remove the double.");
                }
            }
        }
    }
}
