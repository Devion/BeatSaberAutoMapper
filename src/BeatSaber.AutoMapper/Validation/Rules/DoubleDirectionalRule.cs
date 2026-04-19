namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class DoubleDirectionalRule : IValidationRule
{
    public string RuleName => "DoubleDirectional";

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        foreach (NoteHand hand in new[] { NoteHand.Left, NoteHand.Right })
        {
            var notes = beatmap.Notes
                .Where(n => n.Hand == hand)
                .OrderBy(n => n.Beat)
                .ToList();

            for (int i = 1; i < notes.Count; i++)
            {
                if (!PlayabilityHeuristics.IsDoubleDirectional(notes[i - 1], notes[i]))
                    continue;

                yield return new ValidationIssue(
                    RuleName,
                    IssueSeverity.Warning,
                    notes[i].Beat,
                    beatmap.BeatToSeconds(notes[i].Beat),
                    hand,
                    $"{hand} hand repeats {notes[i].CutDirection} too soon at beat {notes[i].Beat:F2}.",
                    "Flip the latter cut direction or add more time between notes.");
            }
        }
    }
}
