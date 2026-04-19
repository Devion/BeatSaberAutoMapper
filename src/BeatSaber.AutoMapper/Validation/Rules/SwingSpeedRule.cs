namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class SwingSpeedRule : IValidationRule
{
    public string RuleName => "SwingSpeed";

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        double bpm = beatmap.Song.BeatsPerMinute > 0 ? beatmap.Song.BeatsPerMinute : 120.0;
        foreach (NoteHand hand in new[] { NoteHand.Left, NoteHand.Right })
        {
            var notes = beatmap.Notes
                .Where(n => n.Hand == hand)
                .OrderBy(n => n.Beat)
                .ToList();

            for (int i = 1; i < notes.Count; i++)
            {
                if (!PlayabilityHeuristics.IsSwingSpeedWarning(notes[i - 1], notes[i], bpm, beatmap.Difficulty.Difficulty))
                    continue;

                double ebpm = PlayabilityHeuristics.ComputeSwingEbpm(notes[i - 1], notes[i], bpm);
                yield return new ValidationIssue(
                    RuleName,
                    IssueSeverity.Warning,
                    notes[i].Beat,
                    beatmap.BeatToSeconds(notes[i].Beat),
                    hand,
                    $"{ebpm:F1} EBPM swing warning for {hand} hand at beat {notes[i].Beat:F2}.",
                    "Increase spacing or simplify the pattern.");
            }
        }
    }
}
