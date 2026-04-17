using BeatSaber.AutoMapper.Canonical.Derived;

namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class ResetRule : IValidationRule
{
    public string RuleName => "Reset";

    private const double ErrorThresholdBeats = 0.75;
    private const double WarningThresholdBeats = 1.5;

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        foreach (NoteHand hand in new[] { NoteHand.Left, NoteHand.Right })
        {
            var handNotes = beatmap.Notes
                .Where(n => n.Hand == hand)
                .OrderBy(n => n.Beat)
                .ToList();

            for (int i = 1; i < handNotes.Count; i++)
            {
                var prev = handNotes[i - 1];
                var curr = handNotes[i];

                // Only flag cases where parity is incompatible AND gap is short
                var prevParity = ParityAnalyzer.CutDirectionParity(prev.CutDirection);
                var currParity = ParityAnalyzer.CutDirectionParity(curr.CutDirection);
                bool compatible = prevParity != currParity
                                  || curr.CutDirection == CutDirection.Dot
                                  || prev.CutDirection == CutDirection.Dot;
                if (compatible) continue;

                double gap = curr.Beat - prev.Beat;
                if (gap < ErrorThresholdBeats)
                {
                    yield return new ValidationIssue(
                        RuleName: RuleName,
                        Severity: IssueSeverity.Error,
                        Beat: curr.Beat,
                        TimeSeconds: beatmap.BeatToSeconds(curr.Beat),
                        Hand: hand,
                        Description: $"{hand} hand reset impossible: same parity cut with only {gap:F2} beats gap.",
                        SuggestedFix: "Remove note or increase gap to at least 0.75 beats."
                    );
                }
                else if (gap < WarningThresholdBeats)
                {
                    yield return new ValidationIssue(
                        RuleName: RuleName,
                        Severity: IssueSeverity.Warning,
                        Beat: curr.Beat,
                        TimeSeconds: beatmap.BeatToSeconds(curr.Beat),
                        Hand: hand,
                        Description: $"{hand} hand reset tight: same parity cut with only {gap:F2} beats gap.",
                        SuggestedFix: "Consider increasing gap to 1.5+ beats for comfortable reset."
                    );
                }
            }
        }
    }
}
