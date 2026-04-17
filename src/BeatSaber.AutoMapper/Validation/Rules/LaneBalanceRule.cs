namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class LaneBalanceRule : IValidationRule
{
    public string RuleName => "LaneBalance";

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        if (beatmap.Notes.Count == 0) yield break;

        // Lane distribution (0-3)
        int total = beatmap.Notes.Count;
        for (int lane = 0; lane <= 3; lane++)
        {
            int laneCount = beatmap.Notes.Count(n => n.Lane == lane);
            double fraction = laneCount / (double)total;
            if (fraction > 0.5)
            {
                yield return new ValidationIssue(
                    RuleName: RuleName,
                    Severity: IssueSeverity.Warning,
                    Beat: null,
                    TimeSeconds: null,
                    Hand: null,
                    Description: $"Lane {lane} contains {fraction:P0} of all notes (should be ≤ 50%).",
                    SuggestedFix: "Spread notes more evenly across lanes."
                );
            }
        }

        // Hand balance
        int leftCount = beatmap.Notes.Count(n => n.Hand == NoteHand.Left);
        int rightCount = beatmap.Notes.Count(n => n.Hand == NoteHand.Right);
        double leftFrac = leftCount / (double)total;
        double rightFrac = rightCount / (double)total;

        if (leftFrac < 0.35 || leftFrac > 0.65)
        {
            var dominantHand = leftFrac > 0.65 ? NoteHand.Left : NoteHand.Right;
            double dominant = Math.Max(leftFrac, rightFrac);
            yield return new ValidationIssue(
                RuleName: RuleName,
                Severity: IssueSeverity.Warning,
                Beat: null,
                TimeSeconds: null,
                Hand: dominantHand,
                Description: $"Hand balance is off: {dominantHand} {dominant:P0} vs {1 - dominant:P0}. Should be between 35/65 and 65/35.",
                SuggestedFix: "Balance note colours more evenly between left and right hand."
            );
        }
    }
}
