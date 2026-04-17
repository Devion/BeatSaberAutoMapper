using BeatSaber.AutoMapper.Canonical.Derived;

namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class ParityRule : IValidationRule
{
    public string RuleName => "Parity";
    public double ParityBreakThresholdBeats { get; set; } = 0.1;

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        var issues = new List<ValidationIssue>();
        foreach (NoteHand hand in new[] { NoteHand.Left, NoteHand.Right })
        {
            var handNotes = beatmap.Notes
                .Where(n => n.Hand == hand)
                .OrderBy(n => n.Beat)
                .ToList();

            var ctx = new SwingContext(hand);
            CanonicalNote? prev = null;

            foreach (var note in handNotes)
            {
                var result = ParityAnalyzer.ClassifyTransition(prev, note, ctx);
                string? fix = null;
                IssueSeverity? sev = null;

                switch (result.Transition)
                {
                    case ParityTransition.RecoveryRequired:
                        sev = IssueSeverity.Warning;
                        fix = "Add more time between notes or adjust cut direction for natural flow.";
                        break;
                    case ParityTransition.AwkwardReset:
                        sev = IssueSeverity.Warning;
                        fix = "Consider increasing the gap or flipping cut direction.";
                        break;
                    case ParityTransition.ParityBreak:
                        sev = IssueSeverity.Error;
                        fix = "Flip cut direction of this note or remove it.";
                        break;
                    case ParityTransition.Invalid:
                        sev = IssueSeverity.Error;
                        fix = "Remove this note – the time gap makes a valid reset physically impossible.";
                        break;
                    case ParityTransition.HighStrain:
                        sev = IssueSeverity.Info;
                        fix = "Consider reducing complexity in this area.";
                        break;
                }

                if (sev is not null)
                {
                    issues.Add(new ValidationIssue(
                        RuleName: RuleName,
                        Severity: sev.Value,
                        Beat: note.Beat,
                        TimeSeconds: beatmap.BeatToSeconds(note.Beat),
                        Hand: hand,
                        Description: $"{hand} hand parity {result.Transition}: {result.Reason}",
                        SuggestedFix: fix
                    ));
                }

                ctx.Update(note);
                prev = note;
            }
        }
        return issues;
    }
}
