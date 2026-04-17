namespace BeatSaber.AutoMapper.Validation.Rules;

public sealed class DensityRule : IValidationRule
{
    public string RuleName => "Density";

    private static readonly Dictionary<DifficultyLevel, double> MaxNps = new()
    {
        [DifficultyLevel.Easy] = 2.5,
        [DifficultyLevel.Normal] = 4.0,
        [DifficultyLevel.Hard] = 5.0,
        [DifficultyLevel.Expert] = 8.0,
        [DifficultyLevel.ExpertPlus] = 14.0
    };

    public IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap)
    {
        if (beatmap.Notes.Count == 0) yield break;

        double max = MaxNps.TryGetValue(beatmap.Difficulty.Difficulty, out double m) ? m : 8.0;

        double firstBeat = beatmap.Notes.Min(n => n.Beat);
        double lastBeat = beatmap.Notes.Max(n => n.Beat);
        double avgNps = beatmap.LocalNps((firstBeat + lastBeat) / 2, lastBeat - firstBeat);

        // Slide a 4-beat window
        for (double beat = firstBeat; beat <= lastBeat; beat += 2.0)
        {
            double localNps = beatmap.LocalNps(beat, 4.0);

            if (localNps > max * 2.0)
            {
                yield return new ValidationIssue(
                    RuleName: RuleName,
                    Severity: IssueSeverity.Error,
                    Beat: beat,
                    TimeSeconds: beatmap.BeatToSeconds(beat),
                    Hand: null,
                    Description: $"Note density {localNps:F1} NPS at beat {beat:F1} far exceeds {max} NPS limit for {beatmap.Difficulty.Difficulty}.",
                    SuggestedFix: "Remove notes in this window to reduce density."
                );
            }
            else if (localNps > max * 1.5)
            {
                yield return new ValidationIssue(
                    RuleName: RuleName,
                    Severity: IssueSeverity.Warning,
                    Beat: beat,
                    TimeSeconds: beatmap.BeatToSeconds(beat),
                    Hand: null,
                    Description: $"Note density {localNps:F1} NPS at beat {beat:F1} exceeds {max} NPS limit for {beatmap.Difficulty.Difficulty}.",
                    SuggestedFix: "Consider reducing density here."
                );
            }

            // Sudden spike check
            if (avgNps > 0 && localNps > avgNps * 3.0)
            {
                yield return new ValidationIssue(
                    RuleName: RuleName,
                    Severity: IssueSeverity.Warning,
                    Beat: beat,
                    TimeSeconds: beatmap.BeatToSeconds(beat),
                    Hand: null,
                    Description: $"Sudden density spike: {localNps:F1} NPS vs avg {avgNps:F1} NPS.",
                    SuggestedFix: "Smooth out this density spike for better flow."
                );
            }
        }
    }
}
