using BeatSaber.AutoMapper.Validation.Rules;

namespace BeatSaber.AutoMapper.Validation;

public sealed class BeatmapValidator
{
    private readonly IReadOnlyList<IValidationRule> _rules;

    public BeatmapValidator()
    {
        _rules =
        [
            new ParityRule(),
            new DoubleDirectionalRule(),
            new ResetRule(),
            new SwingSpeedRule(),
            new ExcessiveDoubleRule(),
            new HandclapRule(),
            new HitboxPathRule(),
            new VisionBlockRule(),
            new DensityRule(),
            new LaneBalanceRule(),
            new DifficultyEnvelopeRule()
        ];
    }

    public BeatmapValidator(IReadOnlyList<IValidationRule> rules)
    {
        _rules = Guard.NotNull(rules, nameof(rules));
    }

    public ValidationReport Validate(CanonicalBeatmap beatmap)
    {
        Guard.NotNull(beatmap, nameof(beatmap));
        var issues = _rules.SelectMany(r => r.Validate(beatmap)).ToList();
        return ValidationReport.Create(issues);
    }
}
