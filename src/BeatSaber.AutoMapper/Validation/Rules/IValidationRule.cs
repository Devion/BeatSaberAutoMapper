namespace BeatSaber.AutoMapper.Validation.Rules;

public interface IValidationRule
{
    string RuleName { get; }
    IEnumerable<ValidationIssue> Validate(CanonicalBeatmap beatmap);
}
