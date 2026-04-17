using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Validation;
using BeatSaber.AutoMapper.Validation.Rules;
using FluentAssertions;

namespace BeatSaber.AutoMapper.Tests;

public sealed class ValidationRuleTests
{
    private static CanonicalBeatmap MakeBeatmap(
        DifficultyLevel difficulty = DifficultyLevel.Hard,
        IReadOnlyList<CanonicalNote>? notes = null) =>
        new()
        {
            Song = new SongMetadata("Test", "Artist", null, 120.0, 0, 10, 30, null, null),
            Difficulty = new DifficultyDescriptor(difficulty, BeatmapCharacteristic.Standard, null, null, null),
            TimingPoints = [new BeatTimingPoint(0, 0, 120.0)],
            Notes = notes ?? [],
            Bombs = [],
            Obstacles = [],
            Sections = []
        };

    // -----------------------------------------------------------------------
    // ParityRule
    // -----------------------------------------------------------------------

    [Fact]
    public void ParityRule_EmptyMap_NoIssues()
    {
        var rule = new ParityRule();
        var issues = rule.Validate(MakeBeatmap()).ToList();
        issues.Should().BeEmpty();
    }

    [Fact]
    public void ParityRule_KnownParityBreak_ProducesError()
    {
        // Two same-hand notes in rapid succession with same parity (Down → Down, 0.1 beats apart)
        var notes = new List<CanonicalNote>
        {
            new(0.0, 1, 1, NoteColor.Red, CutDirection.Down),
            new(0.1, 1, 1, NoteColor.Red, CutDirection.Down)
        };
        var rule = new ParityRule();
        var issues = rule.Validate(MakeBeatmap(notes: notes)).ToList();
        issues.Should().Contain(i => i.Severity == IssueSeverity.Error,
            because: "rapid same-parity cut should produce an error");
    }

    [Fact]
    public void ParityRule_GoodAlternation_NoErrors()
    {
        // Alternating Down / Up every beat (natural parity)
        var notes = Enumerable.Range(0, 8).Select(i =>
            new CanonicalNote(i, 1, 1, NoteColor.Red,
                i % 2 == 0 ? CutDirection.Down : CutDirection.Up)
        ).ToList();
        var rule = new ParityRule();
        var issues = rule.Validate(MakeBeatmap(notes: notes))
            .Where(i => i.Severity == IssueSeverity.Error).ToList();
        issues.Should().BeEmpty(because: "clean alternating parity should produce no errors");
    }

    // -----------------------------------------------------------------------
    // DensityRule
    // -----------------------------------------------------------------------

    [Fact]
    public void DensityRule_TooManyNotes_ProducesWarningOrError()
    {
        // Pack 60 notes into 2 beats (way above any limit)
        var notes = Enumerable.Range(0, 60)
            .Select(i => new CanonicalNote(i * 0.033, 1, 1, NoteColor.Red, CutDirection.Down))
            .ToList();
        var rule = new DensityRule();
        var issues = rule.Validate(MakeBeatmap(notes: notes)).ToList();
        issues.Should().NotBeEmpty(because: "extreme density should raise issues");
    }

    [Fact]
    public void DensityRule_LowDensityMap_NoIssues()
    {
        // One note per beat for 16 beats — well within hard limit
        var notes = Enumerable.Range(0, 16)
            .Select(i => new CanonicalNote(i, i % 4, 1,
                i % 2 == 0 ? NoteColor.Red : NoteColor.Blue, CutDirection.Down))
            .ToList();
        var rule = new DensityRule();
        var densityIssues = rule.Validate(MakeBeatmap(notes: notes))
            .Where(i => i.Severity == IssueSeverity.Error).ToList();
        densityIssues.Should().BeEmpty();
    }

    // -----------------------------------------------------------------------
    // LaneBalanceRule
    // -----------------------------------------------------------------------

    [Fact]
    public void LaneBalanceRule_AllNotesInLane0_ProducesWarning()
    {
        var notes = Enumerable.Range(0, 20)
            .Select(i => new CanonicalNote(i, 0, 1,
                i % 2 == 0 ? NoteColor.Red : NoteColor.Blue, CutDirection.Down))
            .ToList();
        var rule = new LaneBalanceRule();
        var issues = rule.Validate(MakeBeatmap(notes: notes)).ToList();
        issues.Should().Contain(i => i.Severity == IssueSeverity.Warning,
            because: "100% of notes in one lane should warn");
    }

    [Fact]
    public void LaneBalanceRule_BalancedMap_NoWarnings()
    {
        var notes = Enumerable.Range(0, 20).Select(i =>
            new CanonicalNote(i, i % 4, 1,
                i % 2 == 0 ? NoteColor.Red : NoteColor.Blue,
                CutDirection.Down)).ToList();
        var rule = new LaneBalanceRule();
        var issues = rule.Validate(MakeBeatmap(notes: notes)).ToList();
        issues.Where(i => i.RuleName == "LaneBalance").Should().BeEmpty();
    }

    // -----------------------------------------------------------------------
    // Full Validator
    // -----------------------------------------------------------------------

    [Fact]
    public void Validator_EmptyMap_IsValid()
    {
        var validator = new BeatmapValidator();
        var report = validator.Validate(MakeBeatmap());
        // Empty map has no errors (only possible warnings from envelope rule)
        report.ErrorCount.Should().Be(0);
    }

    [Fact]
    public void Validator_SmallCleanMap_HasPositiveScore()
    {
        // Properly alternating parity: each hand alternates Down/Up
        var notes = Enumerable.Range(0, 60).Select(i =>
        {
            bool isLeft = i % 2 == 0;
            int handIndex = i / 2; // index within each hand
            NoteColor color = isLeft ? NoteColor.Red : NoteColor.Blue;
            int lane = isLeft ? (handIndex % 2) : (2 + handIndex % 2);
            // Alternate Down/Up within each hand for good parity
            CutDirection dir = handIndex % 2 == 0 ? CutDirection.Down : CutDirection.Up;
            return new CanonicalNote(i * 0.5, lane, 1, color, dir);
        }).ToList();
        var validator = new BeatmapValidator();
        var report = validator.Validate(MakeBeatmap(notes: notes));
        report.Score.Should().BeGreaterThan(0);
    }
}
