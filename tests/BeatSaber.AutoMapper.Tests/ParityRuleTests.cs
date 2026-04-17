using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Canonical.Derived;
using FluentAssertions;

namespace BeatSaber.AutoMapper.Tests;

public sealed class ParityRuleTests
{
    // ---------------------------------------------------------------------------
    // CutDirectionParity
    // ---------------------------------------------------------------------------

    [Theory]
    [InlineData(CutDirection.Up, ParityClass.Backhand)]
    [InlineData(CutDirection.UpLeft, ParityClass.Backhand)]
    [InlineData(CutDirection.UpRight, ParityClass.Backhand)]
    [InlineData(CutDirection.Left, ParityClass.Backhand)]
    [InlineData(CutDirection.Right, ParityClass.Backhand)]
    [InlineData(CutDirection.Down, ParityClass.Forehand)]
    [InlineData(CutDirection.DownLeft, ParityClass.Forehand)]
    [InlineData(CutDirection.DownRight, ParityClass.Forehand)]
    public void CutDirectionParity_ReturnsExpected(CutDirection dir, ParityClass expected)
    {
        ParityAnalyzer.CutDirectionParity(dir).Should().Be(expected);
    }

    // ---------------------------------------------------------------------------
    // IsNaturalTransition
    // ---------------------------------------------------------------------------

    [Fact]
    public void IsNaturalTransition_UpToDown_IsNatural()
    {
        ParityAnalyzer.IsNaturalTransition(CutDirection.Up, CutDirection.Down).Should().BeTrue();
    }

    [Fact]
    public void IsNaturalTransition_UpToUp_IsNotNatural()
    {
        ParityAnalyzer.IsNaturalTransition(CutDirection.Up, CutDirection.Up).Should().BeFalse();
    }

    [Fact]
    public void IsNaturalTransition_DotIsAlwaysAcceptable()
    {
        ParityAnalyzer.IsNaturalTransition(CutDirection.Up, CutDirection.Dot).Should().BeTrue();
        ParityAnalyzer.IsNaturalTransition(CutDirection.Dot, CutDirection.Down).Should().BeTrue();
    }

    // ---------------------------------------------------------------------------
    // ClassifyTransition
    // ---------------------------------------------------------------------------

    [Fact]
    public void ClassifyTransition_FirstNote_IsGoodFlow()
    {
        var ctx = new SwingContext(NoteHand.Left);
        var note = new CanonicalNote(0.0, 1, 1, NoteColor.Red, CutDirection.Down);
        var result = ParityAnalyzer.ClassifyTransition(null, note, ctx);
        result.Transition.Should().Be(ParityTransition.GoodFlow);
    }

    [Fact]
    public void ClassifyTransition_NaturalAlternation_IsGoodFlow()
    {
        var ctx = new SwingContext(NoteHand.Left);
        var first = new CanonicalNote(0.0, 1, 1, NoteColor.Red, CutDirection.Down);
        ctx.Update(first);

        // After a Down (forehand) cut, wrist is ready for backhand (Up)
        var second = new CanonicalNote(1.0, 1, 1, NoteColor.Red, CutDirection.Up);
        var result = ParityAnalyzer.ClassifyTransition(first, second, ctx);
        result.Transition.Should().Be(ParityTransition.GoodFlow);
    }

    [Fact]
    public void ClassifyTransition_SameDirection_QuickGap_IsInvalid()
    {
        var ctx = new SwingContext(NoteHand.Left);
        var first = new CanonicalNote(0.0, 1, 1, NoteColor.Red, CutDirection.Down);
        ctx.Update(first);

        // Immediately another Down cut — parity break with tiny gap
        var second = new CanonicalNote(0.1, 1, 1, NoteColor.Red, CutDirection.Down);
        var result = ParityAnalyzer.ClassifyTransition(first, second, ctx);
        result.Transition.Should().BeOneOf(
            ParityTransition.ParityBreak,
            ParityTransition.Invalid,
            ParityTransition.AwkwardReset,
            ParityTransition.RecoveryRequired);
    }

    [Fact]
    public void ClassifyTransition_DotNote_IsGoodFlow()
    {
        var ctx = new SwingContext(NoteHand.Left);
        var first = new CanonicalNote(0.0, 1, 1, NoteColor.Red, CutDirection.Down);
        ctx.Update(first);

        var dot = new CanonicalNote(0.5, 1, 1, NoteColor.Red, CutDirection.Dot);
        var result = ParityAnalyzer.ClassifyTransition(first, dot, ctx);
        result.Transition.Should().Be(ParityTransition.GoodFlow);
    }

    [Fact]
    public void ClassifyTransition_SameDirectionWithLargeGap_IsRecoveryOrBetter()
    {
        var ctx = new SwingContext(NoteHand.Left);
        var first = new CanonicalNote(0.0, 1, 1, NoteColor.Red, CutDirection.Down);
        ctx.Update(first);

        // Same cut direction but 3 beats later → enough time to reset
        var second = new CanonicalNote(3.0, 1, 1, NoteColor.Red, CutDirection.Down);
        var result = ParityAnalyzer.ClassifyTransition(first, second, ctx);
        // Should be RecoveryRequired (2+ beats gap allows reset)
        result.Transition.Should().BeOneOf(
            ParityTransition.RecoveryRequired,
            ParityTransition.GoodFlow,
            ParityTransition.Acceptable);
    }
}
