namespace BeatSaber.AutoMapper.Canonical.Derived;

public sealed record ParityTransitionResult(
    ParityTransition Transition,
    string? Reason
);

public static class ParityAnalyzer
{
    /// <summary>
    /// Classify the transition from <paramref name="previous"/> to <paramref name="next"/>
    /// for the hand tracked by <paramref name="context"/>.
    /// </summary>
    public static ParityTransitionResult ClassifyTransition(
        CanonicalNote? previous,
        CanonicalNote next,
        SwingContext context)
    {
        if (previous is null)
            return new(ParityTransition.GoodFlow, "First note");

        var expectedParity = context.CurrentParity;
        var requiredParity = CutDirectionParity(next.CutDirection);

        if (next.CutDirection == CutDirection.Dot)
            return new(ParityTransition.GoodFlow, "Dot note, parity neutral");

        double beatGap = next.Beat - previous.Beat;

        if (requiredParity == expectedParity)
            return new(ParityTransition.GoodFlow, null);

        // Parity mismatch — how bad is it?
        if (beatGap >= 2.0)
            return new(ParityTransition.RecoveryRequired, "Sufficient time to reset");

        if (beatGap >= 1.5)
            return new(ParityTransition.AwkwardReset, "Short reset window");

        if (beatGap >= 0.75)
            return new(ParityTransition.ParityBreak, "Very short gap for parity break");

        return new(ParityTransition.Invalid, $"Impossible reset in {beatGap:F2} beats");
    }

    /// <summary>
    /// Returns the natural entry parity class for a given cut direction.
    /// Up/UpLeft/UpRight/Left/Right → Backhand.
    /// Down/DownLeft/DownRight → Forehand.
    /// Dot → Forehand (neutral/don't-care).
    /// </summary>
    public static ParityClass CutDirectionParity(CutDirection dir) => dir switch
    {
        CutDirection.Up => ParityClass.Backhand,
        CutDirection.UpLeft => ParityClass.Backhand,
        CutDirection.UpRight => ParityClass.Backhand,
        CutDirection.Left => ParityClass.Backhand,
        CutDirection.Right => ParityClass.Backhand,
        CutDirection.Down => ParityClass.Forehand,
        CutDirection.DownLeft => ParityClass.Forehand,
        CutDirection.DownRight => ParityClass.Forehand,
        CutDirection.Dot => ParityClass.Forehand,
        _ => ParityClass.Forehand
    };

    /// <summary>
    /// Returns true when transitioning from <paramref name="from"/> to <paramref name="to"/>
    /// represents a physically natural wrist motion.
    /// </summary>
    public static bool IsNaturalTransition(CutDirection from, CutDirection to)
    {
        var p1 = CutDirectionParity(from);
        var p2 = CutDirectionParity(to);
        // Natural = parity alternates (forehand → backhand or backhand → forehand)
        // Dot is always acceptable
        if (to == CutDirection.Dot || from == CutDirection.Dot) return true;
        return p1 != p2;
    }
}
