using BeatSaber.AutoMapper.Canonical;

namespace BeatSaber.AutoMapper.Training.Patterns;

public enum PatternType
{
    Isolated = 0,
    Stream = 1,
    Chord = 2,
    Burst = 3,
    Reset = 4,
    Anchor = 5
}

public static class PatternModeling
{
    public const int PatternClassCount = 6;

    public static PatternType DerivePatternType(
        double beat,
        IReadOnlyList<CanonicalNote> notesAtBeat,
        IReadOnlyList<CanonicalNote> history)
    {
        if (notesAtBeat.Count >= 2)
            return PatternType.Chord;
        if (notesAtBeat.Count == 0)
            return PatternType.Isolated;

        CanonicalNote note = notesAtBeat[0];
        CanonicalNote? prev = history.OrderByDescending(n => n.Beat).FirstOrDefault();
        CanonicalNote? prevSameHand = history
            .Where(n => n.Hand == note.Hand)
            .OrderByDescending(n => n.Beat)
            .FirstOrDefault();
        double gap = prev is not null ? beat - prev.Beat : 999.0;
        double sameHandGap = prevSameHand is not null ? beat - prevSameHand.Beat : 999.0;

        if (gap >= 1.5)
            return PatternType.Reset;
        if (gap <= 0.25)
            return PatternType.Burst;
        if (gap <= 0.5)
            return PatternType.Stream;
        if (prevSameHand is not null && sameHandGap <= 1.0 && prevSameHand.Lane == note.Lane)
            return PatternType.Anchor;

        return PatternType.Isolated;
    }
}
