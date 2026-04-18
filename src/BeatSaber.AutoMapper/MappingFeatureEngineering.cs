namespace BeatSaber.AutoMapper;

internal static class MappingFeatureEngineering
{
    public static (CanonicalNote? Last, CanonicalNote? Prev2) LastTwoForHand(
        IReadOnlyList<CanonicalNote> notes,
        NoteHand hand)
    {
        CanonicalNote? last = null, prev2 = null;
        for (int i = notes.Count - 1; i >= 0; i--)
        {
            var note = notes[i];
            if (note.Hand != hand) continue;
            if (last is null) last = note;
            else { prev2 = note; break; }
        }
        return (last, prev2);
    }

    public static (double BeatsSinceStart, double BeatsToBoundary) SectionBoundaryFeatures(
        IReadOnlyList<SectionMarker> sections,
        double beat)
    {
        if (sections.Count == 0) return (0, 16);

        SectionMarker current = sections[0];
        SectionMarker? next = null;
        for (int i = 0; i < sections.Count; i++)
        {
            if (sections[i].Beat <= beat) current = sections[i];
            else { next = sections[i]; break; }
        }

        double sinceStart = Math.Max(0, beat - current.Beat);
        double toBoundary = next is not null ? Math.Max(0, next.Beat - beat) : 16.0;
        return (sinceStart, toBoundary);
    }

    public static double RecentChordRate(IReadOnlyList<CanonicalNote> history, double beat, double windowBeats)
    {
        var recent = history.Where(n => n.Beat >= beat - windowBeats && n.Beat < beat).ToList();
        if (recent.Count == 0) return 0;
        int chordBeats = recent
            .GroupBy(n => Math.Round(n.Beat, 3))
            .Count(g => g.Select(n => n.Hand).Distinct().Count() >= 2);
        double denom = Math.Max(1.0, windowBeats * 4.0);
        return Math.Clamp(chordBeats / denom, 0.0, 1.0);
    }

    public static double RecentOffbeatRate(IReadOnlyList<CanonicalNote> history, double beat, double windowBeats)
    {
        var recent = history.Where(n => n.Beat >= beat - windowBeats && n.Beat < beat).ToList();
        if (recent.Count == 0) return 0;
        int offbeats = recent.Count(n => !IsOnQuarterBeat(n.Beat));
        return Math.Clamp(offbeats / (double)recent.Count, 0.0, 1.0);
    }

    public static double RecentStreamRate(IReadOnlyList<CanonicalNote> history, double beat, double windowBeats)
    {
        var recent = history.Where(n => n.Beat >= beat - windowBeats && n.Beat < beat)
            .OrderBy(n => n.Beat)
            .ToList();
        if (recent.Count < 2) return 0;
        int streamTransitions = 0;
        for (int i = 1; i < recent.Count; i++)
            if (recent[i].Beat - recent[i - 1].Beat <= 0.5) streamTransitions++;
        return Math.Clamp(streamTransitions / (double)(recent.Count - 1), 0.0, 1.0);
    }

    public static double RecentAlternation(IReadOnlyList<CanonicalNote> history, int recentNotes)
    {
        var recent = history.OrderByDescending(n => n.Beat).Take(recentNotes).Reverse().ToList();
        if (recent.Count < 2) return 0.5;
        int changes = 0;
        for (int i = 1; i < recent.Count; i++)
            if (recent[i].Hand != recent[i - 1].Hand) changes++;
        return Math.Clamp(changes / (double)(recent.Count - 1), 0.0, 1.0);
    }

    public static double RecentHandBalance(IReadOnlyList<CanonicalNote> history, double beat, double windowBeats)
    {
        var recent = history.Where(n => n.Beat >= beat - windowBeats && n.Beat < beat).ToList();
        if (recent.Count == 0) return 0.5;
        int right = recent.Count(n => n.Hand == NoteHand.Right);
        return Math.Clamp(right / (double)recent.Count, 0.0, 1.0);
    }

    public static double ConsecutiveSameHandCount(IReadOnlyList<CanonicalNote> history)
    {
        if (history.Count == 0) return 0;
        var hand = history[^1].Hand;
        int count = 1;
        for (int i = history.Count - 2; i >= 0; i--)
        {
            if (history[i].Hand != hand) break;
            count++;
        }
        return count;
    }

    public static double BeatsSinceLastAny(IReadOnlyList<CanonicalNote> history, double beat) =>
        history.Count == 0 ? 999 : Math.Max(0, beat - history[^1].Beat);

    public static double NotesAtCurrentBeatSoFar(IReadOnlyList<CanonicalNote> history, double beat) =>
        history.Count(n => Math.Abs(n.Beat - beat) < 0.001);

    public static double InterHandLaneDistance(CanonicalNote? left, CanonicalNote? right) =>
        left is null || right is null ? 1.5 : Math.Abs(right.Lane - left.Lane);

    public static double InterHandRowDistance(CanonicalNote? left, CanonicalNote? right) =>
        left is null || right is null ? 1.0 : Math.Abs(right.Row - left.Row);

    public static double HandsCrossedFlag(CanonicalNote? left, CanonicalNote? right) =>
        left is not null && right is not null && left.Lane > right.Lane ? 1.0 : 0.0;

    public static double RecentTravel(CanonicalNote? prev2, CanonicalNote? last)
    {
        if (prev2 is null || last is null) return 0;
        return Math.Abs(last.Lane - prev2.Lane) + Math.Abs(last.Row - prev2.Row);
    }

    public static double RecentLaneSpan(IReadOnlyList<CanonicalNote> history, double beat, double windowBeats)
    {
        var recent = history.Where(n => n.Beat >= beat - windowBeats && n.Beat < beat).ToList();
        if (recent.Count == 0) return 0;
        return recent.Max(n => n.Lane) - recent.Min(n => n.Lane);
    }

    public static double RecentRowSpan(IReadOnlyList<CanonicalNote> history, double beat, double windowBeats)
    {
        var recent = history.Where(n => n.Beat >= beat - windowBeats && n.Beat < beat).ToList();
        if (recent.Count == 0) return 0;
        return recent.Max(n => n.Row) - recent.Min(n => n.Row);
    }

    private static bool IsOnQuarterBeat(double beat)
    {
        double frac = beat - Math.Floor(beat);
        return frac < 0.01 || Math.Abs(frac - 0.25) < 0.01 || Math.Abs(frac - 0.5) < 0.01 || Math.Abs(frac - 0.75) < 0.01;
    }
}
