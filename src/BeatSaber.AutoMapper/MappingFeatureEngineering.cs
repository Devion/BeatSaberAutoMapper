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

    public static double RecentRestRatio(IReadOnlyList<CanonicalNote> history, double beat, double windowBeats, double stepBeats = 0.25)
    {
        if (windowBeats <= 0 || stepBeats <= 0)
            return 0.0;

        int totalSlots = 0;
        int emptySlots = 0;
        double start = Math.Max(0.0, beat - windowBeats);
        for (double b = start; b < beat; b += stepBeats)
        {
            totalSlots++;
            bool hasNote = history.Any(n => Math.Abs(n.Beat - b) < 0.001);
            if (!hasNote)
                emptySlots++;
        }

        return totalSlots > 0 ? Math.Clamp(emptySlots / (double)totalSlots, 0.0, 1.0) : 0.0;
    }

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

    public static double CurrentBeatVisionBlockRisk(IReadOnlyList<CanonicalNote> history, double beat)
    {
        var notesAtBeat = history
            .Where(n => Math.Abs(n.Beat - beat) < 0.001)
            .ToList();
        if (notesAtBeat.Count == 0)
            return 0.0;

        bool centerOccupied = notesAtBeat.Any(n => n.Lane is 1 or 2);
        bool stackedCenter = notesAtBeat
            .GroupBy(n => n.Lane)
            .Any(g => (g.Key is 1 or 2) && g.Select(n => n.Row).Distinct().Count() > 1);

        if (stackedCenter)
            return 1.0;
        if (centerOccupied)
            return 0.6;
        return 0.2;
    }

    public static double RecentVisionBlockRate(IReadOnlyList<CanonicalNote> history, double beat, double windowBeats)
    {
        var recent = history.Where(n => n.Beat >= beat - windowBeats && n.Beat < beat).ToList();
        if (recent.Count < 2)
            return 0.0;

        int blockedBeats = recent
            .GroupBy(n => Math.Round(n.Beat, 3))
            .Count(g =>
            {
                var notes = g.ToList();
                for (int i = 0; i < notes.Count; i++)
                for (int j = i + 1; j < notes.Count; j++)
                    if (Validation.PlayabilityHeuristics.IsVisionBlock(notes[i], notes[j]))
                        return true;
                return false;
            });

        return Math.Clamp(blockedBeats / Math.Max(1.0, windowBeats), 0.0, 1.0);
    }

    public static double RecentParityBreakRate(IReadOnlyList<CanonicalNote> history, NoteHand hand, int recentTransitions)
    {
        var notes = history
            .Where(n => n.Hand == hand)
            .OrderBy(n => n.Beat)
            .ToList();
        if (notes.Count < 2)
            return 0.0;

        int start = Math.Max(1, notes.Count - recentTransitions);
        int total = 0;
        int bad = 0;
        var ctx = new Canonical.Derived.SwingContext(hand);
        ctx.Update(notes[0]);

        for (int i = 1; i < notes.Count; i++)
        {
            var result = Canonical.Derived.ParityAnalyzer.ClassifyTransition(notes[i - 1], notes[i], ctx);
            if (i >= start)
            {
                total++;
                if (result.Transition is ParityTransition.AwkwardReset or ParityTransition.ParityBreak or ParityTransition.Invalid)
                    bad++;
            }
            ctx.Update(notes[i]);
        }

        return total > 0 ? Math.Clamp(bad / (double)total, 0.0, 1.0) : 0.0;
    }

    public static double ImmediateResetPressure(CanonicalNote? last, double beat, DifficultyLevel difficulty)
    {
        if (last is null)
            return 0.0;

        double gap = Math.Max(0.0, beat - last.Beat);
        double comfortableGap = difficulty switch
        {
            DifficultyLevel.Easy => 1.5,
            DifficultyLevel.Normal => 1.25,
            DifficultyLevel.Hard => 1.0,
            DifficultyLevel.Expert => 0.75,
            DifficultyLevel.ExpertPlus => 0.5,
            _ => 1.0
        };

        return Math.Clamp(1.0 - (gap / comfortableGap), 0.0, 1.0);
    }

    private static bool IsOnQuarterBeat(double beat)
    {
        double frac = beat - Math.Floor(beat);
        return frac < 0.01 || Math.Abs(frac - 0.25) < 0.01 || Math.Abs(frac - 0.5) < 0.01 || Math.Abs(frac - 0.75) < 0.01;
    }
}
