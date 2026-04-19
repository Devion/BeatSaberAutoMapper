using BeatSaber.AutoMapper.Canonical.Derived;

namespace BeatSaber.AutoMapper.Validation;

internal static class PlayabilityHeuristics
{
    public const double BeatTolerance = 0.001;

    public static bool SameBeat(CanonicalNote a, CanonicalNote b) =>
        Math.Abs(a.Beat - b.Beat) < BeatTolerance;

    public static bool IsVisionBlock(CanonicalNote a, CanonicalNote b)
    {
        bool sameLane = a.Lane == b.Lane;
        bool nearCentre = a.Lane is 1 or 2;
        bool differentRows = a.Row != b.Row;
        return sameLane && nearCentre && differentRows;
    }

    public static bool IsHandclap(CanonicalNote a, CanonicalNote b)
    {
        if (!SameBeat(a, b) || a.Hand == b.Hand)
            return false;

        CanonicalNote left = a.Hand == NoteHand.Left ? a : b;
        CanonicalNote right = a.Hand == NoteHand.Right ? a : b;
        bool inwardLeft = left.CutDirection is CutDirection.Right or CutDirection.UpRight or CutDirection.DownRight;
        bool inwardRight = right.CutDirection is CutDirection.Left or CutDirection.UpLeft or CutDirection.DownLeft;
        return inwardLeft && inwardRight && Math.Abs(left.Lane - right.Lane) <= 2;
    }

    public static bool IsExcessiveDouble(IReadOnlyList<CanonicalNote> notesAtBeat) =>
        notesAtBeat.Count >= 2;

    public static bool IsDoubleDirectional(CanonicalNote previous, CanonicalNote next) =>
        previous.Hand == next.Hand &&
        next.CutDirection != CutDirection.Dot &&
        previous.CutDirection == next.CutDirection &&
        next.Beat - previous.Beat <= 1.0;

    public static bool IsHitboxPath(CanonicalNote previous, CanonicalNote next) =>
        next.Beat > previous.Beat &&
        next.Beat - previous.Beat <= 0.5 &&
        Math.Abs(next.Lane - previous.Lane) <= 1 &&
        Math.Abs(next.Row - previous.Row) <= 1;

    public static double ComputeSwingEbpm(CanonicalNote previous, CanonicalNote next, double bpm)
    {
        double gapBeats = Math.Max(0.001, next.Beat - previous.Beat);
        return bpm / gapBeats;
    }

    public static double SwingEbpmWarningThreshold(DifficultyLevel difficulty) => difficulty switch
    {
        DifficultyLevel.Easy => 100.0,
        DifficultyLevel.Normal => 120.0,
        DifficultyLevel.Hard => 140.0,
        DifficultyLevel.Expert => 170.0,
        DifficultyLevel.ExpertPlus => 210.0,
        _ => 140.0,
    };

    public static bool IsSwingSpeedWarning(CanonicalNote previous, CanonicalNote next, double bpm, DifficultyLevel difficulty) =>
        previous.Hand == next.Hand &&
        next.CutDirection != CutDirection.Dot &&
        ComputeSwingEbpm(previous, next, bpm) > SwingEbpmWarningThreshold(difficulty);

    public static bool WouldCauseImmediatePatternIssue(CanonicalNote note, IReadOnlyList<CanonicalNote> placedNotes, double bpm, DifficultyLevel difficulty)
    {
        var sameBeat = placedNotes.Where(n => SameBeat(n, note)).ToList();
        if (sameBeat.Any(existing => existing.Hand == note.Hand))
            return true;
        if (sameBeat.Any(existing => IsVisionBlock(existing, note) || IsHandclap(existing, note)))
            return true;

        CanonicalNote? prevSameHand = placedNotes
            .Where(n => n.Hand == note.Hand && n.Beat < note.Beat)
            .OrderByDescending(n => n.Beat)
            .FirstOrDefault();
        if (prevSameHand is not null)
        {
            if (IsDoubleDirectional(prevSameHand, note))
                return true;
            if (IsSwingSpeedWarning(prevSameHand, note, bpm, difficulty))
                return true;
        }

        if (placedNotes.Any(existing => IsHitboxPath(existing, note)))
            return true;

        return false;
    }

    public static CanonicalNote? PickNoteToRemoveAtBeat(IReadOnlyList<CanonicalNote> notesAtBeat)
    {
        if (notesAtBeat.Count == 0)
            return null;

        return notesAtBeat
            .OrderByDescending(n => n.Row)
            .ThenBy(n => Math.Abs(n.Lane - 1.5))
            .ThenByDescending(n => n.Hand == NoteHand.Right ? 1 : 0)
            .First();
    }
}
