namespace BeatSaber.AutoMapper.Canonical;

/// <summary>
/// Represents a single note in the canonical model.
/// Lane: 0-3 (left to right), Row: 0-2 (bottom to top).
/// </summary>
public sealed record CanonicalNote(
    double Beat,
    int Lane,
    int Row,
    NoteColor Color,
    CutDirection CutDirection
)
{
    public NoteHand Hand => Color == NoteColor.Red ? NoteHand.Left : NoteHand.Right;

    /// <summary>Returns true if this note and <paramref name="other"/> occupy the same beat, lane and row.</summary>
    public bool ConflictsWith(CanonicalNote other) =>
        Math.Abs(Beat - other.Beat) < 0.001 && Lane == other.Lane && Row == other.Row;
}
