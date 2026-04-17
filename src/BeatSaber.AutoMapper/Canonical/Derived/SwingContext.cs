namespace BeatSaber.AutoMapper.Canonical.Derived;

/// <summary>Tracks the swing state for a single hand during note placement or analysis.</summary>
public sealed class SwingContext
{
    public NoteHand Hand { get; }
    public CutDirection? LastCutDirection { get; private set; }
    public double LastBeat { get; private set; } = -1.0;
    public ParityClass CurrentParity { get; private set; } = ParityClass.Forehand;
    public int Lane { get; private set; } = 1;
    public int Row { get; private set; } = 0;

    public SwingContext(NoteHand hand) => Hand = hand;

    /// <summary>Create a copy of this context (for beam-search branching).</summary>
    public SwingContext Clone()
    {
        var c = new SwingContext(Hand)
        {
            LastCutDirection = LastCutDirection,
            LastBeat = LastBeat,
            CurrentParity = CurrentParity,
            Lane = Lane,
            Row = Row
        };
        return c;
    }

    /// <summary>Update the context after placing/processing <paramref name="note"/>.</summary>
    public void Update(CanonicalNote note)
    {
        LastCutDirection = note.CutDirection;
        LastBeat = note.Beat;
        Lane = note.Lane;
        Row = note.Row;
        // After the cut the parity flips (unless it's a dot)
        if (note.CutDirection != CutDirection.Dot)
            CurrentParity = ExpectedParityAfter(note.CutDirection);
    }

    /// <summary>
    /// Returns the parity class the hand will be in after completing a swing
    /// in <paramref name="direction"/>.
    /// Forehand cuts end with the hand ready for a Backhand swing and vice-versa.
    /// </summary>
    public ParityClass ExpectedParityAfter(CutDirection direction)
    {
        var entry = ParityAnalyzer.CutDirectionParity(direction);
        return entry == ParityClass.Forehand ? ParityClass.Backhand : ParityClass.Forehand;
    }
}
