namespace BeatSaber.AutoMapper.Generation.Patterns;

public sealed record PatternNote(
    double BeatOffset,
    int LaneOffset,
    int RowOffset,
    CutDirection Direction,
    NoteColor Color
);

public sealed record NotePattern(string Name, IReadOnlyList<PatternNote> Notes);

public static class PatternLibrary
{
    /// <summary>Simple alternating left-right patterns.</summary>
    public static IReadOnlyList<NotePattern> AltPatterns { get; } =
    [
        new NotePattern("Alt_UD", [
            new PatternNote(0.0, 1, 1, CutDirection.Up,   NoteColor.Red),
            new PatternNote(0.5, 2, 1, CutDirection.Down, NoteColor.Blue)
        ]),
        new NotePattern("Alt_LR", [
            new PatternNote(0.0, 0, 1, CutDirection.Left,  NoteColor.Red),
            new PatternNote(0.5, 3, 1, CutDirection.Right, NoteColor.Blue)
        ]),
        new NotePattern("Alt_Diag", [
            new PatternNote(0.0, 1, 0, CutDirection.UpLeft,   NoteColor.Red),
            new PatternNote(0.5, 2, 2, CutDirection.DownRight, NoteColor.Blue)
        ])
    ];

    /// <summary>Simultaneous two-note patterns.</summary>
    public static IReadOnlyList<NotePattern> DoublePatterns { get; } =
    [
        new NotePattern("Double_UD", [
            new PatternNote(0.0, 1, 1, CutDirection.Up,   NoteColor.Red),
            new PatternNote(0.0, 2, 1, CutDirection.Up,   NoteColor.Blue)
        ]),
        new NotePattern("Double_InOut", [
            new PatternNote(0.0, 1, 1, CutDirection.Left,  NoteColor.Red),
            new PatternNote(0.0, 2, 1, CutDirection.Right, NoteColor.Blue)
        ])
    ];

    /// <summary>Fast burst sequences.</summary>
    public static IReadOnlyList<NotePattern> BurstPatterns { get; } =
    [
        new NotePattern("Burst4", [
            new PatternNote(0.000, 1, 1, CutDirection.Up,   NoteColor.Red),
            new PatternNote(0.25,  2, 1, CutDirection.Down, NoteColor.Blue),
            new PatternNote(0.500, 1, 1, CutDirection.Up,   NoteColor.Red),
            new PatternNote(0.75,  2, 1, CutDirection.Down, NoteColor.Blue)
        ]),
        new NotePattern("Burst8", [
            new PatternNote(0.000, 1, 1, CutDirection.Up,    NoteColor.Red),
            new PatternNote(0.125, 2, 1, CutDirection.Down,  NoteColor.Blue),
            new PatternNote(0.250, 1, 0, CutDirection.UpLeft,NoteColor.Red),
            new PatternNote(0.375, 2, 2, CutDirection.Down,  NoteColor.Blue),
            new PatternNote(0.500, 1, 1, CutDirection.Up,    NoteColor.Red),
            new PatternNote(0.625, 2, 1, CutDirection.Down,  NoteColor.Blue),
            new PatternNote(0.750, 1, 0, CutDirection.UpLeft,NoteColor.Red),
            new PatternNote(0.875, 2, 2, CutDirection.Down,  NoteColor.Blue)
        ])
    ];

    public static NotePattern? GetPattern(string name)
    {
        foreach (var p in AltPatterns) if (p.Name == name) return p;
        foreach (var p in DoublePatterns) if (p.Name == name) return p;
        foreach (var p in BurstPatterns) if (p.Name == name) return p;
        return null;
    }
}
