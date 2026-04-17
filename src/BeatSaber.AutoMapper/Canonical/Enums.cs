namespace BeatSaber.AutoMapper.Canonical;

public enum NoteColor { Red = 0, Blue = 1 }

public enum CutDirection
{
    Up = 0, Down = 1, Left = 2, Right = 3,
    UpLeft = 4, UpRight = 5, DownLeft = 6, DownRight = 7, Dot = 8
}

public enum NoteHand { Left = 0, Right = 1 }

public enum IssueSeverity { Info, Warning, Error }

public enum ParityClass { Forehand, Backhand }

public enum ParityTransition
{
    GoodFlow,
    Acceptable,
    RecoveryRequired,
    ParityBreak,
    AwkwardReset,
    HighStrain,
    Invalid
}

public enum SectionType { Intro, Verse, Chorus, Bridge, Buildup, Drop, Outro, Unknown }
