namespace BeatSaber.AutoMapper.Canonical;

public enum DifficultyLevel { Easy, Normal, Hard, Expert, ExpertPlus }

public enum BeatmapCharacteristic
{
    Standard,
    OneSaber,
    NoArrows,
    NinetyDegree,
    ThreeSixtyDegree,
    Lightshow,
    Lawless
}

public sealed record DifficultyDescriptor(
    DifficultyLevel Difficulty,
    BeatmapCharacteristic Characteristic,
    double? NoteJumpMovementSpeed,
    double? NoteJumpStartBeatOffset,
    string? CustomLabel
);
