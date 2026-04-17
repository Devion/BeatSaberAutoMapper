namespace BeatSaber.AutoMapper.Generation;

public sealed record DifficultyProfile(
    DifficultyLevel Difficulty,
    double MaxNps,
    double TargetNps,
    double BurstFrequency,
    bool AllowCrossovers,
    bool AllowDiagonals,
    double ResetToleranceBeats,
    double EmphasisDensity,
    int MaxConsecutiveSameHand
)
{
    public static DifficultyProfile Easy =>
        new(DifficultyLevel.Easy, 2.5, 1.5, 0.0, false, false, 4.0, 0.1, 2);

    public static DifficultyProfile Normal =>
        new(DifficultyLevel.Normal, 4.0, 2.5, 0.05, false, false, 3.0, 0.2, 2);

    public static DifficultyProfile Hard =>
        new(DifficultyLevel.Hard, 5.0, 3.5, 0.1, false, true, 2.0, 0.3, 3);

    public static DifficultyProfile Expert =>
        new(DifficultyLevel.Expert, 8.0, 6.0, 0.2, true, true, 1.5, 0.5, 4);

    public static DifficultyProfile ExpertPlus =>
        new(DifficultyLevel.ExpertPlus, 14.0, 10.0, 0.35, true, true, 1.0, 0.7, 5);

    public static DifficultyProfile For(DifficultyLevel level) => level switch
    {
        DifficultyLevel.Easy => Easy,
        DifficultyLevel.Normal => Normal,
        DifficultyLevel.Hard => Hard,
        DifficultyLevel.Expert => Expert,
        DifficultyLevel.ExpertPlus => ExpertPlus,
        _ => Hard
    };
}
