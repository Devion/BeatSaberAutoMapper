namespace BeatSaber.AutoMapper.Generation;

public sealed record GenerationSettings(
    DifficultyLevel TargetDifficulty,
    bool AllowBombs,
    bool AllowObstacles,
    double RandomSeed,
    bool UseLearned,
    string? ArtifactsPath
)
{
    public static GenerationSettings Default(DifficultyLevel difficulty) => new(
        TargetDifficulty: difficulty,
        AllowBombs: false,
        AllowObstacles: false,
        RandomSeed: 42,
        UseLearned: false,
        ArtifactsPath: null
    );
}
