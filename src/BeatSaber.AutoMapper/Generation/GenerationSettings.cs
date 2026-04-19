namespace BeatSaber.AutoMapper.Generation;

public sealed record GenerationSettings(
    DifficultyLevel TargetDifficulty,
    bool AllowBombs,
    bool AllowObstacles,
    bool AllowFieldMovement,
    double RandomSeed,
    bool UseLearned,
    string? ArtifactsPath
)
{
    public static GenerationSettings Default(DifficultyLevel difficulty) => new(
        TargetDifficulty: difficulty,
        AllowBombs: false,
        AllowObstacles: false,
        AllowFieldMovement: false,
        RandomSeed: 42,
        UseLearned: false,
        ArtifactsPath: null
    );
}
