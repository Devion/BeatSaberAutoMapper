namespace BeatSaber.AutoMapper.Training;

public sealed record TrainingOptions(
    string DatasetPath,
    string ProfileName,
    string ArtifactsOutputPath,
    int Epochs,
    double TrainFraction,
    double ValidationFraction,
    double TestFraction,
    long RandomSeed,
    int ValidationSongsPerEpoch,
    /// <summary>
    /// How many songs to pre-analyse and cache. Must be ≥ ValidationSongsPerEpoch.
    /// Each training run randomly picks ValidationSongsPerEpoch from this pool.
    /// Grow the pool once (e.g. --validation-cache-size 200) then reuse across restarts.
    /// </summary>
    int ValidationCachePoolSize,
    double InitialLearningRate,
    int EarlyStopPatience,
    // Self-supervised feedback options
    int SelfSupervisedWarmupEpochs,
    int SelfSupervisedEveryNEpochs,
    double SelfSupervisedPositiveWeight,
    double SelfSupervisedNegativeWeight,
    // Periodic checkpoint
    int CheckpointEveryNEpochs = 10
)
{
    public static TrainingOptions Default => new(
        DatasetPath: "dataset",
        ProfileName: "baseline",
        ArtifactsOutputPath: "artifacts",
        Epochs: 100,
        TrainFraction: 0.8,
        ValidationFraction: 0.1,
        TestFraction: 0.1,
        RandomSeed: 42,
        ValidationSongsPerEpoch: 3,
        ValidationCachePoolSize: 3,
        InitialLearningRate: 0.001,
        EarlyStopPatience: 50,
        SelfSupervisedWarmupEpochs: 5,
        SelfSupervisedEveryNEpochs: 1,
        SelfSupervisedPositiveWeight: 4.0,
        SelfSupervisedNegativeWeight: 3.0,
        CheckpointEveryNEpochs: 10
    );
}
