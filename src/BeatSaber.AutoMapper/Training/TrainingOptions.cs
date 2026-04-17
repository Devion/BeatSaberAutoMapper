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
    double InitialLearningRate,
    int EarlyStopPatience,
    // Self-supervised feedback options
    int SelfSupervisedWarmupEpochs,
    int SelfSupervisedEveryNEpochs,
    double SelfSupervisedPositiveWeight,
    double SelfSupervisedNegativeWeight
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
        InitialLearningRate: 0.001,
        EarlyStopPatience: 20,
        SelfSupervisedWarmupEpochs: 5,
        SelfSupervisedEveryNEpochs: 1,
        SelfSupervisedPositiveWeight: 4.0,
        SelfSupervisedNegativeWeight: 3.0
    );
}
