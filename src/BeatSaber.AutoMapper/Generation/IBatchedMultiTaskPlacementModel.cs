namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Optional extension for models that can score multiple placement contexts in one call.
/// This is primarily used to batch training-time validation/generation work on GPU.
/// </summary>
public interface IBatchedMultiTaskPlacementModel : IMultiTaskPlacementModel
{
    IReadOnlyList<NeuralMapPrediction> PredictAllBatch(IReadOnlyList<NeuralPlacementContext> contexts);
}
