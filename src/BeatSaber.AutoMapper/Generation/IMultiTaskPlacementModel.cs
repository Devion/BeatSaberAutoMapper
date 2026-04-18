namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Extended learned placement model that exposes all neural heads, not just placement.
/// </summary>
public interface IMultiTaskPlacementModel : IPlacementScorer
{
    NeuralMapPrediction PredictAll(in NeuralPlacementContext ctx);
}
