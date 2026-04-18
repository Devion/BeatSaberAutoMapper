namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Scores a candidate beat position for note placement.
/// Implemented by both the in-memory trainer (during training) and the loaded model (at generation time).
/// </summary>
public interface IPlacementScorer
{
    /// <summary>Returns a probability-like score in [0, 1] for placing a note here.</summary>
    double ScorePlacement(double onset, double energy, double subdiv,
                          double localNps, double beatStrength, int measureBeat,
                          int difficultyLevel);

    /// <summary>
    /// Extended scoring with full sequential context.
    /// The default implementation delegates to the 7-arg version.
    /// Neural-network models override this to use all 45 features.
    /// </summary>
    double ScorePlacement(in NeuralPlacementContext ctx) =>
        ScorePlacement(ctx.OnsetStrength, ctx.EnergyLevel, ctx.Subdiv, ctx.LocalNps,
                       ctx.BeatStrength, ctx.MeasureBeat, ctx.DifficultyLevel);
}
