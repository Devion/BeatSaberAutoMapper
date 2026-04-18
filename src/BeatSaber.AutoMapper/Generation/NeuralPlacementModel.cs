using BeatSaber.AutoMapper.Training.Models;
using static TorchSharp.torch;

namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// All outputs of one multi-task neural forward pass.
/// Produced by <see cref="NeuralPlacementModel.PredictAll"/>.
/// </summary>
public readonly struct NeuralMapPrediction
{
    public double   PlacementScore { get; init; }  // P(note here)  — sigmoid
    public double   HandScore      { get; init; }  // P(right hand) — sigmoid
    public double[] CutDirProbs    { get; init; }  // softmax over 9 cut directions
    public double[] LaneProbs      { get; init; }  // softmax over 4 lanes (0-3)
    public double[] RowProbs       { get; init; }  // softmax over 3 rows  (0-2)
}

/// <summary>
/// Inference-only multi-task neural model backed by <c>neural_placement.pt</c>.
/// Uses single-step GRU inference (<see cref="BeatSaberMappingNet.ForwardStep"/>) with
/// <see cref="GruState"/> carried between steps to maintain temporal context.
/// </summary>
public sealed class NeuralPlacementModel : IPlacementScorer, IDisposable
{
    private readonly BeatSaberMappingNet _net;

    private NeuralPlacementModel(BeatSaberMappingNet net) => _net = net;

    /// <summary>
    /// Loads the model from <c>neural_placement.pt</c>.
    /// Returns null if the file is absent or loading fails.
    /// </summary>
    public static NeuralPlacementModel? TryLoad(string artifactsPath)
    {
        string path = Path.Combine(artifactsPath, "neural_placement.pt");
        if (!File.Exists(path)) return null;
        try
        {
            var net = new BeatSaberMappingNet();
            net.load(path);
            net.eval();
            return new NeuralPlacementModel(net);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[NeuralModel] Failed to load neural_placement.pt: {ex.Message}");
            return null;
        }
    }

    // ── IPlacementScorer ──────────────────────────────────────────────────────

    public double ScorePlacement(double onset, double energy, double subdiv,
                                 double localNps, double beatStrength,
                                 int measureBeat, int difficultyLevel)
    {
        var ctx = new NeuralPlacementContext
        {
            OnsetStrength = onset, EnergyLevel = energy, Subdiv = subdiv,
            LocalNps = localNps, BeatStrength = beatStrength,
            MeasureBeat = measureBeat, DifficultyLevel = difficultyLevel,
            BeatPhase = 0.0, SectionProgress = 0.5, SpectralCentroid = 0.5,
            PrevLeftLane = 1, PrevLeftRow = 1, PrevLeftCutDir = -1,
            PrevRightLane = 2, PrevRightRow = 1, PrevRightCutDir = -1,
        };
        return ScorePlacement(in ctx);
    }

    public double ScorePlacement(in NeuralPlacementContext ctx)
        => PredictAll(in ctx).PlacementScore;

    // ── Full multi-task prediction ────────────────────────────────────────────

    /// <summary>
    /// Runs one GRU step. If <paramref name="ctx"/> has a non-null
    /// <see cref="NeuralPlacementContext.GruHiddenState"/>, the hidden state is
    /// read before the step and updated with the new hidden state after.
    /// </summary>
    public NeuralMapPrediction PredictAll(in NeuralPlacementContext ctx)
    {
        using var noGrad = no_grad();
        var f = new float[BeatSaberMappingNet.InputDim];
        ctx.FillFeaturesFloat(f);

        using var xT = tensor(f, new long[] { 1, 1, BeatSaberMappingNet.InputDim });

        // Build or restore hidden state
        Tensor h0;
        if (ctx.GruHiddenState is not null)
        {
            h0 = tensor(ctx.GruHiddenState.H,
                        new long[] { BeatSaberMappingNet.GruLayers, 1, BeatSaberMappingNet.GruHiddenDim });
        }
        else
        {
            h0 = zeros(new long[] { BeatSaberMappingNet.GruLayers, 1, BeatSaberMappingNet.GruHiddenDim });
        }

        var (out_, hn) = _net.ForwardStep(xT, h0);
        h0.Dispose();

        // Update GruState in-place if provided
        if (ctx.GruHiddenState is not null)
        {
            var hnData = hn.data<float>();
            for (int i = 0; i < ctx.GruHiddenState.H.Length; i++)
                ctx.GruHiddenState.H[i] = hnData[i];
        }
        hn.Dispose();

        using var plProb = sigmoid(out_.select(1, 0));
        using var haProb = sigmoid(out_.select(1, 1));
        using var cdSm   = softmax(out_.narrow(1, 2,  9), dim: 1);  // [1, 9]
        using var lnSm   = softmax(out_.narrow(1, 11, 4), dim: 1);  // [1, 4]
        using var rwSm   = softmax(out_.narrow(1, 15, 3), dim: 1);  // [1, 3]
        out_.Dispose();

        return new NeuralMapPrediction
        {
            PlacementScore = plProb.item<float>(),
            HandScore      = haProb.item<float>(),
            CutDirProbs    = ToDoubleArray(cdSm.squeeze(0)),
            LaneProbs      = ToDoubleArray(lnSm.squeeze(0)),
            RowProbs       = ToDoubleArray(rwSm.squeeze(0)),
        };
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static double[] ToDoubleArray(Tensor t)
    {
        var data = t.data<float>();
        var arr  = new double[data.Count];
        for (int i = 0; i < arr.Length; i++) arr[i] = data[i];
        return arr;
    }

    public void Dispose() => _net.Dispose();
}
