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
/// Implements both <see cref="IPlacementScorer"/> and <see cref="IAttributeModel"/>
/// for use by the generation pipeline on CPU.
/// </summary>
public sealed class NeuralPlacementModel : IPlacementScorer, IAttributeModel, IDisposable
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
            Onset = onset, Energy = energy, Subdiv = subdiv, LocalNps = localNps,
            BeatStrength = beatStrength, MeasureBeat = measureBeat,
            DifficultyLevel = difficultyLevel,
            BeatPhase = 0.0, SectionProgress = 0.5,
            PrevLeftLane = 1, PrevLeftRow = 1, PrevLeftCutDir = -1,
            PrevRightLane = 2, PrevRightRow = 1, PrevRightCutDir = -1,
            BeatsSinceLastLeft = 999, BeatsSinceLastRight = 999,
            SpectralCentroid = 0.5,
        };
        return ScorePlacement(in ctx);
    }

    public double ScorePlacement(in NeuralPlacementContext ctx)
        => PredictAll(in ctx).PlacementScore;

    // ── Full multi-task prediction ────────────────────────────────────────────

    public NeuralMapPrediction PredictAll(in NeuralPlacementContext ctx)
    {
        using var noGrad = no_grad();
        var f = new float[BeatSaberMappingNet.InputDim];
        ctx.FillFeaturesFloat(f);
        using var xT  = tensor(f, new long[] { 1, BeatSaberMappingNet.InputDim });
        using var out_ = _net.forward(xT);  // [1, 18]

        using var plProb = sigmoid(out_.select(1, 0));
        using var haProb = sigmoid(out_.select(1, 1));
        using var cdSm   = softmax(out_.narrow(1, 2,  9), dim: 1);  // [1, 9]
        using var lnSm   = softmax(out_.narrow(1, 11, 4), dim: 1);  // [1, 4]
        using var rwSm   = softmax(out_.narrow(1, 15, 3), dim: 1);  // [1, 3]

        return new NeuralMapPrediction
        {
            PlacementScore = plProb.item<float>(),
            HandScore      = haProb.item<float>(),
            CutDirProbs    = ToDoubleArray(cdSm.squeeze(0)),
            LaneProbs      = ToDoubleArray(lnSm.squeeze(0)),
            RowProbs       = ToDoubleArray(rwSm.squeeze(0)),
        };
    }

    // ── IAttributeModel ───────────────────────────────────────────────────────

    public CutDirection SampleCutDirection(int prevCutDir, double beatStrength, int hand,
                                           int difficultyLevel, Random rng)
    {
        var ctx = new NeuralPlacementContext
        {
            BeatStrength = beatStrength, DifficultyLevel = difficultyLevel,
            PrevLeftCutDir  = hand == 0 ? prevCutDir : -1,
            PrevRightCutDir = hand == 1 ? prevCutDir : -1,
            PrevLeftLane = 1, PrevLeftRow = 1,
            PrevRightLane = 2, PrevRightRow = 1,
            BeatsSinceLastLeft = 1, BeatsSinceLastRight = 1,
            SpectralCentroid = 0.5, SectionProgress = 0.5,
        };
        return (CutDirection)SampleFromProbs(PredictAll(in ctx).CutDirProbs, rng);
    }

    public (int Lane, int Row) SamplePosition(int cutDir, int hand, double beatStrength,
                                              double spectralCentroid, int difficultyLevel, Random rng)
    {
        var ctx = new NeuralPlacementContext
        {
            BeatStrength = beatStrength, DifficultyLevel = difficultyLevel,
            SpectralCentroid = spectralCentroid, SectionProgress = 0.5,
            PrevLeftLane = 1, PrevLeftRow = 1, PrevLeftCutDir = -1,
            PrevRightLane = 2, PrevRightRow = 1, PrevRightCutDir = cutDir,
            BeatsSinceLastLeft = 1, BeatsSinceLastRight = 1,
        };
        var pred = PredictAll(in ctx);
        return (SampleFromProbs(pred.LaneProbs, rng), SampleFromProbs(pred.RowProbs, rng));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static double[] ToDoubleArray(Tensor t)
    {
        var data = t.data<float>();
        var arr  = new double[data.Count];
        for (int i = 0; i < arr.Length; i++) arr[i] = data[i];
        return arr;
    }

    private static int SampleFromProbs(double[] probs, Random rng)
    {
        double r = rng.NextDouble(), cum = 0;
        for (int i = 0; i < probs.Length; i++) { cum += probs[i]; if (r < cum) return i; }
        return probs.Length - 1;
    }

    public void Dispose() => _net.Dispose();
}

