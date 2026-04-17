using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Training.Features;

namespace BeatSaber.AutoMapper.Training.Models;

/// <summary>
/// Logistic-regression placement model trained with the Adam optimizer.
/// Adam gives each weight its own adaptive learning rate, which avoids the
/// "LR decays to zero" problem that plagues vanilla SGD with decay schedules.
/// TrainEpoch expects examples to be pre-shuffled by the caller so that
/// consecutive mini-batches always cover a different random slice of data.
/// </summary>
public sealed class PlacementModelTrainer : IPlacementScorer
{
    private const int NumWeights    = 8;
    private static int MiniBatchSize => Math.Max(512, Environment.ProcessorCount * 256);

    // Adam hyper-parameters (standard values from the original paper)
    private const double AdamBeta1 = 0.9;
    private const double AdamBeta2 = 0.999;
    private const double AdamEps   = 1e-8;

    // Learned weights — small neutral initialisation; Adam will find the right scale
    private double _w0            =  0.0;
    private double _wOnset        =  0.1;
    private double _wEnergy       =  0.1;
    private double _wSubdiv       =  0.0;
    private double _wNps          =  0.0;
    private double _wBeatStrength =  0.1;
    private double _wMeasureBeat  =  0.0;
    private double _wDifficulty   =  0.1;

    // Adam first/second moment vectors + step counter (persist across epochs)
    private readonly double[] _m = new double[NumWeights];
    private readonly double[] _v = new double[NumWeights];
    private long _adamStep = 0;

    // Best-epoch snapshot
    private double[] _best = new double[NumWeights];

    // -----------------------------------------------------------------------
    // IPlacementScorer
    // -----------------------------------------------------------------------

    public double ScorePlacement(double onset, double energy, double subdiv,
                                 double localNps, double beatStrength, int measureBeat,
                                 int difficultyLevel) =>
        Sigmoid(Linear(onset, energy, subdiv, localNps, beatStrength, measureBeat, difficultyLevel));

    // -----------------------------------------------------------------------
    // Per-epoch training (caller must shuffle examples before each call)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Run one epoch of mini-batch gradient descent with parallel gradient accumulation.
    /// The caller is responsible for shuffling <paramref name="examples"/> before
    /// each epoch so that consecutive batches cover different data.
    /// </summary>
    public (double Loss, double Accuracy) TrainEpoch(
        IReadOnlyList<TrainingExample> examples, double lr)
    {
        int n = examples.Count;
        if (n == 0) return (0, 0);

        // Class-balance weights: prevent the model from collapsing to "always false"
        // when negatives outnumber positives (typical in beat-aligned training grids).
        int posCount = 0;
        for (int i = 0; i < n; i++) if (examples[i].HasNote) posCount++;
        int    negCount   = n - posCount;
        double posWeight  = posCount > 0 && negCount > 0 ? (double)negCount / posCount : 1.0;
        // Cap weight at 10× to avoid extreme magnification on very sparse maps
        posWeight = Math.Min(posWeight, 10.0);

        int P          = Environment.ProcessorCount;
        int batchSize  = MiniBatchSize;
        int numBatches = Math.Max(1, (n + batchSize - 1) / batchSize);

        // Pre-allocate per-thread gradient arrays (reused across batches)
        var grads  = new double[P][];
        var losses = new double[P];
        for (int t = 0; t < P; t++) grads[t] = new double[NumWeights];

        double totalLoss = 0;

        for (int b = 0; b < numBatches; b++)
        {
            int bStart = b * batchSize;
            int bEnd   = Math.Min(bStart + batchSize, n);
            int bSize  = bEnd - bStart;

            // Reset accumulators
            for (int t = 0; t < P; t++)
            {
                Array.Clear(grads[t], 0, NumWeights);
                losses[t] = 0;
            }

            // Parallel gradient computation: divide batch across P threads
            int chunk = Math.Max(1, (bSize + P - 1) / P);
            Parallel.For(0, P, new ParallelOptions { MaxDegreeOfParallelism = P }, t =>
            {
                int s   = bStart + t * chunk;
                int end = Math.Min(s + chunk, bEnd);
                var g   = grads[t];
                double localLoss = 0;

                for (int i = s; i < end; i++)
                {
                    var ex   = examples[i];
                    double p = Sigmoid(PredictEx(ex));
                    double y = ex.HasNote ? 1.0 : 0.0;
                    double w = y > 0.5 ? posWeight : 1.0;  // class-balance weight
                    double e = w * ex.Weight * (p - y);    // ex.Weight: synthetic amplification

                    g[0] += e;
                    g[1] += e * ex.OnsetStrength;
                    g[2] += e * ex.EnergyLevel;
                    g[3] += e * ex.SubdivisionDenominator;
                    g[4] += e * ex.LocalNps;
                    g[5] += e * ex.BeatStrength;
                    g[6] += e * (ex.MeasureBeat / 4.0);
                    g[7] += e * (ex.DifficultyLevel / 4.0);

                    localLoss += w * -(y * Math.Log(p + 1e-10) + (1 - y) * Math.Log(1 - p + 1e-10));
                }
                losses[t] = localLoss;
            });

            // Sum thread gradients
            double g0=0, g1=0, g2=0, g3=0, g4=0, g5=0, g6=0, g7=0;
            for (int t = 0; t < P; t++)
            {
                g0 += grads[t][0]; g1 += grads[t][1]; g2 += grads[t][2];
                g3 += grads[t][3]; g4 += grads[t][4]; g5 += grads[t][5];
                g6 += grads[t][6]; g7 += grads[t][7];
                totalLoss += losses[t];
            }

            // Adam update — average gradient over batch, then apply per-weight
            double invB = 1.0 / bSize;
            double[] avgG = { g0*invB, g1*invB, g2*invB, g3*invB,
                               g4*invB, g5*invB, g6*invB, g7*invB };
            _adamStep++;
            double bc1 = 1.0 - Math.Pow(AdamBeta1, _adamStep);
            double bc2 = 1.0 - Math.Pow(AdamBeta2, _adamStep);

            double[] ws = { _w0, _wOnset, _wEnergy, _wSubdiv,
                             _wNps, _wBeatStrength, _wMeasureBeat, _wDifficulty };
            for (int i = 0; i < NumWeights; i++)
            {
                double g = avgG[i];
                _m[i] = AdamBeta1 * _m[i] + (1 - AdamBeta1) * g;
                _v[i] = AdamBeta2 * _v[i] + (1 - AdamBeta2) * g * g;
                ws[i] -= lr * (_m[i] / bc1) / (Math.Sqrt(_v[i] / bc2) + AdamEps);
            }
            (_w0, _wOnset, _wEnergy, _wSubdiv,
             _wNps, _wBeatStrength, _wMeasureBeat, _wDifficulty) =
                (ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7]);
        }

        return (totalLoss / n, ComputeAccuracy(examples));
    }

    // -----------------------------------------------------------------------
    // Best-weights management
    // -----------------------------------------------------------------------

    public void SaveBestWeights() =>
        _best = [_w0, _wOnset, _wEnergy, _wSubdiv, _wNps, _wBeatStrength, _wMeasureBeat, _wDifficulty];

    public void RestoreBestWeights()
    {
        _w0           = _best[0]; _wOnset        = _best[1];
        _wEnergy      = _best[2]; _wSubdiv       = _best[3];
        _wNps         = _best[4]; _wBeatStrength = _best[5];
        _wMeasureBeat = _best[6]; _wDifficulty   = _best[7];
    }

    public void SaveWeights(string artifactsPath)
    {
        Directory.CreateDirectory(artifactsPath);
        var w = new
        {
            w0            = _w0,   wOnset        = _wOnset,
            wEnergy       = _wEnergy, wSubdiv    = _wSubdiv,
            wNps          = _wNps,    wBeatStrength = _wBeatStrength,
            wMeasureBeat  = _wMeasureBeat,
            wDifficulty   = _wDifficulty
        };
        File.WriteAllText(
            Path.Combine(artifactsPath, "placement_weights.json"),
            JsonSerializer.Serialize(w, new JsonSerializerOptions { WriteIndented = true }));
    }

    // backward-compat for EvaluationRunner
    public double PredictPlacement(TrainingExample ex) => Sigmoid(PredictEx(ex));

    // -----------------------------------------------------------------------

    private double PredictEx(TrainingExample ex) =>
        Linear(ex.OnsetStrength, ex.EnergyLevel, ex.SubdivisionDenominator,
               ex.LocalNps, ex.BeatStrength, ex.MeasureBeat, ex.DifficultyLevel);

    private double Linear(double onset, double energy, double subdiv,
                          double localNps, double beatStrength, int measureBeat,
                          int difficultyLevel) =>
        _w0
        + _wOnset        * onset
        + _wEnergy       * energy
        + _wSubdiv       * subdiv
        + _wNps          * localNps
        + _wBeatStrength * beatStrength
        + _wMeasureBeat  * (measureBeat / 4.0)
        + _wDifficulty   * (difficultyLevel / 4.0);

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private double ComputeAccuracy(IReadOnlyList<TrainingExample> examples)
    {
        int n = examples.Count;
        if (n == 0) return 0;

        int P = Environment.ProcessorCount;
        int chunk = Math.Max(1, (n + P - 1) / P);
        var correct = new long[P];

        Parallel.For(0, P, new ParallelOptions { MaxDegreeOfParallelism = P }, t =>
        {
            int s   = t * chunk;
            int end = Math.Min(s + chunk, n);
            long c  = 0;
            for (int i = s; i < end; i++)
                if ((Sigmoid(PredictEx(examples[i])) > 0.5) == examples[i].HasNote) c++;
            correct[t] = c;
        });

        return correct.Sum() / (double)n;
    }
}

