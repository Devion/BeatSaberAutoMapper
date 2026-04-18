using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Training.Features;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BeatSaber.AutoMapper.Training.Models;

/// <summary>
/// TorchSharp-backed GPU trainer for the GRU-based Beat Saber note placement model.
/// Training is sequence-based with contiguous truncated BPTT over longer chunks.
/// A shadow CPU model is kept in sync after each epoch for inference during generation
/// quality validation — no disk I/O on the hot path.
/// </summary>
public sealed class TorchPlacementTrainer : IMultiTaskPlacementModel, IDisposable
{
    private readonly BeatSaberMappingNet _gpu;
    private readonly BeatSaberMappingNet _cpu;
    private readonly Device              _device;
    private          optim.Optimizer?    _optimizer;
    private          double              _optimizerLr = -1;
    private          string?             _bestWeightsTmp;

    // Sequence training hyper-parameters
    private const int WindowSize  = 64;
    private const int WindowStride = 64;

    public TorchPlacementTrainer()
    {
        _device = cuda.is_available() ? CUDA : CPU;
        string deviceInfo = cuda.is_available()
            ? $"CUDA ({cuda.device_count()} device(s))"
            : "CPU";
        Console.WriteLine($"[TorchSharp] Device: {deviceInfo}");

        _gpu = new BeatSaberMappingNet();
        _gpu.to(_device);
        _cpu = new BeatSaberMappingNet();
        _cpu.eval();
    }

    // ── Checkpoint ────────────────────────────────────────────────────────────

    public bool TryLoadCheckpoint(string artifactsPath)
    {
        string path = Path.Combine(artifactsPath, "neural_placement.pt");
        if (!File.Exists(path)) return false;
        try
        {
            _gpu.load(path);
            _gpu.to(_device);
            SyncCpuShadow();
            Console.WriteLine("[TorchSharp] Warm-start from neural_placement.pt");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[TorchSharp] Checkpoint load failed: {ex.Message}");
            return false;
        }
    }

    // ── Epoch training ────────────────────────────────────────────────────────

    /// <summary>
    /// Sequence-based training epoch over W=16 beat windows with BPTT.
    /// Each inner list is one (song, difficulty) pair in beat order.
    /// </summary>
    public (double Loss, double PlaceBce) TrainEpoch(
        IReadOnlyList<IReadOnlyList<TrainingExample>> sequences, double lr)
    {
        if (sequences.Count == 0) return (0, 0);

        _gpu.train();

        // Lazy-create or update LR
        if (_optimizer is null)
        {
            _optimizer   = optim.Adam(_gpu.parameters(), lr: lr, weight_decay: 1e-5);
            _optimizerLr = lr;
        }
        else if (Math.Abs(_optimizerLr - lr) > 1e-12)
        {
            foreach (var g in _optimizer.ParamGroups)
                g.LearningRate = lr;
            _optimizerLr = lr;
        }

        // Class-balance weight from all examples in all sequences
        int posCount = 0, totalCount = 0;
        foreach (var seq in sequences) foreach (var ex in seq) { totalCount++; if (ex.HasNote) posCount++; }
        float posWt = posCount > 0 ? Math.Min((float)(totalCount - posCount) / posCount, 10f) : 1f;

        const int D = BeatSaberMappingNet.InputDim;
        double totalLoss = 0, totalPlaceBce = 0;
        int totalChunks = 0;

        foreach (var seq in sequences)
        {
            if (seq.Count == 0) continue;
            Tensor? hidden = null;
            for (int start = 0; start < seq.Count; start += WindowStride)
            {
                int len = Math.Min(WindowSize, seq.Count - start);
                var xArr    = new float[len * D];
                var yPlArr  = new float[len];
                var yHaArr  = new float[len];
                var yCdArr  = new long[len];
                var yLnArr  = new long[len];
                var yRwArr  = new long[len];
                var hasMask = new float[len];
                var cdMask  = new float[len];
                var lnMask  = new float[len];
                var rwMask  = new float[len];
                var exWtArr = new float[len];

                for (int t = 0; t < len; t++)
                {
                    var ex  = seq[start + t];
                    int idx = t;
                    FillFeaturesFromExample(ex, xArr, idx * D);
                    yPlArr[idx]  = ex.HasNote ? 1f : 0f;
                    yHaArr[idx]  = ex.NoteHand == 1 ? 1f : 0f;
                    yCdArr[idx]  = ex.NoteCutDir >= 0 ? ex.NoteCutDir : 0;
                    yLnArr[idx]  = ex.NoteLane   >= 0 ? ex.NoteLane   : 0;
                    yRwArr[idx]  = ex.NoteRow    >= 0 ? ex.NoteRow    : 0;
                    hasMask[idx] = ex.HasNote && ex.NoteHand   >= 0 ? 1f : 0f;
                    cdMask[idx]  = ex.HasNote && ex.NoteCutDir >= 0 ? 1f : 0f;
                    lnMask[idx]  = ex.HasNote && ex.NoteLane   >= 0 ? 1f : 0f;
                    rwMask[idx]  = ex.HasNote && ex.NoteRow    >= 0 ? 1f : 0f;
                    exWtArr[idx] = (float)Math.Max(0.05, ex.Weight);
                }

                long N = len;
                using var xT      = tensor(xArr,    new long[] { 1, len, D }, device: _device);
                using var yPlT    = tensor(yPlArr,  new long[] { N },          device: _device);
                using var yHaT    = tensor(yHaArr,  new long[] { N },          device: _device);
                using var yCdT    = tensor(yCdArr,  new long[] { N },          device: _device);
                using var yLnT    = tensor(yLnArr,  new long[] { N },          device: _device);
                using var yRwT    = tensor(yRwArr,  new long[] { N },          device: _device);
                using var posWtT  = tensor(posWt,                             device: _device);
                using var haMaskT = tensor(hasMask, new long[] { N },         device: _device);
                using var cdMaskT = tensor(cdMask,  new long[] { N },         device: _device);
                using var lnMaskT = tensor(lnMask,  new long[] { N },         device: _device);
                using var rwMaskT = tensor(rwMask,  new long[] { N },         device: _device);
                using var exWtT   = tensor(exWtArr, new long[] { N },         device: _device);

                var (outT, hn) = _gpu.ForwardSequence(xT, hidden);
                using (hidden) { }
                hidden = hn.detach();
                hn.Dispose();

                using var plLgt = outT.select(1, 0);
                using var haLgt = outT.select(1, 1);
                using var cdLgt = outT.narrow(1, 2,  9);
                using var lnLgt = outT.narrow(1, 11, 4);
                using var rwLgt = outT.narrow(1, 15, 3);

                var plBceTens = functional.binary_cross_entropy_with_logits(plLgt, yPlT, exWtT, Reduction.Mean, posWtT);
                totalPlaceBce += plBceTens.item<float>();
                var lossTerms = new List<Tensor> { plBceTens };

                using var haMaskIdx = (haMaskT > 0.5f).nonzero().squeeze(1);
                if (haMaskIdx.shape[0] > 0)
                {
                    using var haSelLgt = haLgt.index_select(0, haMaskIdx);
                    using var haSelY   = yHaT.index_select(0, haMaskIdx);
                    using var haSelW   = exWtT.index_select(0, haMaskIdx);
                    using var haRaw    = functional.binary_cross_entropy_with_logits(haSelLgt, haSelY, haSelW);
                lossTerms.Add(0.8 * haRaw);
                }

                using var cdMaskIdx = (cdMaskT > 0.5f).nonzero().squeeze(1);
                if (cdMaskIdx.shape[0] > 0)
                {
                    using var cdSelLgt = cdLgt.index_select(0, cdMaskIdx);
                    using var cdSelY   = yCdT.index_select(0, cdMaskIdx);
                    using var cdSelW   = exWtT.index_select(0, cdMaskIdx);
                    using var cdRaw    = WeightedCrossEntropy(cdSelLgt, cdSelY, cdSelW);
                    lossTerms.Add(1.0 * cdRaw);
                }

                using var lnMaskIdx = (lnMaskT > 0.5f).nonzero().squeeze(1);
                if (lnMaskIdx.shape[0] > 0)
                {
                    using var lnSelLgt  = lnLgt.index_select(0, lnMaskIdx);
                    using var lnSelY    = yLnT.index_select(0, lnMaskIdx);
                    using var lnSelW    = exWtT.index_select(0, lnMaskIdx);
                    var lnLoss          = 1.2 * WeightedCrossEntropy(lnSelLgt, lnSelY, lnSelW);
                    lossTerms.Add(lnLoss);
                }

                using var rwMaskIdx = (rwMaskT > 0.5f).nonzero().squeeze(1);
                if (rwMaskIdx.shape[0] > 0)
                {
                    using var rwSelLgt  = rwLgt.index_select(0, rwMaskIdx);
                    using var rwSelY    = yRwT.index_select(0, rwMaskIdx);
                    using var rwSelW    = exWtT.index_select(0, rwMaskIdx);
                    var rwLoss          = 0.8 * WeightedCrossEntropy(rwSelLgt, rwSelY, rwSelW);
                    lossTerms.Add(rwLoss);
                }

                Tensor loss;
                if (lossTerms.Count == 1)
                {
                    loss = lossTerms[0];
                }
                else
                {
                    using var stacked = stack(lossTerms.ToArray());
                    loss = stacked.sum();
                    foreach (var t in lossTerms) t.Dispose();
                }

                totalLoss += loss.item<float>();
                totalChunks++;
                _optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(_gpu.parameters(), 5.0);
                _optimizer.step();
                outT.Dispose();
                loss.Dispose();
            }
            hidden?.Dispose();
        }

        SyncCpuShadow();
        return (totalChunks > 0 ? totalLoss / totalChunks : 0, totalChunks > 0 ? totalPlaceBce / totalChunks : 0);
    }

    // ── Best-epoch snapshot ───────────────────────────────────────────────────

    public void SaveBestWeights()
    {
        _bestWeightsTmp ??= Path.GetTempFileName() + ".pt";
        _gpu.save(_bestWeightsTmp);
    }

    public void RestoreBestWeights()
    {
        if (_bestWeightsTmp is null || !File.Exists(_bestWeightsTmp)) return;
        _gpu.load(_bestWeightsTmp);
        _gpu.to(_device);
        _optimizer?.Dispose();
        _optimizer   = null;
        _optimizerLr = -1;
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    public void SaveWeights(string artifactsPath)
    {
        string path = Path.Combine(artifactsPath, "neural_placement.pt");
        _gpu.save(path);
        Console.WriteLine($"[TorchSharp] Model saved → {path}");
    }

    // ── IPlacementScorer (CPU shadow model) ───────────────────────────────────

    public double ScorePlacement(double onset, double energy, double subdiv,
                                 double localNps, double beatStrength,
                                 int measureBeat, int difficultyLevel)
    {
        var ctx = new NeuralPlacementContext
        {
            OnsetStrength = onset, EnergyLevel = energy, Subdiv = subdiv,
            LocalNps = localNps, BeatStrength = beatStrength,
            MeasureBeat = measureBeat, DifficultyLevel = difficultyLevel,
            SectionProgress = 0.5, SpectralCentroid = 0.5,
            PrevLeftLane = 1, PrevLeftRow = 1, PrevLeftCutDir = -1,
            PrevRightLane = 2, PrevRightRow = 1, PrevRightCutDir = -1,
        };
        return ScorePlacement(in ctx);
    }

    public double ScorePlacement(in NeuralPlacementContext ctx)
    {
        using var noGrad = no_grad();
        var f = new float[BeatSaberMappingNet.InputDim];
        ctx.FillFeaturesFloat(f);
        // Single-step inference: wrap as [1, 1, D]
        using var xT    = tensor(f, new long[] { 1, 1, BeatSaberMappingNet.InputDim });
        // Use GruState if available; otherwise zeros
        var (outT, _) = ForwardStepCpu(xT, ctx.GruHiddenState);
        using var plProb = sigmoid(outT.select(1, 0));
        double result = plProb.item<float>();
        outT.Dispose();
        return result;
    }

    public NeuralMapPrediction PredictAll(in NeuralPlacementContext ctx)
    {
        using var noGrad = no_grad();
        var f = new float[BeatSaberMappingNet.InputDim];
        ctx.FillFeaturesFloat(f);
        using var xT = tensor(f, new long[] { 1, 1, BeatSaberMappingNet.InputDim });
        var (outT, hn) = ForwardStepCpu(xT, ctx.GruHiddenState);

        if (ctx.GruHiddenState is not null)
        {
            var hnData = hn.data<float>();
            for (int i = 0; i < ctx.GruHiddenState.H.Length; i++)
                ctx.GruHiddenState.H[i] = hnData[i];
        }
        hn.Dispose();

        using var plProb = sigmoid(outT.select(1, 0));
        using var haProb = sigmoid(outT.select(1, 1));
        using var cdSm   = softmax(outT.narrow(1, 2,  9), dim: 1);
        using var lnSm   = softmax(outT.narrow(1, 11, 4), dim: 1);
        using var rwSm   = softmax(outT.narrow(1, 15, 3), dim: 1);
        outT.Dispose();

        return new NeuralMapPrediction
        {
            PlacementScore = plProb.item<float>(),
            HandScore      = haProb.item<float>(),
            CutDirProbs    = ToDoubleArray(cdSm.squeeze(0)),
            LaneProbs      = ToDoubleArray(lnSm.squeeze(0)),
            RowProbs       = ToDoubleArray(rwSm.squeeze(0)),
        };
    }

    private (Tensor output, Tensor hn) ForwardStepCpu(Tensor x, GruState? state)
    {
        using var h0 = state is not null
            ? tensor(state.H, new long[] { BeatSaberMappingNet.GruLayers, 1, BeatSaberMappingNet.GruHiddenDim })
            : zeros(new long[] { BeatSaberMappingNet.GruLayers, 1, BeatSaberMappingNet.GruHiddenDim });
        return _cpu.ForwardStep(x, h0);
    }

    private static Tensor WeightedCrossEntropy(Tensor logits, Tensor targets, Tensor weights)
    {
        using var logProb = functional.log_softmax(logits, dim: 1);
        using var idx     = targets.unsqueeze(1);
        using var picked  = logProb.gather(1, idx).squeeze(1);
        using var numer   = -(picked * weights).sum();
        using var denom   = weights.sum().clamp_min(1e-6);
        return numer / denom;
    }

    private static double[] ToDoubleArray(Tensor t)
    {
        var data = t.data<float>();
        var arr  = new double[data.Count];
        for (int i = 0; i < arr.Length; i++) arr[i] = data[i];
        return arr;
    }

    // ── CPU shadow sync ───────────────────────────────────────────────────────

    private void SyncCpuShadow()
    {
        _gpu.eval();
        string tmp = Path.GetTempFileName() + ".pt";
        try
        {
            _gpu.save(tmp);
            _cpu.load(tmp);
        }
        finally
        {
            try { File.Delete(tmp); } catch { }
        }
        _cpu.eval();
        _gpu.train();
    }

    // ── Feature encoding ──────────────────────────────────────────────────────

    internal static void FillFeaturesFromExample(TrainingExample ex, float[] arr, int offset)
    {
        // Audio [0-13]
        arr[offset + 0]  = (float)ex.OnsetStrength;
        arr[offset + 1]  = (float)ex.EnergyLevel;
        arr[offset + 2]  = (float)ex.SpectralFlux;
        arr[offset + 3]  = (float)ex.TransientStrength;
        arr[offset + 4]  = (float)ex.LowBandEnergy;
        arr[offset + 5]  = (float)ex.MidBandEnergy;
        arr[offset + 6]  = (float)ex.HighBandEnergy;
        arr[offset + 7]  = (float)ex.SpectralCentroid;
        arr[offset + 8]  = (float)Math.Clamp(ex.EnergyDelta,    -1.0, 1.0);
        arr[offset + 9]  = (float)Math.Clamp(ex.HighBandDelta,  -1.0, 1.0);
        arr[offset + 10] = (float)Math.Clamp(ex.EnergyTrend4,    0.0, 1.0);
        arr[offset + 11] = (float)Math.Clamp(ex.EnergyTrend8,    0.0, 1.0);
        arr[offset + 12] = (float)Math.Clamp(ex.LookaheadEnergy, 0.0, 1.0);
        arr[offset + 13] = (float)Math.Clamp(ex.LookaheadOnset,  0.0, 1.0);
        // Beat position [14-20]
        arr[offset + 14] = (float)ex.BeatPhase;
        arr[offset + 15] = (float)ex.SubdivisionDenominator;
        arr[offset + 16] = (float)ex.BeatStrength;
        arr[offset + 17] = (float)(ex.MeasureBeat / 4.0);
        arr[offset + 18] = (float)Math.Clamp(ex.SongFraction,    0.0, 1.0);
        arr[offset + 19] = (float)Math.Clamp(ex.SectionProgress, 0.0, 1.0);
        arr[offset + 20] = (float)ex.BarPosition;
        // Section type one-hot [21-28]
        for (int i = 21; i <= 28; i++) arr[offset + i] = 0f;
        arr[offset + 21 + Math.Clamp(ex.SectionTypeIndex, 0, 7)] = 1f;
        // Context [29-30]
        arr[offset + 29] = (float)(ex.DifficultyLevel / 4.0);
        arr[offset + 30] = (float)Math.Min(ex.LocalNps / 10.0, 1.0);
        // Previous placement [31-44]
        arr[offset + 31] = (float)(ex.PreviousLeftLane  / 3.0);
        arr[offset + 32] = (float)(ex.PreviousLeftRow   / 2.0);
        arr[offset + 33] = ex.PreviousLeftCutDir  >= 0 ? 1f : 0f;
        arr[offset + 34] = ex.PreviousLeftCutDir  >= 0 ? (float)(ex.PreviousLeftCutDir  / 8.0) : 0f;
        arr[offset + 35] = (float)(ex.PreviousRightLane / 3.0);
        arr[offset + 36] = (float)(ex.PreviousRightRow  / 2.0);
        arr[offset + 37] = ex.PreviousRightCutDir >= 0 ? 1f : 0f;
        arr[offset + 38] = ex.PreviousRightCutDir >= 0 ? (float)(ex.PreviousRightCutDir / 8.0) : 0f;
        arr[offset + 39] = (float)Math.Clamp(ex.LeftParityState,        0.0, 1.0);
        arr[offset + 40] = (float)Math.Clamp(ex.RightParityState,       0.0, 1.0);
        arr[offset + 41] = (float)Math.Clamp(ex.BeatsSinceLastLeft  / 8.0, 0.0, 1.0);
        arr[offset + 42] = (float)Math.Clamp(ex.BeatsSinceLastRight / 8.0, 0.0, 1.0);
        arr[offset + 43] = (ex.PreviousLeftCutDir >= 0 || ex.PreviousRightCutDir >= 0) ? 1f : 0f;
        arr[offset + 44] = 0.5f;
        // Phrase / history / geometry [45-67]
        arr[offset + 45] = (float)Math.Clamp(ex.FutureEnergy4, 0.0, 1.0);
        arr[offset + 46] = (float)Math.Clamp(ex.FutureEnergy8, 0.0, 1.0);
        arr[offset + 47] = (float)Math.Clamp(ex.FutureEnergy16, 0.0, 1.0);
        arr[offset + 48] = (float)Math.Clamp(ex.FutureOnset4, 0.0, 1.0);
        arr[offset + 49] = (float)Math.Clamp(ex.FutureOnset8, 0.0, 1.0);
        arr[offset + 50] = (float)Math.Clamp(ex.FutureOnset16, 0.0, 1.0);
        arr[offset + 51] = (float)Math.Clamp(ex.BeatsSinceSectionStart / 16.0, 0.0, 1.0);
        arr[offset + 52] = (float)Math.Clamp(ex.BeatsToSectionBoundary / 16.0, 0.0, 1.0);
        arr[offset + 53] = (float)Math.Clamp(ex.RecentChordRate4, 0.0, 1.0);
        arr[offset + 54] = (float)Math.Clamp(ex.RecentOffbeatRate4, 0.0, 1.0);
        arr[offset + 55] = (float)Math.Clamp(ex.RecentStreamRate4, 0.0, 1.0);
        arr[offset + 56] = (float)Math.Clamp(ex.RecentAlternation8, 0.0, 1.0);
        arr[offset + 57] = (float)Math.Clamp(ex.RecentHandBalance8, 0.0, 1.0);
        arr[offset + 58] = (float)Math.Clamp(ex.ConsecutiveSameHandCount / 8.0, 0.0, 1.0);
        arr[offset + 59] = (float)Math.Clamp(ex.BeatsSinceLastAny / 8.0, 0.0, 1.0);
        arr[offset + 60] = (float)Math.Clamp(ex.NotesAtCurrentBeatSoFar / 2.0, 0.0, 1.0);
        arr[offset + 61] = (float)Math.Clamp(ex.InterHandLaneDistance / 3.0, 0.0, 1.0);
        arr[offset + 62] = (float)Math.Clamp(ex.InterHandRowDistance / 2.0, 0.0, 1.0);
        arr[offset + 63] = (float)Math.Clamp(ex.HandsCrossedFlag, 0.0, 1.0);
        arr[offset + 64] = (float)Math.Clamp(ex.LeftRecentTravel / 5.0, 0.0, 1.0);
        arr[offset + 65] = (float)Math.Clamp(ex.RightRecentTravel / 5.0, 0.0, 1.0);
        arr[offset + 66] = (float)Math.Clamp(ex.RecentLaneSpan4 / 3.0, 0.0, 1.0);
        arr[offset + 67] = (float)Math.Clamp(ex.RecentRowSpan4 / 2.0, 0.0, 1.0);
    }

    public void Dispose()
    {
        _optimizer?.Dispose();
        _gpu.Dispose();
        _cpu.Dispose();
        if (_bestWeightsTmp is not null && File.Exists(_bestWeightsTmp))
            try { File.Delete(_bestWeightsTmp); } catch { }
    }
}
