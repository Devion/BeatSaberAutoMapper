using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Training.Features;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BeatSaber.AutoMapper.Training.Models;

/// <summary>
/// TorchSharp-backed GPU trainer for the multi-task Beat Saber note placement model.
/// Trains on CUDA (automatically falls back to CPU if CUDA is unavailable).
/// A shadow CPU model is kept in sync after each epoch for inference during generation
/// quality validation — no disk I/O on the hot path.
/// </summary>
public sealed class TorchPlacementTrainer : IPlacementScorer, IDisposable
{
    private readonly BeatSaberMappingNet _gpu;
    private readonly BeatSaberMappingNet _cpu;
    private readonly Device              _device;
    private          optim.Optimizer?    _optimizer;
    private          double              _optimizerLr = -1;
    private          string?             _bestWeightsTmp;

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

    public (double Loss, double PlaceBce) TrainEpoch(
        IReadOnlyList<TrainingExample> examples, double lr)
    {
        int n = examples.Count;
        if (n == 0) return (0, 0);

        _gpu.train();

        // Lazy-create or update LR on the existing Adam optimizer.
        // Updating in-place preserves momentum/variance state across LR reductions.
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

        // Class-balance weight for the placement BCE head
        int posCount = 0;
        for (int i = 0; i < n; i++) if (examples[i].HasNote) posCount++;
        float posWt = posCount > 0 ? Math.Min((float)(n - posCount) / posCount, 10f) : 1f;

        // ── Build float / long arrays (CPU) ───────────────────────────────────
        const int D = BeatSaberMappingNet.InputDim;
        var xArr    = new float[n * D];
        var yPlArr  = new float[n];
        var yHaArr  = new float[n];
        var yCdArr  = new long[n];
        var yLnArr  = new long[n];
        var yRwArr  = new long[n];
        var noteArr = new float[n];
        var hasMask = new float[n];  // 1 = has real hand label
        var cdMask  = new float[n];  // 1 = has real cut-dir label
        var lnMask  = new float[n];  // 1 = has real lane label
        var rwMask  = new float[n];  // 1 = has real row label

        for (int i = 0; i < n; i++)
        {
            var ex = examples[i];
            FillFeaturesFromExample(ex, xArr, i * D);
            yPlArr[i]  = ex.HasNote ? 1f : 0f;
            yHaArr[i]  = ex.NoteHand == 1 ? 1f : 0f;
            yCdArr[i]  = ex.NoteCutDir >= 0 ? ex.NoteCutDir : 0;
            yLnArr[i]  = ex.NoteLane  >= 0 ? ex.NoteLane  : 0;
            yRwArr[i]  = ex.NoteRow   >= 0 ? ex.NoteRow   : 0;
            noteArr[i] = ex.HasNote ? 1f : 0f;
            hasMask[i] = ex.HasNote && ex.NoteHand   >= 0 ? 1f : 0f;
            cdMask[i]  = ex.HasNote && ex.NoteCutDir >= 0 ? 1f : 0f;
            lnMask[i]  = ex.HasNote && ex.NoteLane   >= 0 ? 1f : 0f;
            rwMask[i]  = ex.HasNote && ex.NoteRow    >= 0 ? 1f : 0f;
        }

        // ── Move full dataset to device once ──────────────────────────────────
        using var xT     = tensor(xArr,   new long[] { n, D }, device: _device);
        using var yPlT   = tensor(yPlArr, new long[] { n },    device: _device);
        using var yHaT   = tensor(yHaArr, new long[] { n },    device: _device);
        using var yCdT   = tensor(yCdArr, new long[] { n },    device: _device);
        using var yLnT   = tensor(yLnArr, new long[] { n },    device: _device);
        using var yRwT   = tensor(yRwArr, new long[] { n },    device: _device);
        using var noteT  = tensor(noteArr, new long[] { n },   device: _device);
        using var posWtT = tensor(posWt,   device: _device);
        using var haMaskT = tensor(hasMask, new long[] { n },  device: _device);
        using var cdMaskT = tensor(cdMask,  new long[] { n },  device: _device);
        using var lnMaskT = tensor(lnMask,  new long[] { n },  device: _device);
        using var rwMaskT = tensor(rwMask,  new long[] { n },  device: _device);

        const int batchSize = 4096;
        int numBatches = Math.Max(1, (n + batchSize - 1) / batchSize);
        double totalLoss     = 0;
        double totalPlaceBce = 0;  // placement-head BCE component (meaningful regardless of class imbalance)

        for (int b = 0; b < numBatches; b++)
        {
            long s   = (long)b * batchSize;
            long len = Math.Min(batchSize, n - s);

            using var xB      = xT.narrow(0, s, len);
            using var yPlB    = yPlT.narrow(0, s, len);
            using var yHaB    = yHaT.narrow(0, s, len);
            using var yCdB    = yCdT.narrow(0, s, len);
            using var yLnB    = yLnT.narrow(0, s, len);
            using var yRwB    = yRwT.narrow(0, s, len);
            using var noteB   = noteT.narrow(0, s, len);
            using var haMaskB = haMaskT.narrow(0, s, len);
            using var cdMaskB = cdMaskT.narrow(0, s, len);
            using var lnMaskB = lnMaskT.narrow(0, s, len);
            using var rwMaskB = rwMaskT.narrow(0, s, len);

            using var outB  = _gpu.forward(xB);       // [B, 18]
            using var plLgt = outB.select(1, 0);      // [B]
            using var haLgt = outB.select(1, 1);      // [B]
            using var cdLgt = outB.narrow(1, 2,  9);  // [B, 9]
            using var lnLgt = outB.narrow(1, 11, 4);  // [B, 4]
            using var rwLgt = outB.narrow(1, 15, 3);  // [B, 3]

            // ── Multi-task loss ────────────────────────────────────────────────
            var plBceTens = functional.binary_cross_entropy_with_logits(plLgt, yPlB, null, Reduction.Mean, posWtT);
            totalPlaceBce += plBceTens.item<float>();
            var lossTerms = new List<Tensor> { plBceTens };

            // Hand head — only examples with a real hand label
            using var haMaskIdx = (haMaskB > 0.5f).nonzero().squeeze(1);
            if (haMaskIdx.shape[0] > 0)
            {
                using var haSelLgt = haLgt.index_select(0, haMaskIdx);
                using var haSelY   = yHaB.index_select(0, haMaskIdx);
                using var haRaw    = functional.binary_cross_entropy_with_logits(haSelLgt, haSelY);
                lossTerms.Add(0.5 * haRaw);
            }

            // Cut-direction head — only examples with a real cut-dir label
            using var cdMaskIdx = (cdMaskB > 0.5f).nonzero().squeeze(1);
            if (cdMaskIdx.shape[0] > 0)
            {
                using var cdSelLgt = cdLgt.index_select(0, cdMaskIdx);
                using var cdSelY   = yCdB.index_select(0, cdMaskIdx);
                using var cdRaw    = functional.cross_entropy(cdSelLgt, cdSelY);
                lossTerms.Add(1.0 * cdRaw);
            }

            // Lane head — only examples with a real lane label (NOT self-supervised)
            // Entropy regularisation (-α·H) penalises overconfident predictions and
            // prevents the head collapsing to always output lane 0.
            using var lnMaskIdx = (lnMaskB > 0.5f).nonzero().squeeze(1);
            if (lnMaskIdx.shape[0] > 0)
            {
                using var lnSelLgt  = lnLgt.index_select(0, lnMaskIdx);
                using var lnSelY    = yLnB.index_select(0, lnMaskIdx);
                using var lnCE      = functional.cross_entropy(lnSelLgt, lnSelY);
                using var lnLogProb = functional.log_softmax(lnSelLgt, dim: 1);
                using var lnProb    = softmax(lnSelLgt, dim: 1);
                using var lnEntr    = -(lnProb * lnLogProb).sum(1).mean();
                var lnLoss          = 1.2 * lnCE - 0.10 * lnEntr;   // owned by lossTerms
                lossTerms.Add(lnLoss);
            }

            // Row head — only examples with a real row label (NOT self-supervised)
            using var rwMaskIdx = (rwMaskB > 0.5f).nonzero().squeeze(1);
            if (rwMaskIdx.shape[0] > 0)
            {
                using var rwSelLgt  = rwLgt.index_select(0, rwMaskIdx);
                using var rwSelY    = yRwB.index_select(0, rwMaskIdx);
                using var rwCE      = functional.cross_entropy(rwSelLgt, rwSelY);
                using var rwLogProb = functional.log_softmax(rwSelLgt, dim: 1);
                using var rwProb    = softmax(rwSelLgt, dim: 1);
                using var rwEntr    = -(rwProb * rwLogProb).sum(1).mean();
                var rwLoss          = 0.8 * rwCE - 0.08 * rwEntr;   // owned by lossTerms
                lossTerms.Add(rwLoss);
            }

            // Sum all loss terms
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
            _optimizer.zero_grad();
            loss.backward();
            nn.utils.clip_grad_norm_(_gpu.parameters(), 5.0);
            _optimizer.step();
            loss.Dispose();

            // Note-placement F1: track TP/FP/FN on the placement head only.
        }

        SyncCpuShadow();
        return (totalLoss / numBatches, totalPlaceBce / numBatches);
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
        // Invalidate the optimizer so it's rebuilt on next epoch
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
            Onset = onset, Energy = energy, Subdiv = subdiv,
            LocalNps = localNps, BeatStrength = beatStrength,
            MeasureBeat = measureBeat, DifficultyLevel = difficultyLevel,
            SectionProgress = 0.5, SpectralCentroid = 0.5,
            PrevLeftLane  = 1, PrevLeftRow  = 1, PrevLeftCutDir  = -1,
            PrevRightLane = 2, PrevRightRow = 1, PrevRightCutDir = -1,
            BeatsSinceLastLeft = 999, BeatsSinceLastRight = 999,
        };
        return ScorePlacement(in ctx);
    }

    public double ScorePlacement(in NeuralPlacementContext ctx)
    {
        using var noGrad = no_grad();
        var f = new float[BeatSaberMappingNet.InputDim];
        ctx.FillFeaturesFloat(f);
        using var xT     = tensor(f, new long[] { 1, BeatSaberMappingNet.InputDim });
        using var outT   = _cpu.forward(xT);
        using var plLgt  = outT.select(1, 0);
        using var plProb = sigmoid(plLgt);
        return (double)plProb.item<float>();
    }

    // ── CPU shadow sync ───────────────────────────────────────────────────────

    /// <summary>
    /// Saves GPU model to a temp file and reloads into the CPU shadow model.
    /// Called at end of each epoch so inference always uses up-to-date weights.
    /// </summary>
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

    private static void FillFeaturesFromExample(TrainingExample ex, float[] arr, int offset)
    {
        arr[offset + 0]  = (float)ex.OnsetStrength;
        arr[offset + 1]  = (float)ex.EnergyLevel;
        arr[offset + 2]  = (float)ex.SubdivisionDenominator;
        arr[offset + 3]  = (float)Math.Min(ex.LocalNps / 10.0, 1.0);
        arr[offset + 4]  = (float)ex.BeatStrength;
        arr[offset + 5]  = (float)(ex.MeasureBeat / 4.0);
        arr[offset + 6]  = (float)(ex.DifficultyLevel / 4.0);
        arr[offset + 7]  = (float)ex.BeatPhase;
        arr[offset + 8]  = (float)ex.SectionProgress;
        arr[offset + 9]  = (float)(ex.PreviousLeftLane  / 3.0);
        arr[offset + 10] = (float)(ex.PreviousLeftRow   / 2.0);
        arr[offset + 11] = ex.PreviousLeftCutDir  >= 0 ? 1f : 0f;
        arr[offset + 12] = ex.PreviousLeftCutDir  >= 0 ? (float)(ex.PreviousLeftCutDir  / 8.0) : 0f;
        arr[offset + 13] = (float)(ex.PreviousRightLane / 3.0);
        arr[offset + 14] = (float)(ex.PreviousRightRow  / 2.0);
        arr[offset + 15] = ex.PreviousRightCutDir >= 0 ? 1f : 0f;
        arr[offset + 16] = ex.PreviousRightCutDir >= 0 ? (float)(ex.PreviousRightCutDir / 8.0) : 0f;
        arr[offset + 17] = (float)Math.Min(ex.BeatsSinceLastLeft  / 8.0, 1.0);
        arr[offset + 18] = (float)Math.Min(ex.BeatsSinceLastRight / 8.0, 1.0);
        arr[offset + 19] = (float)ex.LowBandEnergy;
        arr[offset + 20] = (float)ex.MidBandEnergy;
        arr[offset + 21] = (float)ex.HighBandEnergy;
        arr[offset + 22] = (float)ex.SpectralCentroid;
        arr[offset + 23] = (float)Math.Clamp(ex.EnergyDelta,      -1.0, 1.0);
        arr[offset + 24] = (float)Math.Clamp(ex.HighBandDelta,    -1.0, 1.0);
        arr[offset + 25] = (float)Math.Clamp(ex.TimeSinceAnyNote,  0.0, 1.0);
        arr[offset + 26] = (float)Math.Clamp(ex.SongFraction,      0.0, 1.0);
        arr[offset + 27] = (float)Math.Clamp(ex.LeftParityState,   0.0, 1.0);
        arr[offset + 28] = (float)Math.Clamp(ex.RightParityState,  0.0, 1.0);
        arr[offset + 29] = ex.Prev2LeftCutDir  >= 0 ? 1f : 0f;
        arr[offset + 30] = ex.Prev2LeftCutDir  >= 0 ? (float)(ex.Prev2LeftCutDir  / 8.0) : 0f;
        arr[offset + 31] = ex.Prev2RightCutDir >= 0 ? 1f : 0f;
        arr[offset + 32] = ex.Prev2RightCutDir >= 0 ? (float)(ex.Prev2RightCutDir / 8.0) : 0f;
        arr[offset + 33] = (float)Math.Clamp(ex.LookaheadEnergy, 0.0, 1.0);
        arr[offset + 34] = (float)Math.Clamp(ex.LookaheadOnset,  0.0, 1.0);
        // Hand hint: 0=left, 1=right, 0.5=unknown (no note or hand undetermined)
        arr[offset + 35] = ex.NoteHand == 0 ? 0f : ex.NoteHand == 1 ? 1f : 0.5f;
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
