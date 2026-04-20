using System.Diagnostics;
using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Diagnostics;
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
public sealed class TorchPlacementTrainer : IBatchedMultiTaskPlacementModel, IDisposable
{
    private sealed class CpuInferenceModelProxy : IBatchedMultiTaskPlacementModel
    {
        private readonly TorchPlacementTrainer _owner;

        public CpuInferenceModelProxy(TorchPlacementTrainer owner)
        {
            _owner = owner;
        }

        public double ScorePlacement(in NeuralPlacementContext ctx)
            => _owner.ScorePlacementCpu(in ctx);

        public double ScorePlacement(double onset, double energy, double subdiv, double localNps, double beatStrength,
            int measureBeat, int difficultyLevel)
            => _owner.ScorePlacement(onset, energy, subdiv, localNps, beatStrength, measureBeat, difficultyLevel);

        public NeuralMapPrediction PredictAll(in NeuralPlacementContext ctx)
            => _owner.PredictAllCpu(in ctx);

        public IReadOnlyList<NeuralMapPrediction> PredictAllBatch(IReadOnlyList<NeuralPlacementContext> contexts)
            => _owner.PredictAllBatchCpu(contexts);
    }

    public enum LoadedCheckpointKind
    {
        None,
        Current,
        Best,
        Legacy
    }

    private readonly BeatSaberMappingNet _gpu;
    private readonly BeatSaberMappingNet _cpu;
    private readonly Device              _device;
    private readonly bool                _useGpuInference;
    private readonly object              _gpuInferenceLock = new();
    private readonly object              _gpuBatchQueueLock = new();
    private readonly IBatchedMultiTaskPlacementModel _cpuInferenceModel;
    private          optim.Optimizer?    _optimizer;
    private          double              _optimizerLr = -1;
    private          string?             _bestWeightsTmp;
    private          bool                _gpuBatchProcessing;
    private readonly List<PendingGpuBatchRequest> _pendingGpuBatchRequests = [];

    // Larger windows reduce host-side tensor/setup overhead and keep the GPU busier.
    private const int WindowSize  = 256;
    private const int WindowStride = 256;
    private const int BatchSize   = 32;
    private const int GpuBatchCoalesceDelayMs = 2;

    private sealed class PendingGpuBatchRequest
    {
        public NeuralPlacementContext[] Contexts { get; }
        public TaskCompletionSource<NeuralMapPrediction[]> Completion { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public PendingGpuBatchRequest(IReadOnlyList<NeuralPlacementContext> contexts)
        {
            Contexts = contexts as NeuralPlacementContext[] ?? contexts.ToArray();
        }
    }

    public TorchPlacementTrainer()
    {
        _useGpuInference = cuda.is_available();
        _device = _useGpuInference ? CUDA : CPU;
        string deviceInfo = _useGpuInference
            ? ConsoleStyler.Colorize($"CUDA ({cuda.device_count()} device(s))", ConsoleColor.Green)
            : ConsoleStyler.Colorize("CPU", ConsoleColor.Yellow);
        Console.WriteLine($"[TorchSharp] Device: {deviceInfo}");
        string validationDevice = _useGpuInference
            ? ConsoleStyler.Colorize("GPU", ConsoleColor.Green)
            : ConsoleStyler.Colorize("CPU", ConsoleColor.Yellow);
        Console.WriteLine($"[TorchSharp] Validation inference default: {validationDevice}");

        _gpu = new BeatSaberMappingNet();
        _gpu.to(_device);
        _cpu = new BeatSaberMappingNet();
        _cpu.eval();
        _cpuInferenceModel = new CpuInferenceModelProxy(this);
    }

    public bool UsesGpuInference => _useGpuInference;
    public IBatchedMultiTaskPlacementModel CpuInferenceModel => _cpuInferenceModel;

    // ── Checkpoint ────────────────────────────────────────────────────────────

    public LoadedCheckpointKind TryLoadCheckpoint(string artifactsPath)
    {
        (string Path, LoadedCheckpointKind Kind)[] candidates =
        {
            (Path.Combine(artifactsPath, "neural_placement.current.pt"), LoadedCheckpointKind.Current),
            (Path.Combine(artifactsPath, "neural_placement.best.pt"), LoadedCheckpointKind.Best),
            (Path.Combine(artifactsPath, "neural_placement.pt"), LoadedCheckpointKind.Legacy),
        };

        foreach (var candidate in candidates)
        {
            if (!File.Exists(candidate.Path))
                continue;

            try
            {
                _gpu.load(candidate.Path);
                _gpu.to(_device);
                SyncCpuShadow();
                Console.WriteLine($"[TorchSharp] Warm-start from {Path.GetFileName(candidate.Path)}");
                return candidate.Kind;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TorchSharp] Checkpoint load failed from {Path.GetFileName(candidate.Path)}: {ex.Message}");
            }
        }

        return LoadedCheckpointKind.None;
    }

    // ── Epoch training ────────────────────────────────────────────────────────

    /// <summary>
    /// Sequence-based training epoch over contiguous beat windows with BPTT.
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
        var batches = BuildSequenceBatches(sequences, BatchSize);
        int totalBatches = batches.Count;
        var epochStopwatch = Stopwatch.StartNew();
        double lastProgressLogSeconds = 0;

        Console.WriteLine(
            $"[TorchSharp] TrainEpoch start: seqs={sequences.Count}  batches={totalBatches}  " +
            $"window={WindowSize}/{WindowStride}  batchSize={BatchSize}");

        for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++)
        {
            var batch = batches[batchIndex];
            int batchCount = batch.Count;
            if (batchCount == 0) continue;

            int maxSeqLen = batch.Max(s => s.Count);
            Tensor? hidden = null;

            for (int start = 0; start < maxSeqLen; start += WindowStride)
            {
                int len = Math.Min(WindowSize, Math.Max(0, maxSeqLen - start));
                if (len <= 0) break;

                var xArr      = new float[batchCount * len * D];
                var yPlArr    = new float[batchCount * len];
                var yHaArr    = new float[batchCount * len];
                var yCdArr    = new long[batchCount * len];
                var yLnArr    = new long[batchCount * len];
                var yRwArr    = new long[batchCount * len];
                var yHlArr    = new long[batchCount * len];
                var yPtArr    = new long[batchCount * len];
                var hasMask   = new float[batchCount * len];
                var cdMask    = new float[batchCount * len];
                var lnMask    = new float[batchCount * len];
                var rwMask    = new float[batchCount * len];
                var hlMask    = new float[batchCount * len];
                var ptMask    = new float[batchCount * len];
                var exWtArr   = new float[batchCount * len];
                var validMask = new float[batchCount * len];

                for (int b = 0; b < batchCount; b++)
                {
                    var seq = batch[b];
                    int remaining = seq.Count - start;
                    int seqLen = Math.Min(len, Math.Max(0, remaining));

                    for (int t = 0; t < seqLen; t++)
                    {
                        var ex = seq[start + t];
                        int idx = b * len + t;
                        bool validHand = ex.NoteHand is 0 or 1;
                        bool validCut = ex.NoteCutDir is >= 0 and <= 8;
                        bool validLane = ex.NoteLane is >= 0 and <= 3;
                        bool validRow = ex.NoteRow is >= 0 and <= 2;
                        bool validHandLane = validHand && validLane;
                        FillFeaturesFromExample(ex, xArr, idx * D);
                        yPlArr[idx]    = ex.HasNote ? 1f : 0f;
                        yHaArr[idx]    = ex.NoteHand == 1 ? 1f : 0f;
                        yCdArr[idx]    = validCut ? ex.NoteCutDir : 0;
                        yLnArr[idx]    = validLane ? ex.NoteLane : 0;
                        yRwArr[idx]    = validRow ? ex.NoteRow : 0;
                        yHlArr[idx]    = ex.HasNote && validHandLane
                            ? ex.NoteHand * 4 + ex.NoteLane
                            : 0;
                        yPtArr[idx]    = ex.PatternTypeId is >= 0 and < 6 ? ex.PatternTypeId : 0;
                        hasMask[idx]   = ex.HasNote && validHand ? 1f : 0f;
                        cdMask[idx]    = ex.HasNote && validCut ? 1f : 0f;
                        lnMask[idx]    = ex.HasNote && validLane ? 1f : 0f;
                        rwMask[idx]    = ex.HasNote && validRow ? 1f : 0f;
                        hlMask[idx]    = ex.HasNote && validHandLane ? 1f : 0f;
                        ptMask[idx]    = ex.HasNote ? 1f : 0f;
                        exWtArr[idx]   = (float)Math.Max(0.05, ex.Weight);
                        validMask[idx] = 1f;
                    }
                }

                long N = batchCount * len;
                using var xT      = tensor(xArr,    new long[] { batchCount, len, D }, device: _device);
                using var yPlT    = tensor(yPlArr,  new long[] { N },          device: _device);
                using var yHaT    = tensor(yHaArr,  new long[] { N },          device: _device);
                using var yCdT    = tensor(yCdArr,  new long[] { N },          device: _device);
                using var yLnT    = tensor(yLnArr,  new long[] { N },          device: _device);
                using var yRwT    = tensor(yRwArr,  new long[] { N },          device: _device);
                using var yHlT    = tensor(yHlArr,  new long[] { N },          device: _device);
                using var yPtT    = tensor(yPtArr,  new long[] { N },          device: _device);
                using var posWtT  = tensor(posWt,                             device: _device);
                using var haMaskT = tensor(hasMask, new long[] { N },         device: _device);
                using var cdMaskT = tensor(cdMask,  new long[] { N },         device: _device);
                using var lnMaskT = tensor(lnMask,  new long[] { N },         device: _device);
                using var rwMaskT = tensor(rwMask,  new long[] { N },         device: _device);
                using var hlMaskT = tensor(hlMask,  new long[] { N },         device: _device);
                using var ptMaskT = tensor(ptMask,  new long[] { N },         device: _device);
                using var exWtT   = tensor(exWtArr, new long[] { N },         device: _device);
                using var validT  = tensor(validMask, new long[] { N },       device: _device);

                var (outT, hn) = _gpu.ForwardSequence(xT, hidden);
                using (hidden) { }
                hidden = hn.detach();
                hn.Dispose();

                using var plLgt = outT.select(1, 0);
                using var haLgt = outT.select(1, 1);
                using var cdLgt = outT.narrow(1, 2,  9);
                using var lnLgt = outT.narrow(1, 11, 4);
                using var rwLgt = outT.narrow(1, 15, 3);
                using var hlLgt = outT.narrow(1, 18, 8);
                using var ptLgt = outT.narrow(1, 26, 6);

                var plBceTens = MaskedBinaryCrossEntropyWithLogits(plLgt, yPlT, exWtT, validT, posWtT);
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

                using var hlMaskIdx = (hlMaskT > 0.5f).nonzero().squeeze(1);
                if (hlMaskIdx.shape[0] > 0)
                {
                    using var hlSelLgt = hlLgt.index_select(0, hlMaskIdx);
                    using var hlSelY   = yHlT.index_select(0, hlMaskIdx);
                    using var hlSelW   = exWtT.index_select(0, hlMaskIdx);
                    var hlLoss         = 1.5 * WeightedCrossEntropy(hlSelLgt, hlSelY, hlSelW);
                    lossTerms.Add(hlLoss);
                }

                using var ptMaskIdx = (ptMaskT > 0.5f).nonzero().squeeze(1);
                if (ptMaskIdx.shape[0] > 0)
                {
                    using var ptSelLgt = ptLgt.index_select(0, ptMaskIdx);
                    using var ptSelY   = yPtT.index_select(0, ptMaskIdx);
                    using var ptSelW   = exWtT.index_select(0, ptMaskIdx);
                    var ptLoss         = 0.7 * WeightedCrossEntropy(ptSelLgt, ptSelY, ptSelW);
                    lossTerms.Add(ptLoss);
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

            double elapsed = epochStopwatch.Elapsed.TotalSeconds;
            bool shouldLogProgress =
                batchIndex == totalBatches - 1 ||
                elapsed - lastProgressLogSeconds >= 15.0;
            if (shouldLogProgress)
            {
                lastProgressLogSeconds = elapsed;
                double pct = totalBatches > 0 ? 100.0 * (batchIndex + 1) / totalBatches : 100.0;
                Console.WriteLine(
                    $"[TorchSharp] TrainEpoch progress: {batchIndex + 1}/{totalBatches} batches " +
                    $"({pct:F0}%)  chunks={totalChunks}  elapsed={elapsed:F0}s");
            }
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
        SyncCpuShadow();
        _optimizer?.Dispose();
        _optimizer   = null;
        _optimizerLr = -1;
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    public void SaveWeights(string artifactsPath)
    {
        SaveWeightsAs(artifactsPath, "neural_placement.pt");
    }

    public void SaveCurrentWeights(string artifactsPath)
    {
        SaveWeightsAs(artifactsPath, "neural_placement.current.pt");
    }

    public void SaveBestArtifact(string artifactsPath)
    {
        SaveWeightsAs(artifactsPath, "neural_placement.best.pt");
    }

    private void SaveWeightsAs(string artifactsPath, string fileName)
    {
        string path = Path.Combine(artifactsPath, fileName);
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
        return _useGpuInference
            ? ScorePlacementGpu(in ctx)
            : ScorePlacementCpu(in ctx);
    }

    public NeuralMapPrediction PredictAll(in NeuralPlacementContext ctx)
        => _useGpuInference
            ? PredictAllGpu(in ctx)
            : PredictAllCpu(in ctx);

    public IReadOnlyList<NeuralMapPrediction> PredictAllBatch(IReadOnlyList<NeuralPlacementContext> contexts)
    {
        if (contexts.Count == 0)
            return [];

        return _useGpuInference
            ? PredictAllBatchGpu(contexts)
            : PredictAllBatchCpu(contexts);
    }

    private double ScorePlacementCpu(in NeuralPlacementContext ctx)
    {
        using var noGrad = no_grad();
        var f = new float[BeatSaberMappingNet.InputDim];
        ctx.FillFeaturesFloat(f);
        using var xT = tensor(f, new long[] { 1, 1, BeatSaberMappingNet.InputDim });
        var (outT, _) = ForwardStepCpu(xT, ctx.GruHiddenState);
        using var plProb = sigmoid(outT.select(1, 0));
        double result = plProb.item<float>();
        outT.Dispose();
        return result;
    }

    private NeuralMapPrediction PredictAllCpu(in NeuralPlacementContext ctx)
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
        using var hlSm   = softmax(outT.narrow(1, 18, 8), dim: 1);
        using var ptSm   = softmax(outT.narrow(1, 26, 6), dim: 1);
        outT.Dispose();

        return new NeuralMapPrediction
        {
            PlacementScore = plProb.item<float>(),
            HandScore      = haProb.item<float>(),
            CutDirProbs    = ToDoubleArray(cdSm.squeeze(0)),
            LaneProbs      = ToDoubleArray(lnSm.squeeze(0)),
            RowProbs       = ToDoubleArray(rwSm.squeeze(0)),
            HandLaneProbs  = ToDoubleArray(hlSm.squeeze(0)),
            PatternTypeProbs = ToDoubleArray(ptSm.squeeze(0)),
        };
    }

    private double ScorePlacementGpu(in NeuralPlacementContext ctx)
        => PredictAllGpu(in ctx).PlacementScore;

    private IReadOnlyList<NeuralMapPrediction> PredictAllBatchCpu(IReadOnlyList<NeuralPlacementContext> contexts)
    {
        var results = new NeuralMapPrediction[contexts.Count];
        for (int i = 0; i < contexts.Count; i++)
        {
            var ctx = contexts[i];
            results[i] = PredictAllCpu(in ctx);
        }
        return results;
    }

    private NeuralMapPrediction PredictAllGpu(in NeuralPlacementContext ctx)
    {
        NeuralPlacementContext boxed = ctx;
        return PredictAllBatchGpu([boxed])[0];
    }

    private IReadOnlyList<NeuralMapPrediction> PredictAllBatchGpu(IReadOnlyList<NeuralPlacementContext> contexts)
    {
        var request = new PendingGpuBatchRequest(contexts);
        bool shouldProcess = false;

        lock (_gpuBatchQueueLock)
        {
            _pendingGpuBatchRequests.Add(request);
            if (!_gpuBatchProcessing)
            {
                _gpuBatchProcessing = true;
                shouldProcess = true;
            }
        }

        if (shouldProcess)
            ProcessPendingGpuBatchRequests();

        return request.Completion.Task.GetAwaiter().GetResult();
    }

    private void ProcessPendingGpuBatchRequests()
    {
        while (true)
        {
            Thread.Sleep(GpuBatchCoalesceDelayMs);

            List<PendingGpuBatchRequest> requests;
            lock (_gpuBatchQueueLock)
            {
                if (_pendingGpuBatchRequests.Count == 0)
                {
                    _gpuBatchProcessing = false;
                    return;
                }

                requests = [.. _pendingGpuBatchRequests];
                _pendingGpuBatchRequests.Clear();
            }

            try
            {
                int totalContexts = requests.Sum(r => r.Contexts.Length);
                var flatContexts = new NeuralPlacementContext[totalContexts];
                var offsets = new int[requests.Count];

                int offset = 0;
                for (int i = 0; i < requests.Count; i++)
                {
                    offsets[i] = offset;
                    Array.Copy(requests[i].Contexts, 0, flatContexts, offset, requests[i].Contexts.Length);
                    offset += requests[i].Contexts.Length;
                }

                var flatResults = PredictAllBatchGpuCore(flatContexts);

                for (int i = 0; i < requests.Count; i++)
                {
                    var req = requests[i];
                    var resultSlice = new NeuralMapPrediction[req.Contexts.Length];
                    Array.Copy(flatResults, offsets[i], resultSlice, 0, req.Contexts.Length);
                    req.Completion.TrySetResult(resultSlice);
                }
            }
            catch (Exception ex)
            {
                foreach (var request in requests)
                    request.Completion.TrySetException(ex);
            }
        }
    }

    private NeuralMapPrediction[] PredictAllBatchGpuCore(IReadOnlyList<NeuralPlacementContext> contexts)
    {
        lock (_gpuInferenceLock)
        {
            using var noGrad = no_grad();
            _gpu.eval();

            int batchSize = contexts.Count;
            int inputDim = BeatSaberMappingNet.InputDim;
            int hiddenSize = BeatSaberMappingNet.GruHiddenDim;
            int layers = BeatSaberMappingNet.GruLayers;

            var xArr = new float[batchSize * inputDim];
            var hArr = new float[layers * batchSize * hiddenSize];

            for (int i = 0; i < batchSize; i++)
            {
                var ctx = contexts[i];
                var features = new float[inputDim];
                ctx.FillFeaturesFloat(features);
                Array.Copy(features, 0, xArr, i * inputDim, inputDim);

                if (ctx.GruHiddenState is null)
                    continue;

                for (int layer = 0; layer < layers; layer++)
                {
                    Array.Copy(
                        ctx.GruHiddenState.H,
                        layer * hiddenSize,
                        hArr,
                        (layer * batchSize + i) * hiddenSize,
                        hiddenSize);
                }
            }

            using var xT = tensor(
                xArr,
                new long[] { batchSize, 1, inputDim },
                device: _device);
            using var h0 = tensor(
                hArr,
                new long[] { layers, batchSize, hiddenSize },
                device: _device);

            var (outT, hn) = _gpu.ForwardSequence(xT, h0);
            var hnData = hn.data<float>();
            var hnArr = hnData.ToArray();

            for (int i = 0; i < batchSize; i++)
            {
                var state = contexts[i].GruHiddenState;
                if (state is null)
                    continue;

                for (int layer = 0; layer < layers; layer++)
                {
                    Array.Copy(
                        hnArr,
                        (layer * batchSize + i) * hiddenSize,
                        state.H,
                        layer * hiddenSize,
                        hiddenSize);
                }
            }

            hn.Dispose();

            using var plProb = sigmoid(outT.select(1, 0));
            using var haProb = sigmoid(outT.select(1, 1));
            using var cdSm   = softmax(outT.narrow(1, 2,  9), dim: 1);
            using var lnSm   = softmax(outT.narrow(1, 11, 4), dim: 1);
            using var rwSm   = softmax(outT.narrow(1, 15, 3), dim: 1);
            using var hlSm   = softmax(outT.narrow(1, 18, 8), dim: 1);
            using var ptSm   = softmax(outT.narrow(1, 26, 6), dim: 1);

            var plData = plProb.data<float>();
            var haData = haProb.data<float>();
            var cdData = cdSm.data<float>().ToArray();
            var lnData = lnSm.data<float>().ToArray();
            var rwData = rwSm.data<float>().ToArray();
            var hlData = hlSm.data<float>().ToArray();
            var ptData = ptSm.data<float>().ToArray();

            var results = new NeuralMapPrediction[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                results[i] = new NeuralMapPrediction
                {
                    PlacementScore = plData[i],
                    HandScore      = haData[i],
                    CutDirProbs    = SliceToDoubleArray(cdData, i * 9, 9),
                    LaneProbs      = SliceToDoubleArray(lnData, i * 4, 4),
                    RowProbs       = SliceToDoubleArray(rwData, i * 3, 3),
                    HandLaneProbs  = SliceToDoubleArray(hlData, i * 8, 8),
                    PatternTypeProbs = SliceToDoubleArray(ptData, i * 6, 6),
                };
            }

            outT.Dispose();
            return results;
        }
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

    private static Tensor MaskedBinaryCrossEntropyWithLogits(
        Tensor logits,
        Tensor targets,
        Tensor weights,
        Tensor validMask,
        Tensor posWeight)
    {
        using var raw = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            weights,
            Reduction.None,
            posWeight);
        using var numer = (raw * validMask).sum();
        using var denom = validMask.sum().clamp_min(1e-6);
        return numer / denom;
    }

    private static IReadOnlyList<IReadOnlyList<IReadOnlyList<TrainingExample>>> BuildSequenceBatches(
        IReadOnlyList<IReadOnlyList<TrainingExample>> sequences,
        int batchSize)
    {
        var ordered = new List<IReadOnlyList<TrainingExample>>(sequences.Count);
        foreach (var seq in sequences)
        {
            if (seq.Count > 0)
                ordered.Add(seq);
        }

        ordered.Sort((a, b) => b.Count.CompareTo(a.Count));

        var batches = new List<IReadOnlyList<IReadOnlyList<TrainingExample>>>();
        for (int i = 0; i < ordered.Count; i += batchSize)
        {
            int count = Math.Min(batchSize, ordered.Count - i);
            var batch = new List<IReadOnlyList<TrainingExample>>(count);
            for (int j = 0; j < count; j++)
                batch.Add(ordered[i + j]);
            batches.Add(batch);
        }
        return batches;
    }

    private static double[] ToDoubleArray(Tensor t)
    {
        var data = t.data<float>();
        var arr  = new double[data.Count];
        for (int i = 0; i < arr.Length; i++) arr[i] = data[i];
        return arr;
    }

    private static double[] SliceToDoubleArray(float[] source, int start, int length)
    {
        var result = new double[length];
        for (int i = 0; i < length; i++)
            result[i] = source[start + i];
        return result;
    }

    // ── CPU shadow sync ───────────────────────────────────────────────────────

    private void SyncCpuShadow()
    {
        if (_useGpuInference)
            return;

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
        // Phrase / history / geometry [45-77]
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
        arr[offset + 68] = (float)Math.Clamp(ex.PhraseBeatPhase32 / 32.0, 0.0, 1.0);
        arr[offset + 69] = (float)Math.Clamp(ex.PhraseProgress32, 0.0, 1.0);
        arr[offset + 70] = (float)Math.Clamp(ex.BeatsSincePhraseStart32 / 32.0, 0.0, 1.0);
        arr[offset + 71] = (float)Math.Clamp(ex.BeatsToPhraseBoundary32 / 32.0, 0.0, 1.0);
        arr[offset + 72] = (float)Math.Clamp(ex.CurrentBeatVisionBlockRisk, 0.0, 1.0);
        arr[offset + 73] = (float)Math.Clamp(ex.RecentVisionBlockRate8, 0.0, 1.0);
        arr[offset + 74] = (float)Math.Clamp(ex.LeftParityBreakRate8, 0.0, 1.0);
        arr[offset + 75] = (float)Math.Clamp(ex.RightParityBreakRate8, 0.0, 1.0);
        arr[offset + 76] = (float)Math.Clamp(ex.ResetPressure, 0.0, 1.0);
        arr[offset + 77] = (float)Math.Clamp(ex.RecentRestRatio8, 0.0, 1.0);
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
