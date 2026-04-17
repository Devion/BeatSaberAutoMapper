using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Training.Features;

namespace BeatSaber.AutoMapper.Training.Models;

/// <summary>
/// Four-layer multi-task neural network for Beat Saber note generation.
/// Architecture: 23 inputs → 256 → 128 → 64 (shared trunk) → 5 heads.
///
/// Heads:
///   placement  : sigmoid(1)  — P(note at this beat position)
///   hand       : sigmoid(1)  — P(right hand | note)
///   cutDir     : softmax(9)  — cut-direction probability distribution
///   lane       : softmax(4)  — lane (0-3) distribution
///   row        : softmax(3)  — row (0-2) distribution
///
/// Input features (23):
///   0-8   onset, energy, subdiv, localNps, beatStrength, measureBeat,
///          difficulty, beatPhase, sectionProgress
///   9-18  prevLeft(lane,row,hasFlag,dirVal), prevRight(lane,row,hasFlag,dirVal),
///          beatsSinceLeft, beatsSinceRight
///   19-22 lowBandEnergy, midBandEnergy, highBandEnergy, spectralCentroid
///
/// Trained with Adam (β1=0.9, β2=0.999), He init, grad clip (max-norm=5),
/// L2 λ=1e-5, parallel mini-batch, multi-task auxiliary losses.
/// Saved to neural_placement.bsam (binary, ~1.4 MB).
/// </summary>
public sealed class NeuralPlacementTrainer : IPlacementScorer
{
    // ── Architecture constants ────────────────────────────────────────────────
    internal const int InputDim = 23;
    internal const int H1       = 256;
    internal const int H2       = 128;
    internal const int H3       = 64;   // shared trunk output size

    private const string ArchTag = "23-256-128-64-mt5";

    private const int NHeadCd = 9;   // cut-direction classes
    private const int NHeadLn = 4;   // lane classes
    private const int NHeadRw = 3;   // row classes

    // Auxiliary task loss weights
    private const double LwHand   = 0.5;
    private const double LwCutDir = 1.0;
    private const double LwLane   = 0.8;
    private const double LwRow    = 0.5;

    // Adam hyper-parameters & regularisation
    private const double AdamBeta1   = 0.9;
    private const double AdamBeta2   = 0.999;
    private const double AdamEps     = 1e-8;
    private const double L2Lambda    = 1e-5;
    private const double MaxGradNorm = 5.0;

    // ── Shared trunk weights ──────────────────────────────────────────────────
    private double[] _W1 = new double[H1 * InputDim];  // 256 × 23 = 5 888
    private double[] _b1 = new double[H1];
    private double[] _W2 = new double[H2 * H1];        // 128 × 256 = 32 768
    private double[] _b2 = new double[H2];
    private double[] _W3 = new double[H3 * H2];        // 64  × 128 = 8 192
    private double[] _b3 = new double[H3];

    // ── Head weights ─────────────────────────────────────────────────────────
    private double[] _Wpl = new double[H3];            // placement
    private double   _bpl;
    private double[] _Wha = new double[H3];            // hand
    private double   _bha;
    private double[] _Wcd = new double[NHeadCd * H3];  // 9 × 64 = 576
    private double[] _bcd = new double[NHeadCd];
    private double[] _Wln = new double[NHeadLn * H3];  // 4 × 64 = 256
    private double[] _bln = new double[NHeadLn];
    private double[] _Wrw = new double[NHeadRw * H3];  // 3 × 64 = 192
    private double[] _brw = new double[NHeadRw];

    // ── Adam first / second moment vectors ───────────────────────────────────
    private double[] _mW1, _vW1, _mb1, _vb1;
    private double[] _mW2, _vW2, _mb2, _vb2;
    private double[] _mW3, _vW3, _mb3, _vb3;
    private double[] _mWpl, _vWpl;  private double _mbpl, _vbpl;
    private double[] _mWha, _vWha;  private double _mbha, _vbha;
    private double[] _mWcd, _vWcd, _mbcd, _vbcd;
    private double[] _mWln, _vWln, _mbln, _vbln;
    private double[] _mWrw, _vWrw, _mbrw, _vbrw;
    private long _adamStep;

    // ── Best-epoch snapshot (early stopping) ─────────────────────────────────
    private double[]? _bW1, _bb1, _bW2, _bb2, _bW3, _bb3;
    private double[]? _bWpl, _bWha, _bWcd, _bcd_s, _bWln, _bln_s, _bWrw, _brw_s;
    private double _bbpl, _bbha;

    // =========================================================================
    // Construction
    // =========================================================================

    public NeuralPlacementTrainer()
    {
        InitWeights();
        AllocMoments();
    }

    private void AllocMoments()
    {
        _mW1  = new double[H1 * InputDim]; _vW1  = new double[H1 * InputDim];
        _mb1  = new double[H1];            _vb1  = new double[H1];
        _mW2  = new double[H2 * H1];       _vW2  = new double[H2 * H1];
        _mb2  = new double[H2];            _vb2  = new double[H2];
        _mW3  = new double[H3 * H2];       _vW3  = new double[H3 * H2];
        _mb3  = new double[H3];            _vb3  = new double[H3];
        _mWpl = new double[H3];            _vWpl = new double[H3];
        _mWha = new double[H3];            _vWha = new double[H3];
        _mWcd = new double[NHeadCd * H3];  _vWcd = new double[NHeadCd * H3];
        _mbcd = new double[NHeadCd];       _vbcd = new double[NHeadCd];
        _mWln = new double[NHeadLn * H3];  _vWln = new double[NHeadLn * H3];
        _mbln = new double[NHeadLn];       _vbln = new double[NHeadLn];
        _mWrw = new double[NHeadRw * H3];  _vWrw = new double[NHeadRw * H3];
        _mbrw = new double[NHeadRw];       _vbrw = new double[NHeadRw];
    }

    // =========================================================================
    // IPlacementScorer
    // =========================================================================

    public double ScorePlacement(double onset, double energy, double subdiv,
                                 double localNps, double beatStrength, int measureBeat,
                                 int difficultyLevel)
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
    {
        var x = new double[InputDim];
        ctx.FillFeatures(x);
        return ForwardPlacement(x);
    }

    // =========================================================================
    // Training
    // =========================================================================

    public (double Loss, double Accuracy) TrainEpoch(
        IReadOnlyList<TrainingExample> examples, double lr)
    {
        int n = examples.Count;
        if (n == 0) return (0, 0);

        // Class-balance weight for the placement head
        int posCount = 0;
        for (int i = 0; i < n; i++) if (examples[i].HasNote) posCount++;
        int    negCount = n - posCount;
        double posWt    = posCount > 0 && negCount > 0
            ? Math.Min((double)negCount / posCount, 10.0)
            : 1.0;

        int P         = Environment.ProcessorCount;
        int batchSize = Math.Max(1024, P * 512);
        int numBatches = Math.Max(1, (n + batchSize - 1) / batchSize);

        // ── Per-thread workspace (allocated once, cleared per-batch) ─────────
        var tX    = Alloc2D(P, InputDim);
        var tZ1   = Alloc2D(P, H1); var tA1   = Alloc2D(P, H1); var tDA1  = Alloc2D(P, H1);
        var tZ2   = Alloc2D(P, H2); var tA2   = Alloc2D(P, H2); var tDA2  = Alloc2D(P, H2);
        var tZ3   = Alloc2D(P, H3); var tA3   = Alloc2D(P, H3); var tDA3  = Alloc2D(P, H3);
        // Head softmax activations (pre-allocated to avoid GC in hot loop)
        var tYcd  = Alloc2D(P, NHeadCd); var tYln = Alloc2D(P, NHeadLn); var tYrw = Alloc2D(P, NHeadRw);
        var tDcd  = Alloc2D(P, NHeadCd); var tDln = Alloc2D(P, NHeadLn); var tDrw = Alloc2D(P, NHeadRw);
        // Trunk + shared-layer gradients
        var tDW1  = Alloc2D(P, H1 * InputDim); var tDb1  = Alloc2D(P, H1);
        var tDW2  = Alloc2D(P, H2 * H1);       var tDb2  = Alloc2D(P, H2);
        var tDW3  = Alloc2D(P, H3 * H2);       var tDb3  = Alloc2D(P, H3);
        // Head weight gradients
        var tDWpl = Alloc2D(P, H3); var tDbpl = new double[P];
        var tDWha = Alloc2D(P, H3); var tDbha = new double[P];
        var tDWcd = Alloc2D(P, NHeadCd * H3); var tDbcd = Alloc2D(P, NHeadCd);
        var tDWln = Alloc2D(P, NHeadLn * H3); var tDbln = Alloc2D(P, NHeadLn);
        var tDWrw = Alloc2D(P, NHeadRw * H3); var tDbrw = Alloc2D(P, NHeadRw);
        var tLoss = new double[P];

        // Batch-level accumulators (summed across threads then Adam-updated)
        var sumDW1  = new double[H1 * InputDim]; var sumDb1  = new double[H1];
        var sumDW2  = new double[H2 * H1];       var sumDb2  = new double[H2];
        var sumDW3  = new double[H3 * H2];       var sumDb3  = new double[H3];
        var sumDWpl = new double[H3]; double sumDbpl = 0;
        var sumDWha = new double[H3]; double sumDbha = 0;
        var sumDWcd = new double[NHeadCd * H3];  var sumDbcd = new double[NHeadCd];
        var sumDWln = new double[NHeadLn * H3];  var sumDbln = new double[NHeadLn];
        var sumDWrw = new double[NHeadRw * H3];  var sumDbrw = new double[NHeadRw];

        double totalLoss = 0;

        for (int b = 0; b < numBatches; b++)
        {
            int bStart = b * batchSize;
            int bEnd   = Math.Min(bStart + batchSize, n);
            int bSize  = bEnd - bStart;

            for (int t = 0; t < P; t++)
            {
                Array.Clear(tDW1[t],  0, H1 * InputDim); Array.Clear(tDb1[t],  0, H1);
                Array.Clear(tDW2[t],  0, H2 * H1);       Array.Clear(tDb2[t],  0, H2);
                Array.Clear(tDW3[t],  0, H3 * H2);       Array.Clear(tDb3[t],  0, H3);
                Array.Clear(tDWpl[t], 0, H3); tDbpl[t] = 0;
                Array.Clear(tDWha[t], 0, H3); tDbha[t] = 0;
                Array.Clear(tDWcd[t], 0, NHeadCd * H3); Array.Clear(tDbcd[t], 0, NHeadCd);
                Array.Clear(tDWln[t], 0, NHeadLn * H3); Array.Clear(tDbln[t], 0, NHeadLn);
                Array.Clear(tDWrw[t], 0, NHeadRw * H3); Array.Clear(tDbrw[t], 0, NHeadRw);
                tLoss[t] = 0;
            }

            int chunk = Math.Max(1, (bSize + P - 1) / P);
            Parallel.For(0, P, new ParallelOptions { MaxDegreeOfParallelism = P }, t =>
            {
                int s   = bStart + t * chunk;
                int end = Math.Min(s + chunk, bEnd);
                if (s >= bEnd) return;

                double localLoss = 0;
                double[] x   = tX[t];
                double[] z1  = tZ1[t];  double[] a1  = tA1[t];  double[] da1 = tDA1[t];
                double[] z2  = tZ2[t];  double[] a2  = tA2[t];  double[] da2 = tDA2[t];
                double[] z3  = tZ3[t];  double[] a3  = tA3[t];  double[] da3 = tDA3[t];
                double[] ycd = tYcd[t]; double[] yln = tYln[t]; double[] yrw = tYrw[t];
                double[] dcd = tDcd[t]; double[] dln = tDln[t]; double[] drw = tDrw[t];
                double[] dW1 = tDW1[t]; double[] db1 = tDb1[t];
                double[] dW2 = tDW2[t]; double[] db2 = tDb2[t];
                double[] dW3 = tDW3[t]; double[] db3 = tDb3[t];
                double[] dWpl = tDWpl[t]; double[] dWha = tDWha[t];
                double[] dWcd = tDWcd[t]; double[] dbcd = tDbcd[t];
                double[] dWln = tDWln[t]; double[] dbln = tDbln[t];
                double[] dWrw = tDWrw[t]; double[] dbrw = tDbrw[t];

                for (int i = s; i < end; i++)
                {
                    var ex = examples[i];
                    FillFromExample(ex, x);

                    // ── Forward pass ──────────────────────────────────────────
                    ForwardTrunk(x, z1, a1, z2, a2, z3, a3);

                    // Placement head
                    double zpl = _bpl;
                    for (int j = 0; j < H3; j++) zpl += _Wpl[j] * a3[j];
                    double ypl = Sigmoid(zpl);

                    // ── Loss & placement gradient ─────────────────────────────
                    double label_place = ex.HasNote ? 1.0 : 0.0;
                    double cw  = label_place > 0.5 ? posWt : 1.0;
                    double ew  = cw * ex.Weight;

                    localLoss += cw * -(label_place * Math.Log(ypl + 1e-10)
                                     + (1 - label_place) * Math.Log(1 - ypl + 1e-10));

                    double dpl = ew * (ypl - label_place);
                    tDbpl[t] += dpl;
                    for (int j = 0; j < H3; j++) dWpl[j] += dpl * a3[j];

                    // ── Trunk gradient from placement head ────────────────────
                    Array.Clear(da3, 0, H3);
                    for (int j = 0; j < H3; j++) da3[j] += dpl * _Wpl[j];

                    // ── Auxiliary heads (note-present examples only) ───────────
                    if (ex.HasNote)
                    {
                        // Hand head
                        double zha = _bha;
                        for (int j = 0; j < H3; j++) zha += _Wha[j] * a3[j];
                        double yha = Sigmoid(zha);
                        double lab_ha = ex.NoteHand == 1 ? 1.0 : 0.0;
                        double dha = LwHand * (yha - lab_ha);
                        tDbha[t] += dha;
                        for (int j = 0; j < H3; j++) { dWha[j] += dha * a3[j]; da3[j] += dha * _Wha[j]; }

                        // CutDir head
                        if (ex.NoteCutDir >= 0)
                        {
                            SoftmaxForward(_Wcd, _bcd, a3, ycd, NHeadCd);
                            for (int k = 0; k < NHeadCd; k++)
                                dcd[k] = LwCutDir * (ycd[k] - (k == ex.NoteCutDir ? 1.0 : 0.0));
                            for (int k = 0; k < NHeadCd; k++)
                            {
                                dbcd[k] += dcd[k];
                                int offK = k * H3;
                                for (int j = 0; j < H3; j++) { dWcd[offK + j] += dcd[k] * a3[j]; da3[j] += dcd[k] * _Wcd[offK + j]; }
                            }
                        }

                        // Lane head
                        if (ex.NoteLane >= 0)
                        {
                            SoftmaxForward(_Wln, _bln, a3, yln, NHeadLn);
                            for (int k = 0; k < NHeadLn; k++)
                                dln[k] = LwLane * (yln[k] - (k == ex.NoteLane ? 1.0 : 0.0));
                            for (int k = 0; k < NHeadLn; k++)
                            {
                                dbln[k] += dln[k];
                                int offK = k * H3;
                                for (int j = 0; j < H3; j++) { dWln[offK + j] += dln[k] * a3[j]; da3[j] += dln[k] * _Wln[offK + j]; }
                            }
                        }

                        // Row head
                        if (ex.NoteRow >= 0)
                        {
                            SoftmaxForward(_Wrw, _brw, a3, yrw, NHeadRw);
                            for (int k = 0; k < NHeadRw; k++)
                                drw[k] = LwRow * (yrw[k] - (k == ex.NoteRow ? 1.0 : 0.0));
                            for (int k = 0; k < NHeadRw; k++)
                            {
                                dbrw[k] += drw[k];
                                int offK = k * H3;
                                for (int j = 0; j < H3; j++) { dWrw[offK + j] += drw[k] * a3[j]; da3[j] += drw[k] * _Wrw[offK + j]; }
                            }
                        }
                    }

                    // ── Backprop through shared trunk ─────────────────────────
                    BackwardLayer(z3, a3, da3, a2, _W3, dW3, db3, da2, H3, H2);
                    BackwardLayer(z2, a2, da2, a1, _W2, dW2, db2, da1, H2, H1);
                    // Layer 1: propagate into x (no further layer, just update W1/b1)
                    for (int ii = 0; ii < H1; ii++)
                    {
                        double dz = z1[ii] > 0 ? da1[ii] : 0;
                        db1[ii] += dz;
                        int off = ii * InputDim;
                        for (int j = 0; j < InputDim; j++) dW1[off + j] += dz * x[j];
                    }
                }
                tLoss[t] = localLoss;
            });

            // ── Sum gradients across threads ──────────────────────────────────
            Array.Clear(sumDW1,  0, H1 * InputDim); Array.Clear(sumDb1,  0, H1);
            Array.Clear(sumDW2,  0, H2 * H1);       Array.Clear(sumDb2,  0, H2);
            Array.Clear(sumDW3,  0, H3 * H2);       Array.Clear(sumDb3,  0, H3);
            Array.Clear(sumDWpl, 0, H3); sumDbpl = 0;
            Array.Clear(sumDWha, 0, H3); sumDbha = 0;
            Array.Clear(sumDWcd, 0, NHeadCd * H3); Array.Clear(sumDbcd, 0, NHeadCd);
            Array.Clear(sumDWln, 0, NHeadLn * H3); Array.Clear(sumDbln, 0, NHeadLn);
            Array.Clear(sumDWrw, 0, NHeadRw * H3); Array.Clear(sumDbrw, 0, NHeadRw);

            for (int t = 0; t < P; t++)
            {
                for (int i = 0; i < H1 * InputDim; i++) sumDW1[i]  += tDW1[t][i];
                for (int i = 0; i < H1;             i++) sumDb1[i]  += tDb1[t][i];
                for (int i = 0; i < H2 * H1;        i++) sumDW2[i]  += tDW2[t][i];
                for (int i = 0; i < H2;             i++) sumDb2[i]  += tDb2[t][i];
                for (int i = 0; i < H3 * H2;        i++) sumDW3[i]  += tDW3[t][i];
                for (int i = 0; i < H3;             i++) sumDb3[i]  += tDb3[t][i];
                for (int i = 0; i < H3;             i++) sumDWpl[i] += tDWpl[t][i];
                sumDbpl += tDbpl[t];
                for (int i = 0; i < H3;             i++) sumDWha[i] += tDWha[t][i];
                sumDbha += tDbha[t];
                for (int i = 0; i < NHeadCd * H3;   i++) sumDWcd[i] += tDWcd[t][i];
                for (int i = 0; i < NHeadCd;         i++) sumDbcd[i] += tDbcd[t][i];
                for (int i = 0; i < NHeadLn * H3;   i++) sumDWln[i] += tDWln[t][i];
                for (int i = 0; i < NHeadLn;         i++) sumDbln[i] += tDbln[t][i];
                for (int i = 0; i < NHeadRw * H3;   i++) sumDWrw[i] += tDWrw[t][i];
                for (int i = 0; i < NHeadRw;         i++) sumDbrw[i] += tDbrw[t][i];
                totalLoss += tLoss[t];
            }

            // ── Gradient clipping ─────────────────────────────────────────────
            double gnorm = GradNorm(sumDW1, sumDb1, sumDW2, sumDb2, sumDW3, sumDb3,
                                    sumDWpl, sumDbpl, sumDWha, sumDbha,
                                    sumDWcd, sumDbcd, sumDWln, sumDbln, sumDWrw, sumDbrw);
            if (gnorm > MaxGradNorm)
            {
                double scale = MaxGradNorm / gnorm;
                Scale(sumDW1, scale); Scale(sumDb1, scale);
                Scale(sumDW2, scale); Scale(sumDb2, scale);
                Scale(sumDW3, scale); Scale(sumDb3, scale);
                Scale(sumDWpl, scale); sumDbpl *= scale;
                Scale(sumDWha, scale); sumDbha *= scale;
                Scale(sumDWcd, scale); Scale(sumDbcd, scale);
                Scale(sumDWln, scale); Scale(sumDbln, scale);
                Scale(sumDWrw, scale); Scale(sumDbrw, scale);
            }

            // ── Adam parameter update ─────────────────────────────────────────
            _adamStep++;
            double bc1 = 1.0 - Math.Pow(AdamBeta1, _adamStep);
            double bc2 = 1.0 - Math.Pow(AdamBeta2, _adamStep);
            double invB = 1.0 / bSize;

            AdamUpdate(_W1, _mW1, _vW1, sumDW1, lr, invB, bc1, bc2, L2Lambda);
            AdamUpdate(_b1, _mb1, _vb1, sumDb1, lr, invB, bc1, bc2, 0.0);
            AdamUpdate(_W2, _mW2, _vW2, sumDW2, lr, invB, bc1, bc2, L2Lambda);
            AdamUpdate(_b2, _mb2, _vb2, sumDb2, lr, invB, bc1, bc2, 0.0);
            AdamUpdate(_W3, _mW3, _vW3, sumDW3, lr, invB, bc1, bc2, L2Lambda);
            AdamUpdate(_b3, _mb3, _vb3, sumDb3, lr, invB, bc1, bc2, 0.0);
            AdamUpdate(_Wpl, _mWpl, _vWpl, sumDWpl, lr, invB, bc1, bc2, L2Lambda);
            AdamScalar(ref _bpl, ref _mbpl, ref _vbpl, sumDbpl * invB, lr, bc1, bc2);
            AdamUpdate(_Wha, _mWha, _vWha, sumDWha, lr, invB, bc1, bc2, L2Lambda);
            AdamScalar(ref _bha, ref _mbha, ref _vbha, sumDbha * invB, lr, bc1, bc2);
            AdamUpdate(_Wcd, _mWcd, _vWcd, sumDWcd, lr, invB, bc1, bc2, L2Lambda);
            AdamUpdate(_bcd, _mbcd, _vbcd, sumDbcd, lr, invB, bc1, bc2, 0.0);
            AdamUpdate(_Wln, _mWln, _vWln, sumDWln, lr, invB, bc1, bc2, L2Lambda);
            AdamUpdate(_bln, _mbln, _vbln, sumDbln, lr, invB, bc1, bc2, 0.0);
            AdamUpdate(_Wrw, _mWrw, _vWrw, sumDWrw, lr, invB, bc1, bc2, L2Lambda);
            AdamUpdate(_brw, _mbrw, _vbrw, sumDbrw, lr, invB, bc1, bc2, 0.0);
        }

        return (totalLoss / n, ComputeAccuracy(examples));
    }

    // =========================================================================
    // Checkpoint load (warm-start from binary .bsam)
    // =========================================================================

    public bool TryLoadCheckpoint(string artifactsPath)
    {
        string path = Path.Combine(artifactsPath, "neural_placement.bsam");
        if (!File.Exists(path)) return false;
        try
        {
            using var br = new BinaryReader(File.OpenRead(path));
            if (br.ReadInt32() != 0x4253414D) return false;  // "BSAM"
            if (br.ReadInt32() != 3)          return false;  // version
            if (br.ReadInt32() != InputDim)   return false;
            if (br.ReadInt32() != H1)         return false;
            if (br.ReadInt32() != H2)         return false;
            if (br.ReadInt32() != H3)         return false;

            ReadDoubles(br, _W1); ReadDoubles(br, _b1);
            ReadDoubles(br, _W2); ReadDoubles(br, _b2);
            ReadDoubles(br, _W3); ReadDoubles(br, _b3);
            ReadDoubles(br, _Wpl); _bpl = br.ReadDouble();
            ReadDoubles(br, _Wha); _bha = br.ReadDouble();
            ReadDoubles(br, _Wcd); ReadDoubles(br, _bcd);
            ReadDoubles(br, _Wln); ReadDoubles(br, _bln);
            ReadDoubles(br, _Wrw); ReadDoubles(br, _brw);

            // Reset Adam moments — re-warm within a few epochs
            ClearMoments();
            _adamStep = 0;
            return true;
        }
        catch { return false; }
    }

    // =========================================================================
    // Best-weights management (early stopping)
    // =========================================================================

    public void SaveBestWeights()
    {
        _bW1 = (double[])_W1.Clone(); _bb1 = (double[])_b1.Clone();
        _bW2 = (double[])_W2.Clone(); _bb2 = (double[])_b2.Clone();
        _bW3 = (double[])_W3.Clone(); _bb3 = (double[])_b3.Clone();
        _bWpl = (double[])_Wpl.Clone(); _bbpl = _bpl;
        _bWha = (double[])_Wha.Clone(); _bbha = _bha;
        _bWcd = (double[])_Wcd.Clone(); _bcd_s = (double[])_bcd.Clone();
        _bWln = (double[])_Wln.Clone(); _bln_s = (double[])_bln.Clone();
        _bWrw = (double[])_Wrw.Clone(); _brw_s = (double[])_brw.Clone();
    }

    public void RestoreBestWeights()
    {
        if (_bW1 is null) return;
        _W1 = (double[])_bW1.Clone(); _b1 = (double[])_bb1!.Clone();
        _W2 = (double[])_bW2!.Clone(); _b2 = (double[])_bb2!.Clone();
        _W3 = (double[])_bW3!.Clone(); _b3 = (double[])_bb3!.Clone();
        _Wpl = (double[])_bWpl!.Clone(); _bpl = _bbpl;
        _Wha = (double[])_bWha!.Clone(); _bha = _bbha;
        _Wcd = (double[])_bWcd!.Clone(); _bcd = (double[])_bcd_s!.Clone();
        _Wln = (double[])_bWln!.Clone(); _bln = (double[])_bln_s!.Clone();
        _Wrw = (double[])_bWrw!.Clone(); _brw = (double[])_brw_s!.Clone();
    }

    // =========================================================================
    // Serialisation (binary .bsam)
    // =========================================================================

    public void SaveWeights(string artifactsPath)
    {
        Directory.CreateDirectory(artifactsPath);
        string path = Path.Combine(artifactsPath, "neural_placement.bsam");
        using var bw = new BinaryWriter(File.OpenWrite(path));
        bw.Write(0x4253414D);   // "BSAM" magic
        bw.Write(3);            // version
        bw.Write(InputDim);
        bw.Write(H1);
        bw.Write(H2);
        bw.Write(H3);
        WriteDoubles(bw, _W1); WriteDoubles(bw, _b1);
        WriteDoubles(bw, _W2); WriteDoubles(bw, _b2);
        WriteDoubles(bw, _W3); WriteDoubles(bw, _b3);
        WriteDoubles(bw, _Wpl); bw.Write(_bpl);
        WriteDoubles(bw, _Wha); bw.Write(_bha);
        WriteDoubles(bw, _Wcd); WriteDoubles(bw, _bcd);
        WriteDoubles(bw, _Wln); WriteDoubles(bw, _bln);
        WriteDoubles(bw, _Wrw); WriteDoubles(bw, _brw);
    }

    // =========================================================================
    // PredictFromExample (used by EvaluationRunner)
    // =========================================================================

    public double PredictFromExample(TrainingExample ex)
    {
        var x = new double[InputDim];
        FillFromExample(ex, x);
        return ForwardPlacement(x);
    }

    // =========================================================================
    // Private: forward passes
    // =========================================================================

    private void ForwardTrunk(
        double[] x,
        double[] z1, double[] a1,
        double[] z2, double[] a2,
        double[] z3, double[] a3)
    {
        for (int i = 0; i < H1; i++)
        {
            double s = _b1[i]; int off = i * InputDim;
            for (int j = 0; j < InputDim; j++) s += _W1[off + j] * x[j];
            z1[i] = s; a1[i] = s > 0 ? s : 0;
        }
        for (int i = 0; i < H2; i++)
        {
            double s = _b2[i]; int off = i * H1;
            for (int j = 0; j < H1; j++) s += _W2[off + j] * a1[j];
            z2[i] = s; a2[i] = s > 0 ? s : 0;
        }
        for (int i = 0; i < H3; i++)
        {
            double s = _b3[i]; int off = i * H2;
            for (int j = 0; j < H2; j++) s += _W3[off + j] * a2[j];
            z3[i] = s; a3[i] = s > 0 ? s : 0;
        }
    }

    private double ForwardPlacement(double[] x)
    {
        var a1 = new double[H1];
        for (int i = 0; i < H1; i++) { double s = _b1[i]; int off = i * InputDim; for (int j = 0; j < InputDim; j++) s += _W1[off + j] * x[j]; a1[i] = s > 0 ? s : 0; }
        var a2 = new double[H2];
        for (int i = 0; i < H2; i++) { double s = _b2[i]; int off = i * H1; for (int j = 0; j < H1; j++) s += _W2[off + j] * a1[j]; a2[i] = s > 0 ? s : 0; }
        var a3 = new double[H3];
        for (int i = 0; i < H3; i++) { double s = _b3[i]; int off = i * H2; for (int j = 0; j < H2; j++) s += _W3[off + j] * a2[j]; a3[i] = s > 0 ? s : 0; }
        double z = _bpl; for (int j = 0; j < H3; j++) z += _Wpl[j] * a3[j];
        return Sigmoid(z);
    }

    internal static void SoftmaxForward(double[] W, double[] b, double[] a3, double[] y, int nOut)
    {
        double maxZ = double.NegativeInfinity;
        for (int k = 0; k < nOut; k++)
        {
            double s = b[k]; int off = k * H3;
            for (int j = 0; j < H3; j++) s += W[off + j] * a3[j];
            y[k] = s;
            if (s > maxZ) maxZ = s;
        }
        double sum = 0;
        for (int k = 0; k < nOut; k++) { y[k] = Math.Exp(y[k] - maxZ); sum += y[k]; }
        if (sum > 0) for (int k = 0; k < nOut; k++) y[k] /= sum;
    }

    // =========================================================================
    // Private: backward helpers
    // =========================================================================

    /// <summary>
    /// Backprop through one ReLU layer.
    /// dz = (z > 0) ? da : 0; accumulates dW, db; sets da_in = W^T dz.
    /// </summary>
    private static void BackwardLayer(
        double[] z, double[] a, double[] da,
        double[] aIn, double[] W, double[] dW, double[] db, double[] daIn,
        int nOut, int nIn)
    {
        Array.Clear(daIn, 0, nIn);
        for (int i = 0; i < nOut; i++)
        {
            double dzi = z[i] > 0 ? da[i] : 0;
            db[i] += dzi;
            int off = i * nIn;
            for (int j = 0; j < nIn; j++)
            {
                dW[off + j] += dzi * aIn[j];
                daIn[j]     += dzi * W[off + j];
            }
        }
    }

    private static double GradNorm(
        double[] dW1, double[] db1,
        double[] dW2, double[] db2,
        double[] dW3, double[] db3,
        double[] dWpl, double dbpl,
        double[] dWha, double dbha,
        double[] dWcd, double[] dbcd,
        double[] dWln, double[] dbln,
        double[] dWrw, double[] dbrw)
    {
        double n = 0;
        foreach (var d in dW1) n += d * d; foreach (var d in db1) n += d * d;
        foreach (var d in dW2) n += d * d; foreach (var d in db2) n += d * d;
        foreach (var d in dW3) n += d * d; foreach (var d in db3) n += d * d;
        foreach (var d in dWpl) n += d * d; n += dbpl * dbpl;
        foreach (var d in dWha) n += d * d; n += dbha * dbha;
        foreach (var d in dWcd) n += d * d; foreach (var d in dbcd) n += d * d;
        foreach (var d in dWln) n += d * d; foreach (var d in dbln) n += d * d;
        foreach (var d in dWrw) n += d * d; foreach (var d in dbrw) n += d * d;
        return Math.Sqrt(n);
    }

    private static void Scale(double[] arr, double s) { for (int i = 0; i < arr.Length; i++) arr[i] *= s; }

    private static void AdamUpdate(
        double[] w, double[] m, double[] v, double[] g,
        double lr, double invB, double bc1, double bc2, double lambda)
    {
        for (int i = 0; i < w.Length; i++)
        {
            double gi = g[i] * invB + lambda * w[i];
            m[i] = AdamBeta1 * m[i] + (1 - AdamBeta1) * gi;
            v[i] = AdamBeta2 * v[i] + (1 - AdamBeta2) * gi * gi;
            w[i] -= lr * (m[i] / bc1) / (Math.Sqrt(v[i] / bc2) + AdamEps);
        }
    }

    private static void AdamScalar(ref double w, ref double m, ref double v,
                                   double g, double lr, double bc1, double bc2)
    {
        m = AdamBeta1 * m + (1 - AdamBeta1) * g;
        v = AdamBeta2 * v + (1 - AdamBeta2) * g * g;
        w -= lr * (m / bc1) / (Math.Sqrt(v / bc2) + AdamEps);
    }

    private double ComputeAccuracy(IReadOnlyList<TrainingExample> examples)
    {
        int n = examples.Count;
        if (n == 0) return 0;
        int P = Environment.ProcessorCount;
        int chunk = Math.Max(1, (n + P - 1) / P);
        var correct = new long[P];
        Parallel.For(0, P, new ParallelOptions { MaxDegreeOfParallelism = P }, t =>
        {
            int s = t * chunk, end = Math.Min(s + chunk, n);
            var x = new double[InputDim];
            long c = 0;
            for (int i = s; i < end; i++)
            {
                FillFromExample(examples[i], x);
                if ((ForwardPlacement(x) > 0.5) == examples[i].HasNote) c++;
            }
            correct[t] = c;
        });
        return correct.Sum() / (double)n;
    }

    // =========================================================================
    // Private: weight init, I/O, helpers
    // =========================================================================

    private void InitWeights()
    {
        var rng = new Random(42);
        He(rng, _W1, InputDim); He(rng, _W2, H1); He(rng, _W3, H2);
        He(rng, _Wpl, H3); He(rng, _Wha, H3);
        He(rng, _Wcd, H3); He(rng, _Wln, H3); He(rng, _Wrw, H3);
    }

    private void ClearMoments()
    {
        ClearArr(_mW1); ClearArr(_vW1); ClearArr(_mb1); ClearArr(_vb1);
        ClearArr(_mW2); ClearArr(_vW2); ClearArr(_mb2); ClearArr(_vb2);
        ClearArr(_mW3); ClearArr(_vW3); ClearArr(_mb3); ClearArr(_vb3);
        ClearArr(_mWpl); ClearArr(_vWpl); _mbpl = 0; _vbpl = 0;
        ClearArr(_mWha); ClearArr(_vWha); _mbha = 0; _vbha = 0;
        ClearArr(_mWcd); ClearArr(_vWcd); ClearArr(_mbcd); ClearArr(_vbcd);
        ClearArr(_mWln); ClearArr(_vWln); ClearArr(_mbln); ClearArr(_vbln);
        ClearArr(_mWrw); ClearArr(_vWrw); ClearArr(_mbrw); ClearArr(_vbrw);
    }

    private static void ClearArr(double[] a) => Array.Clear(a, 0, a.Length);

    private static void He(Random rng, double[] w, int fanIn)
    {
        double std = Math.Sqrt(2.0 / fanIn);
        for (int i = 0; i < w.Length; i++) w[i] = SampleNormal(rng) * std;
    }

    private static double SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double[][] Alloc2D(int rows, int cols)
    {
        var a = new double[rows][];
        for (int i = 0; i < rows; i++) a[i] = new double[cols];
        return a;
    }

    private static void WriteDoubles(BinaryWriter bw, double[] arr)
    {
        var bytes = new byte[arr.Length * sizeof(double)];
        Buffer.BlockCopy(arr, 0, bytes, 0, bytes.Length);
        bw.Write(bytes);
    }

    private static void ReadDoubles(BinaryReader br, double[] arr)
    {
        int byteCount = arr.Length * sizeof(double);
        var bytes = br.ReadBytes(byteCount);
        Buffer.BlockCopy(bytes, 0, arr, 0, byteCount);
    }

    internal static void FillFromExample(TrainingExample ex, double[] f)
    {
        f[0]  = ex.OnsetStrength;
        f[1]  = ex.EnergyLevel;
        f[2]  = ex.SubdivisionDenominator;
        f[3]  = Math.Min(ex.LocalNps / 10.0, 1.0);
        f[4]  = ex.BeatStrength;
        f[5]  = ex.MeasureBeat / 4.0;
        f[6]  = ex.DifficultyLevel / 4.0;
        f[7]  = ex.BeatPhase;
        f[8]  = ex.SectionProgress;
        f[9]  = ex.PreviousLeftLane  / 3.0;
        f[10] = ex.PreviousLeftRow   / 2.0;
        f[11] = ex.PreviousLeftCutDir  >= 0 ? 1.0 : 0.0;
        f[12] = ex.PreviousLeftCutDir  >= 0 ? ex.PreviousLeftCutDir  / 8.0 : 0.0;
        f[13] = ex.PreviousRightLane / 3.0;
        f[14] = ex.PreviousRightRow  / 2.0;
        f[15] = ex.PreviousRightCutDir >= 0 ? 1.0 : 0.0;
        f[16] = ex.PreviousRightCutDir >= 0 ? ex.PreviousRightCutDir / 8.0 : 0.0;
        f[17] = Math.Min(ex.BeatsSinceLastLeft  / 8.0, 1.0);
        f[18] = Math.Min(ex.BeatsSinceLastRight / 8.0, 1.0);
        f[19] = ex.LowBandEnergy;
        f[20] = ex.MidBandEnergy;
        f[21] = ex.HighBandEnergy;
        f[22] = ex.SpectralCentroid;
    }
}
