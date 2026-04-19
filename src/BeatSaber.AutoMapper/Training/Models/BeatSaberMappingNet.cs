using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BeatSaber.AutoMapper.Training.Models;

/// <summary>
/// GRU-based multi-task neural network for Beat Saber note generation.
///
/// Architecture: GRU(input=68, hidden=320, layers=2, causal) → MLP(320→160) → 6 heads
///
/// Training: forward(x=[B, SeqLen, D]) → [B*SeqLen, OutDim]  (BPTT over W=16 windows)
/// Inference: ForwardStep(x=[1,1,D], h=[L,1,H]) → ([1, OutDim], [L,1,H])
///
/// Output layout [B*SeqLen, 32]:
///   [0]      placement logit  (sigmoid → P(note here))
///   [1]      hand logit       (sigmoid → P(right hand))
///   [2..10]  cut-dir logits   (softmax → 9 classes)
///   [11..14] lane logits      (softmax → 4 lanes)
///   [15..17] row logits       (softmax → 3 rows)
///   [18..25] hand+lane logits (softmax → 8 classes: L0..L3,R0..R3)
///   [26..31] pattern logits   (softmax → 6 classes)
///
/// ~540K parameters; float32 .pt file ~2.1 MB.
/// </summary>
internal sealed class BeatSaberMappingNet : Module<Tensor, Tensor>
{
    internal const int InputDim    = 68;
    internal const int GruHiddenDim = 320;
    internal const int GruLayers   = 2;
    internal const int MlpHidden   = 160;
    internal const int OutDim      = 32;   // 1+1+9+4+3+8+6

    private readonly GRU     _gru;
    private readonly Linear  _mlpLin;
    private readonly LayerNorm _mlpLn;
    private readonly Dropout _mlpDrop;
    private readonly Linear _headPl, _headHa, _headCd, _headLn, _headRw, _headHl, _headPt;

    internal BeatSaberMappingNet() : base(nameof(BeatSaberMappingNet))
    {
        _gru     = GRU(InputDim, GruHiddenDim, numLayers: GruLayers,
                       batchFirst: false, dropout: 0.10);
        _mlpLin  = Linear(GruHiddenDim, MlpHidden, hasBias: false);
        _mlpLn   = LayerNorm(MlpHidden);
        _mlpDrop = Dropout(0.15);
        _headPl  = Linear(MlpHidden, 1);
        _headHa  = Linear(MlpHidden, 1);
        _headCd  = Linear(MlpHidden, 9);
        _headLn  = Linear(MlpHidden, 4);
        _headRw  = Linear(MlpHidden, 3);
        _headHl  = Linear(MlpHidden, 8);
        _headPt  = Linear(MlpHidden, 6);
        RegisterComponents();
    }

    /// <summary>
    /// Training forward pass.
    /// Input x: [B, SeqLen, D]. GRU h0 = zeros.
    /// Returns: [B*SeqLen, OutDim]
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        long B = x.shape[0], S = x.shape[1];
        // GRU expects [SeqLen, Batch, D] when batchFirst=false
        using var xT = x.permute(1, 0, 2);           // [S, B, D]
        var (gruOut, _) = _gru.forward(xT, null);    // gruOut: [S, B, H]
        using var gruT = gruOut.permute(1, 0, 2);    // [B, S, H]
        using var flat = gruT.reshape(B * S, GruHiddenDim);  // [B*S, H]
        return ApplyMlpAndHeads(flat);
    }

    internal (Tensor output, Tensor newHidden) ForwardSequence(Tensor x, Tensor? h)
    {
        long B = x.shape[0], S = x.shape[1];
        using var xT = x.permute(1, 0, 2);         // [S, B, D]
        var (gruOut, hn) = _gru.forward(xT, h);    // [S, B, H], [L, B, H]
        using var gruT = gruOut.permute(1, 0, 2);  // [B, S, H]
        using var flat = gruT.reshape(B * S, GruHiddenDim);
        return (ApplyMlpAndHeads(flat), hn);
    }

    /// <summary>
    /// Single-step inference. Input x: [1, 1, D], h: [GruLayers, 1, H].
    /// Returns (output: [1, OutDim], newHidden: [GruLayers, 1, H]).
    /// </summary>
    internal (Tensor output, Tensor newHidden) ForwardStep(Tensor x, Tensor h)
    {
        // x is [1, 1, D] in [B=1, S=1, D] format → need [S=1, B=1, D] for GRU
        using var xT = x.permute(1, 0, 2);         // [1, 1, D]
        var (gruOut, hn) = _gru.forward(xT, h);    // gruOut: [1, 1, H], hn: [L, 1, H]
        using var flat = gruOut.squeeze(0);         // [1, H]
        var output = ApplyMlpAndHeads(flat);        // [1, OutDim]
        return (output, hn);
    }

    private Tensor ApplyMlpAndHeads(Tensor flat)
    {
        using var h0  = _mlpLin.forward(flat);
        using var h1  = _mlpLn.forward(h0);
        using var h   = _mlpDrop.forward(functional.relu(h1));
        using var pl  = _headPl.forward(h);   // [N, 1]
        using var ha  = _headHa.forward(h);   // [N, 1]
        using var cd  = _headCd.forward(h);   // [N, 9]
        using var ln  = _headLn.forward(h);   // [N, 4]
        using var rw  = _headRw.forward(h);   // [N, 3]
        using var hl  = _headHl.forward(h);   // [N, 8]
        using var pt  = _headPt.forward(h);   // [N, 6]
        return cat(new[] { pl, ha, cd, ln, rw, hl, pt }, dim: 1);  // [N, 32]
    }
}

