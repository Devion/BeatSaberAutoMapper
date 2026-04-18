using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace BeatSaber.AutoMapper.Training.Models;

/// <summary>
/// Multi-task neural network for Beat Saber note generation.
/// Architecture: 27 → 1024 → 512 → 256 → 128 trunk (BatchNorm + Dropout) → 5 heads.
///
/// Forward output is a [B, 18] tensor:
///   [B, 0]      placement logit  (sigmoid → P(note here))
///   [B, 1]      hand logit       (sigmoid → P(right hand))
///   [B, 2..10]  cut-dir logits   (softmax → 9 classes)
///   [B, 11..14] lane logits      (softmax → 4 lanes)
///   [B, 15..17] row logits       (softmax → 3 rows)
///
/// Feature layout (27 inputs):
///   [0-22]  original 23 features (rhythm + spectral + sequential note state)
///   [23]    EnergyDelta      — normalised rising/falling energy
///   [24]    HighBandDelta    — hi-hat/cymbal delta (onset sharpness)
///   [25]    TimeSinceAnyNote — min(L,R) beats since last note / 8
///   [26]    SongFraction     — beat / totalBeats (song position context)
///
/// ~725K parameters; float32 .pt file ~2.8 MB.
/// NOTE: architecture change from 23-input v1 — existing v1 .pt files are NOT compatible.
/// </summary>
internal sealed class BeatSaberMappingNet : Module<Tensor, Tensor>
{
    internal const int InputDim = 27;
    internal const int OutDim   = 18;  // 1+1+9+4+3

    private const long H1 = 1024, H2 = 512, H3 = 256, H4 = 128;

    private readonly Sequential _trunk;
    private readonly Linear _headPl, _headHa, _headCd, _headLn, _headRw;

    internal BeatSaberMappingNet() : base(nameof(BeatSaberMappingNet))
    {
        // hasBias=false because BatchNorm absorbs the bias
        _trunk = Sequential(
            ("lin1",  Linear(InputDim, H1, hasBias: false)),
            ("bn1",   BatchNorm1d(H1)),
            ("relu1", ReLU()),
            ("drop1", Dropout(0.20)),
            ("lin2",  Linear(H1, H2, hasBias: false)),
            ("bn2",   BatchNorm1d(H2)),
            ("relu2", ReLU()),
            ("drop2", Dropout(0.20)),
            ("lin3",  Linear(H2, H3, hasBias: false)),
            ("bn3",   BatchNorm1d(H3)),
            ("relu3", ReLU()),
            ("drop3", Dropout(0.20)),
            ("lin4",  Linear(H3, H4, hasBias: false)),
            ("bn4",   BatchNorm1d(H4)),
            ("relu4", ReLU())
        );
        _headPl = Linear(H4, 1);
        _headHa = Linear(H4, 1);
        _headCd = Linear(H4, 9);
        _headLn = Linear(H4, 4);
        _headRw = Linear(H4, 3);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var h  = _trunk.forward(x);
        using var pl = _headPl.forward(h);   // [B, 1]
        using var ha = _headHa.forward(h);   // [B, 1]
        using var cd = _headCd.forward(h);   // [B, 9]
        using var ln = _headLn.forward(h);   // [B, 4]
        using var rw = _headRw.forward(h);   // [B, 3]
        return cat(new[] { pl, ha, cd, ln, rw }, dim: 1);  // [B, 18]
    }
}
