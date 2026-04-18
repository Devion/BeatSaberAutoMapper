using BeatSaber.AutoMapper.Canonical.Derived;

namespace BeatSaber.AutoMapper.Generation;

public sealed class NoteAttributeSelector
{
    public CanonicalNote? SelectAttributes(ProposedEvent proposed, GenerationContext ctx)
    {
        Guard.NotNull(proposed, nameof(proposed));
        Guard.NotNull(ctx, nameof(ctx));

        var hand  = proposed.SuggestedHand;
        var color = hand == NoteHand.Left ? NoteColor.Red : NoteColor.Blue;

        var dir          = SelectCutDirection(hand, proposed, ctx);
        var (lane, row)  = SelectPosition(hand, dir, proposed, ctx);

        var note = new CanonicalNote(proposed.Timing.Beat, lane, row, color, dir);

        if (ctx.PlacedNotes.Any(n => n.ConflictsWith(note)))
            return null;

        return note;
    }

    // -----------------------------------------------------------------------

    private CutDirection SelectCutDirection(
        NoteHand hand, ProposedEvent proposed, GenerationContext ctx)
    {
        // Neural model: sample from learned cut-direction distribution
        if (proposed.NeuralPrediction.HasValue)
        {
            var probs = proposed.NeuralPrediction.Value.CutDirProbs;
            var dir   = (CutDirection)SampleFromProbs(probs, ctx.Rng);
            if (dir == CutDirection.Dot || IsParityValid(dir, ctx.GetContext(hand)))
                return dir;
            // Find best parity-valid direction from neural distribution
            var valid = CutDirectionsByParity(ctx.GetContext(hand).CurrentParity);
            return valid.OrderByDescending(d => probs[(int)d]).First();
        }

        // Parity-based fallback (pre-training or no model loaded)
        var candidates = CutDirectionsByParity(ctx.GetContext(hand).CurrentParity);
        return candidates[ctx.Rng.Next(candidates.Length)];
    }

    private (int Lane, int Row) SelectPosition(
        NoteHand hand, CutDirection dir,
        ProposedEvent proposed, GenerationContext ctx)
    {
        // Neural model: sample lane/row from learned distributions with temperature scaling.
        if (proposed.NeuralPrediction.HasValue)
        {
            int lane = SampleFromProbs(proposed.NeuralPrediction.Value.LaneProbs, ctx.Rng, temperature: 1.4);
            int row  = SampleFromProbs(proposed.NeuralPrediction.Value.RowProbs,  ctx.Rng, temperature: 1.3);
            return (lane, row);
        }

        return HeuristicPosition(hand, dir, proposed.Timing, ctx);
    }

    private (int Lane, int Row) HeuristicPosition(NoteHand hand, CutDirection direction, Audio.Features.TimingCandidate timing, GenerationContext ctx)
    {
        bool isLeft = hand == NoteHand.Left;

        int lane = isLeft ? ctx.Rng.Next(0, 2) : ctx.Rng.Next(2, 4);

        if (ctx.Profile.AllowCrossovers && ctx.Rng.NextDouble() < 0.1)
            lane = isLeft ? ctx.Rng.Next(2, 4) : ctx.Rng.Next(0, 2);

        int row = direction switch
        {
            CutDirection.Down or CutDirection.DownLeft or CutDirection.DownRight => 2,
            CutDirection.Up  or CutDirection.UpLeft   or CutDirection.UpRight    => 0,
            _                                                                     => 1
        };

        return (lane, row);
    }

    // -----------------------------------------------------------------------


    private static int SampleFromProbs(double[] probs, Random rng, double temperature = 1.0)
    {
        if (temperature != 1.0)
        {
            // Apply temperature: p_i ∝ p_i^(1/T)  — T>1 flattens, T<1 sharpens
            var scaled = new double[probs.Length];
            double sum = 0;
            for (int i = 0; i < probs.Length; i++)
            {
                scaled[i] = Math.Pow(Math.Max(probs[i], 1e-10), 1.0 / temperature);
                sum += scaled[i];
            }
            for (int i = 0; i < scaled.Length; i++) scaled[i] /= sum;
            probs = scaled;
        }

        double r = rng.NextDouble(), cum = 0;
        for (int i = 0; i < probs.Length; i++) { cum += probs[i]; if (r < cum) return i; }
        return probs.Length - 1;
    }
    private static bool IsParityValid(CutDirection dir, SwingContext handCtx) =>
        ParityAnalyzer.CutDirectionParity(dir) == handCtx.CurrentParity;

    private static CutDirection[] CutDirectionsByParity(ParityClass parity) =>
        parity == ParityClass.Forehand
            ? [CutDirection.Down, CutDirection.DownLeft, CutDirection.DownRight]
            : [CutDirection.Up,   CutDirection.UpLeft,   CutDirection.UpRight, CutDirection.Left, CutDirection.Right];
}
