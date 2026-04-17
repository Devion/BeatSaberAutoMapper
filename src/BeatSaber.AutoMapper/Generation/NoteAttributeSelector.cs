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
        var handCtx = ctx.GetContext(hand);
        int prevDir = handCtx.LastCutDirection.HasValue ? (int)handCtx.LastCutDirection.Value : -1;
        int handIdx = hand == NoteHand.Left ? 0 : 1;
        double beatStrength = ComputeBeatStrength(proposed.Timing.Beat);

        // Neural model: sample from learned cut-direction distribution
        if (proposed.NeuralPrediction.HasValue)
        {
            var probs = proposed.NeuralPrediction.Value.CutDirProbs;
            var dir   = (CutDirection)SampleFromProbs(probs, ctx.Rng);
            if (dir == CutDirection.Dot || IsParityValid(dir, handCtx))
                return dir;
            // Fallback within neural: find best parity-valid direction
            var valid = CutDirectionsByParity(handCtx.CurrentParity);
            return valid.OrderByDescending(d => probs[(int)d]).First();
        }

        if (ctx.AttributeModel is not null)
        {
            int difficultyLevel = (int)ctx.Profile.Difficulty;
            var dir = ctx.AttributeModel.SampleCutDirection(prevDir, beatStrength, handIdx,
                                                            difficultyLevel, ctx.Rng);
            // Only accept if parity is valid (or it's a dot — always valid)
            if (dir == CutDirection.Dot || IsParityValid(dir, handCtx))
                return dir;
        }

        // Parity-based fallback
        var candidates = CutDirectionsByParity(handCtx.CurrentParity);
        return candidates[ctx.Rng.Next(candidates.Length)];
    }

    private (int Lane, int Row) SelectPosition(
        NoteHand hand, CutDirection dir,
        ProposedEvent proposed, GenerationContext ctx)
    {
        // Neural model: sample lane/row from learned distributions
        if (proposed.NeuralPrediction.HasValue)
        {
            int lane = SampleFromProbs(proposed.NeuralPrediction.Value.LaneProbs, ctx.Rng);
            int row  = SampleFromProbs(proposed.NeuralPrediction.Value.RowProbs,  ctx.Rng);
            return (lane, row);
        }

        if (ctx.AttributeModel is not null)
        {
            double beatStrength = ComputeBeatStrength(proposed.Timing.Beat);
            double centroid     = ctx.AudioAnalysis.GetCentroid(proposed.Timing.TimeSeconds);
            int    handIdx      = hand == NoteHand.Left ? 0 : 1;
            int    diffLevel    = (int)ctx.Profile.Difficulty;
            return ctx.AttributeModel.SamplePosition((int)dir, handIdx, beatStrength, centroid, diffLevel, ctx.Rng);
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


    private static int SampleFromProbs(double[] probs, Random rng)
    {
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

    private static double ComputeBeatStrength(double beat)
    {
        double phase    = beat - Math.Floor(beat);
        int measure     = ((int)Math.Floor(beat) % 4) + 1;
        double strength = measure switch { 1 => 1.00, 3 => 0.75, _ => 0.50 };
        if (phase > 0.01) strength *= 0.5;
        return strength;
    }
}
