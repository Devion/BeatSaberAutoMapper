using BeatSaber.AutoMapper.Canonical.Derived;
using BeatSaber.AutoMapper.Training.Patterns;
using BeatSaber.AutoMapper.Validation;

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
        if (PlayabilityHeuristics.WouldCauseImmediatePatternIssue(
            note,
            ctx.PlacedNotes,
            ctx.Song.BeatsPerMinute,
            ctx.Profile.Difficulty))
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
            var probs = AdjustCutDirectionProbabilities(
                proposed.NeuralPrediction.Value.CutDirProbs,
                ctx,
                hand,
                proposed);
            var dir   = (CutDirection)SampleFromProbs(probs, ctx.Rng, temperature: 0.92);
            if (dir != CutDirection.Dot && IsParityValid(dir, ctx.GetContext(hand)))
                return dir;
            if (dir == CutDirection.Dot && probs[(int)CutDirection.Dot] >= 0.28)
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
            var laneProbs = AdjustLaneProbabilities(
                proposed.NeuralPrediction.Value,
                hand,
                proposed,
                ctx);
            var rowProbs = AdjustRowProbabilities(
                proposed.NeuralPrediction.Value.RowProbs,
                dir,
                proposed,
                ctx);

            int lane = SampleFromProbs(laneProbs, ctx.Rng, temperature: 1.08);
            int row  = SampleFromProbs(rowProbs,  ctx.Rng, temperature: 1.02);
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

    private static double[] AdjustCutDirectionProbabilities(
        double[] probs,
        GenerationContext ctx,
        NoteHand hand,
        ProposedEvent proposed)
    {
        var adjusted = (double[])probs.Clone();
        double localNps = EstimateLocalNps(proposed.Timing.Beat, ctx);

        // Dot notes are often over-produced by immature models; keep them available but expensive.
        adjusted[(int)CutDirection.Dot] *= 0.18;

        // Slightly prefer parity-respecting directional cadence over neutral notes.
        foreach (var dir in CutDirectionsByParity(ctx.GetContext(hand).CurrentParity))
            adjusted[(int)dir] *= 1.08;

        // Diagonals are useful seasoning, but should not dominate normal flow.
        adjusted[(int)CutDirection.UpLeft]    *= 0.82;
        adjusted[(int)CutDirection.UpRight]   *= 0.82;
        adjusted[(int)CutDirection.DownLeft]  *= 0.82;
        adjusted[(int)CutDirection.DownRight] *= 0.82;

        // In denser sections, prefer cleaner directional cadence and reduce dots further.
        if (localNps >= Math.Max(2.5, ctx.Profile.TargetNps * 0.75))
        {
            adjusted[(int)CutDirection.Dot] *= 0.55;
            adjusted[(int)CutDirection.UpLeft]    *= 0.90;
            adjusted[(int)CutDirection.UpRight]   *= 0.90;
            adjusted[(int)CutDirection.DownLeft]  *= 0.90;
            adjusted[(int)CutDirection.DownRight] *= 0.90;
        }

        double sum = adjusted.Sum();
        if (sum <= 0)
            return probs;

        for (int i = 0; i < adjusted.Length; i++)
            adjusted[i] /= sum;
        return adjusted;
    }

    private static double[] AdjustLaneProbabilities(
        NeuralMapPrediction prediction,
        NoteHand hand,
        ProposedEvent proposed,
        GenerationContext ctx)
    {
        var adjusted = BuildLaneDistribution(prediction, hand);
        double localNps = EstimateLocalNps(proposed.Timing.Beat, ctx);
        bool dense = localNps >= Math.Max(2.5, ctx.Profile.TargetNps * 0.75);

        // Wrong-side placements should be rare.
        if (hand == NoteHand.Left)
        {
            adjusted[0] *= 1.18;
            adjusted[1] *= 0.92; // center-left still possible, but less dominant
            adjusted[2] *= 0.36;
            adjusted[3] *= 0.12;
        }
        else
        {
            adjusted[0] *= 0.12;
            adjusted[1] *= 0.36;
            adjusted[2] *= 0.92; // center-right still possible, but less dominant
            adjusted[3] *= 1.18;
        }

        // Center lanes block vision more often; discourage them globally.
        adjusted[1] *= dense ? 0.68 : 0.80;
        adjusted[2] *= dense ? 0.68 : 0.80;
        adjusted[0] *= 1.06;
        adjusted[3] *= 1.06;

        // If hands are already crossing, strongly pull back toward sane sides.
        var lastLeft = ctx.PlacedNotes.LastOrDefault(n => n.Hand == NoteHand.Left);
        var lastRight = ctx.PlacedNotes.LastOrDefault(n => n.Hand == NoteHand.Right);
        bool crossed = lastLeft is not null && lastRight is not null && lastLeft.Lane > lastRight.Lane;
        if (crossed)
        {
            if (hand == NoteHand.Left)
            {
                adjusted[0] *= 1.12;
                adjusted[3] *= 0.40;
            }
            else
            {
                adjusted[3] *= 1.12;
                adjusted[0] *= 0.40;
            }
        }

        PatternType patternType = PredictPatternType(prediction);
        var lastSameHand = ctx.PlacedNotes.LastOrDefault(n => n.Hand == hand);
        if (lastSameHand is not null)
        {
            int prevLane = lastSameHand.Lane;
            for (int lane = 0; lane < adjusted.Length; lane++)
            {
                int laneDistance = Math.Abs(lane - prevLane);
                if (!ctx.Settings.AllowFieldMovement)
                {
                    if (laneDistance >= 2) adjusted[lane] *= 0.20;
                    else if (laneDistance == 1) adjusted[lane] *= 0.72;
                    else adjusted[lane] *= 1.18;
                }

                if (patternType == PatternType.Anchor)
                {
                    adjusted[lane] *= lane == prevLane ? 1.35 : 0.65;
                }
                else if (patternType == PatternType.Stream)
                {
                    adjusted[lane] *= laneDistance <= 1 ? 1.10 : 0.75;
                }
                else if (patternType == PatternType.Reset)
                {
                    adjusted[lane] *= laneDistance <= 1 ? 1.06 : 0.85;
                }
            }
        }

        Normalize(adjusted, prediction.LaneProbs);
        return adjusted;
    }

    private static double[] BuildLaneDistribution(NeuralMapPrediction prediction, NoteHand hand)
    {
        var adjusted = new double[4];
        bool hasJoint = prediction.HandLaneProbs is { Length: 8 };
        int offset = hand == NoteHand.Left ? 0 : 4;

        for (int lane = 0; lane < 4; lane++)
        {
            double marginal = prediction.LaneProbs.Length > lane
                ? prediction.LaneProbs[lane]
                : 0.25;
            double joint = hasJoint ? prediction.HandLaneProbs[offset + lane] : marginal;
            adjusted[lane] = 0.35 * marginal + 0.65 * joint;
        }

        return adjusted;
    }

    private static double[] AdjustRowProbabilities(
        double[] probs,
        CutDirection dir,
        ProposedEvent proposed,
        GenerationContext ctx)
    {
        var adjusted = (double[])probs.Clone();
        double localNps = EstimateLocalNps(proposed.Timing.Beat, ctx);
        bool dense = localNps >= Math.Max(2.5, ctx.Profile.TargetNps * 0.75);

        // Top row gets uncomfortable quickly; discourage it, especially in dense sections.
        adjusted[2] *= dense
            ? (dir == CutDirection.Dot ? 0.68 : 0.38)
            : (dir == CutDirection.Dot ? 0.90 : 0.62);

        // Prefer middle row as the ergonomic default.
        adjusted[1] *= 1.14;

        Normalize(adjusted, probs);
        return adjusted;
    }

    private static double EstimateLocalNps(double beat, GenerationContext ctx)
    {
        double half = 2.0;
        int count = ctx.PlacedNotes.Count(n => n.Beat >= beat - half && n.Beat <= beat + half);
        double windowSec = MathHelpers.BeatToSeconds(half * 2.0, ctx.Song.BeatsPerMinute);
        return windowSec > 0 ? count / windowSec : 0;
    }

    private static void Normalize(double[] adjusted, double[] fallback)
    {
        double sum = adjusted.Sum();
        if (sum <= 0)
        {
            Array.Copy(fallback, adjusted, fallback.Length);
            sum = adjusted.Sum();
            if (sum <= 0) return;
        }

        for (int i = 0; i < adjusted.Length; i++)
            adjusted[i] /= sum;
    }
    private static bool IsParityValid(CutDirection dir, SwingContext handCtx) =>
        ParityAnalyzer.CutDirectionParity(dir) == handCtx.CurrentParity;

    private static CutDirection[] CutDirectionsByParity(ParityClass parity) =>
        parity == ParityClass.Forehand
            ? [CutDirection.Down, CutDirection.DownLeft, CutDirection.DownRight]
            : [CutDirection.Up,   CutDirection.UpLeft,   CutDirection.UpRight, CutDirection.Left, CutDirection.Right];

    private static PatternType PredictPatternType(NeuralMapPrediction prediction)
    {
        if (prediction.PatternTypeProbs is not { Length: >= 6 })
            return PatternType.Isolated;
        int best = 0;
        for (int i = 1; i < prediction.PatternTypeProbs.Length; i++)
        {
            if (prediction.PatternTypeProbs[i] > prediction.PatternTypeProbs[best])
                best = i;
        }
        return (PatternType)best;
    }
}
