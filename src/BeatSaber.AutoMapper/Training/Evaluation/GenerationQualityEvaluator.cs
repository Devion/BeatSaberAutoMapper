using BeatSaber.AutoMapper.Validation;

namespace BeatSaber.AutoMapper.Training.Evaluation;

/// <summary>
/// Evaluates the quality of a generated map against a reference map.
/// Used during training to provide a per-epoch quality signal.
/// </summary>
public sealed class GenerationQualityEvaluator
{
    public sealed record QualityMetrics(
        double PlacementF1,        // note-beat alignment with reference
        double AttributeAgreement, // hand/lane/row/direction agreement on matched notes
        double DensitySimilarity,  // note count similarity to reference
        double NpsCorrelation,     // Pearson correlation of 4-beat NPS profiles
        double ParityBreakRate,    // parity breaks per note (lower is better)
        double ValidationScore,    // BeatmapValidator score 0-1
        double OverallScore        // weighted combination 0-1
    );

    private readonly BeatmapValidator _validator = new();

    public QualityMetrics Evaluate(CanonicalBeatmap generated, CanonicalBeatmap reference)
    {
        Guard.NotNull(generated,  nameof(generated));
        Guard.NotNull(reference,  nameof(reference));

        double placementF1       = ComputePlacementF1(generated, reference);
        double attributeAgreement = ComputeAttributeAgreement(generated, reference);
        double densitySimilarity = ComputeDensitySimilarity(generated, reference);
        double npsCorr           = ComputeNpsCorrelation(generated, reference);
        double parityRate        = ComputeParityBreakRate(generated);
        double valScore          = _validator.Validate(generated).Score / 100.0;

        // Reward direct agreement with the reference map much more than generic diversity.
        double overall = 0.38 * placementF1
                       + 0.24 * attributeAgreement
                       + 0.16 * densitySimilarity
                       + 0.10 * Math.Max(0, npsCorr)
                       + 0.07 * (1.0 - parityRate)
                       + 0.05 * valScore;

        return new QualityMetrics(placementF1, attributeAgreement, densitySimilarity, npsCorr, parityRate, valScore,
                                  Math.Clamp(overall, 0, 1));
    }

    // -----------------------------------------------------------------------

    private static double ComputePlacementF1(CanonicalBeatmap gen, CanonicalBeatmap refer)
    {
        const double window = 0.125; // beats
        var genBeats = gen.Notes.Select(n => n.Beat).OrderBy(b => b).ToList();
        var refBeats = refer.Notes.Select(n => n.Beat).OrderBy(b => b).ToList();
        if (refBeats.Count == 0 || genBeats.Count == 0) return 0;

        int tp = 0;
        var matched = new HashSet<int>();

        foreach (double rb in refBeats)
        {
            for (int i = 0; i < genBeats.Count; i++)
            {
                if (!matched.Contains(i) && Math.Abs(genBeats[i] - rb) <= window)
                {
                    tp++;
                    matched.Add(i);
                    break;
                }
            }
        }

        double precision = tp / (double)genBeats.Count;
        double recall    = tp / (double)refBeats.Count;
        return (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
    }

    private static double ComputeAttributeAgreement(CanonicalBeatmap gen, CanonicalBeatmap refer)
    {
        const double window = 0.125;
        var generated = gen.Notes.OrderBy(n => n.Beat).ToList();
        var reference = refer.Notes.OrderBy(n => n.Beat).ToList();
        if (generated.Count == 0 || reference.Count == 0)
            return 0;

        var matchedGenerated = new HashSet<int>();
        double totalScore = 0;
        int matched = 0;

        foreach (var r in reference)
        {
            int bestIdx = -1;
            double bestDist = double.MaxValue;
            for (int i = 0; i < generated.Count; i++)
            {
                if (matchedGenerated.Contains(i))
                    continue;

                double dist = Math.Abs(generated[i].Beat - r.Beat);
                if (dist <= window && dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0)
                continue;

            matchedGenerated.Add(bestIdx);
            matched++;
            var g = generated[bestIdx];

            double score = 0;
            if (g.Hand == r.Hand) score += 0.25;
            if (g.Lane == r.Lane) score += 0.30;
            if (g.Row == r.Row) score += 0.20;
            if (g.CutDirection == r.CutDirection) score += 0.25;
            totalScore += score;
        }

        return matched > 0 ? totalScore / matched : 0;
    }

    private static double ComputeDensitySimilarity(CanonicalBeatmap gen, CanonicalBeatmap refer)
    {
        int genCount = gen.Notes.Count;
        int refCount = refer.Notes.Count;
        if (genCount == 0 || refCount == 0)
            return 0;

        return Math.Min(genCount, refCount) / (double)Math.Max(genCount, refCount);
    }

    private static double ComputeNpsCorrelation(CanonicalBeatmap gen, CanonicalBeatmap refer)
    {
        double bpm      = gen.Song.BeatsPerMinute > 0 ? gen.Song.BeatsPerMinute : 120;
        double lastBeat = Math.Max(
            gen.Notes.Count   > 0 ? gen.Notes[^1].Beat   : 0,
            refer.Notes.Count > 0 ? refer.Notes[^1].Beat : 0);

        if (lastBeat <= 0) return 0;

        const double windowBeats = 4.0;
        double windowSec = MathHelpers.BeatToSeconds(windowBeats, bpm);
        if (windowSec <= 0) return 0;

        var genNps = new List<double>();
        var refNps = new List<double>();

        for (double b = 0; b < lastBeat; b += windowBeats)
        {
            double end = b + windowBeats;
            genNps.Add(gen.Notes.Count(n   => n.Beat >= b && n.Beat < end) / windowSec);
            refNps.Add(refer.Notes.Count(n => n.Beat >= b && n.Beat < end) / windowSec);
        }

        return PearsonCorrelation(genNps, refNps);
    }

    private static double ComputeParityBreakRate(CanonicalBeatmap beatmap)
    {
        if (beatmap.Notes.Count == 0) return 0;
        int breaks = 0;
        CanonicalNote? lastLeft = null, lastRight = null;

        foreach (var note in beatmap.Notes.OrderBy(n => n.Beat))
        {
            var last = note.Hand == NoteHand.Left ? lastLeft : lastRight;
            if (last is not null && note.Beat - last.Beat < 2.0
                && note.CutDirection == last.CutDirection
                && note.CutDirection != CutDirection.Dot)
                breaks++;

            if (note.Hand == NoteHand.Left) lastLeft  = note;
            else                            lastRight = note;
        }

        return breaks / (double)beatmap.Notes.Count;
    }

    private static double PearsonCorrelation(List<double> x, List<double> y)
    {
        if (x.Count < 2) return 0;
        double mx = x.Average(), my = y.Average();
        double num = x.Zip(y).Sum(p => (p.First - mx) * (p.Second - my));
        double dx  = Math.Sqrt(x.Sum(v => (v - mx) * (v - mx)));
        double dy  = Math.Sqrt(y.Sum(v => (v - my) * (v - my)));
        return dx * dy > 0 ? num / (dx * dy) : 0;
    }

}
