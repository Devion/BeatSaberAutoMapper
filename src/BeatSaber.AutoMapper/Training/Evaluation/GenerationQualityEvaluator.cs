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

        double placementF1    = ComputePlacementF1(generated, reference);
        double npsCorr        = ComputeNpsCorrelation(generated, reference);
        double parityRate     = ComputeParityBreakRate(generated);
        double valScore       = _validator.Validate(generated).Score / 100.0;
        double laneDiversity  = ComputeLaneDiversityScore(generated);
        double rowDiversity   = ComputeRowDiversityScore(generated);

        // Weighted combination — lane/row diversity now explicit (20%)
        double overall = 0.35 * placementF1
                       + 0.15 * Math.Max(0, npsCorr)
                       + 0.15 * (1.0 - parityRate)
                       + 0.15 * valScore
                       + 0.20 * (laneDiversity * 0.6 + rowDiversity * 0.4);

        return new QualityMetrics(placementF1, npsCorr, parityRate, valScore,
                                  Math.Clamp(overall, 0, 1));
    }

    // -----------------------------------------------------------------------

    private static double ComputePlacementF1(CanonicalBeatmap gen, CanonicalBeatmap refer)
    {
        const double window = 0.25; // beats
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

    /// <summary>Normalised entropy of lane usage (0 = all one lane, 1 = perfectly uniform).</summary>
    private static double ComputeLaneDiversityScore(CanonicalBeatmap beatmap)
    {
        if (beatmap.Notes.Count == 0) return 0;
        var counts = new int[4];
        foreach (var n in beatmap.Notes)
            if (n.Lane >= 0 && n.Lane < 4) counts[n.Lane]++;
        double total = beatmap.Notes.Count;
        double entropy = 0;
        foreach (int c in counts)
        {
            double p = c / total;
            if (p > 0) entropy -= p * Math.Log2(p);
        }
        return entropy / 2.0;  // max entropy for 4 uniform classes = log2(4) = 2
    }

    /// <summary>Normalised entropy of row usage (0 = all one row, 1 = perfectly uniform).</summary>
    private static double ComputeRowDiversityScore(CanonicalBeatmap beatmap)
    {
        if (beatmap.Notes.Count == 0) return 0;
        var counts = new int[3];
        foreach (var n in beatmap.Notes)
            if (n.Row >= 0 && n.Row < 3) counts[n.Row]++;
        double total = beatmap.Notes.Count;
        double entropy = 0;
        foreach (int c in counts)
        {
            double p = c / total;
            if (p > 0) entropy -= p * Math.Log2(p);
        }
        return entropy / Math.Log2(3);  // max entropy for 3 uniform classes = log2(3)
    }
}
