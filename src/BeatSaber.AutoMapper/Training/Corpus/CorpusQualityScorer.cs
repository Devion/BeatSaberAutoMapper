using BeatSaber.AutoMapper.Canonical.Derived;
using BeatSaber.AutoMapper.Utilities;
using BeatSaber.AutoMapper.Validation;

namespace BeatSaber.AutoMapper.Training.Corpus;

public sealed class CorpusQualityScorer
{
    private readonly BeatmapValidator _validator = new();

    public sealed record QualityScore(
        double Score,
        double ValidationScore,
        double ParityScore,
        double TimingScore,
        bool IsGimmick,
        bool IsMalformed,
        IReadOnlyList<string> Flags
    );

    public QualityScore Score(CanonicalBeatmap beatmap)
    {
        Guard.NotNull(beatmap, nameof(beatmap));

        var flags = new List<string>();
        bool isMalformed = beatmap.Notes.Count == 0;

        if (isMalformed)
            return new QualityScore(0, 0, 0, 0, false, true, ["No notes"]);

        var report = _validator.Validate(beatmap);
        double validationScore = report.Score;
        double parityScore = ComputeParityScore(beatmap);
        double timingScore = ComputeTimingScore(beatmap);
        bool isGimmick = DetectGimmick(beatmap);

        if (isGimmick) flags.Add("Gimmick map");
        if (parityScore < 0.5) flags.Add("Poor parity");
        if (timingScore < 0.5) flags.Add("Irregular timing");

        double overall = (validationScore * 0.4 + parityScore * 100.0 * 0.4 + timingScore * 100.0 * 0.2);
        overall = MathHelpers.Clamp(overall, 0, 100);

        return new QualityScore(overall, validationScore, parityScore, timingScore, isGimmick, isMalformed, flags);
    }

    private double ComputeParityScore(CanonicalBeatmap beatmap)
    {
        int total = 0, good = 0;
        foreach (NoteHand hand in new[] { NoteHand.Left, NoteHand.Right })
        {
            var handNotes = beatmap.Notes.Where(n => n.Hand == hand).OrderBy(n => n.Beat).ToList();
            var ctx = new SwingContext(hand);
            CanonicalNote? prev = null;
            foreach (var note in handNotes)
            {
                if (prev is not null)
                {
                    total++;
                    var result = ParityAnalyzer.ClassifyTransition(prev, note, ctx);
                    if (result.Transition is ParityTransition.GoodFlow or ParityTransition.Acceptable)
                        good++;
                }
                ctx.Update(note);
                prev = note;
            }
        }
        return total == 0 ? 1.0 : good / (double)total;
    }

    private static double ComputeTimingScore(CanonicalBeatmap beatmap)
    {
        if (beatmap.Notes.Count < 2) return 1.0;
        var gaps = beatmap.Notes
            .OrderBy(n => n.Beat)
            .Zip(beatmap.Notes.OrderBy(n => n.Beat).Skip(1), (a, b) => b.Beat - a.Beat)
            .Where(g => g > 0)
            .ToList();
        if (gaps.Count == 0) return 1.0;

        // Timing regularity: low coefficient of variation = high score
        double mean = gaps.Average();
        double variance = gaps.Select(g => (g - mean) * (g - mean)).Average();
        double cv = mean > 0 ? Math.Sqrt(variance) / mean : 1.0;
        return MathHelpers.Clamp(1.0 - cv / 3.0, 0, 1);
    }

    private static bool DetectGimmick(CanonicalBeatmap beatmap)
    {
        // Heuristic: >90% of notes in same lane/row = likely gimmick
        int total = beatmap.Notes.Count;
        if (total == 0) return false;
        int maxLane = beatmap.Notes.GroupBy(n => n.Lane).Max(g => g.Count());
        return maxLane / (double)total > 0.9;
    }
}
