using BeatSaber.AutoMapper.Audio;

namespace BeatSaber.AutoMapper.Training.SelfSupervised;

/// <summary>
/// Stateless utilities that score how well a single note position is supported
/// by audio evidence. Each component returns 0 (no support) … 1 (perfect support).
/// </summary>
internal static class NoteQualityScorer
{
    private const double OnsetHardWindowSeconds = 0.060; // ±60 ms  → onset score 1.0
    private const double OnsetSoftWindowSeconds = 0.120; // ±120 ms → onset score 0.5

    /// <summary>
    /// Composite score: how well this position is supported by audio.
    /// Weights: onset 35%, beat-align 25%, energy 25%, subdivision 15%.
    /// </summary>
    public static double ScorePosition(
        double noteTimeSeconds,
        double beatPosition,
        AudioAnalysisResult audio)
    {
        double beat    = BeatAlignmentScore(noteTimeSeconds, audio);
        double onset   = OnsetScore(noteTimeSeconds, audio);
        double energy  = EnergyScore(noteTimeSeconds, audio);
        double subdiv  = SubdivisionScore(beatPosition);

        return onset  * 0.35
             + beat   * 0.25
             + energy * 0.25
             + subdiv * 0.15;
    }

    // -----------------------------------------------------------------------

    private static double BeatAlignmentScore(double timeSeconds, AudioAnalysisResult audio)
    {
        if (audio.BeatTimesSeconds.Count == 0) return 0.5; // neutral — no beat data

        double beatLenSec = audio.EstimatedBpm > 0 ? 60.0 / audio.EstimatedBpm : 0.5;
        double minDist    = double.MaxValue;

        foreach (double bt in audio.BeatTimesSeconds)
        {
            double d = Math.Abs(bt - timeSeconds);
            if (d < minDist) minDist = d;
            if (d < 0.005) break; // close enough
        }

        // 0 dist → 1.0; dist >= half-beat → 0.0
        return Math.Max(0.0, 1.0 - minDist / (beatLenSec * 0.5));
    }

    private static double OnsetScore(double timeSeconds, AudioAnalysisResult audio)
    {
        if (audio.OnsetTimesSeconds.Count == 0) return 0.5;

        double minDist = double.MaxValue;
        foreach (double ot in audio.OnsetTimesSeconds)
        {
            double d = Math.Abs(ot - timeSeconds);
            if (d < minDist) minDist = d;
            if (d < 0.005) break;
        }

        if (minDist < OnsetHardWindowSeconds) return 1.0;
        if (minDist < OnsetSoftWindowSeconds) return 0.5;
        return 0.0;
    }

    private static double EnergyScore(double timeSeconds, AudioAnalysisResult audio) =>
        audio.EnergyEnvelope.Count > 0
            ? Math.Clamp(audio.GetEnergy(timeSeconds), 0.0, 1.0)
            : 0.5;

    /// <summary>
    /// Quarter beats score 1.0, 8th beats 0.7, 16th and others 0.3.
    /// </summary>
    private static double SubdivisionScore(double beatPosition)
    {
        double frac = beatPosition - Math.Floor(beatPosition);

        // Quarter-note positions (0, 0.25, 0.5, 0.75)
        double toQuarter = Math.Min(
            Math.Min(frac, Math.Abs(frac - 0.25)),
            Math.Min(Math.Abs(frac - 0.5), Math.Abs(frac - 0.75)));

        // 8th-note positions (0.125, 0.375, 0.625, 0.875)
        double toEighth = Math.Min(
            Math.Min(Math.Abs(frac - 0.125), Math.Abs(frac - 0.375)),
            Math.Min(Math.Abs(frac - 0.625), Math.Abs(frac - 0.875)));

        if (toQuarter < 0.05) return 1.0;
        if (toEighth  < 0.05) return 0.7;
        return 0.3;
    }
}
