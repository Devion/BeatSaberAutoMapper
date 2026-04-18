namespace BeatSaber.AutoMapper.Audio.Features;

public sealed record TimingCandidate(
    double Beat,
    double TimeSeconds,
    double SubdivisionDenominator,
    double OnsetStrength,
    double EnergyLevel
);

public sealed class TimingGridBuilder
{
    public static readonly double[] StandardSubdivisions =
        [1.0, 0.5, 0.25];

    /// <summary>
    /// Build a candidate timing grid by aligning subdivisions to beat times
    /// and annotating each candidate with onset strength and energy.
    /// </summary>
    public TimingCandidate[] Build(
        IReadOnlyList<double> beatTimesSeconds,
        IReadOnlyList<double> onsetTimesSeconds,
        double bpm,
        double songDurationSeconds,
        IReadOnlyList<double> subdivisions,
        IReadOnlyList<double>? energyEnvelope = null,
        double frameRateHz = 43.0)
    {
        Guard.NotNull(beatTimesSeconds, nameof(beatTimesSeconds));
        Guard.NotNull(subdivisions, nameof(subdivisions));

        double beatPeriod = 60.0 / bpm;
        var candidates = new List<TimingCandidate>();

        for (int b = 0; b < beatTimesSeconds.Count - 1; b++)
        {
            double beatStart = beatTimesSeconds[b];
            double beatEnd = beatTimesSeconds[b + 1];
            double actualBeatDuration = beatEnd - beatStart;

            foreach (double sub in subdivisions)
            {
                int steps = (int)Math.Round(1.0 / sub);
                if (steps < 1) steps = 1;

                for (int s = 0; s < steps; s++)
                {
                    double fraction = s * sub;
                    double t = beatStart + fraction * actualBeatDuration;
                    double beat = b + fraction;

                    double onsetStrength = NearestOnsetStrength(t, onsetTimesSeconds, beatPeriod * sub);
                    double energyLevel   = SampleEnergy(t, energyEnvelope, frameRateHz);
                    candidates.Add(new TimingCandidate(beat, t, sub, onsetStrength, energyLevel));
                }
            }
        }

        // Remove duplicates (same beat rounded to 4dp)
        var unique = candidates
            .GroupBy(c => Math.Round(c.Beat, 4))
            .Select(g => g.OrderByDescending(c => c.SubdivisionDenominator).First())
            .OrderBy(c => c.Beat)
            .ToArray();

        return unique;
    }

    private static double NearestOnsetStrength(
        double timeSeconds,
        IReadOnlyList<double> onsets,
        double window)
    {
        double best = 0.0;
        foreach (double o in onsets)
        {
            if (Math.Abs(o - timeSeconds) <= window)
                best = 1.0;
        }
        return best;
    }

    private static double SampleEnergy(
        double timeSeconds,
        IReadOnlyList<double>? envelope,
        double frameRateHz)
    {
        if (envelope is null || envelope.Count == 0 || frameRateHz <= 0) return 0.0;
        int frame = Math.Max(0, Math.Min((int)(timeSeconds * frameRateHz), envelope.Count - 1));
        return envelope[frame];
    }
}
