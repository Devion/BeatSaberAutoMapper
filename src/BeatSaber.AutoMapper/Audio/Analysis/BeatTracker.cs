namespace BeatSaber.AutoMapper.Audio.Analysis;

public sealed class BeatTracker
{
    /// <summary>
    /// Track beats given a BPM estimate.
    /// Creates a regular grid aligned to the strongest onset near each expected beat.
    /// Returns beat times in seconds.
    /// </summary>
    public double[] TrackBeats(float[] samples, int sampleRate, double bpm)
    {
        double beatPeriod = 60.0 / bpm;
        double songDuration = samples.Length / (double)sampleRate;

        var detector = new OnsetDetector { FrameSize = 512, HopSize = 256 };
        double[] onsetTimes = detector.DetectOnsets(samples, sampleRate);

        double snapWindow = beatPeriod * 0.2; // ±20 % of a beat
        var beats = new List<double>();
        double t = 0.0;

        while (t <= songDuration)
        {
            // Snap to nearest onset within window
            double best = t;
            double bestDist = double.MaxValue;
            foreach (double onset in onsetTimes)
            {
                double dist = Math.Abs(onset - t);
                if (dist < snapWindow && dist < bestDist)
                {
                    bestDist = dist;
                    best = onset;
                }
            }
            beats.Add(best);
            t += beatPeriod;
        }

        return [.. beats];
    }
}
