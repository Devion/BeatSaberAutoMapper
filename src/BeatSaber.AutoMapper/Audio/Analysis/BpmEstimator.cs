namespace BeatSaber.AutoMapper.Audio.Analysis;

public sealed class BpmEstimator
{
    public double MinBpm { get; set; } = 60.0;
    public double MaxBpm { get; set; } = 220.0;

    /// <summary>
    /// Estimate BPM from PCM samples using onset-strength autocorrelation.
    /// Returns (estimatedBpm, confidence 0-1).
    /// </summary>
    public (double Bpm, double Confidence) Estimate(float[] samples, int sampleRate)
    {
        const int frameSize = 512;
        const int hopSize = 256;

        float[] onset = ComputeOnsetStrength(samples, sampleRate, frameSize, hopSize);
        float[] ac = Autocorrelate(onset);

        double hopDuration = hopSize / (double)sampleRate;
        // Convert BPM range to lag range in frames
        int minLag = (int)(60.0 / MaxBpm / hopDuration);
        int maxLag = (int)(60.0 / MinBpm / hopDuration);

        if (maxLag >= ac.Length) maxLag = ac.Length - 1;
        if (minLag < 1) minLag = 1;
        if (minLag >= maxLag) return (120.0, 0.0);

        int bestLag = minLag;
        float bestVal = ac[minLag];
        for (int lag = minLag + 1; lag <= maxLag; lag++)
        {
            if (ac[lag] > bestVal) { bestVal = ac[lag]; bestLag = lag; }
        }

        double bpm = 60.0 / (bestLag * hopDuration);
        // Clamp to valid range
        bpm = Math.Max(MinBpm, Math.Min(MaxBpm, bpm));

        // Confidence = peak strength relative to zero-lag
        double confidence = ac[0] > 0 ? Math.Min(1.0, bestVal / ac[0]) : 0.0;

        return (bpm, confidence);
    }

    private static float[] ComputeOnsetStrength(
        float[] samples, int sampleRate, int frameSize = 512, int hopSize = 256)
    {
        var detector = new OnsetDetector { FrameSize = frameSize, HopSize = hopSize };
        return detector.ComputeOnsetStrength(samples, sampleRate);
    }

    private static float[] Autocorrelate(float[] signal)
    {
        int N = signal.Length;
        var ac = new float[N];
        for (int lag = 0; lag < N; lag++)
        {
            double sum = 0;
            for (int i = 0; i < N - lag; i++)
                sum += signal[i] * signal[i + lag];
            ac[lag] = (float)sum;
        }
        return ac;
    }
}
