namespace BeatSaber.AutoMapper.Utilities;

public static class MathHelpers
{
    public static double BeatToSeconds(double beat, double bpm) => beat * (60.0 / bpm);

    public static double SecondsToBeat(double seconds, double bpm) => seconds / (60.0 / bpm);

    public static double Clamp(double val, double min, double max) =>
        val < min ? min : val > max ? max : val;

    public static double LinearInterpolate(double a, double b, double t) => a + (b - a) * t;

    public static int NearestBeat(double timeSeconds, IReadOnlyList<double> beatTimes)
    {
        if (beatTimes.Count == 0) return -1;
        int best = 0;
        double bestDist = Math.Abs(beatTimes[0] - timeSeconds);
        for (int i = 1; i < beatTimes.Count; i++)
        {
            double dist = Math.Abs(beatTimes[i] - timeSeconds);
            if (dist < bestDist) { bestDist = dist; best = i; }
        }
        return best;
    }

    /// <summary>Hamming window: 0.54 - 0.46 * cos(2π·n / (N-1))</summary>
    public static double HammingWindow(int n, int N) =>
        0.54 - 0.46 * Math.Cos(2.0 * Math.PI * n / (N - 1));
}
