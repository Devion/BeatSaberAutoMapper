using BeatSaber.AutoMapper.Utilities;

namespace BeatSaber.AutoMapper.Audio.Analysis;

public sealed class OnsetDetector
{
    public double Threshold { get; set; } = 0.3;
    public int FrameSize { get; set; } = 512;
    public int HopSize { get; set; } = 256;

    /// <summary>Detect onsets via spectral flux. Returns onset times in seconds.</summary>
    public double[] DetectOnsets(float[] samples, int sampleRate)
    {
        float[] strength = ComputeOnsetStrength(samples, sampleRate);
        double hopDuration = HopSize / (double)sampleRate;

        var onsets = new List<double>();
        for (int i = 1; i < strength.Length - 1; i++)
        {
            // Local maximum above threshold
            if (strength[i] > Threshold &&
                strength[i] > strength[i - 1] &&
                strength[i] >= strength[i + 1])
            {
                onsets.Add(i * hopDuration);
            }
        }
        return [.. onsets];
    }

    /// <summary>Returns the raw onset strength signal (one value per hop).</summary>
    public float[] ComputeOnsetStrength(float[] samples, int sampleRate)
    {
        int numFrames = (samples.Length - FrameSize) / HopSize + 1;
        if (numFrames <= 0) return [];

        var strength = new float[numFrames];
        float[]? prevMag = null;

        for (int i = 0; i < numFrames; i++)
        {
            int offset = i * HopSize;
            var frame = new float[FrameSize];
            int copyLen = Math.Min(FrameSize, samples.Length - offset);
            Array.Copy(samples, offset, frame, 0, copyLen);

            frame = ApplyHammingWindow(frame);
            ComputeMagnitudeSpectrum(frame, out float[] mag);

            if (prevMag is not null)
            {
                float flux = 0f;
                for (int j = 0; j < mag.Length; j++)
                {
                    float diff = mag[j] - prevMag[j];
                    if (diff > 0) flux += diff; // half-wave rectification
                }
                strength[i] = flux;
            }
            prevMag = mag;
        }

        // Normalise
        float max = strength.Max();
        if (max > 0)
            for (int i = 0; i < strength.Length; i++)
                strength[i] /= max;

        return strength;
    }

    private static float[] ApplyHammingWindow(float[] frame)
    {
        int N = frame.Length;
        var result = new float[N];
        for (int n = 0; n < N; n++)
            result[n] = frame[n] * (float)MathHelpers.HammingWindow(n, N);
        return result;
    }

    private static void ComputeMagnitudeSpectrum(float[] frame, out float[] magnitude)
    {
        int N = frame.Length;
        int half = N / 2 + 1;
        magnitude = new float[half];

        // Naive DFT – only needs to be fast enough for offline processing
        for (int k = 0; k < half; k++)
        {
            double re = 0, im = 0;
            for (int n = 0; n < N; n++)
            {
                double angle = -2.0 * Math.PI * k * n / N;
                re += frame[n] * Math.Cos(angle);
                im += frame[n] * Math.Sin(angle);
            }
            magnitude[k] = (float)Math.Sqrt(re * re + im * im);
        }
    }
}
