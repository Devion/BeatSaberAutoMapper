namespace BeatSaber.AutoMapper.Audio.Analysis;

/// <summary>
/// Computes per-frame spectral features using a Cooley-Tukey radix-2 FFT.
/// Frame size 1024, hop 512.  Works at 22050 Hz sample rate.
/// </summary>
public sealed class SpectralAnalyzer
{
    public int FrameSize { get; set; } = 1024;
    public int HopSize   { get; set; } = 512;

    // Frequency band boundaries (Hz) — tuned for beat-saber mapping relevance
    // Low  :   20 – 250 Hz  → kick, bass sub — typically downward cuts
    // Mid  :  250 – 2000 Hz → snare, vocals, melody
    // High : 2000+      Hz  → hi-hat, cymbals, bright attacks — typically upward cuts
    private const double LowMaxHz  =  250.0;
    private const double MidMaxHz  = 2000.0;

    public sealed class SpectralFrames
    {
        public float[] LowBandEnergy  { get; init; } = [];
        public float[] MidBandEnergy  { get; init; } = [];
        public float[] HighBandEnergy { get; init; } = [];
        /// <summary>Normalised spectral centroid 0-1 (0=bass, 1=treble).</summary>
        public float[] SpectralCentroid { get; init; } = [];
        public double FrameRateHz { get; init; }
    }

    public SpectralFrames Analyze(float[] samples, int sampleRate)
    {
        int numFrames = Math.Max(0, (samples.Length - FrameSize) / HopSize + 1);
        var low      = new float[numFrames];
        var mid      = new float[numFrames];
        var high     = new float[numFrames];
        var centroid = new float[numFrames];

        int half = FrameSize / 2 + 1;
        int binLow  = FreqToBin(LowMaxHz,  sampleRate);
        int binMid  = FreqToBin(MidMaxHz,  sampleRate);

        var re = new double[FrameSize];
        var im = new double[FrameSize];

        for (int f = 0; f < numFrames; f++)
        {
            int offset = f * HopSize;
            Array.Clear(re, 0, FrameSize);
            Array.Clear(im, 0, FrameSize);

            // Fill + Hann window
            for (int n = 0; n < FrameSize; n++)
            {
                int si = offset + n;
                double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * n / (FrameSize - 1)));
                re[n] = si < samples.Length ? samples[si] * w : 0.0;
            }

            Fft(re, im);

            // Compute bands from magnitude spectrum
            double sumLow = 0, sumMid = 0, sumHigh = 0;
            double sumMag = 0, sumWeighted = 0;

            for (int k = 1; k < half; k++)
            {
                double mag = Math.Sqrt(re[k] * re[k] + im[k] * im[k]);
                if (k <= binLow)         sumLow  += mag;
                else if (k <= binMid)    sumMid  += mag;
                else                     sumHigh += mag;

                sumMag      += mag;
                sumWeighted += k * mag;
            }

            low[f]      = (float)sumLow;
            mid[f]      = (float)sumMid;
            high[f]     = (float)sumHigh;
            centroid[f] = sumMag > 0 ? (float)(sumWeighted / (sumMag * half)) : 0f;
        }

        Normalize(low);
        Normalize(mid);
        Normalize(high);
        // centroid is already 0-1 by construction

        return new SpectralFrames
        {
            LowBandEnergy   = low,
            MidBandEnergy   = mid,
            HighBandEnergy  = high,
            SpectralCentroid = centroid,
            FrameRateHz     = sampleRate / (double)HopSize
        };
    }

    // -----------------------------------------------------------------------
    // Cooley-Tukey radix-2 in-place FFT (iterative, bit-reversal permutation)
    // -----------------------------------------------------------------------
    private static void Fft(double[] re, double[] im)
    {
        int n = re.Length;

        // Bit-reversal permutation
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        // Butterfly passes
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang  = -2.0 * Math.PI / len;
            double wRe  = Math.Cos(ang);
            double wIm  = Math.Sin(ang);
            for (int i = 0; i < n; i += len)
            {
                double curRe = 1.0, curIm = 0.0;
                int half2 = len >> 1;
                for (int j = 0; j < half2; j++)
                {
                    double uRe = re[i + j];
                    double uIm = im[i + j];
                    double vRe = re[i + j + half2] * curRe - im[i + j + half2] * curIm;
                    double vIm = re[i + j + half2] * curIm + im[i + j + half2] * curRe;
                    re[i + j]         = uRe + vRe;
                    im[i + j]         = uIm + vIm;
                    re[i + j + half2] = uRe - vRe;
                    im[i + j + half2] = uIm - vIm;
                    double nr = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = nr;
                }
            }
        }
    }

    private int FreqToBin(double hz, int sampleRate) =>
        (int)Math.Round(hz * FrameSize / sampleRate);

    private static void Normalize(float[] arr)
    {
        float max = 0f;
        foreach (float v in arr) if (v > max) max = v;
        if (max > 0f)
            for (int i = 0; i < arr.Length; i++) arr[i] /= max;
    }
}
