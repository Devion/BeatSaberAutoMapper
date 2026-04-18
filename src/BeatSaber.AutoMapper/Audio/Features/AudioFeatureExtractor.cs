using BeatSaber.AutoMapper.Audio.Analysis;

namespace BeatSaber.AutoMapper.Audio.Features;

public sealed class AudioFeatureExtractor
{
    private readonly BpmEstimator _bpmEstimator = new();
    private readonly BeatTracker _beatTracker = new();
    private readonly OnsetDetector _onsetDetector = new();
    private readonly SectionSegmenter _sectionSegmenter = new();
    private readonly SpectralAnalyzer _spectral = new() { FrameSize = 1024, HopSize = 512 };

    // Energy envelope uses the same hop size as the spectral analyzer so
    // all per-frame arrays share the same time base.
    private const int HopSize = 512;

    public AudioAnalysisResult Extract(string audioFilePath)
    {
        Guard.NotNullOrEmpty(audioFilePath, nameof(audioFilePath));

        var (samples, sampleRate, duration) = AudioDecoder.Decode(audioFilePath);

        // Phase 1: all steps that depend only on raw samples — run in parallel.
        double bpm = 0, confidence = 0;
        double[] onsetTimes = [];
        float[] energyRaw = [];
        SpectralAnalyzer.SpectralFrames spectral = null!;

        Parallel.Invoke(
            () => (bpm, confidence) = _bpmEstimator.Estimate(samples, sampleRate),
            () => onsetTimes         = _onsetDetector.DetectOnsets(samples, sampleRate),
            () => energyRaw          = ComputeEnergyEnvelope(samples, HopSize),
            () => spectral           = _spectral.Analyze(samples, sampleRate));

        // Phase 2: sequential — BeatTracker needs bpm, SectionSegmenter needs beatTimes.
        double[] beatTimes = _beatTracker.TrackBeats(samples, sampleRate, bpm);
        var sections       = _sectionSegmenter.Segment(samples, sampleRate, beatTimes, bpm);

        // Align spectral arrays to the energy-envelope length by resampling if needed
        double frameRate = sampleRate / (double)HopSize;
        int targetLen    = energyRaw.Length;

        return new AudioAnalysisResult
        {
            AudioFilePath    = audioFilePath,
            DurationSeconds  = duration,
            SampleRate       = sampleRate,
            Channels         = 1,
            EstimatedBpm     = bpm,
            BpmConfidence    = confidence,
            BeatTimesSeconds = beatTimes,
            OnsetTimesSeconds = onsetTimes,
            EnergyEnvelope   = energyRaw.Select(f => (double)f).ToArray(),
            Sections         = sections,
            FrameRateHz      = frameRate,
            LowBandEnergy    = ResampleToLength(spectral.LowBandEnergy,          targetLen),
            MidBandEnergy    = ResampleToLength(spectral.MidBandEnergy,          targetLen),
            HighBandEnergy   = ResampleToLength(spectral.HighBandEnergy,         targetLen),
            SpectralCentroid = ResampleToLength(spectral.SpectralCentroid,       targetLen),
            SpectralFlux     = ResampleToLength(spectral.SpectralFlux,           targetLen),
            TransientStrength = ResampleToLength(spectral.TransientStrength,     targetLen),
            OnsetStrengthEnvelope = ResampleToLength(spectral.OnsetStrengthEnvelope, targetLen),
        };
    }

    private static float[] ComputeEnergyEnvelope(float[] samples, int hopSize)
    {
        int numFrames = samples.Length / hopSize;
        var energy = new float[numFrames];
        for (int i = 0; i < numFrames; i++)
        {
            int start = i * hopSize;
            int end   = Math.Min(start + hopSize, samples.Length);
            double sum = 0;
            for (int j = start; j < end; j++) sum += samples[j] * samples[j];
            energy[i] = (float)Math.Sqrt(sum / (end - start));
        }
        return energy;
    }

    // Nearest-neighbour resample so all per-frame arrays share the same length.
    private static double[] ResampleToLength(float[] source, int targetLen)
    {
        if (source.Length == targetLen) return source.Select(f => (double)f).ToArray();
        var result = new double[targetLen];
        for (int i = 0; i < targetLen; i++)
        {
            int si = (int)Math.Round((double)i * (source.Length - 1) / Math.Max(1, targetLen - 1));
            result[i] = source[Math.Clamp(si, 0, source.Length - 1)];
        }
        return result;
    }
}
