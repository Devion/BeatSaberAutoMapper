using BeatSaber.AutoMapper.Audio.Analysis;
using BeatSaber.AutoMapper.Utilities;
using FluentAssertions;

namespace BeatSaber.AutoMapper.Tests;

public sealed class TimingConversionTests
{
    [Theory]
    [InlineData(1.0, 120.0, 0.5)]
    [InlineData(4.0, 120.0, 2.0)]
    [InlineData(1.0, 60.0, 1.0)]
    [InlineData(1.0, 240.0, 0.25)]
    public void BeatToSeconds_MatchesExpected(double beat, double bpm, double expectedSecs)
    {
        MathHelpers.BeatToSeconds(beat, bpm).Should().BeApproximately(expectedSecs, 0.001);
    }

    [Theory]
    [InlineData(0.5, 120.0, 1.0)]
    [InlineData(2.0, 120.0, 4.0)]
    [InlineData(1.0, 60.0, 1.0)]
    public void SecondsToBeat_MatchesExpected(double seconds, double bpm, double expectedBeat)
    {
        MathHelpers.SecondsToBeat(seconds, bpm).Should().BeApproximately(expectedBeat, 0.001);
    }

    [Theory]
    [InlineData(120.0)]
    [InlineData(180.0)]
    [InlineData(90.0)]
    public void BeatToSeconds_SecondsToBeat_RoundTrip(double bpm)
    {
        for (double beat = 0; beat <= 16; beat += 0.25)
        {
            double secs = MathHelpers.BeatToSeconds(beat, bpm);
            double roundTrip = MathHelpers.SecondsToBeat(secs, bpm);
            roundTrip.Should().BeApproximately(beat, 0.001,
                because: $"round-trip should hold at {bpm} BPM for beat {beat}");
        }
    }

    [Fact]
    public void BpmEstimator_PlausibleResult_ForSineTone()
    {
        // Generate a 120 BPM click track as a sine tone burst
        int sampleRate = 22050;
        float[] samples = GenerateClickTrack(120.0, sampleRate, durationSeconds: 10);

        var estimator = new BpmEstimator { MinBpm = 60, MaxBpm = 200 };
        var (bpm, confidence) = estimator.Estimate(samples, sampleRate);

        bpm.Should().BeInRange(60, 200, "BPM estimator should return a value in the valid range");
        confidence.Should().BeInRange(0, 1, "confidence should be normalised 0-1");
    }

    private static float[] GenerateClickTrack(double bpm, int sampleRate, double durationSeconds)
    {
        int totalSamples = (int)(durationSeconds * sampleRate);
        float[] samples = new float[totalSamples];
        double beatPeriod = 60.0 / bpm;
        int beatPeriodSamples = (int)(beatPeriod * sampleRate);

        for (int beat = 0; beat * beatPeriodSamples < totalSamples; beat++)
        {
            int onset = beat * beatPeriodSamples;
            for (int i = 0; i < 200 && onset + i < totalSamples; i++)
            {
                double t = i / (double)sampleRate;
                samples[onset + i] += (float)(Math.Sin(2 * Math.PI * 440 * t) * Math.Exp(-i * 0.01));
            }
        }
        return samples;
    }
}
