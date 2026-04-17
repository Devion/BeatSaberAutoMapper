using BeatSaber.AutoMapper.Utilities;

namespace BeatSaber.AutoMapper.Audio.Analysis;

public sealed class SectionSegmenter
{
    private const double EnergyChangeThr = 0.25; // relative RMS change to call a boundary

    /// <summary>
    /// Segment audio into sections based on per-beat RMS energy.
    /// Returns an array of SectionMarkers.
    /// </summary>
    public SectionMarker[] Segment(
        float[] samples,
        int sampleRate,
        IReadOnlyList<double> beatTimes,
        double bpm)
    {
        if (beatTimes.Count < 2)
            return [new SectionMarker(0, 0, "Full Song", SectionType.Unknown, 0.5)];

        double[] rmsPerBeat = ComputeRmsPerBeat(samples, sampleRate, beatTimes);
        double maxRms = rmsPerBeat.Max();
        if (maxRms <= 0) maxRms = 1;
        double[] normRms = rmsPerBeat.Select(r => r / maxRms).ToArray();

        // Detect boundary beats (significant energy change)
        var boundaryIndices = new List<int> { 0 };
        for (int i = 1; i < normRms.Length - 1; i++)
        {
            double change = Math.Abs(normRms[i] - normRms[i - 1]);
            if (change > EnergyChangeThr)
                boundaryIndices.Add(i);
        }
        if (boundaryIndices[^1] != normRms.Length - 1)
            boundaryIndices.Add(normRms.Length - 1);

        // Label sections heuristically
        var markers = new List<SectionMarker>();
        for (int s = 0; s < boundaryIndices.Count - 1; s++)
        {
            int startIdx = boundaryIndices[s];
            int endIdx = boundaryIndices[s + 1];
            double avgEnergy = normRms[startIdx..endIdx].Average();
            double beatNum = startIdx; // beat index
            double timeSeconds = beatTimes[startIdx];

            SectionType type = LabelSection(s, boundaryIndices.Count - 1, avgEnergy, bpm, beatTimes, startIdx, endIdx);
            string label = $"Section {s + 1} ({type})";
            markers.Add(new SectionMarker(beatNum, timeSeconds, label, type, avgEnergy));
        }

        return [.. markers];
    }

    private static SectionType LabelSection(
        int sectionIndex, int totalSections, double avgEnergy,
        double bpm, IReadOnlyList<double> beatTimes, int startBeat, int endBeat)
    {
        if (sectionIndex == 0) return SectionType.Intro;
        if (sectionIndex == totalSections - 1) return SectionType.Outro;

        double sectionProgress = sectionIndex / (double)totalSections;

        if (avgEnergy > 0.7) return sectionProgress < 0.6 ? SectionType.Chorus : SectionType.Drop;
        if (avgEnergy > 0.4) return SectionType.Verse;
        if (sectionProgress > 0.4 && sectionProgress < 0.6) return SectionType.Bridge;
        if (avgEnergy < 0.25) return SectionType.Buildup;
        return SectionType.Unknown;
    }

    private static double[] ComputeRmsPerBeat(
        float[] samples, int sampleRate, IReadOnlyList<double> beatTimes)
    {
        var rms = new double[beatTimes.Count];
        for (int i = 0; i < beatTimes.Count; i++)
        {
            double start = beatTimes[i];
            double end = i + 1 < beatTimes.Count ? beatTimes[i + 1] : start + 0.5;
            int startSample = (int)(start * sampleRate);
            int endSample = Math.Min((int)(end * sampleRate), samples.Length);

            if (endSample <= startSample) continue;
            double sum = 0;
            for (int j = startSample; j < endSample; j++)
                sum += samples[j] * samples[j];
            rms[i] = Math.Sqrt(sum / (endSample - startSample));
        }
        return rms;
    }
}
