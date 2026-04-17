using BeatSaber.AutoMapper.Audio;

namespace BeatSaber.AutoMapper.Diagnostics;

public static class BeatGridCsvWriter
{
    /// <summary>
    /// Writes beat grid and onset data to a CSV file.
    /// Columns: beat_index, time_seconds, is_downbeat, energy, is_onset
    /// </summary>
    public static void Write(AudioAnalysisResult analysis, string outputPath)
    {
        Guard.NotNull(analysis, nameof(analysis));
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));

        var onsetSet = new HashSet<double>(analysis.OnsetTimesSeconds.Select(o => Math.Round(o, 3)));

        using var sw = new StreamWriter(outputPath, false, Encoding.UTF8);
        sw.WriteLine("beat_index,time_seconds,is_downbeat,energy,is_onset");

        for (int i = 0; i < analysis.BeatTimesSeconds.Count; i++)
        {
            double t = analysis.BeatTimesSeconds[i];
            bool isDownbeat = i % 4 == 0;
            double energy = GetEnergy(t, analysis);
            bool isOnset = onsetSet.Contains(Math.Round(t, 3));

            sw.WriteLine($"{i},{t:F4},{(isDownbeat ? 1 : 0)},{energy:F4},{(isOnset ? 1 : 0)}");
        }
    }

    private static double GetEnergy(double timeSeconds, AudioAnalysisResult analysis)
    {
        if (analysis.EnergyEnvelope.Count == 0 || analysis.FrameRateHz <= 0) return 0;
        int idx = (int)(timeSeconds * analysis.FrameRateHz);
        idx = Math.Max(0, Math.Min(idx, analysis.EnergyEnvelope.Count - 1));
        return analysis.EnergyEnvelope[idx];
    }
}
