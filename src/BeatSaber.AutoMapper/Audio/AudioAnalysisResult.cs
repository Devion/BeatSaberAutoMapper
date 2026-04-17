namespace BeatSaber.AutoMapper.Audio;

public sealed class AudioAnalysisResult
{
    public string AudioFilePath { get; init; } = "";
    public double DurationSeconds { get; init; }
    public int SampleRate { get; init; }
    public int Channels { get; init; }
    public double EstimatedBpm { get; init; }
    public double BpmConfidence { get; init; }
    public IReadOnlyList<double> BeatTimesSeconds { get; init; } = [];
    public IReadOnlyList<double> OnsetTimesSeconds { get; init; } = [];
    /// <summary>RMS energy per frame (one value per hop).</summary>
    public IReadOnlyList<double> EnergyEnvelope { get; init; } = [];
    public IReadOnlyList<SectionMarker> Sections { get; init; } = [];
    /// <summary>Frames per second matching all per-frame arrays.</summary>
    public double FrameRateHz { get; init; }

    // --- Spectral features (per frame, aligned to EnergyEnvelope) -----------
    /// <summary>Normalised low-band energy (20–250 Hz). High = kick/bass.</summary>
    public IReadOnlyList<double> LowBandEnergy { get; init; } = [];
    /// <summary>Normalised mid-band energy (250–2000 Hz). High = snare/vocals.</summary>
    public IReadOnlyList<double> MidBandEnergy { get; init; } = [];
    /// <summary>Normalised high-band energy (2000+ Hz). High = hi-hat/cymbal.</summary>
    public IReadOnlyList<double> HighBandEnergy { get; init; } = [];
    /// <summary>Normalised spectral centroid 0-1 (0=bass, 1=treble).</summary>
    public IReadOnlyList<double> SpectralCentroid { get; init; } = [];

    // --- Frame-level helpers -------------------------------------------------

    /// <summary>Returns the frame index for a given time in seconds.</summary>
    public int TimeToFrame(double timeSeconds) =>
        FrameRateHz > 0
            ? Math.Max(0, Math.Min((int)(timeSeconds * FrameRateHz), EnergyEnvelope.Count - 1))
            : 0;

    public double GetLowBand(double timeSeconds)      => LowBandEnergy.Count  > 0 ? LowBandEnergy[TimeToFrame(timeSeconds)]  : 0;
    public double GetMidBand(double timeSeconds)      => MidBandEnergy.Count  > 0 ? MidBandEnergy[TimeToFrame(timeSeconds)]  : 0;
    public double GetHighBand(double timeSeconds)     => HighBandEnergy.Count > 0 ? HighBandEnergy[TimeToFrame(timeSeconds)] : 0;
    public double GetCentroid(double timeSeconds)     => SpectralCentroid.Count > 0 ? SpectralCentroid[TimeToFrame(timeSeconds)] : 0.5;
    public double GetEnergy(double timeSeconds)       => EnergyEnvelope.Count > 0 ? EnergyEnvelope[TimeToFrame(timeSeconds)] : 0;
}
