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
    /// <summary>
    /// Normalised spectral flux: sum of positive spectral magnitude differences.
    /// High = rapid spectral change, onset transient.
    /// </summary>
    public IReadOnlyList<double> SpectralFlux { get; init; } = [];
    /// <summary>
    /// Transient strength: ratio of high-frequency flux to total flux (0-1).
    /// High = sharp percussive attack.
    /// </summary>
    public IReadOnlyList<double> TransientStrength { get; init; } = [];
    /// <summary>
    /// Continuous onset strength envelope (normalised 0-1).
    /// More informative than binary OnsetTimesSeconds for model features.
    /// </summary>
    public IReadOnlyList<double> OnsetStrengthEnvelope { get; init; } = [];

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
    public double GetSpectralFlux(double timeSeconds) => SpectralFlux.Count > 0 ? SpectralFlux[TimeToFrame(timeSeconds)] : 0;
    public double GetTransient(double timeSeconds)    => TransientStrength.Count > 0 ? TransientStrength[TimeToFrame(timeSeconds)] : 0;
    public double GetOnsetStrength(double timeSeconds)=> OnsetStrengthEnvelope.Count > 0 ? OnsetStrengthEnvelope[TimeToFrame(timeSeconds)] : 0;

    /// <summary>
    /// Mean energy over the N beats immediately preceding <paramref name="timeSeconds"/>.
    /// Returns 0 if no energy data is available.
    /// </summary>
    public double GetMeanEnergyBeforeBeats(double timeSeconds, double bpm, int beatCount)
    {
        if (EnergyEnvelope.Count == 0 || bpm <= 0 || beatCount <= 0) return 0;
        double beatSec   = 60.0 / bpm;
        double startTime = Math.Max(0, timeSeconds - beatCount * beatSec);
        int    f0        = TimeToFrame(startTime);
        int    f1        = TimeToFrame(timeSeconds);
        if (f1 <= f0) return GetEnergy(timeSeconds);
        double sum = 0;
        for (int i = f0; i <= f1; i++) sum += EnergyEnvelope[i];
        return sum / (f1 - f0 + 1);
    }
}
