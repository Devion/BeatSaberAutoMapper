namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Rich scoring context for the neural placement model.
/// 23 features: rhythmic, audio, spectral, and sequential note-history.
/// </summary>
public readonly struct NeuralPlacementContext
{
    // ── audio features ──
    public double Onset           { get; init; }
    public double Energy          { get; init; }
    public double Subdiv          { get; init; }   // 0.25 / 0.5 / 1.0
    public double LocalNps        { get; init; }
    public double BeatStrength    { get; init; }
    public int    MeasureBeat     { get; init; }   // 1-4
    public int    DifficultyLevel { get; init; }   // 0=Easy … 4=ExpertPlus

    // ── rhythmic context ──
    public double BeatPhase       { get; init; }   // 0-1 fractional beat position
    public double SectionProgress { get; init; }   // 0-1 progress within current section

    // ── previous-note state ──
    public int    PrevLeftLane        { get; init; }
    public int    PrevLeftRow         { get; init; }
    public int    PrevLeftCutDir      { get; init; }  // -1 = no previous note
    public int    PrevRightLane       { get; init; }
    public int    PrevRightRow        { get; init; }
    public int    PrevRightCutDir     { get; init; }  // -1 = no previous note
    public double BeatsSinceLastLeft  { get; init; }
    public double BeatsSinceLastRight { get; init; }

    // ── spectral features (per-frame from AudioAnalysisResult) ──
    public double LowBandEnergy    { get; init; }  // 20-250 Hz — kick / bass
    public double MidBandEnergy    { get; init; }  // 250-2000 Hz — snare / vocals
    public double HighBandEnergy   { get; init; }  // 2000+ Hz — hi-hat / cymbal
    public double SpectralCentroid { get; init; }  // 0-1 brightness (0=bass, 1=treble)

    /// <summary>
    /// Fills <paramref name="f"/> with 23 normalised feature values.
    /// The caller must supply an array of length ≥ 23.
    /// </summary>
    /// <summary>Float32 version for TorchSharp tensor building.</summary>
    internal void FillFeaturesFloat(float[] f)
    {
        f[0]  = (float)Onset;
        f[1]  = (float)Energy;
        f[2]  = (float)Subdiv;
        f[3]  = (float)Math.Min(LocalNps / 10.0, 1.0);
        f[4]  = (float)BeatStrength;
        f[5]  = (float)(MeasureBeat / 4.0);
        f[6]  = (float)(DifficultyLevel / 4.0);
        f[7]  = (float)BeatPhase;
        f[8]  = (float)SectionProgress;
        f[9]  = (float)(PrevLeftLane  / 3.0);
        f[10] = (float)(PrevLeftRow   / 2.0);
        f[11] = PrevLeftCutDir  >= 0 ? 1f : 0f;
        f[12] = PrevLeftCutDir  >= 0 ? (float)(PrevLeftCutDir  / 8.0) : 0f;
        f[13] = (float)(PrevRightLane / 3.0);
        f[14] = (float)(PrevRightRow  / 2.0);
        f[15] = PrevRightCutDir >= 0 ? 1f : 0f;
        f[16] = PrevRightCutDir >= 0 ? (float)(PrevRightCutDir / 8.0) : 0f;
        f[17] = (float)Math.Min(BeatsSinceLastLeft  / 8.0, 1.0);
        f[18] = (float)Math.Min(BeatsSinceLastRight / 8.0, 1.0);
        f[19] = (float)LowBandEnergy;
        f[20] = (float)MidBandEnergy;
        f[21] = (float)HighBandEnergy;
        f[22] = (float)SpectralCentroid;
    }

    internal void FillFeatures(double[] f)
    {
        f[0]  = Onset;
        f[1]  = Energy;
        f[2]  = Subdiv;
        f[3]  = Math.Min(LocalNps / 10.0, 1.0);
        f[4]  = BeatStrength;
        f[5]  = MeasureBeat / 4.0;
        f[6]  = DifficultyLevel / 4.0;
        f[7]  = BeatPhase;
        f[8]  = SectionProgress;
        f[9]  = PrevLeftLane  / 3.0;
        f[10] = PrevLeftRow   / 2.0;
        f[11] = PrevLeftCutDir  >= 0 ? 1.0 : 0.0;            // has-prev-left flag
        f[12] = PrevLeftCutDir  >= 0 ? PrevLeftCutDir  / 8.0 : 0.0;
        f[13] = PrevRightLane / 3.0;
        f[14] = PrevRightRow  / 2.0;
        f[15] = PrevRightCutDir >= 0 ? 1.0 : 0.0;            // has-prev-right flag
        f[16] = PrevRightCutDir >= 0 ? PrevRightCutDir / 8.0 : 0.0;  // value (was missing!)
        f[17] = Math.Min(BeatsSinceLastLeft  / 8.0, 1.0);
        f[18] = Math.Min(BeatsSinceLastRight / 8.0, 1.0);
        f[19] = LowBandEnergy;
        f[20] = MidBandEnergy;
        f[21] = HighBandEnergy;
        f[22] = SpectralCentroid;
    }
}
