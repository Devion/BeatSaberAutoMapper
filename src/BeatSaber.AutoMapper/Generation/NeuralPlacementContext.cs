namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Rich scoring context for the neural placement model.
/// 45 features matching BeatSaberMappingNet.InputDim exactly.
/// Feature layout:
///   [0-13]  Audio (onset, energy, flux, transient, bands, centroid, deltas, trends, lookahead)
///   [14-20] Beat position (phase, subdiv, strength, measure beat, song fraction, section progress, bar pos)
///   [21-28] Section type one-hot (8 classes)
///   [29-30] Context (difficulty, local NPS)
///   [31-44] Previous placement (14 dims)
/// </summary>
public readonly struct NeuralPlacementContext
{
    // Audio features [0-13]
    public double OnsetStrength     { get; init; }
    public double EnergyLevel       { get; init; }
    public double SpectralFlux      { get; init; }
    public double TransientStrength { get; init; }
    public double LowBandEnergy     { get; init; }
    public double MidBandEnergy     { get; init; }
    public double HighBandEnergy    { get; init; }
    public double SpectralCentroid  { get; init; } = 0.5;
    public double EnergyDelta       { get; init; }
    public double HighBandDelta     { get; init; }
    public double EnergyTrend4      { get; init; }
    public double EnergyTrend8      { get; init; }
    public double LookaheadEnergy   { get; init; }
    public double LookaheadOnset    { get; init; }

    // Beat position [14-20]
    public double BeatPhase       { get; init; }
    public double Subdiv          { get; init; }
    public double BeatStrength    { get; init; }
    public int    MeasureBeat     { get; init; }
    public double SongFraction    { get; init; }
    public double SectionProgress { get; init; }
    public double BarPosition     { get; init; }

    // Section type one-hot (8 classes) [21-28]
    // 0=Intro 1=Verse 2=Chorus 3=Bridge 4=Buildup 5=Drop 6=Outro 7=Unknown
    public int SectionTypeIndex { get; init; } = 7;

    // Context [29-30]
    public int    DifficultyLevel { get; init; }
    public double LocalNps        { get; init; }

    // Previous placement [31-44]
    public int    PrevLeftLane        { get; init; }
    public int    PrevLeftRow         { get; init; }
    public int    PrevLeftCutDir      { get; init; } = -1;
    public int    PrevRightLane       { get; init; }
    public int    PrevRightRow        { get; init; }
    public int    PrevRightCutDir     { get; init; } = -1;
    public double LeftParityState     { get; init; } = 0.5;
    public double RightParityState    { get; init; } = 0.5;
    public double BeatsSinceLastLeft  { get; init; } = 999;
    public double BeatsSinceLastRight { get; init; } = 999;
    // slot 43: HasAnyPrevNote (computed in FillFeaturesFloat)
    // slot 44: NoteHandHint
    public double NoteHandHint { get; init; } = 0.5;

    // GRU hidden state (not part of feature vector)
    public GruState? GruHiddenState { get; init; }

    public NeuralPlacementContext() { }

    /// <summary>Fills <paramref name="f"/> with exactly 45 normalised float features.</summary>
    internal void FillFeaturesFloat(float[] f)
    {
        // Audio [0-13]
        f[0]  = (float)OnsetStrength;
        f[1]  = (float)EnergyLevel;
        f[2]  = (float)SpectralFlux;
        f[3]  = (float)TransientStrength;
        f[4]  = (float)LowBandEnergy;
        f[5]  = (float)MidBandEnergy;
        f[6]  = (float)HighBandEnergy;
        f[7]  = (float)SpectralCentroid;
        f[8]  = (float)Math.Clamp(EnergyDelta,    -1.0, 1.0);
        f[9]  = (float)Math.Clamp(HighBandDelta,  -1.0, 1.0);
        f[10] = (float)Math.Clamp(EnergyTrend4,    0.0, 1.0);
        f[11] = (float)Math.Clamp(EnergyTrend8,    0.0, 1.0);
        f[12] = (float)Math.Clamp(LookaheadEnergy, 0.0, 1.0);
        f[13] = (float)Math.Clamp(LookaheadOnset,  0.0, 1.0);
        // Beat position [14-20]
        f[14] = (float)BeatPhase;
        f[15] = (float)Subdiv;
        f[16] = (float)BeatStrength;
        f[17] = (float)(MeasureBeat / 4.0);
        f[18] = (float)Math.Clamp(SongFraction,    0.0, 1.0);
        f[19] = (float)Math.Clamp(SectionProgress, 0.0, 1.0);
        f[20] = (float)BarPosition;
        // Section one-hot [21-28]
        for (int i = 21; i <= 28; i++) f[i] = 0f;
        f[21 + Math.Clamp(SectionTypeIndex, 0, 7)] = 1f;
        // Context [29-30]
        f[29] = (float)(DifficultyLevel / 4.0);
        f[30] = (float)Math.Min(LocalNps / 10.0, 1.0);
        // Previous placement [31-44]
        f[31] = (float)(PrevLeftLane  / 3.0);
        f[32] = (float)(PrevLeftRow   / 2.0);
        f[33] = PrevLeftCutDir  >= 0 ? 1f : 0f;
        f[34] = PrevLeftCutDir  >= 0 ? (float)(PrevLeftCutDir  / 8.0) : 0f;
        f[35] = (float)(PrevRightLane / 3.0);
        f[36] = (float)(PrevRightRow  / 2.0);
        f[37] = PrevRightCutDir >= 0 ? 1f : 0f;
        f[38] = PrevRightCutDir >= 0 ? (float)(PrevRightCutDir / 8.0) : 0f;
        f[39] = (float)Math.Clamp(LeftParityState,       0.0, 1.0);
        f[40] = (float)Math.Clamp(RightParityState,      0.0, 1.0);
        f[41] = (float)Math.Clamp(BeatsSinceLastLeft  / 8.0, 0.0, 1.0);
        f[42] = (float)Math.Clamp(BeatsSinceLastRight / 8.0, 0.0, 1.0);
        f[43] = (PrevLeftCutDir >= 0 || PrevRightCutDir >= 0) ? 1f : 0f;
        f[44] = (float)Math.Clamp(NoteHandHint, 0.0, 1.0);
    }
}