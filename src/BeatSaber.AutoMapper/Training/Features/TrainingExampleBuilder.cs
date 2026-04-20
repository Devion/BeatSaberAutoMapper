using BeatSaber.AutoMapper.Audio;
using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Training.Patterns;

namespace BeatSaber.AutoMapper.Training.Features;

/// <summary>
/// One beat-position training example. Features map 1-to-1 to the GRU input vector.
/// Feature layout (see BeatSaberMappingNet.InputDim):
///   [0-13]  Audio features (onset, energy, spectral flux, transient, bands, centroid, deltas, trends, lookahead)
///   [14-19] Beat position (phase, subdiv, strength, measure beat, song fraction, section progress)
///   [20-27] Section type one-hot (Intro/Verse/Chorus/Bridge/Buildup/Drop/Outro/Unknown)
///   [28-30] Context (difficulty, local NPS, bar position)
///   [31-44] Previous placement (teacher-forced: has-note, hand, L lane/row/dir, R lane/row/dir, parity, beats-since)
///   [45-77] Phrase, rhythm-history, geometry, and comfort features.
/// </summary>
public sealed record TrainingExample(
    double Beat,
    // ── Audio features ─────────────────────────────────────────────────
    double OnsetStrength,           // continuous magnitude (0-1), not binary
    double EnergyLevel,
    double SpectralFlux,
    double TransientStrength,
    double LowBandEnergy,
    double MidBandEnergy,
    double HighBandEnergy,
    double SpectralCentroid,
    double EnergyDelta,             // normalised energy rise since 100 ms ago [-1,1]
    double HighBandDelta,           // hi-hat/cymbal delta since 100 ms ago [-1,1]
    double EnergyTrend4,            // mean energy over previous 4 beats
    double EnergyTrend8,            // mean energy over previous 8 beats
    double LookaheadEnergy,         // energy 1 beat ahead
    double LookaheadOnset,          // onset strength 1 beat ahead
    // ── Beat position ──────────────────────────────────────────────────
    double BeatPhase,               // fractional part of beat (0-1)
    double SubdivisionDenominator,  // 1.0 / 0.5 / 0.25
    double BeatStrength,            // downbeat=1.0, beat3=0.75, etc.
    int    MeasureBeat,             // 1-4
    double SongFraction,            // beat / totalBeats [0,1]
    double SectionProgress,         // position within current section [0,1]
    // ── Section type one-hot (8 classes) ───────────────────────────────
    int    SectionTypeIndex,        // 0=Intro 1=Verse 2=Chorus 3=Bridge 4=Buildup 5=Drop 6=Outro 7=Unknown
    // ── Context ────────────────────────────────────────────────────────
    int    DifficultyLevel,         // 0=Easy … 4=ExpertPlus
    double LocalNps,                // notes-per-second in 4-beat window
    double BarPosition,             // (beat % 4) / 4 — continuous position in bar [0,1]
    // ── Previous placement (teacher-forced) ────────────────────────────
    // These are the ground-truth placements from step t-1 (used to condition step t).
    int    PreviousLeftLane,
    int    PreviousLeftRow,
    int    PreviousLeftCutDir,      // -1 = no previous note
    int    PreviousRightLane,
    int    PreviousRightRow,
    int    PreviousRightCutDir,     // -1 = no previous note
    double LeftParityState,         // 0=FH next, 1=BH next, 0.5=unknown
    double RightParityState,
    double BeatsSinceLastLeft,
    double BeatsSinceLastRight,
    // ── Labels ─────────────────────────────────────────────────────────
    bool HasNote   = false,
    int NoteHand   = -1,   // 0=left, 1=right, -1=none
    int NoteLane   = -1,
    int NoteRow    = -1,
    int NoteCutDir = -1,
    int PatternTypeId = 0,
    // ── Per-example gradient multiplier ────────────────────────────────
    double Weight = 1.0,
    // ── Extended phrase / history / geometry features ────────────────
    double FutureEnergy4 = 0,
    double FutureEnergy8 = 0,
    double FutureEnergy16 = 0,
    double FutureOnset4 = 0,
    double FutureOnset8 = 0,
    double FutureOnset16 = 0,
    double BeatsSinceSectionStart = 0,
    double BeatsToSectionBoundary = 16,
    double RecentChordRate4 = 0,
    double RecentOffbeatRate4 = 0,
    double RecentStreamRate4 = 0,
    double RecentAlternation8 = 0.5,
    double RecentHandBalance8 = 0.5,
    double ConsecutiveSameHandCount = 0,
    double BeatsSinceLastAny = 999,
    double NotesAtCurrentBeatSoFar = 0,
    double InterHandLaneDistance = 1.5,
    double InterHandRowDistance = 1.0,
    double HandsCrossedFlag = 0,
    double LeftRecentTravel = 0,
    double RightRecentTravel = 0,
    double RecentLaneSpan4 = 0,
    double RecentRowSpan4 = 0,
    double PhraseBeatPhase32 = 0,
    double PhraseProgress32 = 0,
    double BeatsSincePhraseStart32 = 0,
    double BeatsToPhraseBoundary32 = 32,
    double CurrentBeatVisionBlockRisk = 0,
    double RecentVisionBlockRate8 = 0,
    double LeftParityBreakRate8 = 0,
    double RightParityBreakRate8 = 0,
    double ResetPressure = 0,
    double RecentRestRatio8 = 0
);

public sealed class TrainingExampleBuilder
{
    public IReadOnlyList<TrainingExample> Build(
        CanonicalBeatmap beatmap,
        AudioAnalysisResult audio)
    {
        Guard.NotNull(beatmap, nameof(beatmap));
        Guard.NotNull(audio, nameof(audio));

        double bpm      = audio.EstimatedBpm > 0 ? audio.EstimatedBpm : beatmap.Song.BeatsPerMinute;
        int    diffLevel = (int)beatmap.Difficulty.Difficulty;

        // Derive song length in beats
        double songDurationBeats;
        if (audio.DurationSeconds > 0)
            songDurationBeats = MathHelpers.SecondsToBeat(audio.DurationSeconds, bpm);
        else if (beatmap.Notes.Count > 0)
            songDurationBeats = beatmap.Notes[^1].Beat + 4.0;
        else
            return [];

        var beatTimes = audio.BeatTimesSeconds;

        var examples = new List<TrainingExample>((int)(songDurationBeats * 4) + 8);
        var history  = new List<CanonicalNote>();

        for (double beat = 0; beat < songDurationBeats; beat += 0.25)
        {
            double timeSeconds = MathHelpers.BeatToSeconds(beat, bpm);

            // ── Beat position ─────────────────────────────────────────────
            double frac   = beat - Math.Floor(beat);
            double subdiv = frac < 0.01 ? 1.0
                          : Math.Abs(frac - 0.5) < 0.01 ? 0.5
                          : 0.25;
            (double beatPhase, double beatStrength, int measureBeat) =
                ComputeBeatFeatures(beat, beatTimes);

            double barPosition  = (beat % 4) / 4.0;
            double songFraction = songDurationBeats > 0
                ? Math.Clamp(beat / songDurationBeats, 0.0, 1.0) : 0.0;
            double sectionProg  = GetSectionProgress(timeSeconds, audio);

            // ── Section type one-hot ───────────────────────────────────────
            int sectionTypeIdx = GetSectionTypeIndex(timeSeconds, audio);

            // ── Audio features ────────────────────────────────────────────
            double energy       = audio.GetEnergy(timeSeconds);
            double onsetStr     = audio.GetOnsetStrength(timeSeconds);
            double spectralFlux = audio.GetSpectralFlux(timeSeconds);
            double transient    = audio.GetTransient(timeSeconds);
            double lowBand      = audio.GetLowBand(timeSeconds);
            double midBand      = audio.GetMidBand(timeSeconds);
            double highBand     = audio.GetHighBand(timeSeconds);
            double centroid     = audio.GetCentroid(timeSeconds);

            double prevTime     = Math.Max(0, timeSeconds - 0.1);
            double prevEnergy   = audio.GetEnergy(prevTime);
            double prevHigh     = audio.GetHighBand(prevTime);
            double energyMax    = Math.Max(0.01, Math.Max(energy, prevEnergy));
            double energyDelta  = Math.Clamp((energy - prevEnergy) / energyMax, -1.0, 1.0);
            double highDelta    = Math.Clamp(highBand - prevHigh, -1.0, 1.0);

            double energyTrend4 = audio.GetMeanEnergyBeforeBeats(timeSeconds, bpm, 4);
            double energyTrend8 = audio.GetMeanEnergyBeforeBeats(timeSeconds, bpm, 8);

            double lookaheadTime  = MathHelpers.BeatToSeconds(Math.Min(beat + 1.0, songDurationBeats), bpm);
            double lookaheadEnergy = audio.GetEnergy(lookaheadTime);
            double lookaheadOnset  = audio.GetOnsetStrength(lookaheadTime);

            // ── Context ───────────────────────────────────────────────────
            double localNps = beatmap.LocalNps(beat, 4.0);
            var (lastLeft, prev2Left)   = MappingFeatureEngineering.LastTwoForHand(history, NoteHand.Left);
            var (lastRight, prev2Right) = MappingFeatureEngineering.LastTwoForHand(history, NoteHand.Right);
            var (beatsSinceSectionStart, beatsToSectionBoundary) =
                MappingFeatureEngineering.SectionBoundaryFeatures(audio.Sections, beat);
            double futureEnergy4 = audio.GetMeanEnergyAfterBeats(timeSeconds, bpm, 4);
            double futureEnergy8 = audio.GetMeanEnergyAfterBeats(timeSeconds, bpm, 8);
            double futureEnergy16 = audio.GetMeanEnergyAfterBeats(timeSeconds, bpm, 16);
            double futureOnset4 = audio.GetMeanOnsetAfterBeats(timeSeconds, bpm, 4);
            double futureOnset8 = audio.GetMeanOnsetAfterBeats(timeSeconds, bpm, 8);
            double futureOnset16 = audio.GetMeanOnsetAfterBeats(timeSeconds, bpm, 16);
            double recentChordRate4 = MappingFeatureEngineering.RecentChordRate(history, beat, 4.0);
            double recentOffbeatRate4 = MappingFeatureEngineering.RecentOffbeatRate(history, beat, 4.0);
            double recentStreamRate4 = MappingFeatureEngineering.RecentStreamRate(history, beat, 4.0);
            double recentAlternation8 = MappingFeatureEngineering.RecentAlternation(history, 8);
            double recentHandBalance8 = MappingFeatureEngineering.RecentHandBalance(history, beat, 8.0);
            double consecutiveSameHandCount = MappingFeatureEngineering.ConsecutiveSameHandCount(history);
            double beatsSinceLastAny = MappingFeatureEngineering.BeatsSinceLastAny(history, beat);
            double notesAtCurrentBeatSoFar = MappingFeatureEngineering.NotesAtCurrentBeatSoFar(history, beat);
            double interHandLaneDistance = MappingFeatureEngineering.InterHandLaneDistance(lastLeft, lastRight);
            double interHandRowDistance = MappingFeatureEngineering.InterHandRowDistance(lastLeft, lastRight);
            double handsCrossedFlag = MappingFeatureEngineering.HandsCrossedFlag(lastLeft, lastRight);
            double leftRecentTravel = MappingFeatureEngineering.RecentTravel(prev2Left, lastLeft);
            double rightRecentTravel = MappingFeatureEngineering.RecentTravel(prev2Right, lastRight);
            double recentLaneSpan4 = MappingFeatureEngineering.RecentLaneSpan(history, beat, 4.0);
            double recentRowSpan4 = MappingFeatureEngineering.RecentRowSpan(history, beat, 4.0);
            double phraseBeatPhase32 = beat % 32.0;
            double phraseProgress32 = phraseBeatPhase32 / 32.0;
            double beatsSincePhraseStart32 = phraseBeatPhase32;
            double beatsToPhraseBoundary32 = 32.0 - phraseBeatPhase32;
            double currentBeatVisionBlockRisk = MappingFeatureEngineering.CurrentBeatVisionBlockRisk(history, beat);
            double recentVisionBlockRate8 = MappingFeatureEngineering.RecentVisionBlockRate(history, beat, 8.0);
            double leftParityBreakRate8 = MappingFeatureEngineering.RecentParityBreakRate(history, NoteHand.Left, 8);
            double rightParityBreakRate8 = MappingFeatureEngineering.RecentParityBreakRate(history, NoteHand.Right, 8);
            double recentRestRatio8 = MappingFeatureEngineering.RecentRestRatio(history, beat, 8.0);

            var notesAtBeat = beatmap.Notes
                .Where(n => Math.Abs(n.Beat - beat) < 0.13)
                .OrderBy(n => n.Hand == NoteHand.Left ? 0 : 1)
                .ToList();
            int patternTypeId = (int)PatternModeling.DerivePatternType(beat, notesAtBeat, history);

            if (notesAtBeat.Count == 0)
            {
                double leftParity  = ParityStateFromCutDir(lastLeft?.CutDirection);
                double rightParity = ParityStateFromCutDir(lastRight?.CutDirection);
                double beatsL = lastLeft  is not null ? beat - lastLeft.Beat  : 999;
                double beatsR = lastRight is not null ? beat - lastRight.Beat : 999;
                double resetPressure = Math.Max(
                    MappingFeatureEngineering.ImmediateResetPressure(lastLeft, beat, beatmap.Difficulty.Difficulty),
                    MappingFeatureEngineering.ImmediateResetPressure(lastRight, beat, beatmap.Difficulty.Difficulty));

                examples.Add(new TrainingExample(
                    Beat:                   beat,
                    OnsetStrength:          onsetStr,
                    EnergyLevel:            energy,
                    SpectralFlux:           spectralFlux,
                    TransientStrength:      transient,
                    LowBandEnergy:          lowBand,
                    MidBandEnergy:          midBand,
                    HighBandEnergy:         highBand,
                    SpectralCentroid:       centroid,
                    EnergyDelta:            energyDelta,
                    HighBandDelta:          highDelta,
                    EnergyTrend4:           energyTrend4,
                    EnergyTrend8:           energyTrend8,
                    LookaheadEnergy:        lookaheadEnergy,
                    LookaheadOnset:         lookaheadOnset,
                    BeatPhase:              beatPhase,
                    SubdivisionDenominator: subdiv,
                    BeatStrength:           beatStrength,
                    MeasureBeat:            measureBeat,
                    SongFraction:           songFraction,
                    SectionProgress:        sectionProg,
                    SectionTypeIndex:       sectionTypeIdx,
                    DifficultyLevel:        diffLevel,
                    LocalNps:               localNps,
                    BarPosition:            barPosition,
                    PreviousLeftLane:       lastLeft?.Lane  ?? 1,
                    PreviousLeftRow:        lastLeft?.Row   ?? 1,
                    PreviousLeftCutDir:     lastLeft  is not null ? (int)lastLeft.CutDirection  : -1,
                    PreviousRightLane:      lastRight?.Lane ?? 2,
                    PreviousRightRow:       lastRight?.Row  ?? 1,
                    PreviousRightCutDir:    lastRight is not null ? (int)lastRight.CutDirection : -1,
                    LeftParityState:        leftParity,
                    RightParityState:       rightParity,
                    BeatsSinceLastLeft:     beatsL,
                    BeatsSinceLastRight:    beatsR,
                    HasNote:                false,
                    NoteHand:               -1,
                    NoteLane:               -1,
                    NoteRow:                -1,
                    NoteCutDir:             -1,
                    PatternTypeId:          patternTypeId,
                    FutureEnergy4:          futureEnergy4,
                    FutureEnergy8:          futureEnergy8,
                    FutureEnergy16:         futureEnergy16,
                    FutureOnset4:           futureOnset4,
                    FutureOnset8:           futureOnset8,
                    FutureOnset16:          futureOnset16,
                    BeatsSinceSectionStart: beatsSinceSectionStart,
                    BeatsToSectionBoundary: beatsToSectionBoundary,
                    RecentChordRate4:       recentChordRate4,
                    RecentOffbeatRate4:     recentOffbeatRate4,
                    RecentStreamRate4:      recentStreamRate4,
                    RecentAlternation8:     recentAlternation8,
                    RecentHandBalance8:     recentHandBalance8,
                    ConsecutiveSameHandCount: consecutiveSameHandCount,
                    BeatsSinceLastAny:      beatsSinceLastAny,
                    NotesAtCurrentBeatSoFar: notesAtCurrentBeatSoFar,
                    InterHandLaneDistance:  interHandLaneDistance,
                    InterHandRowDistance:   interHandRowDistance,
                    HandsCrossedFlag:       handsCrossedFlag,
                    LeftRecentTravel:       leftRecentTravel,
                    RightRecentTravel:      rightRecentTravel,
                    RecentLaneSpan4:        recentLaneSpan4,
                    RecentRowSpan4:         recentRowSpan4,
                    PhraseBeatPhase32:      phraseBeatPhase32,
                    PhraseProgress32:       phraseProgress32,
                    BeatsSincePhraseStart32: beatsSincePhraseStart32,
                    BeatsToPhraseBoundary32: beatsToPhraseBoundary32,
                    CurrentBeatVisionBlockRisk: currentBeatVisionBlockRisk,
                    RecentVisionBlockRate8: recentVisionBlockRate8,
                    LeftParityBreakRate8: leftParityBreakRate8,
                    RightParityBreakRate8: rightParityBreakRate8,
                    ResetPressure: resetPressure,
                    RecentRestRatio8: recentRestRatio8
                ));
                continue;
            }

            foreach (var note in notesAtBeat)
            {
                double leftParity  = ParityStateFromCutDir(lastLeft?.CutDirection);
                double rightParity = ParityStateFromCutDir(lastRight?.CutDirection);
                double beatsL = lastLeft  is not null ? beat - lastLeft.Beat  : 999;
                double beatsR = lastRight is not null ? beat - lastRight.Beat : 999;
                double resetPressure = Math.Max(
                    MappingFeatureEngineering.ImmediateResetPressure(lastLeft, beat, beatmap.Difficulty.Difficulty),
                    MappingFeatureEngineering.ImmediateResetPressure(lastRight, beat, beatmap.Difficulty.Difficulty));
                int noteHand = note.Hand == NoteHand.Left ? 0 : 1;

                examples.Add(new TrainingExample(
                    Beat:                   beat,
                    OnsetStrength:          onsetStr,
                    EnergyLevel:            energy,
                    SpectralFlux:           spectralFlux,
                    TransientStrength:      transient,
                    LowBandEnergy:          lowBand,
                    MidBandEnergy:          midBand,
                    HighBandEnergy:         highBand,
                    SpectralCentroid:       centroid,
                    EnergyDelta:            energyDelta,
                    HighBandDelta:          highDelta,
                    EnergyTrend4:           energyTrend4,
                    EnergyTrend8:           energyTrend8,
                    LookaheadEnergy:        lookaheadEnergy,
                    LookaheadOnset:         lookaheadOnset,
                    BeatPhase:              beatPhase,
                    SubdivisionDenominator: subdiv,
                    BeatStrength:           beatStrength,
                    MeasureBeat:            measureBeat,
                    SongFraction:           songFraction,
                    SectionProgress:        sectionProg,
                    SectionTypeIndex:       sectionTypeIdx,
                    DifficultyLevel:        diffLevel,
                    LocalNps:               localNps,
                    BarPosition:            barPosition,
                    PreviousLeftLane:       lastLeft?.Lane  ?? 1,
                    PreviousLeftRow:        lastLeft?.Row   ?? 1,
                    PreviousLeftCutDir:     lastLeft  is not null ? (int)lastLeft.CutDirection  : -1,
                    PreviousRightLane:      lastRight?.Lane ?? 2,
                    PreviousRightRow:       lastRight?.Row  ?? 1,
                    PreviousRightCutDir:    lastRight is not null ? (int)lastRight.CutDirection : -1,
                    LeftParityState:        leftParity,
                    RightParityState:       rightParity,
                    BeatsSinceLastLeft:     beatsL,
                    BeatsSinceLastRight:    beatsR,
                    HasNote:                true,
                    NoteHand:               noteHand,
                    NoteLane:               note.Lane,
                    NoteRow:                note.Row,
                    NoteCutDir:             (int)note.CutDirection,
                    PatternTypeId:          patternTypeId,
                    FutureEnergy4:          futureEnergy4,
                    FutureEnergy8:          futureEnergy8,
                    FutureEnergy16:         futureEnergy16,
                    FutureOnset4:           futureOnset4,
                    FutureOnset8:           futureOnset8,
                    FutureOnset16:          futureOnset16,
                    BeatsSinceSectionStart: beatsSinceSectionStart,
                    BeatsToSectionBoundary: beatsToSectionBoundary,
                    RecentChordRate4:       recentChordRate4,
                    RecentOffbeatRate4:     recentOffbeatRate4,
                    RecentStreamRate4:      recentStreamRate4,
                    RecentAlternation8:     recentAlternation8,
                    RecentHandBalance8:     recentHandBalance8,
                    ConsecutiveSameHandCount: consecutiveSameHandCount,
                    BeatsSinceLastAny:      beatsSinceLastAny,
                    NotesAtCurrentBeatSoFar: notesAtCurrentBeatSoFar,
                    InterHandLaneDistance:  interHandLaneDistance,
                    InterHandRowDistance:   interHandRowDistance,
                    HandsCrossedFlag:       handsCrossedFlag,
                    LeftRecentTravel:       leftRecentTravel,
                    RightRecentTravel:      rightRecentTravel,
                    RecentLaneSpan4:        recentLaneSpan4,
                    RecentRowSpan4:         recentRowSpan4,
                    PhraseBeatPhase32:      phraseBeatPhase32,
                    PhraseProgress32:       phraseProgress32,
                    BeatsSincePhraseStart32: beatsSincePhraseStart32,
                    BeatsToPhraseBoundary32: beatsToPhraseBoundary32,
                    CurrentBeatVisionBlockRisk: currentBeatVisionBlockRisk,
                    RecentVisionBlockRate8: recentVisionBlockRate8,
                    LeftParityBreakRate8: leftParityBreakRate8,
                    RightParityBreakRate8: rightParityBreakRate8,
                    ResetPressure: resetPressure,
                    RecentRestRatio8: recentRestRatio8
                ));

                history.Add(note);
            }
        }

        return examples;
    }

    // ── Beat / rhythm helpers ────────────────────────────────────────────────

    private static (double Phase, double Strength, int MeasureBeat) ComputeBeatFeatures(
        double beat, IReadOnlyList<double> beatTimes)
    {
        double phase = beat - Math.Floor(beat);
        int measureBeat = ((int)Math.Floor(beat) % 4) + 1;
        double strength = measureBeat switch
        {
            1 => 1.00,
            3 => 0.75,
            _ => 0.50
        };
        if (phase > 0.01) strength *= 0.5;
        return (phase, strength, measureBeat);
    }

    private static double GetSectionProgress(double timeSeconds, AudioAnalysisResult audio)
    {
        if (audio.Sections.Count == 0) return 0;
        int secIdx = 0;
        for (int i = 0; i < audio.Sections.Count; i++)
            if (audio.Sections[i].TimeSeconds <= timeSeconds) secIdx = i;

        double start = audio.Sections[secIdx].TimeSeconds;
        double end   = secIdx + 1 < audio.Sections.Count
            ? audio.Sections[secIdx + 1].TimeSeconds
            : audio.DurationSeconds;

        double span = end - start;
        return span > 0 ? (timeSeconds - start) / span : 0;
    }

    private static int GetSectionTypeIndex(double timeSeconds, AudioAnalysisResult audio)
    {
        if (audio.Sections.Count == 0) return 7; // Unknown

        SectionMarker? current = null;
        foreach (var sec in audio.Sections)
            if (sec.TimeSeconds <= timeSeconds) current = sec;

        if (current is null) return 7;

        return current.Label?.ToUpperInvariant() switch
        {
            "INTRO"   or "I"               => 0,
            "VERSE"   or "V"               => 1,
            "CHORUS"  or "C" or "HOOK"     => 2,
            "BRIDGE"  or "B"               => 3,
            "BUILDUP" or "BUILD" or "PRE"  => 4,
            "DROP"    or "D"               => 5,
            "OUTRO"   or "O" or "OUTRO"    => 6,
            _                              => 7
        };
    }

    private static double ParityStateFromCutDir(CutDirection? dir) => dir switch
    {
        CutDirection.Down      or
        CutDirection.DownLeft  or
        CutDirection.DownRight => 1.0,
        CutDirection.Up        or
        CutDirection.UpLeft    or
        CutDirection.UpRight   or
        CutDirection.Left      or
        CutDirection.Right     => 0.0,
        _                      => 0.5,
    };
}
