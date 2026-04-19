using BeatSaber.AutoMapper.Audio;
using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Training.Features;
using BeatSaber.AutoMapper.Training.Patterns;
using BeatSaber.AutoMapper.Utilities;

namespace BeatSaber.AutoMapper.Training.SelfSupervised;

/// <summary>
/// Generates synthetic training examples from a generated beatmap by scoring
/// each placed note against audio evidence. This creates a self-reinforcing
/// feedback loop:
/// <list type="bullet">
///   <item>Well-supported note placements (strong beat + onset + energy) are
///         reinforced as positive examples with elevated weight.</item>
///   <item>Poorly-supported placements (note in silence, no onset) are injected
///         as negative examples so the model learns to avoid them.</item>
///   <item>Strong onsets that the model missed are added as positive examples.</item>
///   <item>Difficulty-appropriate density correction adds downbeat positives when
///         the generated map is too sparse.</item>
/// </list>
/// </summary>
public sealed class SelfSupervisedExampleGenerator
{
    // Placement quality thresholds
    private const double StrongPositiveThreshold = 0.65; // clearly good
    private const double WeakPositiveThreshold   = 0.50; // marginal
    private const double NegativeThreshold       = 0.35; // bad
    private const double StrongNegativeThreshold = 0.20; // very bad

    // Missed-onset criteria
    private const double MissedOnsetEnergyMin    = 0.50; // minimum local energy
    private const double MissedOnsetScoreMin     = 0.65; // minimum NoteQualityScorer score
    private const double OnsetCoverageSeconds    = 0.08; // note within 80 ms "covers" an onset

    /// <summary>
    /// Generate synthetic examples from a completed generation pass.
    /// </summary>
    /// <param name="generated">The beatmap the model just produced.</param>
    /// <param name="audio">Pre-analysed audio for the same song.</param>
    /// <param name="difficultyLevel">0=Easy … 4=ExpertPlus.</param>
    /// <param name="positiveWeight">Weight applied to reinforced positive examples.</param>
    /// <param name="negativeWeight">Weight applied to penalising negative examples.</param>
    public IReadOnlyList<TrainingExample> Generate(
        CanonicalBeatmap generated,
        AudioAnalysisResult audio,
        int difficultyLevel,
        double positiveWeight = 4.0,
        double negativeWeight = 3.0)
    {
        Guard.NotNull(generated, nameof(generated));
        Guard.NotNull(audio, nameof(audio));

        var result = new List<TrainingExample>();
        double bpm = audio.EstimatedBpm > 0 ? audio.EstimatedBpm : 120.0;
        double songDurationBeats = audio.DurationSeconds > 0
            ? MathHelpers.SecondsToBeat(audio.DurationSeconds, bpm) : 1.0;
        var orderedNotes = generated.Notes
            .OrderBy(n => n.Beat)
            .ThenBy(n => n.Hand == NoteHand.Left ? 0 : 1)
            .ToList();

        // Sorted note times for fast "covered?" lookup
        var noteTimes = orderedNotes
            .Select(n => MathHelpers.BeatToSeconds(n.Beat, bpm))
            .OrderBy(t => t)
            .ToList();

        // ------------------------------------------------------------------
        // 1. Score every placed note — reinforce good, penalise bad
        // ------------------------------------------------------------------
        var history = new List<CanonicalNote>();
        foreach (var note in orderedNotes)
        {
            double noteTime = MathHelpers.BeatToSeconds(note.Beat, bpm);
            double quality  = NoteQualityScorer.ScorePosition(noteTime, note.Beat, audio);
            double localNps = generated.LocalNps(note.Beat, 4.0);

            // Use the actual note attributes (hand, lane, row, dir) for positive examples
            int hand   = note.Color == NoteColor.Blue ? 1 : 0;
            int lane   = note.Lane;
            int row    = note.Row;
            int cutDir = (int)note.CutDirection;

            if (quality >= StrongPositiveThreshold)
            {
                result.Add(MakeExample(note.Beat, noteTime, audio, bpm, difficultyLevel,
                    history,
                    localNps, songDurationBeats, hasNote: true, weight: positiveWeight * quality,
                    noteHand: hand, noteLane: lane, noteRow: row, noteCutDir: cutDir));
            }
            else if (quality >= WeakPositiveThreshold)
            {
                result.Add(MakeExample(note.Beat, noteTime, audio, bpm, difficultyLevel,
                    history,
                    localNps, songDurationBeats, hasNote: true, weight: positiveWeight * 0.5,
                    noteHand: hand, noteLane: lane, noteRow: row, noteCutDir: cutDir));
            }
            else if (quality <= NegativeThreshold)
            {
                double w = quality <= StrongNegativeThreshold
                    ? negativeWeight * 1.5
                    : negativeWeight;
                result.Add(MakeExample(note.Beat, noteTime, audio, bpm, difficultyLevel,
                    history,
                    localNps, songDurationBeats, hasNote: false, weight: w));
            }
            // Middle range (0.35–0.50): ambiguous, skip to avoid noisy signal
            history.Add(note);
        }

        // ------------------------------------------------------------------
        // 2. Find strong onsets the model missed — add as positive examples
        // ------------------------------------------------------------------
        foreach (double onset in audio.OnsetTimesSeconds)
        {
            // Does a placed note already cover this onset?
            int idx = noteTimes.BinarySearch(onset);
            if (idx < 0) idx = ~idx;

            bool covered = false;
            for (int k = idx - 1; k <= idx && k < noteTimes.Count; k++)
            {
                if (k >= 0 && Math.Abs(noteTimes[k] - onset) < OnsetCoverageSeconds)
                { covered = true; break; }
            }
            if (covered) continue;

            double energy = audio.GetEnergy(onset);
            if (energy < MissedOnsetEnergyMin) continue;

            double beat  = MathHelpers.SecondsToBeat(onset, bpm);
            double score = NoteQualityScorer.ScorePosition(onset, beat, audio);
            if (score < MissedOnsetScoreMin) continue;

            result.Add(MakeExample(beat, onset, audio, bpm, difficultyLevel,
                HistoryBeforeBeat(orderedNotes, beat),
                localNps: 0, songDurationBeats, hasNote: true, weight: positiveWeight));
        }

        // ------------------------------------------------------------------
        // 3. Density correction: when map is too sparse, reinforce downbeats
        //    in energetic sections so the model learns to place MORE notes
        // ------------------------------------------------------------------
        double expectedNps = ExpectedNps(difficultyLevel);
        double actualNps   = generated.Notes.Count > 0 && audio.DurationSeconds > 0
            ? generated.Notes.Count / audio.DurationSeconds : 0;

        if (actualNps < expectedNps * 0.5 && audio.BeatTimesSeconds.Count > 0)
        {
            foreach (double bt in audio.BeatTimesSeconds)
            {
                double energy = audio.GetEnergy(bt);
                if (energy < 0.5) continue;

                double beat = MathHelpers.SecondsToBeat(bt, bpm);
                int    mb   = ((int)Math.Floor(beat) % 4) + 1; // 1-4
                if (mb != 1 && mb != 3) continue;              // downbeats only

                // Is this beat already covered?
                int idx = noteTimes.BinarySearch(bt);
                if (idx < 0) idx = ~idx;
                bool covered = false;
                for (int k = idx - 1; k <= idx && k < noteTimes.Count; k++)
                    if (k >= 0 && Math.Abs(noteTimes[k] - bt) < OnsetCoverageSeconds)
                    { covered = true; break; }
                if (covered) continue;

                result.Add(MakeExample(beat, bt, audio, bpm, difficultyLevel,
                    HistoryBeforeBeat(orderedNotes, beat),
                    localNps: 0, songDurationBeats, hasNote: true, weight: positiveWeight * 0.7));
            }
        }

        return result;
    }

    // -----------------------------------------------------------------------

    private static TrainingExample MakeExample(
        double beat, double timeSeconds,
        AudioAnalysisResult audio, double bpm, int difficultyLevel,
        IReadOnlyList<CanonicalNote> history,
        double localNps, double songDurationBeats, bool hasNote, double weight,
        int noteHand = -1, int noteLane = -1, int noteRow = -1, int noteCutDir = -1)
    {
        double frac   = beat - Math.Floor(beat);
        double subdiv = frac < 0.01 ? 1.0 : Math.Abs(frac - 0.5) < 0.01 ? 0.5 : 0.25;

        int    mb       = ((int)Math.Floor(beat) % 4) + 1;
        double strength = mb switch { 1 => 1.00, 3 => 0.75, 2 or 4 => 0.50, _ => 0.50 };
        if (frac > 0.01) strength *= 0.5;

        double energy  = audio.GetEnergy(timeSeconds);
        double onset   = audio.GetOnsetStrength(timeSeconds);
        double flux    = audio.GetSpectralFlux(timeSeconds);
        double trans   = audio.GetTransient(timeSeconds);

        double prevTime      = Math.Max(0, timeSeconds - 0.1);
        double prevEnergy    = audio.GetEnergy(prevTime);
        double prevHighBand  = audio.GetHighBand(prevTime);
        double energyMax     = Math.Max(0.01, Math.Max(energy, prevEnergy));
        double energyDelta   = Math.Clamp((energy - prevEnergy) / energyMax, -1.0, 1.0);
        double highBandDelta = Math.Clamp(audio.GetHighBand(timeSeconds) - prevHighBand, -1.0, 1.0);
        double energyTrend4  = audio.GetMeanEnergyBeforeBeats(timeSeconds, bpm, 4);
        double energyTrend8  = audio.GetMeanEnergyBeforeBeats(timeSeconds, bpm, 8);
        double songFraction  = songDurationBeats > 0 ? Math.Clamp(beat / songDurationBeats, 0.0, 1.0) : 0.0;
        double barPosition   = (beat % 4) / 4.0;

        double lookaheadTime   = Math.Min(timeSeconds + MathHelpers.BeatToSeconds(1.0, bpm), audio.DurationSeconds);
        double lookaheadEnergy = audio.GetEnergy(lookaheadTime);
        double lookaheadOnset  = audio.GetOnsetStrength(lookaheadTime);

        int sectionTypeIdx     = GetSectionTypeIndex(timeSeconds, audio);
        double sectionProgress = GetSectionProgress(timeSeconds, audio);
        var (lastLeft, prev2Left)   = MappingFeatureEngineering.LastTwoForHand(history, NoteHand.Left);
        var (lastRight, prev2Right) = MappingFeatureEngineering.LastTwoForHand(history, NoteHand.Right);
        double leftParity  = lastLeft is not null ? (lastLeft.CutDirection is CutDirection.Up or CutDirection.UpLeft or CutDirection.UpRight or CutDirection.Left or CutDirection.Right ? 1.0 : 0.0) : 0.5;
        double rightParity = lastRight is not null ? (lastRight.CutDirection is CutDirection.Up or CutDirection.UpLeft or CutDirection.UpRight or CutDirection.Left or CutDirection.Right ? 1.0 : 0.0) : 0.5;
        double beatsSinceLeft = lastLeft is not null ? Math.Max(0, beat - lastLeft.Beat) : 999;
        double beatsSinceRight = lastRight is not null ? Math.Max(0, beat - lastRight.Beat) : 999;
        var (beatsSinceSectionStart, beatsToSectionBoundary) = MappingFeatureEngineering.SectionBoundaryFeatures(audio.Sections, beat);
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
        int patternTypeId = hasNote
            ? (int)PatternModeling.DerivePatternType(
                beat,
                [new CanonicalNote(
                    beat,
                    Math.Max(0, noteLane),
                    Math.Max(0, noteRow),
                    noteHand == 1 ? NoteColor.Blue : NoteColor.Red,
                    noteCutDir >= 0 ? (CutDirection)noteCutDir : CutDirection.Down)],
                history)
            : (int)PatternType.Isolated;

        return new TrainingExample(
            Beat:                   beat,
            OnsetStrength:          onset,
            EnergyLevel:            energy,
            SpectralFlux:           flux,
            TransientStrength:      trans,
            LowBandEnergy:          audio.GetLowBand(timeSeconds),
            MidBandEnergy:          audio.GetMidBand(timeSeconds),
            HighBandEnergy:         audio.GetHighBand(timeSeconds),
            SpectralCentroid:       audio.GetCentroid(timeSeconds),
            EnergyDelta:            energyDelta,
            HighBandDelta:          highBandDelta,
            EnergyTrend4:           energyTrend4,
            EnergyTrend8:           energyTrend8,
            LookaheadEnergy:        lookaheadEnergy,
            LookaheadOnset:         lookaheadOnset,
            BeatPhase:              frac,
            SubdivisionDenominator: subdiv,
            BeatStrength:           strength,
            MeasureBeat:            mb,
            SongFraction:           songFraction,
            SectionProgress:        sectionProgress,
            SectionTypeIndex:       sectionTypeIdx,
            DifficultyLevel:        difficultyLevel,
            LocalNps:               localNps,
            BarPosition:            barPosition,
            PreviousLeftLane:       lastLeft?.Lane ?? 1,
            PreviousLeftRow:        lastLeft?.Row ?? 1,
            PreviousLeftCutDir:     lastLeft is not null ? (int)lastLeft.CutDirection : -1,
            PreviousRightLane:      lastRight?.Lane ?? 2,
            PreviousRightRow:       lastRight?.Row ?? 1,
            PreviousRightCutDir:    lastRight is not null ? (int)lastRight.CutDirection : -1,
            LeftParityState:        leftParity,
            RightParityState:       rightParity,
            BeatsSinceLastLeft:     beatsSinceLeft,
            BeatsSinceLastRight:    beatsSinceRight,
            HasNote:                hasNote,
            NoteHand:               noteHand,
            NoteLane:               noteLane,
            NoteRow:                noteRow,
            NoteCutDir:             noteCutDir,
            PatternTypeId:          patternTypeId,
            Weight:                 weight,
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
            RecentRowSpan4:         recentRowSpan4);
    }

    private static List<CanonicalNote> HistoryBeforeBeat(IReadOnlyList<CanonicalNote> orderedNotes, double beat) =>
        orderedNotes
            .Where(n => n.Beat < beat)
            .OrderBy(n => n.Beat)
            .ThenBy(n => n.Hand == NoteHand.Left ? 0 : 1)
            .ToList();

    private static int GetSectionTypeIndex(double timeSeconds, AudioAnalysisResult audio)
    {
        if (audio.Sections.Count == 0) return 7;
        SectionMarker? current = null;
        foreach (var s in audio.Sections)
            if (s.TimeSeconds <= timeSeconds) current = s;
        if (current is null) return 7;
        return current.Type switch
        {
            SectionType.Intro   => 0,
            SectionType.Verse   => 1,
            SectionType.Chorus  => 2,
            SectionType.Bridge  => 3,
            SectionType.Buildup => 4,
            SectionType.Drop    => 5,
            SectionType.Outro   => 6,
            _                   => 7
        };
    }

    private static double GetSectionProgress(double timeSeconds, AudioAnalysisResult audio)
    {
        if (audio.Sections.Count == 0) return 0;
        int idx = 0;
        for (int i = 0; i < audio.Sections.Count; i++)
            if (audio.Sections[i].TimeSeconds <= timeSeconds) idx = i;

        double start = audio.Sections[idx].TimeSeconds;
        double end   = idx + 1 < audio.Sections.Count
            ? audio.Sections[idx + 1].TimeSeconds
            : audio.DurationSeconds;
        double span = end - start;
        return span > 0 ? (timeSeconds - start) / span : 0;
    }

    private static double ExpectedNps(int diff) => diff switch
    {
        0 => 1.5,
        1 => 2.5,
        2 => 3.5,
        3 => 5.0,
        4 => 7.0,
        _ => 3.5
    };
}
