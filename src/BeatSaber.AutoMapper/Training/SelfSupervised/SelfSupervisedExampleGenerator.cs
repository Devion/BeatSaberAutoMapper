using BeatSaber.AutoMapper.Audio;
using BeatSaber.AutoMapper.Training.Features;
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

        // Sorted note times for fast "covered?" lookup
        var noteTimes = generated.Notes
            .Select(n => MathHelpers.BeatToSeconds(n.Beat, bpm))
            .OrderBy(t => t)
            .ToList();

        // ------------------------------------------------------------------
        // 1. Score every placed note — reinforce good, penalise bad
        // ------------------------------------------------------------------
        foreach (var note in generated.Notes)
        {
            double noteTime = MathHelpers.BeatToSeconds(note.Beat, bpm);
            double quality  = NoteQualityScorer.ScorePosition(noteTime, note.Beat, audio);
            double localNps = generated.LocalNps(note.Beat, 4.0);

            if (quality >= StrongPositiveThreshold)
            {
                result.Add(MakeExample(note.Beat, noteTime, audio, bpm, difficultyLevel,
                    localNps, hasNote: true, weight: positiveWeight * quality));
            }
            else if (quality >= WeakPositiveThreshold)
            {
                result.Add(MakeExample(note.Beat, noteTime, audio, bpm, difficultyLevel,
                    localNps, hasNote: true, weight: positiveWeight * 0.5));
            }
            else if (quality <= NegativeThreshold)
            {
                double w = quality <= StrongNegativeThreshold
                    ? negativeWeight * 1.5
                    : negativeWeight;
                result.Add(MakeExample(note.Beat, noteTime, audio, bpm, difficultyLevel,
                    localNps, hasNote: false, weight: w));
            }
            // Middle range (0.35–0.50): ambiguous, skip to avoid noisy signal
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
                localNps: 0, hasNote: true, weight: positiveWeight));
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
                    localNps: 0, hasNote: true, weight: positiveWeight * 0.7));
            }
        }

        return result;
    }

    // -----------------------------------------------------------------------

    private static TrainingExample MakeExample(
        double beat, double timeSeconds,
        AudioAnalysisResult audio, double bpm, int difficultyLevel,
        double localNps, bool hasNote, double weight)
    {
        double frac   = beat - Math.Floor(beat);
        double subdiv = frac < 0.01 ? 1.0
                      : Math.Abs(frac - 0.5) < 0.01 ? 0.5
                      : 0.25;

        int    mb       = ((int)Math.Floor(beat) % 4) + 1;
        double strength = mb switch { 1 => 1.00, 3 => 0.75, 2 or 4 => 0.50, _ => 0.50 };
        if (frac > 0.01) strength *= 0.5;

        double energy  = audio.GetEnergy(timeSeconds);
        double onset   = GetOnsetStrength(timeSeconds, audio);

        return new TrainingExample(
            Beat:                   beat,
            SubdivisionDenominator: subdiv,
            OnsetStrength:          onset,
            EnergyLevel:            energy,
            LocalNps:               localNps,
            SectionProgress:        GetSectionProgress(timeSeconds, audio),
            BeatPhase:              frac,
            BeatStrength:           strength,
            MeasureBeat:            mb,
            DifficultyLevel:        difficultyLevel,
            PreviousLeftLane:       1,
            PreviousLeftRow:        1,
            PreviousLeftCutDir:     -1,
            PreviousRightLane:      2,
            PreviousRightRow:       1,
            PreviousRightCutDir:    -1,
            BeatsSinceLastLeft:     999,
            BeatsSinceLastRight:    999,
            LowBandEnergy:          audio.GetLowBand(timeSeconds),
            MidBandEnergy:          audio.GetMidBand(timeSeconds),
            HighBandEnergy:         audio.GetHighBand(timeSeconds),
            SpectralCentroid:       audio.GetCentroid(timeSeconds),
            HasNote:                hasNote,
            NoteHand:               -1,
            NoteLane:               -1,
            NoteRow:                -1,
            NoteCutDir:             -1,
            Weight:                 weight);
    }

    private static double GetOnsetStrength(double timeSeconds, AudioAnalysisResult audio)
    {
        const double window = 0.05;
        foreach (double o in audio.OnsetTimesSeconds)
            if (Math.Abs(o - timeSeconds) < window) return 1.0;
        return 0.0;
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
