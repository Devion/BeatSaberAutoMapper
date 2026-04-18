using BeatSaber.AutoMapper.Audio;

namespace BeatSaber.AutoMapper.Training.Features;

public sealed record TrainingExample(
    double Beat,
    double SubdivisionDenominator,
    double OnsetStrength,
    double EnergyLevel,
    double LocalNps,
    double SectionProgress,
    // Beat / rhythmic context
    double BeatPhase,
    double BeatStrength,
    int    MeasureBeat,
    // Difficulty context (0=Easy … 4=ExpertPlus)
    int    DifficultyLevel,
    // Previous note context
    int PreviousLeftLane,
    int PreviousLeftRow,
    int PreviousLeftCutDir,
    int PreviousRightLane,
    int PreviousRightRow,
    int PreviousRightCutDir,
    double BeatsSinceLastLeft,
    double BeatsSinceLastRight,
    // Spectral features (per beat, from AudioAnalysisResult)
    double LowBandEnergy    = 0.0,  // 20-250 Hz — kick / bass
    double MidBandEnergy    = 0.0,  // 250-2000 Hz — snare / vocals
    double HighBandEnergy   = 0.0,  // 2000+ Hz — hi-hat / cymbal
    double SpectralCentroid = 0.5,  // 0-1 brightness
    // Derived temporal features
    double EnergyDelta      = 0.0,  // normalised energy rise since 100 ms ago [-1,1]
    double HighBandDelta    = 0.0,  // hi-hat/cymbal delta since 100 ms ago [-1,1]
    double TimeSinceAnyNote = 0.5,  // min(L,R) beats-since-last-note / 8, [0,1]
    double SongFraction     = 0.0,  // beat / totalBeats [0,1]
    // Labels
    bool HasNote   = false,
    int NoteHand   = -1,   // 0=left, 1=right, -1=none
    int NoteLane   = -1,
    int NoteRow    = -1,
    int NoteCutDir = -1,
    // Per-example gradient multiplier (default 1.0; synthetic examples use higher values)
    double Weight = 1.0
);

public sealed class TrainingExampleBuilder
{
    public IReadOnlyList<TrainingExample> Build(
        CanonicalBeatmap beatmap,
        AudioAnalysisResult audio)
    {
        Guard.NotNull(beatmap, nameof(beatmap));
        Guard.NotNull(audio, nameof(audio));

        double bpm       = audio.EstimatedBpm > 0 ? audio.EstimatedBpm : beatmap.Song.BeatsPerMinute;
        double frameRate = audio.FrameRateHz > 0 ? audio.FrameRateHz : 43.0;
        int    diffLevel = (int)beatmap.Difficulty.Difficulty;

        // Derive song length in beats
        double songDurationBeats;
        if (audio.DurationSeconds > 0)
            songDurationBeats = MathHelpers.SecondsToBeat(audio.DurationSeconds, bpm);
        else if (beatmap.Notes.Count > 0)
            songDurationBeats = beatmap.Notes[^1].Beat + 4.0;
        else
            return [];

        // Pre-sort beat times for quick nearest-beat lookup
        var beatTimes = audio.BeatTimesSeconds;

        var examples  = new List<TrainingExample>((int)(songDurationBeats * 4) + 8);
        CanonicalNote? lastLeft = null, lastRight = null;

        for (double beat = 0; beat < songDurationBeats; beat += 0.25)
        {
            double timeSeconds = MathHelpers.BeatToSeconds(beat, bpm);

            // Compute the true subdivision: whole beat=1.0, half=0.5, quarter=0.25
            double frac   = beat - Math.Floor(beat);
            double subdiv = frac < 0.01 ? 1.0
                          : Math.Abs(frac - 0.5) < 0.01 ? 0.5
                          : 0.25;

            // ---- Beat / rhythm features ----
            (double beatPhase, double beatStrength, int measureBeat) =
                ComputeBeatFeatures(beat, bpm, beatTimes);

            // ---- Audio features (available when audio was actually analyzed) ----
            double onsetStrength = GetOnsetStrength(timeSeconds, audio);
            double energy        = audio.GetEnergy(timeSeconds);
            double localNps      = beatmap.LocalNps(beat, 4.0);
            double sectionProg   = GetSectionProgress(timeSeconds, audio);
            double lowBand       = audio.GetLowBand(timeSeconds);
            double midBand       = audio.GetMidBand(timeSeconds);
            double highBand      = audio.GetHighBand(timeSeconds);
            double centroid      = audio.GetCentroid(timeSeconds);

            // Derived temporal features
            double prevTime      = Math.Max(0, timeSeconds - 0.1);
            double prevEnergy    = audio.GetEnergy(prevTime);
            double prevHighBand  = audio.GetHighBand(prevTime);
            double energyMax     = Math.Max(0.01, Math.Max(energy, prevEnergy));
            double energyDelta   = Math.Clamp((energy - prevEnergy) / energyMax, -1.0, 1.0);
            double highBandDelta = Math.Clamp(highBand - prevHighBand, -1.0, 1.0);
            double timeSinceAny  = Math.Min(
                lastLeft  != null ? beat - lastLeft.Beat  : 999.0,
                lastRight != null ? beat - lastRight.Beat : 999.0);
            timeSinceAny = Math.Clamp(timeSinceAny / 8.0, 0.0, 1.0);
            double songFraction  = songDurationBeats > 0
                ? Math.Clamp(beat / songDurationBeats, 0.0, 1.0) : 0.0;

            // ---- Find the note nearest to this beat (within ±0.13 beats) ----
            var note    = beatmap.Notes.FirstOrDefault(n => Math.Abs(n.Beat - beat) < 0.13);
            bool hasNote = note is not null;
            int noteHand = note?.Hand == NoteHand.Left ? 0
                         : note?.Hand == NoteHand.Right ? 1 : -1;

            examples.Add(new TrainingExample(
                Beat:                  beat,
                SubdivisionDenominator: subdiv,
                OnsetStrength:         onsetStrength,
                EnergyLevel:           energy,
                LocalNps:              localNps,
                SectionProgress:       sectionProg,
                BeatPhase:             beatPhase,
                BeatStrength:          beatStrength,
                MeasureBeat:           measureBeat,
                DifficultyLevel:       diffLevel,
                PreviousLeftLane:      lastLeft?.Lane  ?? 1,
                PreviousLeftRow:       lastLeft?.Row   ?? 1,
                PreviousLeftCutDir:    lastLeft  is not null ? (int)lastLeft.CutDirection  : -1,
                PreviousRightLane:     lastRight?.Lane ?? 2,
                PreviousRightRow:      lastRight?.Row  ?? 1,
                PreviousRightCutDir:   lastRight is not null ? (int)lastRight.CutDirection : -1,
                BeatsSinceLastLeft:    lastLeft  is not null ? beat - lastLeft.Beat  : 999,
                BeatsSinceLastRight:   lastRight is not null ? beat - lastRight.Beat : 999,
                LowBandEnergy:         lowBand,
                MidBandEnergy:         midBand,
                HighBandEnergy:        highBand,
                SpectralCentroid:      centroid,
                EnergyDelta:           energyDelta,
                HighBandDelta:         highBandDelta,
                TimeSinceAnyNote:      timeSinceAny,
                SongFraction:          songFraction,
                HasNote:               hasNote,
                NoteHand:              noteHand,
                NoteLane:              note?.Lane ?? -1,
                NoteRow:               note?.Row  ?? -1,
                NoteCutDir:            note is not null ? (int)note.CutDirection : -1
            ));

            if (note is not null)
            {
                if (note.Hand == NoteHand.Left)  lastLeft  = note;
                else                             lastRight = note;
            }
        }

        return examples;
    }

    // -------------------------------------------------------------------------
    // Beat / rhythm feature computation
    // -------------------------------------------------------------------------

    private static (double Phase, double Strength, int MeasureBeat) ComputeBeatFeatures(
        double beat, double bpm, IReadOnlyList<double> beatTimes)
    {
        // Fractional part of beat position → phase within current beat
        double phase = beat - Math.Floor(beat);   // 0-1

        // Measure beat: which 1-of-4 beats in a 4/4 measure
        int measureBeat = ((int)Math.Floor(beat) % 4) + 1;   // 1-4

        // Beat strength based on position within measure
        double strength = measureBeat switch
        {
            1 => 1.00,   // downbeat — strongest
            3 => 0.75,   // beat 3
            2 or 4 => 0.50,
            _ => 0.50
        };
        // Subdivisions are weaker
        if (phase > 0.01) strength *= 0.5;

        return (phase, strength, measureBeat);
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
}
