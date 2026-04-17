using BeatSaber.AutoMapper.Audio.Features;

namespace BeatSaber.AutoMapper.Generation;

public sealed record ProposedEvent(
    TimingCandidate Timing,
    double PlacementScore,
    NoteHand SuggestedHand,
    double HandScore,
    NeuralMapPrediction? NeuralPrediction = null  // cached full prediction for attribute selection
);

public sealed class CandidateEventProposer
{
    public IReadOnlyList<ProposedEvent> ProposeEvents(GenerationContext ctx)
    {
        Guard.NotNull(ctx, nameof(ctx));

        // Pre-compute the most-recent note per hand from already-placed notes.
        // This sequential state is passed to the neural context so the network
        // can learn phrasing patterns (e.g., left follows right, direction flow).
        CanonicalNote? lastLeft = null, lastRight = null;
        foreach (var n in ctx.PlacedNotes)
        {
            if (n.Hand == NoteHand.Left  && (lastLeft  == null || n.Beat > lastLeft.Beat))  lastLeft  = n;
            if (n.Hand == NoteHand.Right && (lastRight == null || n.Beat > lastRight.Beat)) lastRight = n;
        }

        var proposed  = new List<ProposedEvent>();
        bool nextLeft = true;

        // Learned model: sigmoid output centred at 0.5; heuristic: ~0.15 floor.
        double threshold = ctx.PlacementScorer is not null ? 0.5 : 0.15;

        foreach (var candidate in ctx.CandidateGrid)
        {
            double score = ScoreCandidate(candidate, ctx, lastLeft, lastRight,
                                          out var neuralPred);
            if (score < threshold) continue;

            var hand = nextLeft ? NoteHand.Left : NoteHand.Right;
            nextLeft = !nextLeft;

            proposed.Add(new ProposedEvent(candidate, score, hand, score, neuralPred));
        }

        return ApplyDensityControl(proposed, ctx);
    }

    private double ScoreCandidate(TimingCandidate candidate, GenerationContext ctx,
                                  CanonicalNote? lastLeft, CanonicalNote? lastRight,
                                  out NeuralMapPrediction? neuralPred)
    {
        double localNps  = EstimateLocalNps(candidate.Beat, ctx);
        double targetNps = ctx.Profile.TargetNps;

        if (ctx.NeuralModel is not null || ctx.PlacementScorer is not null)
        {
            double beatStrength = ComputeBeatStrength(candidate.Beat);
            int    measureBeat  = ((int)Math.Floor(candidate.Beat) % 4) + 1;
            double energy       = ctx.AudioAnalysis.GetEnergy(candidate.TimeSeconds);
            double phase        = candidate.Beat - Math.Floor(candidate.Beat);
            double secProg      = GetSectionProgress(candidate.TimeSeconds, ctx);

            var nctx = new NeuralPlacementContext
            {
                Onset           = candidate.OnsetStrength,
                Energy          = energy,
                Subdiv          = candidate.SubdivisionDenominator,
                LocalNps        = localNps,
                BeatStrength    = beatStrength,
                MeasureBeat     = measureBeat,
                DifficultyLevel = (int)ctx.Profile.Difficulty,
                BeatPhase       = phase,
                SectionProgress = secProg,
                PrevLeftLane    = lastLeft?.Lane  ?? 1,
                PrevLeftRow     = lastLeft?.Row   ?? 1,
                PrevLeftCutDir  = lastLeft  != null ? (int)lastLeft.CutDirection  : -1,
                PrevRightLane   = lastRight?.Lane ?? 2,
                PrevRightRow    = lastRight?.Row  ?? 1,
                PrevRightCutDir = lastRight != null ? (int)lastRight.CutDirection : -1,
                BeatsSinceLastLeft  = lastLeft  != null ? candidate.Beat - lastLeft.Beat  : 999,
                BeatsSinceLastRight = lastRight != null ? candidate.Beat - lastRight.Beat : 999,
                LowBandEnergy    = ctx.AudioAnalysis.GetLowBand(candidate.TimeSeconds),
                MidBandEnergy    = ctx.AudioAnalysis.GetMidBand(candidate.TimeSeconds),
                HighBandEnergy   = ctx.AudioAnalysis.GetHighBand(candidate.TimeSeconds),
                SpectralCentroid = ctx.AudioAnalysis.GetCentroid(candidate.TimeSeconds),
            };

            double score;
            if (ctx.NeuralModel is not null)
            {
                var pred = ctx.NeuralModel.PredictAll(in nctx);
                neuralPred = pred;
                score = pred.PlacementScore;
            }
            else
            {
                neuralPred = null;
                score = ctx.PlacementScorer!.ScorePlacement(in nctx);
            }

            if (localNps > targetNps)
                score *= Math.Max(0.1, 1.0 - (localNps - targetNps) / targetNps * 0.3);

            return score;
        }

        neuralPred = null;
        return HeuristicScore(candidate, localNps, targetNps, ctx);
    }

    private double HeuristicScore(TimingCandidate candidate, double localNps,
                                  double targetNps, GenerationContext ctx)
    {
        double score = candidate.OnsetStrength * 0.5;

        score -= candidate.SubdivisionDenominator switch
        {
            1.0    => 0.0,
            0.5    => 0.05,
            0.25   => 0.1,
            _      => 0.2
        };

        var section = FindSection(candidate.Beat, ctx);
        if (section is not null)
        {
            score += section.Type switch
            {
                SectionType.Chorus => 0.1,
                SectionType.Drop   => 0.15,
                SectionType.Verse  => 0.05,
                _                  => 0.0
            };
        }

        if (localNps > targetNps)
            score -= (localNps - targetNps) / targetNps * 0.3;

        return Math.Max(0, score);
    }

    // Beat strength based solely on beat position within a 4/4 measure
    private static double ComputeBeatStrength(double beat)
    {
        double phase       = beat - Math.Floor(beat);
        int    measureBeat = ((int)Math.Floor(beat) % 4) + 1;
        double strength    = measureBeat switch { 1 => 1.00, 3 => 0.75, _ => 0.50 };
        if (phase > 0.01) strength *= 0.5;
        return strength;
    }

    private static double EstimateLocalNps(double beat, GenerationContext ctx)
    {
        double half      = 2.0;
        int    count     = ctx.PlacedNotes.Count(n => n.Beat >= beat - half && n.Beat <= beat + half);
        double windowSec = MathHelpers.BeatToSeconds(half * 2.0, ctx.Song.BeatsPerMinute);
        return windowSec > 0 ? count / windowSec : 0;
    }

    private static double GetSectionProgress(double timeSeconds, GenerationContext ctx)
    {
        var sections = ctx.AudioAnalysis.Sections;
        if (sections.Count == 0) return 0;
        int idx = 0;
        for (int i = 0; i < sections.Count; i++)
            if (sections[i].TimeSeconds <= timeSeconds) idx = i;
        double start = sections[idx].TimeSeconds;
        double end   = idx + 1 < sections.Count
            ? sections[idx + 1].TimeSeconds
            : ctx.AudioAnalysis.DurationSeconds;
        double span  = end - start;
        return span > 0 ? (timeSeconds - start) / span : 0;
    }

    private static SectionMarker? FindSection(double beat, GenerationContext ctx)
    {
        SectionMarker? last = null;
        foreach (var s in ctx.AudioAnalysis.Sections)
        {
            if (s.Beat <= beat) last = s;
            else break;
        }
        return last;
    }

    private static IReadOnlyList<ProposedEvent> ApplyDensityControl(
        List<ProposedEvent> events, GenerationContext ctx)
    {
        double windowBeats = 4.0;
        double windowSec   = MathHelpers.BeatToSeconds(windowBeats, ctx.Song.BeatsPerMinute);
        double maxNps      = ctx.Profile.MaxNps;

        // Sort by score descending so highest-value candidates survive density pruning first
        var sorted = events.OrderByDescending(e => e.PlacementScore).ToList();
        var kept   = new List<ProposedEvent>();

        foreach (var ev in sorted)
        {
            // Energy-proportional density: louder sections can sustain more notes.
            // Scale the max NPS between 60 % and 140 % of the difficulty cap.
            double energy       = ctx.AudioAnalysis.GetEnergy(ev.Timing.TimeSeconds);
            double energyFactor = 0.6 + 0.8 * energy;          // [0.6, 1.4]
            double effectiveMax = maxNps * energyFactor;

            double countInWindow = kept.Count(k =>
                Math.Abs(k.Timing.Beat - ev.Timing.Beat) <= windowBeats / 2);

            if (windowSec > 0 && (countInWindow + 1) / windowSec <= effectiveMax)
                kept.Add(ev);
        }

        // Restore chronological order for the beam decoder
        return kept.OrderBy(e => e.Timing.Beat).ToList();
    }
}
