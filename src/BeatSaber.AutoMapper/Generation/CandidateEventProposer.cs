using BeatSaber.AutoMapper.Audio.Features;
using BeatSaber.AutoMapper.Canonical.Derived;

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

        CanonicalNote? lastLeft = null, lastRight = null;
        foreach (var n in ctx.PlacedNotes)
        {
            if (n.Hand == NoteHand.Left)
            {
                if (lastLeft == null || n.Beat > lastLeft.Beat) lastLeft = n;
            }
            else
            {
                if (lastRight == null || n.Beat > lastRight.Beat) lastRight = n;
            }
        }

        var proposed  = new List<ProposedEvent>();
        double baseThreshold = ctx.PlacementScorer is not null
            ? ctx.Profile.Difficulty switch
            {
                DifficultyLevel.Easy       => 0.22,
                DifficultyLevel.Normal     => 0.18,
                DifficultyLevel.Hard       => 0.15,
                DifficultyLevel.Expert     => 0.12,
                DifficultyLevel.ExpertPlus => 0.10,
                _                          => 0.16
            }
            : 0.15;

        var scored = new List<ProposedEvent>(ctx.CandidateGrid.Length);
        var proposalHidden = ctx.GruHiddenState?.Clone();
        foreach (var candidate in ctx.CandidateGrid)
        {
            double score = ScoreCandidate(
                candidate,
                ctx,
                lastLeft,
                lastRight,
                ctx.LeftHandContext,
                ctx.RightHandContext,
                proposalHidden,
                out var neuralPred);
            var hand = ChooseSuggestedHand(neuralPred, ctx);
            scored.Add(new ProposedEvent(candidate, score, hand, neuralPred?.HandScore ?? score, neuralPred));
        }

        double threshold = DetermineAdaptiveThreshold(scored, ctx, baseThreshold);
        foreach (var ev in scored)
        {
            if (ev.PlacementScore < threshold)
                continue;

            proposed.Add(ev);
        }

        return ApplyDensityControl(proposed, ctx);
    }

    private static double DetermineAdaptiveThreshold(
        List<ProposedEvent> scored,
        GenerationContext ctx,
        double baseThreshold)
    {
        if (scored.Count == 0)
            return baseThreshold;

        double minThreshold = ctx.Profile.Difficulty switch
        {
            DifficultyLevel.Easy       => 0.05,
            DifficultyLevel.Normal     => 0.045,
            DifficultyLevel.Hard       => 0.04,
            DifficultyLevel.Expert     => 0.035,
            DifficultyLevel.ExpertPlus => 0.03,
            _                          => 0.04
        };

        double durationSeconds = Math.Max(1.0, ctx.AudioAnalysis.DurationSeconds);
        double estimatedFinalNotes = durationSeconds * ctx.Profile.TargetNps;
        int desiredProposals = (int)Math.Round(estimatedFinalNotes * 1.8);
        desiredProposals = Math.Clamp(desiredProposals, 24, Math.Max(24, (int)Math.Round(scored.Count * 0.35)));
        desiredProposals = Math.Min(desiredProposals, scored.Count);

        var sortedScores = scored
            .Select(e => e.PlacementScore)
            .OrderByDescending(s => s)
            .ToList();

        double quantileThreshold = sortedScores[desiredProposals - 1];
        return Math.Clamp(Math.Min(baseThreshold, quantileThreshold), minThreshold, baseThreshold);
    }

    internal static NeuralPlacementContext BuildNeuralContext(
        TimingCandidate candidate,
        GenerationContext ctx,
        CanonicalNote? lastLeft,
        CanonicalNote? lastRight,
        SwingContext leftCtx,
        SwingContext rightCtx,
        GruState? gruState)
    {
        double beatStrength = ComputeBeatStrength(candidate.Beat);
        int    measureBeat  = ((int)Math.Floor(candidate.Beat) % 4) + 1;
        double energy       = ctx.AudioAnalysis.GetEnergy(candidate.TimeSeconds);
        double phase        = candidate.Beat - Math.Floor(candidate.Beat);
        double secProg      = GetSectionProgress(candidate.TimeSeconds, ctx);
        double barPosition  = (candidate.Beat % 4) / 4.0;
        double localNps     = EstimateLocalNps(candidate.Beat, ctx);
        double prevTime      = Math.Max(0, candidate.TimeSeconds - 0.1);
        double prevEnergy    = ctx.AudioAnalysis.GetEnergy(prevTime);
        double prevHighBand  = ctx.AudioAnalysis.GetHighBand(prevTime);
        double energyMax     = Math.Max(0.01, Math.Max(energy, prevEnergy));
        double energyDelta   = Math.Clamp((energy - prevEnergy) / energyMax, -1.0, 1.0);
        double highBandDelta = Math.Clamp(ctx.AudioAnalysis.GetHighBand(candidate.TimeSeconds) - prevHighBand, -1.0, 1.0);
        double totalBeats    = MathHelpers.SecondsToBeat(ctx.AudioAnalysis.DurationSeconds, ctx.Song.BeatsPerMinute);
        double songFraction  = totalBeats > 0 ? Math.Clamp(candidate.Beat / totalBeats, 0.0, 1.0) : 0.0;
        double spectralFlux  = ctx.AudioAnalysis.GetSpectralFlux(candidate.TimeSeconds);
        double transient     = ctx.AudioAnalysis.GetTransient(candidate.TimeSeconds);
        double energyTrend4  = ctx.AudioAnalysis.GetMeanEnergyBeforeBeats(candidate.TimeSeconds, ctx.Song.BeatsPerMinute, 4);
        double energyTrend8  = ctx.AudioAnalysis.GetMeanEnergyBeforeBeats(candidate.TimeSeconds, ctx.Song.BeatsPerMinute, 8);
        int    sectionType   = GetSectionTypeIndex(candidate.TimeSeconds, ctx);
        double leftParity    = ParityStateFromSwingContext(leftCtx);
        double rightParity   = ParityStateFromSwingContext(rightCtx);
        var (beatsSinceSectionStart, beatsToSectionBoundary) =
            MappingFeatureEngineering.SectionBoundaryFeatures(ctx.AudioAnalysis.Sections, candidate.Beat);
        var (_, prev2Left)   = MappingFeatureEngineering.LastTwoForHand(ctx.PlacedNotes, NoteHand.Left);
        var (_, prev2Right)  = MappingFeatureEngineering.LastTwoForHand(ctx.PlacedNotes, NoteHand.Right);
        double lookaheadBeat   = Math.Min(candidate.Beat + 1.0, totalBeats);
        double lookaheadSec    = MathHelpers.BeatToSeconds(lookaheadBeat, ctx.Song.BeatsPerMinute);
        double lookaheadEnergy = ctx.AudioAnalysis.GetEnergy(lookaheadSec);
        double lookaheadOnset  = ctx.AudioAnalysis.GetOnsetStrength(lookaheadSec);
        double futureEnergy4   = ctx.AudioAnalysis.GetMeanEnergyAfterBeats(candidate.TimeSeconds, ctx.Song.BeatsPerMinute, 4);
        double futureEnergy8   = ctx.AudioAnalysis.GetMeanEnergyAfterBeats(candidate.TimeSeconds, ctx.Song.BeatsPerMinute, 8);
        double futureEnergy16  = ctx.AudioAnalysis.GetMeanEnergyAfterBeats(candidate.TimeSeconds, ctx.Song.BeatsPerMinute, 16);
        double futureOnset4    = ctx.AudioAnalysis.GetMeanOnsetAfterBeats(candidate.TimeSeconds, ctx.Song.BeatsPerMinute, 4);
        double futureOnset8    = ctx.AudioAnalysis.GetMeanOnsetAfterBeats(candidate.TimeSeconds, ctx.Song.BeatsPerMinute, 8);
        double futureOnset16   = ctx.AudioAnalysis.GetMeanOnsetAfterBeats(candidate.TimeSeconds, ctx.Song.BeatsPerMinute, 16);
        double recentChordRate4 = MappingFeatureEngineering.RecentChordRate(ctx.PlacedNotes, candidate.Beat, 4.0);
        double recentOffbeatRate4 = MappingFeatureEngineering.RecentOffbeatRate(ctx.PlacedNotes, candidate.Beat, 4.0);
        double recentStreamRate4 = MappingFeatureEngineering.RecentStreamRate(ctx.PlacedNotes, candidate.Beat, 4.0);
        double recentAlternation8 = MappingFeatureEngineering.RecentAlternation(ctx.PlacedNotes, 8);
        double recentHandBalance8 = MappingFeatureEngineering.RecentHandBalance(ctx.PlacedNotes, candidate.Beat, 8.0);
        double consecutiveSameHandCount = MappingFeatureEngineering.ConsecutiveSameHandCount(ctx.PlacedNotes);
        double beatsSinceLastAny = MappingFeatureEngineering.BeatsSinceLastAny(ctx.PlacedNotes, candidate.Beat);
        double notesAtCurrentBeatSoFar = MappingFeatureEngineering.NotesAtCurrentBeatSoFar(ctx.PlacedNotes, candidate.Beat);
        double interHandLaneDistance = MappingFeatureEngineering.InterHandLaneDistance(lastLeft, lastRight);
        double interHandRowDistance = MappingFeatureEngineering.InterHandRowDistance(lastLeft, lastRight);
        double handsCrossedFlag = MappingFeatureEngineering.HandsCrossedFlag(lastLeft, lastRight);
        double leftRecentTravel = MappingFeatureEngineering.RecentTravel(prev2Left, lastLeft);
        double rightRecentTravel = MappingFeatureEngineering.RecentTravel(prev2Right, lastRight);
        double recentLaneSpan4 = MappingFeatureEngineering.RecentLaneSpan(ctx.PlacedNotes, candidate.Beat, 4.0);
        double recentRowSpan4 = MappingFeatureEngineering.RecentRowSpan(ctx.PlacedNotes, candidate.Beat, 4.0);
        double phraseBeatPhase32 = candidate.Beat % 32.0;
        double phraseProgress32 = phraseBeatPhase32 / 32.0;
        double beatsSincePhraseStart32 = phraseBeatPhase32;
        double beatsToPhraseBoundary32 = 32.0 - phraseBeatPhase32;
        double currentBeatVisionBlockRisk = MappingFeatureEngineering.CurrentBeatVisionBlockRisk(ctx.PlacedNotes, candidate.Beat);
        double recentVisionBlockRate8 = MappingFeatureEngineering.RecentVisionBlockRate(ctx.PlacedNotes, candidate.Beat, 8.0);
        double leftParityBreakRate8 = MappingFeatureEngineering.RecentParityBreakRate(ctx.PlacedNotes, NoteHand.Left, 8);
        double rightParityBreakRate8 = MappingFeatureEngineering.RecentParityBreakRate(ctx.PlacedNotes, NoteHand.Right, 8);
        double resetPressure = Math.Max(
            MappingFeatureEngineering.ImmediateResetPressure(lastLeft, candidate.Beat, ctx.Profile.Difficulty),
            MappingFeatureEngineering.ImmediateResetPressure(lastRight, candidate.Beat, ctx.Profile.Difficulty));
        double recentRestRatio8 = MappingFeatureEngineering.RecentRestRatio(ctx.PlacedNotes, candidate.Beat, 8.0);

        return new NeuralPlacementContext
        {
            OnsetStrength    = ctx.AudioAnalysis.GetOnsetStrength(candidate.TimeSeconds),
            EnergyLevel      = energy,
            SpectralFlux     = spectralFlux,
            TransientStrength = transient,
            LowBandEnergy    = ctx.AudioAnalysis.GetLowBand(candidate.TimeSeconds),
            MidBandEnergy    = ctx.AudioAnalysis.GetMidBand(candidate.TimeSeconds),
            HighBandEnergy   = ctx.AudioAnalysis.GetHighBand(candidate.TimeSeconds),
            SpectralCentroid = ctx.AudioAnalysis.GetCentroid(candidate.TimeSeconds),
            EnergyDelta      = energyDelta,
            HighBandDelta    = highBandDelta,
            EnergyTrend4     = energyTrend4,
            EnergyTrend8     = energyTrend8,
            LookaheadEnergy  = lookaheadEnergy,
            LookaheadOnset   = lookaheadOnset,
            BeatPhase        = phase,
            Subdiv           = candidate.SubdivisionDenominator,
            BeatStrength     = beatStrength,
            MeasureBeat      = measureBeat,
            SongFraction     = songFraction,
            SectionProgress  = secProg,
            BarPosition      = barPosition,
            SectionTypeIndex = sectionType,
            DifficultyLevel  = (int)ctx.Profile.Difficulty,
            LocalNps         = localNps,
            PrevLeftLane     = lastLeft?.Lane  ?? 1,
            PrevLeftRow      = lastLeft?.Row   ?? 1,
            PrevLeftCutDir   = lastLeft  != null ? (int)lastLeft.CutDirection  : -1,
            PrevRightLane    = lastRight?.Lane ?? 2,
            PrevRightRow     = lastRight?.Row  ?? 1,
            PrevRightCutDir  = lastRight != null ? (int)lastRight.CutDirection : -1,
            LeftParityState  = leftParity,
            RightParityState = rightParity,
            BeatsSinceLastLeft  = lastLeft  != null ? candidate.Beat - lastLeft.Beat  : 999,
            BeatsSinceLastRight = lastRight != null ? candidate.Beat - lastRight.Beat : 999,
            NoteHandHint     = 0.5,
            FutureEnergy4    = futureEnergy4,
            FutureEnergy8    = futureEnergy8,
            FutureEnergy16   = futureEnergy16,
            FutureOnset4     = futureOnset4,
            FutureOnset8     = futureOnset8,
            FutureOnset16    = futureOnset16,
            BeatsSinceSectionStart = beatsSinceSectionStart,
            BeatsToSectionBoundary = beatsToSectionBoundary,
            RecentChordRate4  = recentChordRate4,
            RecentOffbeatRate4 = recentOffbeatRate4,
            RecentStreamRate4 = recentStreamRate4,
            RecentAlternation8 = recentAlternation8,
            RecentHandBalance8 = recentHandBalance8,
            ConsecutiveSameHandCount = consecutiveSameHandCount,
            BeatsSinceLastAny = beatsSinceLastAny,
            NotesAtCurrentBeatSoFar = notesAtCurrentBeatSoFar,
            InterHandLaneDistance = interHandLaneDistance,
            InterHandRowDistance = interHandRowDistance,
            HandsCrossedFlag = handsCrossedFlag,
            LeftRecentTravel = leftRecentTravel,
            RightRecentTravel = rightRecentTravel,
            RecentLaneSpan4 = recentLaneSpan4,
            RecentRowSpan4 = recentRowSpan4,
            PhraseBeatPhase32 = phraseBeatPhase32,
            PhraseProgress32 = phraseProgress32,
            BeatsSincePhraseStart32 = beatsSincePhraseStart32,
            BeatsToPhraseBoundary32 = beatsToPhraseBoundary32,
            CurrentBeatVisionBlockRisk = currentBeatVisionBlockRisk,
            RecentVisionBlockRate8 = recentVisionBlockRate8,
            LeftParityBreakRate8 = leftParityBreakRate8,
            RightParityBreakRate8 = rightParityBreakRate8,
            ResetPressure = resetPressure,
            RecentRestRatio8 = recentRestRatio8,
            GruHiddenState   = gruState,
        };
    }

    private double ScoreCandidate(
        TimingCandidate candidate,
        GenerationContext ctx,
        CanonicalNote? lastLeft,
        CanonicalNote? lastRight,
        SwingContext leftCtx,
        SwingContext rightCtx,
        GruState? gruState,
        out NeuralMapPrediction? neuralPred)
    {
        double localNps  = EstimateLocalNps(candidate.Beat, ctx);
        double targetNps = ctx.Profile.TargetNps;

            if (ctx.MultiTaskModel is not null || ctx.PlacementScorer is not null)
        {
            var nctx = BuildNeuralContext(candidate, ctx, lastLeft, lastRight, leftCtx, rightCtx, gruState);
            double score;
            if (ctx.MultiTaskModel is not null)
            {
                var pred = ctx.MultiTaskModel.PredictAll(in nctx);
                neuralPred = pred;
                score = pred.PlacementScore;
            }
            else
            {
                neuralPred = null;
                score = ctx.PlacementScorer!.ScorePlacement(in nctx);
            }

            score = ApplyMusicalityPrior(score, in nctx, ctx);

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

    private static double ApplyMusicalityPrior(double score, in NeuralPlacementContext ctx, GenerationContext genCtx)
    {
        double onset = Math.Clamp(ctx.OnsetStrength, 0.0, 1.0);
        double transient = Math.Clamp(ctx.TransientStrength, 0.0, 1.0);
        double energy = Math.Clamp(ctx.EnergyLevel, 0.0, 1.0);
        double emphasis = 0.55 * onset + 0.30 * transient + 0.15 * energy;

        if (emphasis < 0.12)
            score *= 0.18;
        else if (emphasis < 0.20)
            score *= 0.40;
        else if (emphasis < 0.30)
            score *= 0.68;

        if (ctx.SectionTypeIndex is 0 or 6 && emphasis < 0.28) // intro/outro
            score *= 0.78;

        if (ctx.SectionTypeIndex == 1 && emphasis < 0.22) // verse
            score *= 0.88;

        return score;
    }

    private static NoteHand ChooseSuggestedHand(NeuralMapPrediction? prediction, GenerationContext ctx)
    {
        if (!prediction.HasValue)
            return NoteHand.Left;

        double handScore = prediction.Value.HandScore;
        double jointRightMass = 0.5;
        if (prediction.Value.HandLaneProbs is { Length: 8 } handLane)
            jointRightMass = handLane[4] + handLane[5] + handLane[6] + handLane[7];

        double blended = 0.40 * handScore + 0.60 * jointRightMass;
        double recentRightBias = MappingFeatureEngineering.RecentHandBalance(ctx.PlacedNotes, ctx.PlacedNotes.Count > 0 ? ctx.PlacedNotes[^1].Beat : 0, 8.0);
        double handBalanceAdjust = (0.5 - recentRightBias) * 0.18;
        double adjusted = Math.Clamp(blended + handBalanceAdjust, 0.0, 1.0);

        return adjusted >= 0.5 ? NoteHand.Right : NoteHand.Left;
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

    /// <summary>
    /// Converts the current parity from a swing context to the scalar used as a model feature.
    /// Forehand position (forehand expected next) → 0.0; Backhand position → 1.0; unknown → 0.5.
    /// </summary>
    private static double ParityStateFromSwingContext(SwingContext swingCtx) =>
        swingCtx.LastCutDirection is null ? 0.5
        : swingCtx.CurrentParity == ParityClass.Backhand ? 1.0 : 0.0;

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

    private static int GetSectionTypeIndex(double timeSeconds, GenerationContext ctx)
    {
        if (ctx.AudioAnalysis.Sections.Count == 0) return 7;
        SectionMarker? current = null;
        foreach (var s in ctx.AudioAnalysis.Sections)
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
        if (events.Count <= 1)
            return events;

        double windowBeats = 4.0;
        double halfWindow  = windowBeats / 2;
        double windowSec   = MathHelpers.BeatToSeconds(windowBeats, ctx.Song.BeatsPerMinute);
        double maxNps      = ctx.Profile.MaxNps;

        // Sort by score descending so highest-value candidates survive density pruning first
        var sorted = events.OrderByDescending(e => e.PlacementScore).ToList();
        var kept   = new List<ProposedEvent>();
        var keptBeats = new List<double>(sorted.Count);

        foreach (var ev in sorted)
        {
            // Energy-proportional density: louder sections can sustain more notes.
            // Scale the max NPS between 60 % and 140 % of the difficulty cap.
            double energy       = ctx.AudioAnalysis.GetEnergy(ev.Timing.TimeSeconds);
            double energyFactor = 0.6 + 0.8 * energy;          // [0.6, 1.4]
            double effectiveMax = maxNps * energyFactor;

            int start = LowerBound(keptBeats, ev.Timing.Beat - halfWindow);
            int end = UpperBound(keptBeats, ev.Timing.Beat + halfWindow);
            double countInWindow = end - start;

            if (windowSec > 0 && (countInWindow + 1) / windowSec <= effectiveMax)
            {
                kept.Add(ev);
                keptBeats.Insert(UpperBound(keptBeats, ev.Timing.Beat), ev.Timing.Beat);
            }
        }

        // Restore chronological order for the beam decoder
        return kept.OrderBy(e => e.Timing.Beat).ToList();
    }

    private static int LowerBound(List<double> values, double target)
    {
        int lo = 0;
        int hi = values.Count;
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (values[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private static int UpperBound(List<double> values, double target)
    {
        int lo = 0;
        int hi = values.Count;
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (values[mid] <= target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}
