using BeatSaber.AutoMapper.Canonical.Derived;
using BeatSaber.AutoMapper.Training.Patterns;
using BeatSaber.AutoMapper.Validation;

namespace BeatSaber.AutoMapper.Generation.Decoding;

internal sealed record BeamState(
    ImmutableList<CanonicalNote> Notes,
    SwingContext LeftCtx,
    SwingContext RightCtx,
    CanonicalNote? LastLeft,
    CanonicalNote? LastRight,
    GruState? Hidden,
    double Score
);

/// <summary>Beam-search decoder that ensures parity validity across the note sequence.</summary>
public sealed class ConstrainedDecoder
{
    public int BeamWidth { get; set; } = 12;

    public IReadOnlyList<CanonicalNote> Decode(
        IReadOnlyList<ProposedEvent> candidates,
        GenerationContext ctx)
    {
        Guard.NotNull(candidates, nameof(candidates));
        Guard.NotNull(ctx, nameof(ctx));

        var beam = new List<BeamState>
        {
            new(
                ImmutableList<CanonicalNote>.Empty,
                new SwingContext(NoteHand.Left),
                new SwingContext(NoteHand.Right),
                null,
                null,
                ctx.GruHiddenState?.Clone(),
                0.0)
        };

        var selector = new NoteAttributeSelector();

        foreach (var candidate in candidates)
        {
            var nextBeam = new List<BeamState>();
            var batchedPredictions = ctx.MultiTaskModel is IBatchedMultiTaskPlacementModel batchedModel
                ? PredictBeamBatch(candidate, beam, ctx, batchedModel)
                : null;
            int beamPredictionIndex = 0;

            foreach (var state in beam)
            {
                var tempCtx = CloneContextWith(ctx, state);
                var updatedHidden = state.Hidden?.Clone();
                NeuralMapPrediction? beamPred = null;
                double placementScore = candidate.PlacementScore;

                if (ctx.MultiTaskModel is not null)
                {
                    if (batchedPredictions is not null)
                    {
                        var prediction = batchedPredictions[beamPredictionIndex++];
                        beamPred = prediction.Prediction;
                        updatedHidden = prediction.Hidden;
                        placementScore = prediction.Prediction.PlacementScore;
                    }
                    else
                    {
                        var nctx = CandidateEventProposer.BuildNeuralContext(
                            candidate.Timing,
                            tempCtx,
                            state.LastLeft,
                            state.LastRight,
                            state.LeftCtx,
                            state.RightCtx,
                            updatedHidden);
                        beamPred = ctx.MultiTaskModel.PredictAll(in nctx);
                        placementScore = beamPred.Value.PlacementScore;
                    }
                }

                nextBeam.Add(state with
                {
                    Hidden = updatedHidden?.Clone(),
                    Score = state.Score - placementScore * 0.15
                });

                if (placementScore < 0.18)
                    continue;

                foreach (var hand in GetHandOrder(candidate, beamPred))
                {
                    var proposed = candidate with
                    {
                        SuggestedHand = hand,
                        PlacementScore = placementScore,
                        HandScore = beamPred?.HandScore ?? candidate.HandScore,
                        NeuralPrediction = beamPred
                    };

                    var note = selector.SelectAttributes(proposed, tempCtx);
                    if (note is null)
                        continue;

                    var transition = ParityAnalyzer.ClassifyTransition(
                        note.Hand == NoteHand.Left ? state.LastLeft : state.LastRight,
                        note,
                        note.Hand == NoteHand.Left ? state.LeftCtx : state.RightCtx);

                    if (transition.Transition == ParityTransition.Invalid)
                        continue;

                    var newLeft = state.LeftCtx.Clone();
                    var newRight = state.RightCtx.Clone();
                    if (note.Hand == NoteHand.Left) newLeft.Update(note);
                    else newRight.Update(note);

                    var singleState = new BeamState(
                        state.Notes.Add(note),
                        newLeft,
                        newRight,
                        note.Hand == NoteHand.Left ? note : state.LastLeft,
                        note.Hand == NoteHand.Right ? note : state.LastRight,
                        updatedHidden?.Clone(),
                        state.Score + ScoreState(proposed, note, transition));
                    nextBeam.Add(singleState);

                    if (!ShouldAttemptChord(candidate, proposed, beamPred, tempCtx))
                        continue;

                    var chordCtx = CloneContextWith(ctx, singleState);
                    var oppositeHand = hand == NoteHand.Left ? NoteHand.Right : NoteHand.Left;
                    var chordProposed = proposed with { SuggestedHand = oppositeHand };
                    var chordNote = selector.SelectAttributes(chordProposed, chordCtx);
                    if (chordNote is null || chordNote.Hand == note.Hand)
                        continue;
                    if (PlayabilityHeuristics.IsHandclap(note, chordNote) ||
                        PlayabilityHeuristics.IsVisionBlock(note, chordNote) ||
                        PlayabilityHeuristics.IsHitboxPath(note, chordNote))
                        continue;

                    var chordTransition = ParityAnalyzer.ClassifyTransition(
                        chordNote.Hand == NoteHand.Left ? singleState.LastLeft : singleState.LastRight,
                        chordNote,
                        chordNote.Hand == NoteHand.Left ? singleState.LeftCtx : singleState.RightCtx);
                    if (chordTransition.Transition == ParityTransition.Invalid)
                        continue;

                    var chordLeft = singleState.LeftCtx.Clone();
                    var chordRight = singleState.RightCtx.Clone();
                    if (chordNote.Hand == NoteHand.Left) chordLeft.Update(chordNote);
                    else chordRight.Update(chordNote);

                    nextBeam.Add(new BeamState(
                        singleState.Notes.Add(chordNote),
                        chordLeft,
                        chordRight,
                        chordNote.Hand == NoteHand.Left ? chordNote : singleState.LastLeft,
                        chordNote.Hand == NoteHand.Right ? chordNote : singleState.LastRight,
                        updatedHidden?.Clone(),
                        singleState.Score + ScoreState(chordProposed, chordNote, chordTransition) + ScoreChordBonus(candidate, tempCtx)));
                }
            }

            beam = nextBeam.OrderByDescending(s => s.Score).Take(BeamWidth).ToList();
        }

        var best = beam.OrderByDescending(s => s.Score).First();
        return best.Notes.OrderBy(n => n.Beat).ToList();
    }

    private static IReadOnlyList<(NeuralMapPrediction Prediction, GruState? Hidden)> PredictBeamBatch(
        ProposedEvent candidate,
        IReadOnlyList<BeamState> beam,
        GenerationContext ctx,
        IBatchedMultiTaskPlacementModel batchedModel)
    {
        var contexts = new NeuralPlacementContext[beam.Count];
        var hiddenStates = new GruState?[beam.Count];

        for (int i = 0; i < beam.Count; i++)
        {
            var state = beam[i];
            var tempCtx = CloneContextWith(ctx, state);
            var updatedHidden = state.Hidden?.Clone();
            hiddenStates[i] = updatedHidden;

            contexts[i] = CandidateEventProposer.BuildNeuralContext(
                candidate.Timing,
                tempCtx,
                state.LastLeft,
                state.LastRight,
                state.LeftCtx,
                state.RightCtx,
                updatedHidden);
        }

        var predictions = batchedModel.PredictAllBatch(contexts);
        var results = new (NeuralMapPrediction Prediction, GruState? Hidden)[beam.Count];
        for (int i = 0; i < beam.Count; i++)
            results[i] = (predictions[i], hiddenStates[i]);
        return results;
    }

    private static NoteHand[] GetHandOrder(ProposedEvent candidate, NeuralMapPrediction? prediction)
    {
        if (!prediction.HasValue)
            return [candidate.SuggestedHand];

        double rightMass = prediction.Value.HandLaneProbs is { Length: 8 } handLane
            ? handLane[4] + handLane[5] + handLane[6] + handLane[7]
            : prediction.Value.HandScore;

        return rightMass >= 0.5
            ? [NoteHand.Right, NoteHand.Left]
            : [NoteHand.Left, NoteHand.Right];
    }

    private static bool ShouldAttemptChord(
        ProposedEvent candidate,
        ProposedEvent proposed,
        NeuralMapPrediction? prediction,
        GenerationContext ctx)
    {
        double placement = proposed.PlacementScore;
        if (placement < 0.48)
            return false;

        double onset = Math.Clamp(candidate.Timing.OnsetStrength, 0.0, 1.0);
        double energy = Math.Clamp(candidate.Timing.EnergyLevel, 0.0, 1.0);
        double beatPhase = candidate.Timing.Beat - Math.Floor(candidate.Timing.Beat);
        bool strongBeat = beatPhase < 0.01 || Math.Abs(beatPhase - 0.5) < 0.01;
        double recentChordRate = MappingFeatureEngineering.RecentChordRate(ctx.PlacedNotes, candidate.Timing.Beat, 4.0);
        double currentBeatCount = MappingFeatureEngineering.NotesAtCurrentBeatSoFar(ctx.PlacedNotes, candidate.Timing.Beat);
        if (currentBeatCount >= 1)
            return false;
        if (recentChordRate > 0.18)
            return false;

        double rightMass = prediction.HasValue && prediction.Value.HandLaneProbs is { Length: 8 } handLane
            ? handLane[4] + handLane[5] + handLane[6] + handLane[7]
            : proposed.HandScore;
        PatternType patternType = PredictPatternType(prediction);
        if (patternType is PatternType.Stream or PatternType.Anchor or PatternType.Reset)
            return false;
        double handAmbiguity = 1.0 - Math.Abs(rightMass - 0.5) * 2.0;

        double propensity = 0.38 * placement
            + 0.20 * onset
            + 0.12 * energy
            + 0.16 * recentChordRate
            + 0.14 * handAmbiguity;

        if (strongBeat)
            propensity += 0.08;

        propensity -= ctx.Profile.Difficulty switch
        {
            DifficultyLevel.Easy => 0.28,
            DifficultyLevel.Normal => 0.20,
            DifficultyLevel.Hard => 0.10,
            _ => 0.0
        };

        return propensity >= 0.50;
    }

    private static PatternType PredictPatternType(NeuralMapPrediction? prediction)
    {
        if (!prediction.HasValue || prediction.Value.PatternTypeProbs is not { Length: >= 6 } probs)
            return PatternType.Isolated;
        int best = 0;
        for (int i = 1; i < probs.Length; i++)
        {
            if (probs[i] > probs[best])
                best = i;
        }
        return (PatternType)best;
    }

    private static double ScoreChordBonus(ProposedEvent candidate, GenerationContext ctx)
    {
        double onset = Math.Clamp(candidate.Timing.OnsetStrength, 0.0, 1.0);
        double recentChordRate = MappingFeatureEngineering.RecentChordRate(ctx.PlacedNotes, candidate.Timing.Beat, 4.0);
        double phase = candidate.Timing.Beat - Math.Floor(candidate.Timing.Beat);
        bool strongBeat = phase < 0.01 || Math.Abs(phase - 0.5) < 0.01;

        double bonus = 0.12 + 0.20 * onset + 0.12 * recentChordRate;
        if (strongBeat)
            bonus += 0.08;
        return bonus;
    }

    private static GenerationContext CloneContextWith(GenerationContext original, BeamState state)
    {
        var clone = new GenerationContext
        {
            Song             = original.Song,
            AudioAnalysis    = original.AudioAnalysis,
            Settings         = original.Settings,
            Profile          = original.Profile,
            CandidateGrid    = original.CandidateGrid,
            PlacementScorer  = original.PlacementScorer,
            MultiTaskModel   = original.MultiTaskModel,
            NeuralModel      = original.NeuralModel,
            LeftHandContext  = state.LeftCtx.Clone(),
            RightHandContext = state.RightCtx.Clone(),
            Rng              = original.Rng
        };
        clone.PlacedNotes.AddRange(state.Notes);
        return clone;
    }

    private static double ScoreState(ProposedEvent proposed, CanonicalNote note, ParityTransitionResult transition)
    {
        double score = proposed.PlacementScore * 1.2;
        if (proposed.NeuralPrediction.HasValue)
        {
            double handPrior = note.Hand == NoteHand.Right
                ? proposed.NeuralPrediction.Value.HandScore
                : 1.0 - proposed.NeuralPrediction.Value.HandScore;
            score += (handPrior - 0.5) * 0.4;
        }

        score += transition.Transition switch
        {
            ParityTransition.GoodFlow => 0.5,
            ParityTransition.Acceptable => 0.2,
            ParityTransition.RecoveryRequired => -0.2,
            ParityTransition.AwkwardReset => -0.4,
            ParityTransition.ParityBreak => -1.0,
            _ => -2.0
        };
        return score;
    }
}
