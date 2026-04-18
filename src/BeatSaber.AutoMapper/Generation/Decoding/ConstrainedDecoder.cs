using BeatSaber.AutoMapper.Canonical.Derived;

namespace BeatSaber.AutoMapper.Generation.Decoding;

internal sealed record BeamState(
    ImmutableList<CanonicalNote> Notes,
    SwingContext LeftCtx,
    SwingContext RightCtx,
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

        // Initialise beam with empty state
        var beam = new List<BeamState>
        {
            new BeamState(
                ImmutableList<CanonicalNote>.Empty,
                new SwingContext(NoteHand.Left),
                new SwingContext(NoteHand.Right),
                0.0)
        };

        var selector = new NoteAttributeSelector();

        foreach (var candidate in candidates)
        {
            var nextBeam = new List<BeamState>();

            foreach (var state in beam)
            {
                // Option 1: skip this candidate
                nextBeam.Add(state);

                // Build generation context with current state's placed notes
                var tempCtx = CloneContextWith(ctx, state);

                // Try placing a note
                var proposed = candidate;
                var note = selector.SelectAttributes(proposed, tempCtx);
                if (note is not null)
                {
                    var transition = ParityAnalyzer.ClassifyTransition(
                        GetLastNote(state, note.Hand),
                        note,
                        note.Hand == NoteHand.Left ? state.LeftCtx : state.RightCtx);

                    if (transition.Transition != ParityTransition.Invalid)
                    {
                        var newLeft = note.Hand == NoteHand.Left ? state.LeftCtx.Clone() : state.LeftCtx;
                        var newRight = note.Hand == NoteHand.Right ? state.RightCtx.Clone() : state.RightCtx;

                        if (note.Hand == NoteHand.Left) newLeft.Update(note);
                        else newRight.Update(note);

                        double delta = ScoreState(state, note, transition);
                        nextBeam.Add(new BeamState(
                            state.Notes.Add(note),
                            newLeft,
                            newRight,
                            state.Score + delta));
                    }
                }
            }

            // Prune to BeamWidth
            beam = nextBeam.OrderByDescending(s => s.Score).Take(BeamWidth).ToList();
        }

        // Return best path
        var best = beam.OrderByDescending(s => s.Score).First();
        return best.Notes.OrderBy(n => n.Beat).ToList();
    }

    private static CanonicalNote? GetLastNote(BeamState state, NoteHand hand) =>
        state.Notes.Where(n => n.Hand == hand).OrderByDescending(n => n.Beat).FirstOrDefault();

    private static GenerationContext CloneContextWith(GenerationContext original, BeamState state)
    {
        var clone = new GenerationContext
        {
            Song            = original.Song,
            AudioAnalysis   = original.AudioAnalysis,
            Settings        = original.Settings,
            Profile         = original.Profile,
            CandidateGrid   = original.CandidateGrid,
            PlacementScorer = original.PlacementScorer,
            Rng             = original.Rng
        };
        clone.PlacedNotes.AddRange(state.Notes);
        return clone;
    }

    private static double ScoreState(BeamState state, CanonicalNote note, ParityTransitionResult transition)
    {
        double score = 1.0; // base placement score
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
