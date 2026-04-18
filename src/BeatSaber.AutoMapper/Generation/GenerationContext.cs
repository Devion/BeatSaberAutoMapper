using BeatSaber.AutoMapper.Audio;
using BeatSaber.AutoMapper.Audio.Features;
using BeatSaber.AutoMapper.Canonical.Derived;

namespace BeatSaber.AutoMapper.Generation;

public sealed class GenerationContext
{
    public required SongMetadata Song { get; init; }
    public required AudioAnalysisResult AudioAnalysis { get; init; }
    public required GenerationSettings Settings { get; init; }
    public required DifficultyProfile Profile { get; init; }
    public required TimingCandidate[] CandidateGrid { get; init; }

    // Optional learned models — null = fall back to heuristics
    public IPlacementScorer?     PlacementScorer { get; init; }
    /// <summary>
    /// Multi-task neural model — serves as placement scorer and provides cut-direction,
    /// lane and row distributions. Cached per-beat predictions are stored in ProposedEvent.
    /// </summary>
    public NeuralPlacementModel? NeuralModel { get; init; }

    public List<CanonicalNote> PlacedNotes { get; } = [];
    public SwingContext LeftHandContext  { get; } = new(NoteHand.Left);
    public SwingContext RightHandContext { get; } = new(NoteHand.Right);
    public Random Rng { get; init; } = new(42);

    public SwingContext GetContext(NoteHand hand) =>
        hand == NoteHand.Left ? LeftHandContext : RightHandContext;
}
