using BeatSaber.AutoMapper.Audio;
using BeatSaber.AutoMapper.Audio.Features;
using BeatSaber.AutoMapper.Generation.Decoding;
using BeatSaber.AutoMapper.Training.Models;
using BeatSaber.AutoMapper.Utilities;
using BeatSaber.AutoMapper.Validation;
using BeatSaber.AutoMapper.Validation.Repair;

namespace BeatSaber.AutoMapper.Generation;

public sealed class MapGenerationService
{
    public sealed record GenerationTelemetry(
        int CandidateCount,
        int ProposedCount,
        int DecodedNoteCount,
        int RepairedNoteCount);

    private readonly AudioFeatureExtractor _audioExtractor = new();
    private readonly TimingGridBuilder     _gridBuilder    = new();
    private readonly CandidateEventProposer _proposer      = new();
    private readonly NoteAttributeSelector  _selector      = new();
    private readonly ConstrainedDecoder     _decoder       = new();
    private readonly BeatmapValidator       _validator     = new();
    private readonly RepairEngine           _repair        = new();

    public sealed record GenerationResult(
        CanonicalBeatmap Beatmap,
        AudioAnalysisResult AudioAnalysis,
        ValidationReport ValidationReport,
        RepairEngine.RepairResult RepairResult,
        GenerationTelemetry Telemetry
    );

    /// <summary>Generate from an audio file path (audio is analysed on the fly).</summary>
    public GenerationResult Generate(
        string audioPath,
        SongMetadata song,
        GenerationSettings settings,
        IPlacementScorer? placementScorer = null)
    {
        Guard.NotNullOrEmpty(audioPath, nameof(audioPath));
        Guard.NotNull(song,     nameof(song));
        Guard.NotNull(settings, nameof(settings));
        var audio = _audioExtractor.Extract(audioPath);
        return Generate(audio, song, settings, placementScorer);
    }

    /// <summary>
    /// Generate one map per requested difficulty from a single audio file.
    /// Audio is analysed once and reused for all difficulties.
    /// </summary>
    public IReadOnlyList<GenerationResult> GenerateAll(
        string audioPath,
        SongMetadata song,
        IReadOnlyList<DifficultyLevel> difficulties,
        GenerationSettings templateSettings,
        IPlacementScorer? placementScorer = null)
    {
        Guard.NotNullOrEmpty(audioPath, nameof(audioPath));
        Guard.NotNull(song, nameof(song));
        Guard.NotNull(difficulties, nameof(difficulties));
        Guard.NotNull(templateSettings, nameof(templateSettings));

        var audio = _audioExtractor.Extract(audioPath);
        // Keep BPM consistent between audio analysis and song metadata
        if (audio.EstimatedBpm > 0)
            song = song with { BeatsPerMinute = audio.EstimatedBpm };

        return GenerateAll(audio, song, difficulties, templateSettings, placementScorer);
    }

    /// <summary>
    /// Generate one map per requested difficulty using a pre-computed audio analysis.
    /// Avoids re-analysis when called from a training loop or web service.
    /// </summary>
    public IReadOnlyList<GenerationResult> GenerateAll(
        AudioAnalysisResult audio,
        SongMetadata song,
        IReadOnlyList<DifficultyLevel> difficulties,
        GenerationSettings templateSettings,
        IPlacementScorer? placementScorer = null)
    {
        Guard.NotNull(audio, nameof(audio));
        Guard.NotNull(song, nameof(song));
        Guard.NotNull(difficulties, nameof(difficulties));
        Guard.NotNull(templateSettings, nameof(templateSettings));

        return difficulties
            .Select(diff => Generate(
                audio,
                song,
                templateSettings with { TargetDifficulty = diff },
                placementScorer))
            .ToList();
    }

    /// <summary>
    /// Generate using a pre-computed <see cref="AudioAnalysisResult"/> — avoids re-analysing
    /// audio every epoch during training.
    /// </summary>
    public GenerationResult Generate(
        AudioAnalysisResult audio,
        SongMetadata song,
        GenerationSettings settings,
        IPlacementScorer? placementScorer = null)
    {
        Guard.NotNull(audio,    nameof(audio));
        Guard.NotNull(song,     nameof(song));
        Guard.NotNull(settings, nameof(settings));

        NeuralPlacementModel? neuralModel = null;
        IMultiTaskPlacementModel? multiTaskModel = placementScorer as IMultiTaskPlacementModel;
        if (settings.UseLearned && settings.ArtifactsPath is not null)
        {
            neuralModel = NeuralPlacementModel.TryLoad(settings.ArtifactsPath);
            placementScorer ??= neuralModel
                             ?? (IPlacementScorer?)PlacementModel.TryLoad(settings.ArtifactsPath);
            multiTaskModel ??= neuralModel;
        }

        var grid = _gridBuilder.Build(
            audio.BeatTimesSeconds,
            audio.OnsetTimesSeconds,
            audio.EstimatedBpm,
            audio.DurationSeconds,
            TimingGridBuilder.StandardSubdivisions,
            audio.EnergyEnvelope,
            audio.FrameRateHz);

        var profile = DifficultyProfile.For(settings.TargetDifficulty);
        var ctx = new GenerationContext
        {
            Song            = song,
            AudioAnalysis   = audio,
            Settings        = settings,
            Profile         = profile,
            CandidateGrid   = grid,
            PlacementScorer = placementScorer,
            MultiTaskModel  = multiTaskModel,
            NeuralModel     = neuralModel,
            Rng             = new Random((int)settings.RandomSeed),
            GruHiddenState  = neuralModel is not null
                ? new GruState(BeatSaberMappingNet.GruLayers, BeatSaberMappingNet.GruHiddenDim)
                : null
        };

        var proposed = _proposer.ProposeEvents(ctx);
        var notes    = _decoder.Decode(proposed, ctx);

        var difficulty = new DifficultyDescriptor(
            settings.TargetDifficulty,
            BeatmapCharacteristic.Standard,
            NoteJumpMovementSpeed:       null,
            NoteJumpStartBeatOffset:     null,
            CustomLabel:                 null);

        var beatmap = new CanonicalBeatmap
        {
            Song         = song,
            Difficulty   = difficulty,
            TimingPoints = [new BeatTimingPoint(0.0, 0.0, audio.EstimatedBpm)],
            Notes        = notes,
            Bombs        = [],
            Obstacles    = [],
            Sections     = audio.Sections
        };

        var validationReport = _validator.Validate(beatmap);
        var repairResult     = _repair.Repair(beatmap);
        var telemetry = new GenerationTelemetry(
            CandidateCount: grid.Length,
            ProposedCount: proposed.Count,
            DecodedNoteCount: notes.Count,
            RepairedNoteCount: repairResult.RepairedBeatmap.Notes.Count);

        return new GenerationResult(
            repairResult.RepairedBeatmap,
            audio,
            validationReport,
            repairResult,
            telemetry);
    }
}
