# Architecture and Implementation Notes

## Solution Layout

```text
BeatSaberAutoMapper.sln
/src
  BeatSaber.AutoMapper
  BeatSaber.AutoMapper.Cli
/tests
  BeatSaber.AutoMapper.Tests
/docs
  agent.md
  architecture.md
  cli-contract.md
  sqlite-schema.md
```

## Namespace Layout

```text
BeatSaber.AutoMapper
  Audio
  Audio.Analysis
  Audio.Features
  Beatmap
  Beatmap.Import
  Beatmap.Export
  Canonical
  Canonical.Derived
  Training
  Training.Corpus
  Training.Features
  Training.Models
  Training.Evaluation
  Generation
  Generation.Decoding
  Generation.Patterns
  Validation
  Validation.Rules
  Validation.Repair
  Persistence
  Diagnostics
  Packaging
  Utilities
```

## Core Runtime Pipeline
1. Read input song and optional metadata.
2. Convert or verify audio packaging requirements.
3. Analyze waveform and derive beat grid, onset grid, energy and sections.
4. Create a candidate timing grid at valid subdivisions.
5. Produce candidate event probabilities or scores.
6. Predict note attributes or fetch heuristic candidates.
7. Decode through parity-aware constrained search.
8. Validate the result.
9. Repair where safe.
10. Export Beat Saber files and package assets.

## Recommended Core Classes

### Beatmap / Canonical
- `SongMetadata`
- `DifficultyDescriptor`
- `CanonicalBeatmap`
- `CanonicalNote`
- `CanonicalBomb`
- `CanonicalObstacle`
- `BeatTimingPoint`
- `SectionMarker`
- `SwingContext`
- `ParityState`

### Import / Export
- `BeatmapImporter`
- `BeatmapExporter`
- `V2BeatmapAdapter`
- `V3BeatmapAdapter`
- `V4BeatmapAdapter`
- `InfoDatReader`
- `BeatmapPackager`

### Audio
- `AudioDecoder`
- `AudioConverter`
- `BpmEstimator`
- `BeatTracker`
- `OnsetDetector`
- `SectionSegmenter`
- `AudioFeatureExtractor`
- `AudioAnalysisResult`
- `TimingGridBuilder`

### Training
- `CorpusIngestionService`
- `CorpusQualityScorer`
- `CorpusDeduplicator`
- `TrainingExampleBuilder`
- `DatasetManifestBuilder`
- `TrainingPipeline`
- `ModelArtifactRegistry`
- `PlacementModelTrainer`
- `AttributeModelTrainer`
- `EvaluationRunner`

### Generation
- `MapGenerationService`
- `CandidateEventProposer`
- `NoteAttributeSelector`
- `PatternLibrary`
- `ConstrainedDecoder`
- `GenerationSettings`
- `GenerationContext`
- `DifficultyProfile`

### Validation
- `BeatmapValidator`
- `ValidationIssue`
- `ValidationReport`
- `ParityRule`
- `ResetRule`
- `VisionBlockRule`
- `DensityRule`
- `LaneBalanceRule`
- `DifficultyEnvelopeRule`
- `RepairEngine`

### Persistence
- `AutoMapperDbContext`
- `SongEntity`
- `AudioAnalysisEntity`
- `BeatmapSetEntity`
- `DifficultyEntity`
- `TrainingRunEntity`
- `ModelArtifactEntity`
- `GenerationRunEntity`
- `ValidationRunEntity`

### Diagnostics
- `BeatGridCsvWriter`
- `ParityTraceWriter`
- `ValidationReportWriter`
- `GenerationDiffWriter`

## Suggested Directory Layout Inside Core Project

```text
BeatSaber.AutoMapper
  /Audio
    AudioDecoder.cs
    AudioConverter.cs
    AudioAnalysisResult.cs
    /Analysis
      BpmEstimator.cs
      BeatTracker.cs
      OnsetDetector.cs
      SectionSegmenter.cs
    /Features
      AudioFeatureExtractor.cs
      TimingGridBuilder.cs
  /Beatmap
    BeatmapImporter.cs
    BeatmapExporter.cs
    InfoDatReader.cs
    /Import
      V2BeatmapAdapter.cs
      V3BeatmapAdapter.cs
      V4BeatmapAdapter.cs
    /Export
      BeatmapPackager.cs
  /Canonical
    CanonicalBeatmap.cs
    CanonicalNote.cs
    CanonicalBomb.cs
    CanonicalObstacle.cs
    SongMetadata.cs
    DifficultyDescriptor.cs
    BeatTimingPoint.cs
    SectionMarker.cs
    /Derived
      SwingContext.cs
      ParityState.cs
  /Training
    TrainingPipeline.cs
    TrainingOptions.cs
    /Corpus
      CorpusIngestionService.cs
      CorpusQualityScorer.cs
      CorpusDeduplicator.cs
    /Features
      TrainingExampleBuilder.cs
      DatasetManifestBuilder.cs
    /Models
      PlacementModelTrainer.cs
      AttributeModelTrainer.cs
      ModelArtifactRegistry.cs
    /Evaluation
      EvaluationRunner.cs
  /Generation
    MapGenerationService.cs
    CandidateEventProposer.cs
    NoteAttributeSelector.cs
    GenerationSettings.cs
    GenerationContext.cs
    DifficultyProfile.cs
    /Decoding
      ConstrainedDecoder.cs
    /Patterns
      PatternLibrary.cs
  /Validation
    BeatmapValidator.cs
    ValidationIssue.cs
    ValidationReport.cs
    /Rules
      ParityRule.cs
      ResetRule.cs
      VisionBlockRule.cs
      DensityRule.cs
      LaneBalanceRule.cs
      DifficultyEnvelopeRule.cs
    /Repair
      RepairEngine.cs
  /Persistence
    AutoMapperDbContext.cs
    SongEntity.cs
    AudioAnalysisEntity.cs
    BeatmapSetEntity.cs
    DifficultyEntity.cs
    TrainingRunEntity.cs
    ModelArtifactEntity.cs
    GenerationRunEntity.cs
    ValidationRunEntity.cs
  /Diagnostics
    BeatGridCsvWriter.cs
    ParityTraceWriter.cs
    ValidationReportWriter.cs
    GenerationDiffWriter.cs
  /Packaging
    OggPackagingService.cs
  /Utilities
    Guard.cs
    MathHelpers.cs
```

## Training Artifacts
Keep model artifacts external to the DB as files with indexed metadata.

Recommended artifact types:
- placement model
- attribute model
- feature normalization metadata
- vocabulary or label map
- quality thresholds
- experiment config snapshot

## Design Rules
- Keep import/export isolated from canonical logic.
- Keep parity logic deterministic and testable.
- Keep generation settings explicit and serializable.
- Keep every repair step idempotent where possible.
- Keep debug output available for every major stage.

## First Critical Invariants
- Imported map round-trips without object loss for supported features.
- Beat-to-time conversion and time-to-beat conversion are stable.
- Decoder never knowingly emits impossible parity transitions when a valid alternative exists.
- Repair never makes a valid map less valid in another rule category without recording it.
