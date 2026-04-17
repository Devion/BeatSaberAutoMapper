# SQLite Schema Draft

## Purpose
Persist corpus metadata, cached analysis, training runs, artifacts, generation runs, and validation results.

The database should index the pipeline, not become the sole storage for all heavy binary data.
Large blobs should remain on disk with paths stored in SQLite.

## Tables

### Songs
Stores normalized song identity and source metadata.

```sql
CREATE TABLE Songs (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    SongHash TEXT NULL,
    Title TEXT NOT NULL,
    Artist TEXT NOT NULL,
    SubTitle TEXT NULL,
    SourcePath TEXT NOT NULL,
    ManagedAudioPath TEXT NULL,
    OriginalAudioFormat TEXT NULL,
    DurationSeconds REAL NULL,
    CreatedUtc TEXT NOT NULL,
    UNIQUE(SourcePath)
);
```

### AudioAnalyses
Stores cached analysis results for a song.

```sql
CREATE TABLE AudioAnalyses (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    SongId INTEGER NOT NULL,
    AnalysisVersion TEXT NOT NULL,
    BpmEstimate REAL NULL,
    BpmConfidence REAL NULL,
    BeatCount INTEGER NULL,
    OnsetCount INTEGER NULL,
    SectionCount INTEGER NULL,
    FeaturesPath TEXT NULL,
    BeatGridPath TEXT NULL,
    OnsetPath TEXT NULL,
    SectionsPath TEXT NULL,
    DiagnosticsPath TEXT NULL,
    CreatedUtc TEXT NOT NULL,
    FOREIGN KEY (SongId) REFERENCES Songs(Id)
);
```

### BeatmapSets
Stores imported map-set level metadata.

```sql
CREATE TABLE BeatmapSets (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    SongId INTEGER NOT NULL,
    SourcePath TEXT NOT NULL,
    SourceArchiveHash TEXT NULL,
    MapperName TEXT NULL,
    BeatmapSchemaVersion TEXT NULL,
    Mode TEXT NOT NULL,
    QualityScore REAL NULL,
    IsCurated INTEGER NOT NULL DEFAULT 0,
    IsRankedLike INTEGER NOT NULL DEFAULT 0,
    IsGimmick INTEGER NOT NULL DEFAULT 0,
    IsMalformed INTEGER NOT NULL DEFAULT 0,
    ImportedUtc TEXT NOT NULL,
    FOREIGN KEY (SongId) REFERENCES Songs(Id)
);
```

### Difficulties
Stores difficulty-level metadata.

```sql
CREATE TABLE Difficulties (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    BeatmapSetId INTEGER NOT NULL,
    DifficultyName TEXT NOT NULL,
    Characteristic TEXT NOT NULL,
    NoteCount INTEGER NULL,
    BombCount INTEGER NULL,
    ObstacleCount INTEGER NULL,
    Nps REAL NULL,
    DurationSeconds REAL NULL,
    ValidationScore REAL NULL,
    CanonicalPath TEXT NULL,
    CreatedUtc TEXT NOT NULL,
    FOREIGN KEY (BeatmapSetId) REFERENCES BeatmapSets(Id)
);
```

### ValidationRuns
Stores validation summaries for imported or generated maps.

```sql
CREATE TABLE ValidationRuns (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    DifficultyId INTEGER NULL,
    GenerationRunId INTEGER NULL,
    RuleSetVersion TEXT NOT NULL,
    TotalIssues INTEGER NOT NULL,
    ErrorCount INTEGER NOT NULL,
    WarningCount INTEGER NOT NULL,
    InfoCount INTEGER NOT NULL,
    Score REAL NULL,
    ReportPath TEXT NULL,
    CreatedUtc TEXT NOT NULL,
    FOREIGN KEY (DifficultyId) REFERENCES Difficulties(Id),
    FOREIGN KEY (GenerationRunId) REFERENCES GenerationRuns(Id)
);
```

### ValidationIssues
Stores per-issue details.

```sql
CREATE TABLE ValidationIssues (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    ValidationRunId INTEGER NOT NULL,
    RuleName TEXT NOT NULL,
    Severity TEXT NOT NULL,
    Beat REAL NULL,
    TimeSeconds REAL NULL,
    Hand TEXT NULL,
    Description TEXT NOT NULL,
    SuggestedFix TEXT NULL,
    FOREIGN KEY (ValidationRunId) REFERENCES ValidationRuns(Id)
);
```

### DatasetManifests
Tracks built datasets for training and evaluation.

```sql
CREATE TABLE DatasetManifests (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Version TEXT NOT NULL,
    ManifestPath TEXT NOT NULL,
    ExampleCount INTEGER NOT NULL,
    TrainCount INTEGER NOT NULL,
    ValidationCount INTEGER NOT NULL,
    TestCount INTEGER NOT NULL,
    CreatedUtc TEXT NOT NULL,
    UNIQUE(Name, Version)
);
```

### TrainingRuns
Tracks model training execution.

```sql
CREATE TABLE TrainingRuns (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    DatasetManifestId INTEGER NOT NULL,
    ProfileName TEXT NOT NULL,
    Status TEXT NOT NULL,
    MetricsJsonPath TEXT NULL,
    StartedUtc TEXT NOT NULL,
    FinishedUtc TEXT NULL,
    FOREIGN KEY (DatasetManifestId) REFERENCES DatasetManifests(Id)
);
```

### ModelArtifacts
Tracks saved model outputs.

```sql
CREATE TABLE ModelArtifacts (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    TrainingRunId INTEGER NOT NULL,
    ArtifactType TEXT NOT NULL,
    ArtifactVersion TEXT NOT NULL,
    FilePath TEXT NOT NULL,
    MetadataPath TEXT NULL,
    CreatedUtc TEXT NOT NULL,
    FOREIGN KEY (TrainingRunId) REFERENCES TrainingRuns(Id)
);
```

### GenerationRuns
Tracks generated outputs for songs.

```sql
CREATE TABLE GenerationRuns (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    SongId INTEGER NOT NULL,
    DifficultyName TEXT NOT NULL,
    ArtifactBundle TEXT NULL,
    SettingsJsonPath TEXT NULL,
    OutputPath TEXT NULL,
    RepairSummaryPath TEXT NULL,
    Status TEXT NOT NULL,
    StartedUtc TEXT NOT NULL,
    FinishedUtc TEXT NULL,
    FOREIGN KEY (SongId) REFERENCES Songs(Id)
);
```

### CorpusDuplicates
Tracks deduplication decisions.

```sql
CREATE TABLE CorpusDuplicates (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    BeatmapSetId INTEGER NOT NULL,
    DuplicateOfBeatmapSetId INTEGER NOT NULL,
    Reason TEXT NOT NULL,
    CreatedUtc TEXT NOT NULL,
    FOREIGN KEY (BeatmapSetId) REFERENCES BeatmapSets(Id),
    FOREIGN KEY (DuplicateOfBeatmapSetId) REFERENCES BeatmapSets(Id)
);
```

## Indexes

```sql
CREATE INDEX IX_AudioAnalyses_SongId ON AudioAnalyses(SongId);
CREATE INDEX IX_BeatmapSets_SongId ON BeatmapSets(SongId);
CREATE INDEX IX_Difficulties_BeatmapSetId ON Difficulties(BeatmapSetId);
CREATE INDEX IX_ValidationRuns_DifficultyId ON ValidationRuns(DifficultyId);
CREATE INDEX IX_ValidationRuns_GenerationRunId ON ValidationRuns(GenerationRunId);
CREATE INDEX IX_ValidationIssues_ValidationRunId ON ValidationIssues(ValidationRunId);
CREATE INDEX IX_TrainingRuns_DatasetManifestId ON TrainingRuns(DatasetManifestId);
CREATE INDEX IX_ModelArtifacts_TrainingRunId ON ModelArtifacts(TrainingRunId);
CREATE INDEX IX_GenerationRuns_SongId ON GenerationRuns(SongId);
```

## Notes
- Keep heavy feature blobs on disk, not inline in SQLite.
- Consider adding `Checksum` columns to tables that reference artifact files.
- Add migration support from the start if EF Core is used.
- Keep rule set versioning explicit so validator regressions are traceable.
