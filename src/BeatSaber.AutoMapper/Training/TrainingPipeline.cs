using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json;
using BeatSaber.AutoMapper.Audio;
using BeatSaber.AutoMapper.Audio.Features;
using BeatSaber.AutoMapper.Beatmap;
using BeatSaber.AutoMapper.Diagnostics;
using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Training.Corpus;
using BeatSaber.AutoMapper.Training.Evaluation;
using BeatSaber.AutoMapper.Training.Features;
using BeatSaber.AutoMapper.Training.Models;
using BeatSaber.AutoMapper.Training.SelfSupervised;

namespace BeatSaber.AutoMapper.Training;

public sealed class TrainingPipeline : IDisposable
{
    private sealed record SongFolderInfo(
        string FolderPath,
        IReadOnlyList<DifficultyLevel> Difficulties)
    {
        public DifficultyLevel HighestDifficulty =>
            Difficulties.Count > 0 ? Difficulties.Max() : DifficultyLevel.Easy;
    }

    private sealed record SplitPlan(
        int TrainSongs,
        int ValidationSongs,
        int TestSongs,
        int ValidationSongsPerEpoch,
        int ValidationCachePoolSize);

    private readonly CorpusIngestionService _ingestion        = new();
    private readonly TrainingExampleBuilder _exampleBuilder   = new();
    private readonly DatasetManifestBuilder _manifestBuilder  = new();
    private readonly TorchPlacementTrainer   _placementTrainer = new();
    private readonly EvaluationRunner       _evaluator        = new();
    private TrainingConsoleDashboard? _dashboard;
    private TrainingFileLogger? _logger;

    public void Dispose() => _placementTrainer.Dispose();

    // Pre-analysed audio kept alive for all epochs (analysis is expensive, result is immutable)
    private sealed record CachedPair(
        ValidationSongFinder.ValidationPair Pair,
        AudioAnalysisResult Audio);

    // One entry per unique song in the pool; flattened to CachedPair[] per epoch
    private sealed record CachedSong(
        ValidationSongFinder.SongGroup Group,
        AudioAnalysisResult Audio);

    private sealed record ValidationSchedulePlan(
        CachedPair[] CorePairs,
        CachedPair[] FullPairs,
        int CoreSongCount,
        int FullSongCap,
        int StageStartEpoch,
        int StageStepEpochs)
    {
        public int ActiveFullSongCountForEpoch(int absoluteEpochCompleted)
        {
            if (FullSongCap <= CoreSongCount)
                return CoreSongCount;
            if (absoluteEpochCompleted < StageStartEpoch)
                return CoreSongCount;

            int stages = 1 + Math.Max(0, (absoluteEpochCompleted - StageStartEpoch) / Math.Max(1, StageStepEpochs));
            int count = CoreSongCount + stages * CoreSongCount;
            return Math.Clamp(count, CoreSongCount, FullSongCap);
        }
    }

    private sealed record PersistedValidationSchedule(
        long RandomSeed,
        int CoreSongCount,
        int FullSongCap,
        int StageStartEpoch,
        int StageStepEpochs,
        int LastCompletedEpoch,
        string[] CoreFolderPaths,
        string[] FullFolderPaths);

    /// <summary>
    /// Survives a process restart so warm-start resumes with the correct
    /// best-quality gate, stagnation counters, and learning rate.
    /// </summary>
    private sealed record PersistedTrainingState(
        double BestQuality,
        int    BestEpochNumber,
        int    StagnationEpochs,
        int    LrReductions,
        int    PlateauRestartsUsed,
        double SmoothedQuality,
        double CurrentLr,
        int    TotalEpochsCompleted);

    private sealed record SequenceDifficultySnapshot(
        int SequenceCount,
        int ExampleCount,
        string Mix);

    private sealed record ValidationPairSnapshot(
        double Score,
        double RawScore,
        int RawNoteCount,
        int RepairedNoteCount,
        ulong RawFingerprint);

    private sealed record ValidationPairOutcome(
        string PairKey,
        DifficultyLevel Difficulty,
        double Score,
        double RawScore,
        double RepairedScore,
        int RawNoteCount,
        int CandidateCount,
        int ProposedCount,
        int DecodedNoteCount,
        int RepairedNoteCount,
        int RepairCount,
        double RepairNoteDeltaRatio,
        IReadOnlyList<string> RawIssueRules,
        double RightHandFraction,
        double DotFraction,
        double ChordBeatFraction,
        int SyntheticExampleCount,
        ulong RawFingerprint,
        bool Failed);

    private sealed record ValidationDiagnostics(
        int TotalPairs,
        int SuccessfulPairs,
        int FailedPairs,
        int CoreSuccessfulPairs,
        int CoreFailedPairs,
        int TotalSyntheticExamples,
        double AvgCandidateSurvival,
        double AvgDecodeRate,
        double AvgRawScore,
        double AvgRepairedScore,
        double AvgRawNotes,
        double AvgRepairedNotes,
        double AvgRepairCount,
        double AvgRepairNoteDeltaRatio,
        double AvgRightHandFraction,
        double AvgDotFraction,
        double AvgChordBeatFraction,
        double AvgScoreAllPairs,
        double AvgSyntheticExamplesPerAcceptedPair,
        Dictionary<DifficultyLevel, double> AvgGeneratedNotesByDifficulty,
        Dictionary<DifficultyLevel, double> DiffScores,
        Dictionary<DifficultyLevel, double> PreviousDiffScores,
        Dictionary<string, int> RawIssueCounts,
        int ChangedPairs,
        int UnchangedPairs,
        int FingerprintChangedPairs,
        int ScoreImprovedPairs,
        int ScoreWorsenedPairs,
        int NoteCountChangedPairs,
        double AvgAbsScoreDelta,
        double AvgAbsRawScoreDelta,
        double AvgAbsRawNoteDelta,
        Dictionary<string, ValidationPairSnapshot> PairSnapshots);

    private readonly record struct SyntheticBudget(
        int MaxSequenceCount,
        int MaxExampleCount,
        double FractionOfRealExamples,
        int StableRefreshStreak);

    private readonly record struct NegativeSupervisionBudget(
        int MaxSequenceCount,
        int MaxExampleCount,
        double FractionOfRealExamples);

    public void Run(TrainingOptions options)
    {
        Guard.NotNull(options, nameof(options));
        Directory.CreateDirectory(options.ArtifactsOutputPath);
        using var logger = new TrainingFileLogger(options.ArtifactsOutputPath);
        _logger = logger;
        TrainingConsoleDashboard? dashboard = TrainingConsoleDashboard.TryCreate();
        _dashboard = dashboard;
        _placementTrainer.Dashboard = dashboard;
        dashboard?.OnStageStarted("startup", 1, "initializing training pipeline");

        // Warm-start from existing checkpoint
        var loadedCheckpointKind = _placementTrainer.TryLoadCheckpoint(options.ArtifactsOutputPath);
        bool warmStarted = loadedCheckpointKind != TorchPlacementTrainer.LoadedCheckpointKind.None;
        if (warmStarted)
        {
            if (dashboard is not null) dashboard.AddNotice($"Warm-start from {loadedCheckpointKind}");
            else Console.WriteLine($"[Training] Warm-start: loaded {loadedCheckpointKind} checkpoint.");
        }

        // Load persisted counters (bestQ, stagnation, LR) so warm restarts never
        // mistake a lower first-epoch quality for a new best and overwrite the checkpoint.
        string trainingStatePath = Path.Combine(options.ArtifactsOutputPath, "training_state.json");
        PersistedTrainingState? persistedTrainingState =
            warmStarted ? TryLoadTrainingState(trainingStatePath) : null;
        if (persistedTrainingState is not null)
        {
            string msg =
                $"bestQ={persistedTrainingState.BestQuality:F3}  bestEp={persistedTrainingState.BestEpochNumber}" +
                $"  stag={persistedTrainingState.StagnationEpochs}  lr={persistedTrainingState.CurrentLr:G4}" +
                $"  lrReductions={persistedTrainingState.LrReductions}  plateauRestarts={persistedTrainingState.PlateauRestartsUsed}";
            logger.Log("RESTORE", msg);
            if (dashboard is not null)
                dashboard.AddNotice(
                    $"State: bestQ={persistedTrainingState.BestQuality:F3} @ep{persistedTrainingState.BestEpochNumber}  lr={persistedTrainingState.CurrentLr:G4}");
            else
                Console.WriteLine($"[Training] Restored training state: {msg}");
        }

        // ----------------------------------------------------------------
        // 1. Resolve map folders
        // ----------------------------------------------------------------
        string[] mapFolders;
        dashboard?.OnStageStarted("dataset", 1, Path.GetFileName(options.DatasetPath));

        if (IsMapFolder(options.DatasetPath))
        {
            mapFolders = [options.DatasetPath];
            if (dashboard is not null) dashboard.AddNotice($"Dataset single folder {Path.GetFileName(options.DatasetPath)}");
            else Console.WriteLine($"[Training] Using single map folder '{options.DatasetPath}'.");
        }
        else if (ContainsUnpackedMaps(options.DatasetPath))
        {
            mapFolders = Directory.GetDirectories(options.DatasetPath)
                .Where(IsMapFolder).ToArray();
            if (dashboard is not null) dashboard.AddNotice($"Found {mapFolders.Length} unpacked map folders");
            else
                Console.WriteLine(
                    $"[Training] Found {mapFolders.Length} unpacked map folders in '{options.DatasetPath}'.");
        }
        else
        {
            string libraryPath = Path.Combine(options.ArtifactsOutputPath, "library");
            if (dashboard is not null) dashboard.OnStageStarted("ingest", 1, Path.GetFileName(options.DatasetPath));
            else Console.WriteLine($"[Training] Ingesting corpus from '{options.DatasetPath}'...");
            var summary = _ingestion.IngestFolder(options.DatasetPath, libraryPath, options.DatasetPath);
            if (dashboard is not null)
                dashboard.AddNotice(
                    $"Ingested {summary.Imported} maps, skipped {summary.Skipped}, dup {summary.Duplicate}");
            else
                Console.WriteLine(
                    $"[Training] Ingested {summary.Imported} maps, {summary.Skipped} skipped, " +
                    $"{summary.Malformed} malformed, {summary.Duplicate} duplicates.");

            if (!Directory.Exists(libraryPath))
            {
                if (dashboard is not null) dashboard.AddNotice("No maps found");
                else Console.WriteLine("[Training] No maps found.");
                return;
            }
            mapFolders = Directory.GetDirectories(libraryPath);
        }
        dashboard?.OnStageProgress(1, 1, 0, $"{mapFolders.Length} map folder(s)");

        // ----------------------------------------------------------------
        // 2. Song-level train / val / test split
        //    Splits are difficulty-aware and remain song-separated:
        //    one map folder and all its difficulties stay in exactly one split.
        // ----------------------------------------------------------------
        int totalSongs = mapFolders.Length;
        dashboard?.OnStageStarted("inspect", mapFolders.Length, "reading song difficulties");
        var folderInfos = InspectSongFolders(mapFolders, dashboard);
        var splitPlan = ResolveSplitPlan(totalSongs, options);

        if (splitPlan.TrainSongs <= 0)
        {
            if (dashboard is not null) dashboard.AddNotice("Not enough songs to split train/val/test");
            else Console.WriteLine("[Training] Not enough songs to split into train / val / test. Aborting.");
            return;
        }

        var (trainFoldersRaw, valFoldersRaw, testFoldersRaw) = CreateDifficultyAwareSplit(
            folderInfos,
            splitPlan,
            options.RandomSeed);
        var (trainFolders, valFolders, testFolders, coverageMoves) = EnsureTrainingDifficultyCoverage(
            folderInfos,
            trainFoldersRaw,
            valFoldersRaw,
            testFoldersRaw);

        foreach (string move in coverageMoves)
            if (dashboard is not null) dashboard.AddNotice($"Coverage fix: {move}");
            else Console.WriteLine($"[Training] Coverage fix: {move}");

        if (dashboard is not null)
        {
            dashboard.AddNotice(
                $"Split {trainFolders.Length} train / {valFolders.Length} val / {testFolders.Length} test");
            dashboard.AddNotice(
                $"Mix train[{FormatDifficultyMix(trainFolders)}] val[{FormatDifficultyMix(valFolders)}]");
        }
        else
        {
            Console.WriteLine(
                $"[Training] Split: {trainFolders.Length} train / " +
                $"{valFolders.Length} val / {testFolders.Length} test songs " +
                $"(total {totalSongs}, val-songs/epoch={splitPlan.ValidationSongsPerEpoch}, " +
                $"validation-cache-size={splitPlan.ValidationCachePoolSize}).");

            Console.WriteLine(
                $"[Training] Difficulty mix: train[{FormatDifficultyMix(trainFolders)}]  " +
                $"val[{FormatDifficultyMix(valFolders)}]  " +
                $"test[{FormatDifficultyMix(testFolders)}]");
        }

        // ----------------------------------------------------------------
        // 3. Build training examples — parallel across trainFolders only
        // Each (folder, difficulty) pair becomes one sequence (beat-ordered).
        // ----------------------------------------------------------------
        if (dashboard is not null)
            dashboard.OnStageStarted("train-audio", trainFolders.Length, "analysing audio + building examples");
        else
            Console.WriteLine(
                $"[Training] Analysing audio + building examples from {trainFolders.Length} train folders " +
                $"using {Environment.ProcessorCount} threads (audio cached to disk)...");

        var allSequences   = new List<List<TrainingExample>>();
        var printLock      = new object();
        int doneCount      = 0;
        var listLock       = new object();
        var buildStopwatch = Stopwatch.StartNew();

        // Separate cache directory for training audio (never touches the validation cache).
        string trainingCacheDir = Path.Combine(options.ArtifactsOutputPath, "training_audio_cache");
        string validationCacheDir = Path.Combine(options.ArtifactsOutputPath, "validation_audio_cache");
        var trainAudioCache = new ValidationAudioCache(
            trainingCacheDir,
            exactDirectory: true,
            validationCacheDir,
            Path.Combine(trainingCacheDir, "validation_audio_cache"));

        Parallel.ForEach(
            trainFolders,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => new List<List<TrainingExample>>(),
            (folder, _, localSeqs) =>
            {
                try
                {
                    var maps = BeatmapImporter.Import(folder);
                    if (maps.Count == 0) return localSeqs;

                    // Resolve and analyse the audio file (cached after first run).
                    string? audioPath = FindAudioFile(folder);
                    AudioAnalysisResult? audio = null;
                    if (audioPath != null)
                    {
                        audio = trainAudioCache.TryLoad(audioPath);
                        if (audio is null)
                        {
                            audio = new AudioFeatureExtractor().Extract(audioPath);
                            trainAudioCache.Store(audioPath, audio);
                        }
                    }

                    var builder = new TrainingExampleBuilder();
                    foreach (var map in maps)
                    {
                        // Fall back to stub audio (zeros for spectral features) when the
                        // audio file is missing — better than skipping the map entirely.
                        var audioForMap = audio ?? MakeStubAudio(map);
                        var examples    = builder.Build(map, audioForMap);
                        if (examples.Count > 0)
                            localSeqs.Add(new List<TrainingExample>(examples));
                    }
                }
                catch (Exception ex)
                {
                    lock (printLock)
                        Console.WriteLine($"[Training] Warning: '{folder}': {ex.Message}");
                }

                int n = Interlocked.Increment(ref doneCount);
                if (n % 50 == 0 || n == trainFolders.Length)
                {
                    double elapsed = buildStopwatch.Elapsed.TotalSeconds;
                    double eta     = n < trainFolders.Length && elapsed > 0
                        ? elapsed / n * (trainFolders.Length - n) : 0;
                    lock (printLock)
                    {
                        if (dashboard is not null)
                            dashboard.OnStageProgress(
                                n,
                                trainFolders.Length,
                                elapsed,
                                Path.GetFileName(folder));
                        else
                            Console.WriteLine(
                                $"[Training]   {n}/{trainFolders.Length} folders " +
                                $"({100.0 * n / trainFolders.Length:F0}%)" +
                                $" — {elapsed:F0}s elapsed" +
                                (eta > 1 ? $", ~{eta:F0}s remaining" : string.Empty));
                    }
                }

                return localSeqs;
            },
            localSeqs =>
            {
                lock (listLock) allSequences.AddRange(localSeqs);
            });

        int totalExamples = allSequences.Sum(s => s.Count);
        if (dashboard is not null)
            dashboard.AddNotice($"Built {totalExamples} train examples in {allSequences.Count} sequences");
        else
            Console.WriteLine(
                $"[Training] Built {totalExamples} training examples in {allSequences.Count} sequences.");
        if (allSequences.Count == 0) return;

        var trainSeqs = allSequences;
        var realTrainSnapshot = SummarizeSequences(trainSeqs);
        Console.WriteLine(
            $"[Training] Real train mix: seq={realTrainSnapshot.SequenceCount}  ex={realTrainSnapshot.ExampleCount}  " +
            $"{realTrainSnapshot.Mix}");
        logger.Log("CORPUS",
            $"train seq={realTrainSnapshot.SequenceCount}  ex={realTrainSnapshot.ExampleCount}  {realTrainSnapshot.Mix}");

        var badNegativeSeqs = new List<List<TrainingExample>>();
        if (!string.IsNullOrWhiteSpace(options.BadLibraryPath))
        {
            var badFolders = ResolveMapFolders(
                options.BadLibraryPath,
                Path.Combine(options.ArtifactsOutputPath, "bad_library"),
                "[Training] Bad-lib");

            if (badFolders.Length > 0)
            {
                if (dashboard is not null)
                    dashboard.OnStageStarted("bad-lib", badFolders.Length, "building negative supervision");
                else
                    Console.WriteLine(
                        $"[Training] Building explicit negative supervision from {badFolders.Length} bad-map folders...");
                var (badSeqs, badFilteredSeqs, badFallbackSeqs) = BuildNegativeExamplesFromFolders(
                    badFolders,
                    printLock,
                    trainAudioCache,
                    options.SelfSupervisedNegativeWeight,
                    dashboard);
                badNegativeSeqs = badSeqs.ToList();

                var badSnapshot = SummarizeSequences(badNegativeSeqs);
                var badBudget = ComputeBadNegativeBudget(trainSeqs);
                double badFilteredPct = badNegativeSeqs.Count > 0
                    ? 100.0 * badFilteredSeqs / badNegativeSeqs.Count : 0;
                string badLibMsg =
                    $"neg pool: seq={badSnapshot.SequenceCount}  ex={badSnapshot.ExampleCount}  {badSnapshot.Mix}" +
                    $"  filtered={badFilteredSeqs}/{badNegativeSeqs.Count} ({badFilteredPct:F0}%)  fallback={badFallbackSeqs}" +
                    $"  cap/epoch seq={badBudget.MaxSequenceCount} ex={badBudget.MaxExampleCount}" +
                    $"  frac={badBudget.FractionOfRealExamples:F3}";
                Console.WriteLine(
                    $"[Training] Bad negative pool: seq={badSnapshot.SequenceCount}  ex={badSnapshot.ExampleCount}  " +
                    $"{badSnapshot.Mix}  filtered={badFilteredSeqs}/{badNegativeSeqs.Count} ({badFilteredPct:F0}%)  " +
                    $"fallback={badFallbackSeqs}  cap/epoch={badBudget.MaxSequenceCount} seq {badBudget.MaxExampleCount} ex");
                logger.Log("BADLIB", badLibMsg);
            }
            else
            {
                Console.WriteLine("[Training] Bad-lib resolved to 0 usable map folders; skipping explicit negative supervision.");
            }
        }

        // Build test examples from held-out test folders (used only for final evaluation)
        var testEx = testFolders.Length > 0
            ? BuildExamplesFromFolders(testFolders, printLock, trainAudioCache)
            : (IReadOnlyList<TrainingExample>)[];

        // ----------------------------------------------------------------
        // 4. Pre-cache audio analysis for validation songs (runs ONCE, reused every epoch)
        //
        // Only songs from valFolders are used — training songs are never evaluated.
        // Each song contributes ALL its valid difficulties as separate pairs.
        // 'songsPerEpoch' is sampled from the pool once and fixed for the whole run.
        // ----------------------------------------------------------------
        int songsPerEpoch = splitPlan.ValidationSongsPerEpoch;
        string validationStatePath = Path.Combine(options.ArtifactsOutputPath, "validation_schedule.json");
        int resumeEpochOffset = 0;
        PersistedValidationSchedule? persistedSchedule = TryLoadValidationSchedule(validationStatePath);
        if (persistedSchedule is not null)
        {
            resumeEpochOffset = Math.Max(0, persistedSchedule.LastCompletedEpoch);
            Console.WriteLine($"[Training] Loaded validation schedule state (last completed epoch {resumeEpochOffset}).");
        }

        var poolGroups = new ValidationSongFinder()
            .Find(valFolders, splitPlan.ValidationCachePoolSize, options.RandomSeed);

        ValidationSchedulePlan? validationPlan = null;
        if (poolGroups.Count > 0)
        {
            var audioCache  = new ValidationAudioCache(
                validationCacheDir,
                exactDirectory: true,
                trainingCacheDir,
                Path.Combine(trainingCacheDir, "validation_audio_cache"));
            var poolTmp     = new CachedSong?[poolGroups.Count];
            int cacheHits   = 0;
            int cacheMisses = 0;
            int valDone     = 0;
            var valStopwatch = Stopwatch.StartNew();

            if (dashboard is not null)
                dashboard.OnStageStarted("val-audio", poolGroups.Count, "warming validation audio cache");
            else
                Console.WriteLine(
                    $"[Training] Analysing {poolGroups.Count} validation songs " +
                    $"(cached in '{options.ArtifactsOutputPath}')...");

            Parallel.For(0, poolGroups.Count,
                new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                i =>
                {
                    var group = poolGroups[i];
                    bool hit = false;
                    try
                    {
                        // One audio cache entry per unique song (shared across all its difficulties)
                        var audio = audioCache.TryLoad(group.AudioPath);
                        if (audio is not null)
                        {
                            hit = true;
                            Interlocked.Increment(ref cacheHits);
                        }
                        else
                        {
                            // New extractor per thread — AudioFeatureExtractor has mutable inner state
                            audio = new AudioFeatureExtractor().Extract(group.AudioPath);
                            audioCache.Store(group.AudioPath, audio);
                            Interlocked.Increment(ref cacheMisses);
                        }
                        poolTmp[i] = new CachedSong(group, audio);
                    }
                    catch (Exception ex)
                    {
                        lock (printLock)
                            Console.WriteLine(
                                $"[Training] Warning: could not analyse '{group.AudioPath}': {ex.Message}");
                    }

                    int n = Interlocked.Increment(ref valDone);
                    if (!hit || n % 25 == 0 || n == poolGroups.Count)
                    {
                        double elapsed = valStopwatch.Elapsed.TotalSeconds;
                        double eta     = n < poolGroups.Count && elapsed > 0
                            ? elapsed / n * (poolGroups.Count - n) : 0;
                        lock (printLock)
                        {
                            if (dashboard is not null)
                                dashboard.OnStageProgress(
                                    n,
                                    poolGroups.Count,
                                    elapsed,
                                    $"{Path.GetFileName(group.AudioPath)} [{(hit ? "cache" : "fresh")}]");
                            else
                                Console.WriteLine(
                                    $"[Training]   Val {n}/{poolGroups.Count}" +
                                    $" — {Path.GetFileName(group.AudioPath)}" +
                                    $" [{(hit ? "cache" : "fresh")}]" +
                                    $" {elapsed:F0}s elapsed" +
                                    (eta > 1 ? $" ~{eta:F0}s remaining" : string.Empty));
                        }
                    }
                });

            var pool = poolTmp.Where(s => s is not null).Select(s => s!).ToArray();
            int poolPairCount = pool.Sum(s => s.Group.Difficulties.Count);
            Console.WriteLine(
                $"[Training] Validation pool: {pool.Length}/{poolGroups.Count} unique songs ready " +
                $"({poolPairCount} total pairs, {cacheHits} from cache, {cacheMisses} freshly analysed, " +
                $"{valStopwatch.Elapsed.TotalSeconds:F0}s).");

            validationPlan = BuildValidationSchedule(
                pool,
                songsPerEpoch,
                options,
                persistedSchedule);

            SaveValidationSchedule(validationStatePath, options.RandomSeed, validationPlan, resumeEpochOffset);

            if (dashboard is not null)
                dashboard.AddNotice(
                    $"Val schedule core {validationPlan.CoreSongCount} full {validationPlan.FullSongCap} pairs {validationPlan.FullPairs.Length}");
            else
                Console.WriteLine(
                    $"[Training] Validation schedule: core={validationPlan.CorePairs.Length} pair(s) / {validationPlan.CoreSongCount} song(s), " +
                    $"full-cap={validationPlan.FullPairs.Length} pair(s) / {validationPlan.FullSongCap} song(s), " +
                    $"stage-start={validationPlan.StageStartEpoch}, stage-step={validationPlan.StageStepEpochs}.");
        }

        // ----------------------------------------------------------------
        // 5. Placement model — epoch loop with shuffle + parallel mini-batch GD
        // ----------------------------------------------------------------
        double lr = persistedTrainingState?.CurrentLr > 0
            ? persistedTrainingState.CurrentLr
            : loadedCheckpointKind switch
            {
                TorchPlacementTrainer.LoadedCheckpointKind.Current => options.InitialLearningRate,
                TorchPlacementTrainer.LoadedCheckpointKind.Best => options.InitialLearningRate * 0.5,
                TorchPlacementTrainer.LoadedCheckpointKind.Legacy => options.InitialLearningRate * 0.5,
                _ => options.InitialLearningRate
            };
        if (dashboard is null)
            Console.WriteLine(
                $"[Training] LR setup: requested={ConsoleStyler.Colorize(options.InitialLearningRate.ToString("G4"), ConsoleColor.Cyan)}  " +
                $"effective={ConsoleStyler.Colorize(lr.ToString("G4"), ConsoleColor.Cyan)}  " +
                $"checkpoint={FormatCheckpointKind(loadedCheckpointKind)}");
        logger.Log("LR", $"Initial effective lr={lr:G4}  requestedLr={options.InitialLearningRate:G4}  checkpoint={loadedCheckpointKind}  fromState={persistedTrainingState is not null && persistedTrainingState.CurrentLr > 0}");
        const double MinLr   = 1e-5;
        const double LrDecay = 0.5;
        // Reduce LR after this many consecutive stagnation epochs.
        // Default: patience/5 → faster LR drops than the old patience/3.
        int lrPatience = options.LrPatience > 0
            ? options.LrPatience
            : Math.Max(1, options.EarlyStopPatience / 5);

        if (dashboard is not null)
        {
            dashboard.SetConfig(new TrainingConsoleDashboard.Config(
                Device: _placementTrainer.UsesGpuInference ? "CUDA" : "CPU",
                ValidationDevice: options.ValidationUseGpuInference ? "GPU" : "CPU",
                InputDim: BeatSaberMappingNet.InputDim,
                HiddenDim: BeatSaberMappingNet.GruHiddenDim,
                Layers: BeatSaberMappingNet.GruLayers,
                MlpHidden: BeatSaberMappingNet.MlpHidden,
                LearningRate: lr,
                Epochs: options.Epochs,
                ValidationEvery: Math.Max(1, options.ValidationEveryNEpochs),
                ValidationSongs: validationPlan?.CoreSongCount ?? 0,
                ValidationPairs: validationPlan?.FullPairs.Length ?? 0,
                PlateauRestarts: options.PlateauRestartCount,
                PlateauRestartScale: options.PlateauRestartLrScale,
                CheckpointKind: loadedCheckpointKind.ToString()));
            dashboard.AddNotice($"Synthetic {(options.EnableSyntheticTraining ? "on" : "off")} warmup {options.SelfSupervisedWarmupEpochs}");
        }
        else
        {
            Console.WriteLine(
                $"[Training] Training GRU placement model {BeatSaberMappingNet.InputDim}→GRU({BeatSaberMappingNet.GruHiddenDim}×{BeatSaberMappingNet.GruLayers})→{BeatSaberMappingNet.MlpHidden} " +
                $"({options.Epochs} epochs max, Adam lr={lr:G4}, patience={options.EarlyStopPatience}, lr-patience={lrPatience})...[warmStart={warmStarted}]");
            Console.WriteLine(
                $"[Training] Generation validation cadence: every {Math.Max(1, options.ValidationEveryNEpochs)} epoch(s).");
            Console.WriteLine(
                $"[Training] Validation inference device: {(options.ValidationUseGpuInference
                    ? ConsoleStyler.Colorize("GPU", ConsoleColor.Green)
                    : ConsoleStyler.Colorize("CPU", ConsoleColor.Yellow))}.");
            Console.WriteLine(
                $"[Training] Plateau restarts: max={options.PlateauRestartCount}  lrScale={options.PlateauRestartLrScale:F2}");
            Console.WriteLine(
                $"[Training] Synthetic training: enabled={options.EnableSyntheticTraining}  " +
                $"unlockCoreQ={options.SyntheticUnlockCoreQ:F3}  warmup={options.SelfSupervisedWarmupEpochs}");
        }

        double bestQuality     = persistedTrainingState?.BestQuality ?? -1.0;
        double smoothedQuality = persistedTrainingState is not null && persistedTrainingState.SmoothedQuality > 0
            ? persistedTrainingState.SmoothedQuality
            : -1.0;
        const double EmaAlpha  = 0.4;
        int    stagnationEpochs = persistedTrainingState?.StagnationEpochs ?? 0;
        int    lrReductions     = persistedTrainingState?.LrReductions ?? 0;
        int    plateauRestartsUsed = persistedTrainingState?.PlateauRestartsUsed ?? 0;

        const int SlopeWindow = 10;
        var lossHistory = new List<double>(SlopeWindow + 1);
        var genQHistory = new List<double>(SlopeWindow + 1);
        Dictionary<string, ValidationPairSnapshot>? previousValidationSnapshots = null;
        double? previousLoss = null;
        double? previousPlBce = null;
        double? previousCoreQuality = null;
        double? previousFullQuality = null;
        double? previousValidationSeconds = null;
        int bestEpochNumber = persistedTrainingState?.BestEpochNumber ?? 0;

        if (dashboard is not null) dashboard.AddNotice("Saving initial best snapshot");
        else Console.WriteLine("[Training] Saving initial best-weight snapshot...");
        _placementTrainer.SaveBestWeights();
        if (dashboard is not null) dashboard.AddNotice("Initial best snapshot ready");
        else Console.WriteLine("[Training] Initial best-weight snapshot ready.");

        // Restore dashboard Best panel from persisted state so it is never blank on warm restart
        if (persistedTrainingState is not null && bestQuality >= 0)
        {
            dashboard?.SetRestoredBestEpoch(bestEpochNumber, bestQuality, 0);
            logger.Log("RESTORE", $"Best quality gate restored: {bestQuality:F3} at epoch {bestEpochNumber}");
        }
        int stableSyntheticRefreshes = 0;

        var epochRng = new Random((int)(options.RandomSeed & int.MaxValue));

        // Self-supervised synthetic examples — refreshed each epoch after warmup.
        // Stored as single-step sequences (sorted by beat within each synthetic batch).
        var currentSyntheticSeqs = new List<List<TrainingExample>>();

        for (int epoch = 0; epoch < options.Epochs; epoch++)
        {
            var epochStopwatch = Stopwatch.StartNew();
            dashboard?.BeginEpoch(epoch + 1, options.Epochs);

            // Shuffle sequences (not individual examples) each epoch
            var shuffledSeqs = FisherYatesShuffleSeqs(trainSeqs, epochRng.Next());
            bool syntheticMerged = false;
            bool badNegativesMerged = false;

            if (badNegativeSeqs.Count > 0)
            {
                var badBudget = ComputeBadNegativeBudget(trainSeqs);
                shuffledSeqs = MergeSequences(
                    shuffledSeqs,
                    LimitSyntheticSequencesBalanced(
                        badNegativeSeqs,
                        badBudget.MaxSequenceCount,
                        badBudget.MaxExampleCount,
                        epochRng.Next()),
                    epochRng.Next());
                badNegativesMerged = true;
            }

            bool syntheticTrainingUnlocked = options.EnableSyntheticTraining &&
                stableSyntheticRefreshes > 0;

            if (syntheticTrainingUnlocked &&
                currentSyntheticSeqs.Count > 0 &&
                epoch >= options.SelfSupervisedWarmupEpochs)
            {
                var syntheticBudget = ComputeSyntheticBudget(trainSeqs, stableSyntheticRefreshes);
                shuffledSeqs = MergeSequences(
                    shuffledSeqs,
                    LimitSyntheticSequencesBalanced(
                        currentSyntheticSeqs,
                        syntheticBudget.MaxSequenceCount,
                        syntheticBudget.MaxExampleCount,
                        epochRng.Next()),
                    epochRng.Next());
                syntheticMerged = true;
            }

            var epochMix = SummarizeSequences(shuffledSeqs);
            ValidateDifficultyCoverageOrThrow(epochMix);

            // Log class balance so we can diagnose posWt and bad-neg coverage per epoch
            int epochPosCount  = shuffledSeqs.Sum(s => s.Count(ex => ex.HasNote));
            int epochRealNeg   = shuffledSeqs.Sum(s => s.Count(ex => !ex.HasNote && !ex.IsNegativeSupervision));
            int epochBadNeg    = shuffledSeqs.Sum(s => s.Count(ex => ex.IsNegativeSupervision));
            logger.Log("BALANCE",
                $"ep={epoch + 1}  pos={epochPosCount}  realNeg={epochRealNeg}  badNeg={epochBadNeg}" +
                $"  posRatio={epochRealNeg / (double)Math.Max(1, epochPosCount):F2}x" +
                $"  badNeg%={100.0 * epochBadNeg / Math.Max(1, epochPosCount + epochRealNeg + epochBadNeg):F1}");

            var trainStopwatch = Stopwatch.StartNew();
            var (loss, plBce) = _placementTrainer.TrainEpoch(shuffledSeqs, lr);
            double trainSeconds = trainStopwatch.Elapsed.TotalSeconds;

            double coreQualityScore = 0;
            double fullQualityScore = 0;
            double validationSeconds = 0;
            int activeValidationPairs = 0;
            int activeValidationSongs = 0;
            ValidationDiagnostics? diagnostics = null;
            var currentValidationPlan = validationPlan;
            bool shouldRunValidation = currentValidationPlan is not null &&
                ((epoch + 1) % Math.Max(1, options.ValidationEveryNEpochs) == 0);
            if (shouldRunValidation && currentValidationPlan is not null)
            {
                var validationStopwatch = Stopwatch.StartNew();
                int absoluteEpoch = resumeEpochOffset + epoch + 1;
                activeValidationSongs = currentValidationPlan.ActiveFullSongCountForEpoch(absoluteEpoch);
                activeValidationPairs = CountPairsForSongs(currentValidationPlan.FullPairs, activeValidationSongs);
                var activePairs = currentValidationPlan.FullPairs.Take(activeValidationPairs).ToArray();
                int corePairCount = Math.Min(currentValidationPlan.CorePairs.Length, activePairs.Length);
                int validationWorkers = DetermineValidationParallelism(options);
                if (dashboard is not null)
                    dashboard.OnValidationStarted(activeValidationPairs, activeValidationSongs, validationWorkers);
                else
                    Console.WriteLine(
                        $"[Training] Starting generation validation: pairs={activeValidationPairs}  " +
                        $"songs={activeValidationSongs}  workers={validationWorkers}");
                var (coreScore, fullScore, newSynthetics, newDiagnostics) =
                    RunGenerationValidationWithFeedback(activePairs, corePairCount, options, previousValidationSnapshots);
                coreQualityScore = coreScore;
                fullQualityScore = fullScore;
                diagnostics = newDiagnostics;
                previousValidationSnapshots = newDiagnostics.PairSnapshots;
                validationSeconds = validationStopwatch.Elapsed.TotalSeconds;

                // Per-difficulty breakdown (logged after the main epoch line)
                if (diagnostics.DiffScores.Count > 1)
                {
                    var parts = diagnostics.DiffScores
                        .OrderBy(kv => (int)kv.Key)
                        .Select(kv =>
                        {
                            diagnostics.PreviousDiffScores.TryGetValue(kv.Key, out double previous);
                            bool hasPrevious = diagnostics.PreviousDiffScores.ContainsKey(kv.Key);
                            string scoreText = hasPrevious
                                ? FormatHigherIsBetter(kv.Value, previous, "F3")
                                : kv.Value.ToString("F3");
                            return $"{kv.Key}={scoreText}";
                        });
                    if (dashboard is null)
                        Console.WriteLine($"[Training]   genQ by diff: {string.Join("  ", parts)}");
                }

                // Refresh synthetic pool on the configured cadence.
                // Each batch from a generated map becomes one sequence (sorted by beat).
                if ((epoch + 1) % Math.Max(1, options.SelfSupervisedEveryNEpochs) == 0)
                {
                    if (ShouldAcceptSyntheticRefresh(coreQualityScore, fullQualityScore, diagnostics) &&
                        options.EnableSyntheticTraining &&
                        coreQualityScore >= options.SyntheticUnlockCoreQ)
                    {
                        stableSyntheticRefreshes++;
                        var syntheticBudget = ComputeSyntheticBudget(trainSeqs, stableSyntheticRefreshes);
                        currentSyntheticSeqs = LimitSyntheticSequencesBalanced(
                            new List<List<TrainingExample>>(newSynthetics),
                            syntheticBudget.MaxSequenceCount,
                            syntheticBudget.MaxExampleCount,
                            epochRng.Next());
                        if (dashboard is not null)
                            dashboard.AddNotice($"Synthetic refresh accepted streak {stableSyntheticRefreshes}");
                        else
                            Console.WriteLine(
                                $"[Training]   synthetic refresh accepted: streak={stableSyntheticRefreshes}  " +
                                $"budgetSeq={syntheticBudget.MaxSequenceCount}  budgetEx={syntheticBudget.MaxExampleCount}  " +
                                $"frac={syntheticBudget.FractionOfRealExamples:F3}");
                    }
                    else
                    {
                        stableSyntheticRefreshes = 0;
                        string reason = !options.EnableSyntheticTraining
                            ? "disabled"
                            : coreQualityScore < options.SyntheticUnlockCoreQ
                                ? $"coreQ<{options.SyntheticUnlockCoreQ:F3}"
                                : "quality gate";
                        if (dashboard is not null)
                            dashboard.AddNotice($"Synthetic skipped {reason}");
                        else
                            Console.WriteLine(
                                $"[Training]   synthetic refresh skipped: reason={reason}  coreQ={coreQualityScore:F3}  " +
                                $"fullQ={fullQualityScore:F3}  okPairs={diagnostics.SuccessfulPairs}/{diagnostics.TotalPairs}");
                    }
                }
            }
            else if (currentValidationPlan is not null)
            {
                if (dashboard is not null) dashboard.AddNotice("Validation skipped this epoch");
                else
                    Console.WriteLine(
                        $"[Training]   validation skipped this epoch (cadence={Math.Max(1, options.ValidationEveryNEpochs)}).");
            }

            // ── Improvement check (raw quality) ──────────────────────────────────
            // neural_placement.pt is ONLY written when a new best is reached.
            bool isNewBest = shouldRunValidation && coreQualityScore > bestQuality + 1e-4;
            if (isNewBest)
            {
                bestQuality = coreQualityScore;
                bestEpochNumber = epoch + 1;
                _placementTrainer.SaveBestWeights();
                _placementTrainer.SaveBestArtifact(options.ArtifactsOutputPath);
                _placementTrainer.SaveWeights(options.ArtifactsOutputPath);
                dashboard?.AddNotice($"New best epoch {bestEpochNumber} coreQ {bestQuality:F3}");
                logger.Log("BEST", $"New best: epoch={bestEpochNumber}  coreQ={bestQuality:F3}  fullQ={fullQualityScore:F3}");
            }

            _placementTrainer.SaveCurrentWeights(options.ArtifactsOutputPath);

            // EMA for display only (no longer drives LR/stop decisions)
            if (shouldRunValidation)
            {
                smoothedQuality = smoothedQuality < 0
                    ? coreQualityScore
                    : EmaAlpha * coreQualityScore + (1 - EmaAlpha) * smoothedQuality;
            }

            // ── Trend detection — linear regression over last SlopeWindow epochs ─
            lossHistory.Add(loss);
            if (lossHistory.Count > SlopeWindow) lossHistory.RemoveAt(0);
            if (shouldRunValidation)
            {
                genQHistory.Add(coreQualityScore);
                if (genQHistory.Count > SlopeWindow) genQHistory.RemoveAt(0);
            }

            double lossSlope = ComputeLinearSlope(lossHistory); // negative = loss improving
            double genQSlope = ComputeLinearSlope(genQHistory);  // positive = quality improving

            // Stagnation: neither loss nor quality is trending in the right direction.
            // Thresholds are intentionally lenient: loss slope < -0.001/epoch and
            // quality slope > 0.0005/epoch represent genuine slow-but-real learning
            // that should NOT trigger stagnation (especially near the 0.4xx plateau).
            bool isLossImproving = lossSlope < -0.001;
            bool isGenQImproving = genQSlope > 0.0005;
            bool isStagnating    = shouldRunValidation && !isLossImproving && !isGenQImproving;

            if (shouldRunValidation)
            {
                if (isStagnating)
                    stagnationEpochs++;
                else
                    stagnationEpochs = Math.Max(0, stagnationEpochs - 1);
            }

            // ── Epoch log ────────────────────────────────────────────────────────
            string statusTag = isNewBest
                ? $"  {ConsoleStyler.Colorize("** NEW BEST **", ConsoleColor.Green)}"
                : stagnationEpochs > 0
                    ? $"  {ConsoleStyler.Colorize($"[stag: {stagnationEpochs}/{options.EarlyStopPatience}]", ConsoleColor.Yellow)}"
                    : string.Empty;

            string lossText = FormatLowerIsBetter(loss, previousLoss, "F4");
            string plBceText = FormatLowerIsBetter(plBce, previousPlBce, "F4");
            string coreQText = shouldRunValidation
                ? FormatHigherIsBetter(coreQualityScore, previousCoreQuality, "F3")
                : "skip";
            string fullQText = shouldRunValidation
                ? FormatHigherIsBetter(fullQualityScore, previousFullQuality, "F3")
                : "skip";
            string valText = shouldRunValidation
                ? FormatLowerIsBetter(validationSeconds, previousValidationSeconds, "F1")
                : validationSeconds.ToString("F1");

            string diffLine = string.Empty;
            string validationInfo = string.Empty;
            string validationDelta = string.Empty;
            if (currentValidationPlan is not null)
            {
                var epochSyntheticBudget = ComputeSyntheticBudget(trainSeqs, stableSyntheticRefreshes);
                validationInfo =
                    $"val {activeValidationSongs}/{currentValidationPlan.FullSongCap} synth {epochSyntheticBudget.MaxSequenceCount}/{epochSyntheticBudget.MaxExampleCount}";
                if (diagnostics is not null)
                {
                    var noteParts = diagnostics.AvgGeneratedNotesByDifficulty.Count > 0
                        ? string.Join("  ", diagnostics.AvgGeneratedNotesByDifficulty
                            .OrderBy(kv => (int)kv.Key)
                            .Select(kv => $"{kv.Key}Notes={kv.Value:F1}"))
                        : "none";
                    diffLine = diagnostics.DiffScores.Count > 0
                        ? string.Join("  ", diagnostics.DiffScores.OrderBy(kv => (int)kv.Key).Select(kv => $"{kv.Key}={kv.Value:F3}"))
                        : string.Empty;
                    validationInfo =
                        $"ok {diagnostics.SuccessfulPairs}/{diagnostics.TotalPairs} cand {diagnostics.AvgCandidateSurvival:F3} notes {diagnostics.AvgDecodeRate:F3}";
                    validationDelta =
                        $"chg {diagnostics.ChangedPairs}/{diagnostics.TotalPairs} up {diagnostics.ScoreImprovedPairs} dn {diagnostics.ScoreWorsenedPairs} dQ {diagnostics.AvgAbsScoreDelta:F4}";
                    if (diagnostics.RawIssueCounts.Count > 0)
                    {
                        string topRules = string.Join("  ",
                            diagnostics.RawIssueCounts
                                .OrderByDescending(kv => kv.Value)
                                .Take(5)
                                .Select(kv => $"{kv.Key}={kv.Value}"));
                        if (dashboard is not null)
                            dashboard.AddNotice($"Issues {topRules}");
                        else
                            Console.WriteLine($"[Training]   raw issue mix: {topRules}");
                    }
                    if (dashboard is null)
                    {
                        Console.WriteLine(
                            $"[Training] Epoch {epoch + 1}/{options.Epochs}: " +
                            $"loss={lossText}  plBCE={plBceText}  " +
                            $"coreQ={coreQText}  " +
                            $"fullQ={fullQText}  " +
                            $"best={(bestQuality >= 0 ? bestQuality.ToString("F3") : "n/a")}  " +
                            $"sm={(smoothedQuality >= 0 ? smoothedQuality.ToString("F3") : "n/a")}  lr={lr:G4}" +
                            $"  lSlp={lossSlope:+0.0000;-0.0000}  qSlp={genQSlope:+0.0000;-0.0000}" +
                            $"  train={trainSeconds:F1}s  val={valText}s  epoch={epochStopwatch.Elapsed.TotalSeconds:F1}s" +
                            (activeValidationPairs > 0 ? $"  valPairs={activeValidationPairs}" : string.Empty) +
                            (currentSyntheticSeqs.Count > 0 ? $"  synthEx={currentSyntheticSeqs.Sum(s => s.Count)}" : string.Empty) +
                            statusTag);

                        Console.WriteLine(
                            $"[Training]   train mix: seq={epochMix.SequenceCount}  ex={epochMix.ExampleCount}  " +
                            $"synthetic={(syntheticMerged ? "on" : "off")}  badNeg={(badNegativesMerged ? "on" : "off")}  {epochMix.Mix}");
                        Console.WriteLine($"[Training]   validation stage: coreSongs={currentValidationPlan.CoreSongCount}  activeFullSongs={activeValidationSongs}/{currentValidationPlan.FullSongCap}");
                        Console.WriteLine(
                            $"[Training]   synthetic budget: enabled={options.EnableSyntheticTraining}  " +
                            $"streak={stableSyntheticRefreshes}  " +
                            $"seqCap={epochSyntheticBudget.MaxSequenceCount}  exCap={epochSyntheticBudget.MaxExampleCount}  " +
                            $"frac={epochSyntheticBudget.FractionOfRealExamples:F3}");
                        Console.WriteLine(
                            $"[Training]   val diag: ok={diagnostics.SuccessfulPairs}/{diagnostics.TotalPairs}  " +
                            $"coreOk={diagnostics.CoreSuccessfulPairs}/{Math.Max(1, diagnostics.CoreSuccessfulPairs + diagnostics.CoreFailedPairs)}  " +
                            $"avgAll={diagnostics.AvgScoreAllPairs:F3}  cand->prop={diagnostics.AvgCandidateSurvival:F3}  " +
                            $"rawQ={diagnostics.AvgRawScore:F3}  repairedQ={diagnostics.AvgRepairedScore:F3}  " +
                            $"prop->notes={diagnostics.AvgDecodeRate:F3}  rawNotes={diagnostics.AvgRawNotes:F1}  repairedNotes={diagnostics.AvgRepairedNotes:F1}  " +
                            $"repairs={diagnostics.AvgRepairCount:F1}  noteDelta={diagnostics.AvgRepairNoteDeltaRatio:F3}  " +
                            $"newSynthEx={diagnostics.TotalSyntheticExamples}  synth/pair={diagnostics.AvgSyntheticExamplesPerAcceptedPair:F1}");
                        Console.WriteLine($"[Training]   val notes by diff: {noteParts}");
                        Console.WriteLine(
                            $"[Training]   val shape: right={diagnostics.AvgRightHandFraction:F3}  " +
                            $"dot={diagnostics.AvgDotFraction:F3}  chordBeats={diagnostics.AvgChordBeatFraction:F3}");
                        Console.WriteLine(
                            $"[Training]   val delta: changed={FormatCountComparison(diagnostics.ChangedPairs)}" +
                            $"/{diagnostics.TotalPairs}  fpChanged={FormatCountComparison(diagnostics.FingerprintChangedPairs)}  " +
                            $"notesChanged={FormatCountComparison(diagnostics.NoteCountChangedPairs)}  " +
                            $"scoreUp={FormatCountComparison(diagnostics.ScoreImprovedPairs)}  " +
                            $"scoreDown={FormatCountRegression(diagnostics.ScoreWorsenedPairs)}  " +
                            $"|dScore|={FormatMovementMagnitude(diagnostics.AvgAbsScoreDelta, 0.0005, 0.0050, "F4")}  " +
                            $"|dRawQ|={FormatMovementMagnitude(diagnostics.AvgAbsRawScoreDelta, 0.0005, 0.0050, "F4")}  " +
                            $"|dNotes|={FormatMovementMagnitude(diagnostics.AvgAbsRawNoteDelta, 0.1, 2.0, "F1")}");
                    }
                }
            }
            else if (dashboard is null)
            {
                Console.WriteLine(
                    $"[Training] Epoch {epoch + 1}/{options.Epochs}: " +
                    $"loss={lossText}  plBCE={plBceText}  " +
                    $"coreQ={coreQText}  " +
                    $"fullQ={fullQText}  " +
                    $"best={(bestQuality >= 0 ? bestQuality.ToString("F3") : "n/a")}  " +
                    $"sm={(smoothedQuality >= 0 ? smoothedQuality.ToString("F3") : "n/a")}  lr={lr:G4}" +
                    $"  lSlp={lossSlope:+0.0000;-0.0000}  qSlp={genQSlope:+0.0000;-0.0000}" +
                    $"  train={trainSeconds:F1}s  val={valText}s  epoch={epochStopwatch.Elapsed.TotalSeconds:F1}s" +
                    (activeValidationPairs > 0 ? $"  valPairs={activeValidationPairs}" : string.Empty) +
                    (currentSyntheticSeqs.Count > 0 ? $"  synthEx={currentSyntheticSeqs.Sum(s => s.Count)}" : string.Empty) +
                    statusTag);
                Console.WriteLine(
                    $"[Training]   train mix: seq={epochMix.SequenceCount}  ex={epochMix.ExampleCount}  " +
                    $"synthetic={(syntheticMerged ? "on" : "off")}  badNeg={(badNegativesMerged ? "on" : "off")}  {epochMix.Mix}");
            }

            dashboard?.CompleteEpoch(new TrainingConsoleDashboard.EpochSummary(
                Epoch: epoch + 1,
                Epochs: options.Epochs,
                ValidationRan: shouldRunValidation,
                Loss: loss,
                PlaceBce: plBce,
                CoreQ: coreQualityScore,
                FullQ: fullQualityScore,
                BestQ: bestQuality,
                SmoothedQ: smoothedQuality,
                LearningRate: lr,
                LossSlope: lossSlope,
                QualitySlope: genQSlope,
                TrainSeconds: trainSeconds,
                ValidationSeconds: validationSeconds,
                EpochSeconds: epochStopwatch.Elapsed.TotalSeconds,
                ValidationPairs: activeValidationPairs,
                SyntheticExamples: currentSyntheticSeqs.Sum(s => s.Count),
                StagnationEpochs: stagnationEpochs,
                EarlyStopPatience: options.EarlyStopPatience,
                TrainMix: epochMix.Mix,
                ValidationInfo: validationInfo,
                ValidationDelta: validationDelta,
                DiffScores: diffLine,
                Status: isNewBest ? $"NEW BEST #{bestEpochNumber}" : (stagnationEpochs > 0 ? $"stagnation {stagnationEpochs}/{options.EarlyStopPatience}" : "running")));

            // ── File log: unified epoch summary (always written, regardless of dashboard) ─
            logger.Log("EPOCH",
                $"ep={epoch + 1}/{options.Epochs}" +
                $"  loss={loss:F4}  plBCE={plBce:F4}" +
                $"  coreQ={coreQualityScore:F3}  fullQ={fullQualityScore:F3}  bestQ={bestQuality:F3}" +
                $"  sm={smoothedQuality:F3}  lr={lr:G4}" +
                $"  lSlp={lossSlope:+0.0000;-0.0000}  qSlp={genQSlope:+0.0000;-0.0000}" +
                $"  stag={stagnationEpochs}  train={trainSeconds:F1}s  val={validationSeconds:F1}s" +
                (isNewBest ? "  *** NEW BEST ***" : string.Empty));
            if (shouldRunValidation && diagnostics is not null)
            {
                logger.Log("VALID",
                    $"ep={epoch + 1}  ok={diagnostics.SuccessfulPairs}/{diagnostics.TotalPairs}" +
                    $"  rawQ={diagnostics.AvgRawScore:F3}  repQ={diagnostics.AvgRepairedScore:F3}" +
                    $"  cand={diagnostics.AvgCandidateSurvival:F3}  decode={diagnostics.AvgDecodeRate:F3}" +
                    $"  rawNotes={diagnostics.AvgRawNotes:F1}  repairedNotes={diagnostics.AvgRepairedNotes:F1}" +
                    $"  repairs={diagnostics.AvgRepairCount:F1}  noteDelta={diagnostics.AvgRepairNoteDeltaRatio:F3}" +
                    $"  synth={diagnostics.TotalSyntheticExamples}  right={diagnostics.AvgRightHandFraction:F3}");
                if (diagnostics.DiffScores.Count > 0)
                    logger.Log("VALID",
                        $"ep={epoch + 1}  diffScores: " +
                        string.Join("  ", diagnostics.DiffScores.OrderBy(kv => (int)kv.Key)
                            .Select(kv => $"{kv.Key}={kv.Value:F3}")));
                if (diagnostics.RawIssueCounts.Count > 0)
                    logger.Log("VALID",
                        $"ep={epoch + 1}  topIssues: " +
                        string.Join("  ", diagnostics.RawIssueCounts.OrderByDescending(kv => kv.Value)
                            .Take(8).Select(kv => $"{kv.Key}={kv.Value}")));
            }

            // ── Stagnation-driven LR reduction and early stop ────────────────────
            if (stagnationEpochs > 0)
            {
                // Periodic checkpoint marker (no file write)
                if (options.CheckpointEveryNEpochs > 0
                    && (epoch + 1) % options.CheckpointEveryNEpochs == 0)
                {
                    if (dashboard is not null) dashboard.AddNotice($"Checkpoint marker epoch {epoch + 1}");
                    else
                        Console.WriteLine(
                            $"[Training] Checkpoint epoch {epoch + 1} — best so far: {bestQuality:F3}");
                }

                // Reduce LR when stagnation counter hits the lr-patience threshold
                if (stagnationEpochs % lrPatience == 0 && lr > MinLr * 1.1)
                {
                    lr = Math.Max(lr * LrDecay, MinLr);
                    lrReductions++;
                    if (dashboard is not null) dashboard.AddNotice($"LR reduced to {lr:G4} (#{lrReductions})");
                    else
                        Console.WriteLine(
                            $"[Training] Stagnation plateau — reducing lr to {lr:G4} (reduction #{lrReductions})");
                    logger.Log("LR",
                        $"Reduced to {lr:G4} (reduction #{lrReductions})  stagnation={stagnationEpochs}  epoch={epoch + 1}");
                }

                if (options.EarlyStopPatience > 0 && stagnationEpochs >= options.EarlyStopPatience)
                {
                    if (plateauRestartsUsed < Math.Max(0, options.PlateauRestartCount))
                    {
                        plateauRestartsUsed++;
                        double restartScale = Math.Clamp(options.PlateauRestartLrScale, 0.05, 0.95);
                        double restartedLr = Math.Max(lr * restartScale, MinLr);
                        if (dashboard is not null)
                            dashboard.AddNotice($"Hot restart {plateauRestartsUsed}/{options.PlateauRestartCount} lr {lr:G4}->{restartedLr:G4}");
                        else
                            Console.WriteLine(
                                $"[Training] Plateau detected at epoch {epoch + 1} — hot restart {plateauRestartsUsed}/{options.PlateauRestartCount} " +
                                $"from best weights, lr {lr:G4} -> {restartedLr:G4}.");
                        logger.Log("RESTART",
                            $"Hot restart #{plateauRestartsUsed}/{options.PlateauRestartCount}" +
                            $"  lr={lr:G4} -> {restartedLr:G4}  bestQ={bestQuality:F3}  epoch={epoch + 1}");

                        _placementTrainer.RestoreBestWeights();
                        _placementTrainer.SaveCurrentWeights(options.ArtifactsOutputPath);

                        lr = restartedLr;
                        stagnationEpochs = 0;
                        stableSyntheticRefreshes = 0;
                        lossHistory.Clear();
                        genQHistory.Clear();
                        previousLoss = null;
                        previousPlBce = null;
                        previousCoreQuality = null;
                        previousFullQuality = null;
                        previousValidationSeconds = null;
                        previousValidationSnapshots = null;

                        if (validationPlan is not null)
                            SaveValidationSchedule(validationStatePath, options.RandomSeed, validationPlan, resumeEpochOffset + epoch + 1);
                        SaveTrainingState(trainingStatePath, new PersistedTrainingState(
                            BestQuality:         bestQuality,
                            BestEpochNumber:     bestEpochNumber,
                            StagnationEpochs:    0,
                            LrReductions:        lrReductions,
                            PlateauRestartsUsed: plateauRestartsUsed,
                            SmoothedQuality:     smoothedQuality >= 0 ? smoothedQuality : bestQuality,
                            CurrentLr:           lr,
                            TotalEpochsCompleted: resumeEpochOffset + epoch + 1));
                        continue;
                    }

                    if (dashboard is not null) dashboard.AddNotice($"Early stop epoch {epoch + 1} best {bestQuality:F3}");
                    else
                        Console.WriteLine(
                            $"[Training] Early stopping at epoch {epoch + 1} " +
                            $"(stagnation for {options.EarlyStopPatience} epochs). " +
                            $"Best genQuality={bestQuality:F3}");
                    logger.Log("STOP",
                        $"Early stop: epoch={epoch + 1}  stagnation={stagnationEpochs}  bestQ={bestQuality:F3}  bestEpoch={bestEpochNumber}");
                    break;
                }
            }

            if (validationPlan is not null)
                SaveValidationSchedule(validationStatePath, options.RandomSeed, validationPlan, resumeEpochOffset + epoch + 1);

            SaveTrainingState(trainingStatePath, new PersistedTrainingState(
                BestQuality:         bestQuality,
                BestEpochNumber:     bestEpochNumber,
                StagnationEpochs:    stagnationEpochs,
                LrReductions:        lrReductions,
                PlateauRestartsUsed: plateauRestartsUsed,
                SmoothedQuality:     smoothedQuality >= 0 ? smoothedQuality : bestQuality,
                CurrentLr:           lr,
                TotalEpochsCompleted: resumeEpochOffset + epoch + 1));

            previousLoss = loss;
            previousPlBce = plBce;
            if (shouldRunValidation)
            {
                previousCoreQuality = coreQualityScore;
                previousFullQuality = fullQualityScore;
                previousValidationSeconds = validationSeconds;
            }
        }

        _placementTrainer.RestoreBestWeights();

        // ----------------------------------------------------------------
        // 6. Save artifacts
        // ----------------------------------------------------------------
        _placementTrainer.SaveBestArtifact(options.ArtifactsOutputPath);
        _placementTrainer.SaveWeights(options.ArtifactsOutputPath);
        _placementTrainer.SaveCurrentWeights(options.ArtifactsOutputPath);

        // ----------------------------------------------------------------
        // 7. Final test-set evaluation
        // ----------------------------------------------------------------
        if (testEx.Count > 0)
        {
            var metrics = _evaluator.Evaluate(testEx, _placementTrainer, options.ArtifactsOutputPath);
            Console.WriteLine(
                $"[Training] F1={metrics.PlacementF1:F3}  " +
                $"Precision={metrics.PlacementPrecision:F3}  " +
                $"Recall={metrics.PlacementRecall:F3}");
            logger.Log("TEST",
                $"F1={metrics.PlacementF1:F3}  Precision={metrics.PlacementPrecision:F3}  Recall={metrics.PlacementRecall:F3}");
        }

        logger.Log("DONE",
            $"bestQ={bestQuality:F3}  bestEpoch={bestEpochNumber}  lrReductions={lrReductions}  plateauRestarts={plateauRestartsUsed}");

        if (dashboard is not null)
            dashboard.MarkComplete($"Done. Best generation quality={bestQuality:F3}");
        else
            Console.WriteLine(
                $"[Training] Done. Best generation quality={bestQuality:F3}. " +
                $"Artifacts in '{options.ArtifactsOutputPath}'.");
        dashboard?.Dispose();
        _dashboard = null;
        _placementTrainer.Dashboard = null;
        _logger = null;
        // logger disposed by 'using' — writes the session-end marker
    }

    // ----------------------------------------------------------------
    // Per-epoch generation validation + self-supervised example generation
    // Both happen in the same generation pass — no extra cost.
    // ----------------------------------------------------------------

    private (double CoreQuality, double FullQuality, IReadOnlyList<List<TrainingExample>> Synthetics,
             ValidationDiagnostics Diagnostics)
        RunGenerationValidationWithFeedback(
            CachedPair[] cachedPairs,
            int corePairCount,
            TrainingOptions options,
            IReadOnlyDictionary<string, ValidationPairSnapshot>? previousSnapshots)
    {
        var outcomes = new ValidationPairOutcome[cachedPairs.Length];
        var synthBag = new ConcurrentBag<List<TrainingExample>>();
        int validationParallelism = DetermineValidationParallelism(options);
        const double MinSyntheticQuality = 0.24;
        var progressStopwatch = Stopwatch.StartNew();
        int completedPairs = 0;
        long nextProgressMs = 15_000;

        Parallel.For(0, cachedPairs.Length,
            new ParallelOptions { MaxDegreeOfParallelism = validationParallelism },
            i =>
            {
                var cached = cachedPairs[i];
                try
                {
                    var svc  = new MapGenerationService();
                    var eval = new GenerationQualityEvaluator();
                    var gen  = new SelfSupervisedExampleGenerator();

                    var settings = new GenerationSettings(
                        TargetDifficulty: cached.Pair.ReferenceMap.Difficulty.Difficulty,
                        AllowBombs:       false,
                        AllowObstacles:   false,
                        AllowFieldMovement: false,
                        RandomSeed:       options.RandomSeed,
                        UseLearned:       true,
                        ArtifactsPath:    null);

                    var result = svc.Generate(
                        cached.Audio,
                        cached.Pair.ReferenceMap.Song,
                        settings,
                        placementScorer: GetValidationPlacementModel(options));

                    double rawScore = eval.Evaluate(result.RawBeatmap, cached.Pair.ReferenceMap).OverallScore;
                    double repairedScore = eval.Evaluate(result.Beatmap, cached.Pair.ReferenceMap).OverallScore;
                    double score = ComputeEffectiveValidationScore(
                        rawScore,
                        repairedScore,
                        result.Telemetry.RepairCount,
                        result.Telemetry.RepairNoteDeltaRatio);

                    // Derive synthetic training examples from this generation pass
                    var synthetics = gen.Generate(
                        result.RawBeatmap,
                        cached.Audio,
                        (int)settings.TargetDifficulty,
                        options.SelfSupervisedPositiveWeight,
                        options.SelfSupervisedNegativeWeight);
                    bool acceptSynthetic = rawScore >= MinSyntheticQuality && synthetics.Count > 0;

                    int rawNoteCount = result.RawBeatmap.Notes.Count;
                    double rightHandFraction = rawNoteCount > 0
                        ? result.RawBeatmap.Notes.Count(n => n.Hand == NoteHand.Right) / (double)rawNoteCount
                        : 0.5;
                    double dotFraction = rawNoteCount > 0
                        ? result.RawBeatmap.Notes.Count(n => n.CutDirection == CutDirection.Dot) / (double)rawNoteCount
                        : 0.0;
                    double chordBeatFraction = ComputeChordBeatFraction(result.RawBeatmap);
                    ulong rawFingerprint = ComputeBeatmapFingerprint(result.RawBeatmap);
                    string pairKey = BuildValidationPairKey(cached.Pair);

                    if (acceptSynthetic)
                        synthBag.Add(synthetics.OrderBy(s => s.Beat).ToList());

                    outcomes[i] = new ValidationPairOutcome(
                        pairKey,
                        cached.Pair.ReferenceMap.Difficulty.Difficulty,
                        score,
                        rawScore,
                        repairedScore,
                        rawNoteCount,
                        result.Telemetry.CandidateCount,
                        result.Telemetry.ProposedCount,
                        result.Telemetry.DecodedNoteCount,
                        result.Telemetry.RepairedNoteCount,
                        result.Telemetry.RepairCount,
                        result.Telemetry.RepairNoteDeltaRatio,
                        result.ValidationReport.Issues.Select(issue => issue.RuleName).ToArray(),
                        rightHandFraction,
                        dotFraction,
                        chordBeatFraction,
                        acceptSynthetic ? synthetics.Count : 0,
                        rawFingerprint,
                        Failed: false);
                }
                catch
                {
                    outcomes[i] = new ValidationPairOutcome(
                        BuildValidationPairKey(cached.Pair),
                        cached.Pair.ReferenceMap.Difficulty.Difficulty,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        [],
                        0.5,
                        0.0,
                        0.0,
                        0,
                        0,
                        Failed: true);
                }

                int done = Interlocked.Increment(ref completedPairs);
                long elapsedMs = progressStopwatch.ElapsedMilliseconds;
                while (true)
                {
                    long targetMs = Interlocked.Read(ref nextProgressMs);
                    bool finalUpdate = done == cachedPairs.Length;
                    if (!finalUpdate && elapsedMs < targetMs)
                        break;

                    long nextTargetMs = finalUpdate ? targetMs : targetMs + 15_000;
                    if (Interlocked.CompareExchange(ref nextProgressMs, nextTargetMs, targetMs) != targetMs)
                        continue;

                    double elapsedSeconds = progressStopwatch.Elapsed.TotalSeconds;
                    if (_dashboard is not null)
                        _dashboard.OnValidationProgress(done, cachedPairs.Length, elapsedSeconds);
                    else
                        Console.WriteLine(
                            $"[Training]   validation progress: {done}/{cachedPairs.Length} pair(s)  " +
                            $"elapsed={elapsedSeconds:F0}s");
                    break;
                }
            });

        var valid = outcomes.Where(o => !o.Failed).ToArray();
        double fullQuality = valid.Length > 0 ? valid.Average(x => x.Score) : 0;

        var coreOutcomes = outcomes.Take(Math.Min(corePairCount, outcomes.Length)).ToArray();
        var coreValid = coreOutcomes.Where(o => !o.Failed).ToArray();
        double coreQuality = coreValid.Length > 0 ? coreValid.Average(x => x.Score) : 0;

        // Per-difficulty breakdown
        var diffScores = valid
            .GroupBy(x => x.Difficulty)
            .ToDictionary(g => g.Key, g => g.Average(x => x.Score));

        double avgCandidateSurvival = valid.Length > 0
            ? valid.Average(x => x.CandidateCount > 0 ? x.ProposedCount / (double)x.CandidateCount : 0)
            : 0;
        double avgDecodeRate = valid.Length > 0
            ? valid.Average(x => x.ProposedCount > 0 ? x.DecodedNoteCount / (double)x.ProposedCount : 0)
            : 0;
        double avgRawScore = valid.Length > 0 ? valid.Average(x => x.RawScore) : 0;
        double avgRepairedScore = valid.Length > 0 ? valid.Average(x => x.RepairedScore) : 0;
        double avgRawNotes = valid.Length > 0 ? valid.Average(x => x.RawNoteCount) : 0;
        double avgRepairedNotes = valid.Length > 0 ? valid.Average(x => x.RepairedNoteCount) : 0;
        double avgRepairCount = valid.Length > 0 ? valid.Average(x => x.RepairCount) : 0;
        double avgRepairNoteDeltaRatio = valid.Length > 0 ? valid.Average(x => x.RepairNoteDeltaRatio) : 0;
        double avgRightHandFraction = valid.Length > 0 ? valid.Average(x => x.RightHandFraction) : 0.5;
        double avgDotFraction = valid.Length > 0 ? valid.Average(x => x.DotFraction) : 0.0;
        double avgChordBeatFraction = valid.Length > 0 ? valid.Average(x => x.ChordBeatFraction) : 0.0;
        double avgScoreAllPairs = outcomes.Length > 0 ? outcomes.Average(x => x.Score) : 0;
        var avgGeneratedNotesByDifficulty = valid
            .GroupBy(x => x.Difficulty)
            .ToDictionary(g => g.Key, g => g.Average(x => x.RepairedNoteCount));
        var rawIssueCounts = valid
            .SelectMany(x => x.RawIssueRules)
            .GroupBy(rule => rule, StringComparer.OrdinalIgnoreCase)
            .OrderByDescending(g => g.Count())
            .ToDictionary(g => g.Key, g => g.Count(), StringComparer.OrdinalIgnoreCase);
        var pairSnapshots = valid
            .GroupBy(x => x.PairKey, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                g => g.Key,
                g =>
                {
                    var x = g.First();
                    return new ValidationPairSnapshot(
                        x.Score,
                        x.RawScore,
                        x.RawNoteCount,
                        x.RepairedNoteCount,
                        x.RawFingerprint);
                },
                StringComparer.OrdinalIgnoreCase);
        var previousDiffScores = previousSnapshots is not null
            ? previousSnapshots
                .Where(kv => pairSnapshots.ContainsKey(kv.Key))
                .Join(
                    valid,
                    kv => kv.Key,
                    outcome => outcome.PairKey,
                    (kv, outcome) => new { outcome.Difficulty, kv.Value.Score })
                .GroupBy(x => x.Difficulty)
                .ToDictionary(g => g.Key, g => g.Average(x => x.Score))
            : new Dictionary<DifficultyLevel, double>();

        int changedPairs = 0;
        int unchangedPairs = 0;
        int fingerprintChangedPairs = 0;
        int scoreImprovedPairs = 0;
        int scoreWorsenedPairs = 0;
        int noteCountChangedPairs = 0;
        double totalAbsScoreDelta = 0;
        double totalAbsRawScoreDelta = 0;
        double totalAbsRawNoteDelta = 0;
        int comparedPairs = 0;

        if (previousSnapshots is not null)
        {
            foreach (var outcome in valid)
            {
                if (!previousSnapshots.TryGetValue(outcome.PairKey, out var previous))
                    continue;

                comparedPairs++;
                double scoreDelta = outcome.Score - previous.Score;
                double rawScoreDelta = outcome.RawScore - previous.RawScore;
                double rawNoteDelta = outcome.RawNoteCount - previous.RawNoteCount;
                bool fingerprintChanged = outcome.RawFingerprint != previous.RawFingerprint;
                bool noteCountChanged = outcome.RawNoteCount != previous.RawNoteCount ||
                    outcome.RepairedNoteCount != previous.RepairedNoteCount;
                bool changed = fingerprintChanged ||
                    Math.Abs(scoreDelta) > 1e-6 ||
                    Math.Abs(rawScoreDelta) > 1e-6 ||
                    noteCountChanged;

                if (changed) changedPairs++;
                else unchangedPairs++;
                if (fingerprintChanged) fingerprintChangedPairs++;
                if (noteCountChanged) noteCountChangedPairs++;
                if (scoreDelta > 1e-6) scoreImprovedPairs++;
                else if (scoreDelta < -1e-6) scoreWorsenedPairs++;

                totalAbsScoreDelta += Math.Abs(scoreDelta);
                totalAbsRawScoreDelta += Math.Abs(rawScoreDelta);
                totalAbsRawNoteDelta += Math.Abs(rawNoteDelta);
            }
        }

        var diagnostics = new ValidationDiagnostics(
            TotalPairs: outcomes.Length,
            SuccessfulPairs: outcomes.Count(x => !x.Failed),
            FailedPairs: outcomes.Count(x => x.Failed),
            TotalSyntheticExamples: outcomes.Sum(x => x.SyntheticExampleCount),
            AvgCandidateSurvival: avgCandidateSurvival,
            AvgDecodeRate: avgDecodeRate,
            AvgRawScore: avgRawScore,
            AvgRepairedScore: avgRepairedScore,
            AvgRawNotes: avgRawNotes,
            AvgRepairedNotes: avgRepairedNotes,
            AvgRepairCount: avgRepairCount,
            AvgRepairNoteDeltaRatio: avgRepairNoteDeltaRatio,
            AvgRightHandFraction: avgRightHandFraction,
            AvgDotFraction: avgDotFraction,
            AvgChordBeatFraction: avgChordBeatFraction,
            AvgScoreAllPairs: avgScoreAllPairs,
            AvgSyntheticExamplesPerAcceptedPair: valid.Length > 0 ? valid.Average(x => x.SyntheticExampleCount) : 0,
            AvgGeneratedNotesByDifficulty: avgGeneratedNotesByDifficulty,
            DiffScores: diffScores,
            PreviousDiffScores: previousDiffScores,
            RawIssueCounts: rawIssueCounts,
            ChangedPairs: changedPairs,
            UnchangedPairs: unchangedPairs,
            FingerprintChangedPairs: fingerprintChangedPairs,
            ScoreImprovedPairs: scoreImprovedPairs,
            ScoreWorsenedPairs: scoreWorsenedPairs,
            NoteCountChangedPairs: noteCountChangedPairs,
            AvgAbsScoreDelta: comparedPairs > 0 ? totalAbsScoreDelta / comparedPairs : 0,
            AvgAbsRawScoreDelta: comparedPairs > 0 ? totalAbsRawScoreDelta / comparedPairs : 0,
            AvgAbsRawNoteDelta: comparedPairs > 0 ? totalAbsRawNoteDelta / comparedPairs : 0,
            PairSnapshots: pairSnapshots,
            CoreSuccessfulPairs: coreOutcomes.Count(x => !x.Failed),
            CoreFailedPairs: coreOutcomes.Count(x => x.Failed));

        return (coreQuality, fullQuality, synthBag.ToList(), diagnostics);
    }

    private IBatchedMultiTaskPlacementModel GetValidationPlacementModel(TrainingOptions options)
    {
        return options.ValidationUseGpuInference
            ? _placementTrainer
            : _placementTrainer.CpuInferenceModel;
    }

    private int DetermineValidationParallelism(TrainingOptions options)
    {
        if (options.ValidationWorkers > 0)
            return Math.Clamp(options.ValidationWorkers, 1, Environment.ProcessorCount);

        if (!options.ValidationUseGpuInference || !_placementTrainer.UsesGpuInference)
            return Environment.ProcessorCount;

        // GPU inference calls are still serialized at the trainer boundary, but much of
        // generation validation is CPU-side orchestration. On higher-core machines the
        // old 2-4 worker cap underutilizes available CPU, so default to a wider pool.
        return Math.Clamp(Environment.ProcessorCount / 2, 4, 64);
    }

    private static double ComputeEffectiveValidationScore(
        double rawScore,
        double repairedScore,
        int repairCount,
        double repairNoteDeltaRatio)
    {
        // Repair penalties are intentionally light: a model that generates decent maps
        // but needs some post-processing should not be scored the same as a model that
        // generates unplayable maps. Max combined penalty: 0.05 + 0.08 = 0.13.
        double repairCountPenalty  = Math.Min(0.05, repairCount * 0.005);
        double repairVolumePenalty = Math.Min(0.08, repairNoteDeltaRatio * 0.10);
        double effective = 0.75 * rawScore + 0.25 * repairedScore - repairCountPenalty - repairVolumePenalty;
        return Math.Clamp(effective, 0.0, 1.0);
    }

    private static double ComputeChordBeatFraction(CanonicalBeatmap beatmap)
    {
        if (beatmap.Notes.Count == 0)
            return 0.0;

        var groups = beatmap.Notes
            .GroupBy(n => Math.Round(n.Beat, 3))
            .ToArray();
        if (groups.Length == 0)
            return 0.0;

        return groups.Count(g => g.Count() >= 2) / (double)groups.Length;
    }

    private static string BuildValidationPairKey(ValidationSongFinder.ValidationPair pair)
    {
        var difficulty = pair.ReferenceMap.Difficulty;
        string customLabel = string.IsNullOrWhiteSpace(difficulty.CustomLabel)
            ? "-"
            : difficulty.CustomLabel.Trim();
        ulong referenceFingerprint = ComputeBeatmapFingerprint(pair.ReferenceMap);
        return $"{pair.MapFolder}|{difficulty.Characteristic}|{difficulty.Difficulty}|{customLabel}|ref={referenceFingerprint}";
    }

    private static ulong ComputeBeatmapFingerprint(CanonicalBeatmap beatmap)
    {
        const ulong OffsetBasis = 14695981039346656037UL;
        const ulong Prime = 1099511628211UL;

        ulong hash = OffsetBasis;
        foreach (var note in beatmap.Notes.OrderBy(n => n.Beat))
        {
            hash ^= (ulong)Math.Round(note.Beat * 1000.0);
            hash *= Prime;
            hash ^= (ulong)(note.Lane + 1);
            hash *= Prime;
            hash ^= (ulong)(note.Row + 1);
            hash *= Prime;
            hash ^= (ulong)((int)note.Hand + 1);
            hash *= Prime;
            hash ^= (ulong)((int)note.CutDirection + 1);
            hash *= Prime;
        }

        return hash;
    }

    private static string FormatHigherIsBetter(double value, double? previous, string format)
    {
        string text = value.ToString(format);
        if (!previous.HasValue)
            return text;

        double delta = value - previous.Value;
        if (delta > 1e-6)
            return ConsoleStyler.Colorize(text, ConsoleColor.Green);
        if (delta < -1e-6)
            return ConsoleStyler.Colorize(text, ConsoleColor.Red);
        return ConsoleStyler.Colorize(text, ConsoleColor.Yellow);
    }

    private static string FormatLowerIsBetter(double value, double? previous, string format)
    {
        string text = value.ToString(format);
        if (!previous.HasValue)
            return text;

        double delta = value - previous.Value;
        if (delta < -1e-6)
            return ConsoleStyler.Colorize(text, ConsoleColor.Green);
        if (delta > 1e-6)
            return ConsoleStyler.Colorize(text, ConsoleColor.Red);
        return ConsoleStyler.Colorize(text, ConsoleColor.Yellow);
    }

    private static string FormatCountComparison(int count)
    {
        string text = count.ToString();
        if (count > 0)
            return ConsoleStyler.Colorize(text, ConsoleColor.Green);
        return ConsoleStyler.Colorize(text, ConsoleColor.Yellow);
    }

    private static string FormatCountRegression(int count)
    {
        string text = count.ToString();
        if (count > 0)
            return ConsoleStyler.Colorize(text, ConsoleColor.Red);
        return ConsoleStyler.Colorize(text, ConsoleColor.Yellow);
    }

    private static string FormatMovementMagnitude(double value, double flatThreshold, double movingThreshold, string format)
    {
        string text = value.ToString(format);
        if (value >= movingThreshold)
            return ConsoleStyler.Colorize(text, ConsoleColor.Green);
        if (value <= flatThreshold)
            return ConsoleStyler.Colorize(text, ConsoleColor.Yellow);
        return text;
    }

    private static string FormatCheckpointKind(TorchPlacementTrainer.LoadedCheckpointKind checkpointKind)
    {
        string text = checkpointKind.ToString();
        return checkpointKind switch
        {
            TorchPlacementTrainer.LoadedCheckpointKind.None => ConsoleStyler.Colorize(text, ConsoleColor.Yellow),
            TorchPlacementTrainer.LoadedCheckpointKind.Current => ConsoleStyler.Colorize(text, ConsoleColor.Green),
            TorchPlacementTrainer.LoadedCheckpointKind.Best => ConsoleStyler.Colorize(text, ConsoleColor.Cyan),
            TorchPlacementTrainer.LoadedCheckpointKind.Legacy => ConsoleStyler.Colorize(text, ConsoleColor.Magenta),
            _ => text
        };
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private static ValidationSchedulePlan BuildValidationSchedule(
        CachedSong[] pool,
        int coreSongCount,
        TrainingOptions options,
        PersistedValidationSchedule? persisted)
    {
        int normalizedCoreSongs = Math.Clamp(coreSongCount, 1, Math.Max(1, pool.Length));
        int fullSongCap = Math.Min(
            pool.Length,
            Math.Max(normalizedCoreSongs, Math.Min(64, normalizedCoreSongs * 4)));
        int stageStartEpoch = Math.Max(15, options.SelfSupervisedWarmupEpochs + 10);
        int stageStepEpochs = 15;

        var savedCore = persisted is not null && persisted.RandomSeed == options.RandomSeed
            ? RestoreSongs(pool, persisted.CoreFolderPaths, normalizedCoreSongs)
            : [];
        var savedFull = persisted is not null && persisted.RandomSeed == options.RandomSeed
            ? RestoreSongs(pool, persisted.FullFolderPaths, fullSongCap)
            : [];

        var rng = new Random((int)(options.RandomSeed >> 1));
        var coreSongs = savedCore.Count == normalizedCoreSongs
            ? savedCore
            : SampleSongs(pool, normalizedCoreSongs, rng);

        var fullSongs = savedFull.Count >= coreSongs.Count
            ? EnsureContainsCore(savedFull, coreSongs, fullSongCap, pool, rng)
            : EnsureContainsCore([], coreSongs, fullSongCap, pool, rng);

        return new ValidationSchedulePlan(
            FlattenPairs(coreSongs),
            FlattenPairs(fullSongs),
            coreSongs.Count,
            fullSongs.Count,
            stageStartEpoch,
            stageStepEpochs);
    }

    private static List<CachedSong> RestoreSongs(CachedSong[] pool, string[] folderPaths, int cap)
    {
        var byFolder = pool.ToDictionary(s => s.Group.FolderPath, StringComparer.OrdinalIgnoreCase);
        var restored = new List<CachedSong>();
        foreach (var folder in folderPaths)
        {
            if (restored.Count >= cap)
                break;
            if (byFolder.TryGetValue(folder, out var song))
                restored.Add(song);
        }
        return restored;
    }

    private static List<CachedSong> SampleSongs(CachedSong[] pool, int count, Random rng) =>
        FisherYatesSample(pool, count, rng).ToList();

    private static List<CachedSong> EnsureContainsCore(
        IReadOnlyList<CachedSong> fullBase,
        IReadOnlyList<CachedSong> coreSongs,
        int fullSongCap,
        CachedSong[] pool,
        Random rng)
    {
        var selected = new List<CachedSong>(coreSongs);
        var seen = new HashSet<string>(coreSongs.Select(s => s.Group.FolderPath), StringComparer.OrdinalIgnoreCase);

        foreach (var song in fullBase)
        {
            if (selected.Count >= fullSongCap)
                break;
            if (seen.Add(song.Group.FolderPath))
                selected.Add(song);
        }

        foreach (var song in FisherYatesSample(pool, pool.Length, rng))
        {
            if (selected.Count >= fullSongCap)
                break;
            if (seen.Add(song.Group.FolderPath))
                selected.Add(song);
        }

        return selected;
    }

    private static CachedPair[] FlattenPairs(IReadOnlyList<CachedSong> songs) =>
        songs.SelectMany(cs => cs.Group.Difficulties.Select(diff =>
            new CachedPair(
                new ValidationSongFinder.ValidationPair(cs.Group.FolderPath, cs.Group.AudioPath, diff),
                cs.Audio)))
        .ToArray();

    private static int CountPairsForSongs(IReadOnlyList<CachedPair> pairs, int songCount)
    {
        if (songCount <= 0 || pairs.Count == 0)
            return 0;

        int count = 0;
        string? currentFolder = null;
        int seenSongs = 0;
        foreach (var pair in pairs)
        {
            if (!string.Equals(currentFolder, pair.Pair.MapFolder, StringComparison.OrdinalIgnoreCase))
            {
                currentFolder = pair.Pair.MapFolder;
                seenSongs++;
                if (seenSongs > songCount)
                    break;
            }
            count++;
        }
        return count;
    }

    private static PersistedValidationSchedule? TryLoadValidationSchedule(string path)
    {
        try
        {
            if (!File.Exists(path))
                return null;
            return JsonSerializer.Deserialize<PersistedValidationSchedule>(File.ReadAllText(path));
        }
        catch
        {
            return null;
        }
    }

    private static PersistedTrainingState? TryLoadTrainingState(string path)
    {
        try
        {
            if (!File.Exists(path))
                return null;
            return JsonSerializer.Deserialize<PersistedTrainingState>(File.ReadAllText(path));
        }
        catch
        {
            return null;
        }
    }

    private static void SaveTrainingState(string path, PersistedTrainingState state)
    {
        try
        {
            File.WriteAllText(
                path,
                JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch { }
    }

    private static void SaveValidationSchedule(
        string path,
        long randomSeed,
        ValidationSchedulePlan? plan,
        int lastCompletedEpoch)
    {
        if (plan is null)
            return;

        var payload = new PersistedValidationSchedule(
            randomSeed,
            plan.CoreSongCount,
            plan.FullSongCap,
            plan.StageStartEpoch,
            plan.StageStepEpochs,
            lastCompletedEpoch,
            plan.CorePairs.Select(p => p.Pair.MapFolder).Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
            plan.FullPairs.Select(p => p.Pair.MapFolder).Distinct(StringComparer.OrdinalIgnoreCase).ToArray());

        File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>Fisher-Yates in-place shuffle of sequences (not individual examples).</summary>
    private static List<List<TrainingExample>> FisherYatesShuffleSeqs(
        List<List<TrainingExample>> source, int seed)
    {
        var copy = new List<List<TrainingExample>>(source);
        var rng  = new Random(seed);
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }
        return copy;
    }

    /// <summary>Merge real sequences with synthetic sequences and shuffle the combined list.</summary>
    private static List<List<TrainingExample>> MergeSequences(
        List<List<TrainingExample>> real,
        List<List<TrainingExample>> synthetics,
        int seed)
    {
        var merged = new List<List<TrainingExample>>(real.Count + synthetics.Count);
        merged.AddRange(real);
        merged.AddRange(synthetics);
        var rng = new Random(seed);
        for (int i = merged.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (merged[i], merged[j]) = (merged[j], merged[i]);
        }
        return merged;
    }

    private static List<List<TrainingExample>> LimitSyntheticSequencesBalanced(
        List<List<TrainingExample>> synthetics,
        int maxCount,
        int maxExamples,
        int seed)
    {
        if (synthetics.Count <= maxCount && synthetics.Sum(s => s.Count) <= maxExamples)
            return synthetics;

        var copy = new List<List<TrainingExample>>(synthetics);
        var rng = new Random(seed);
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }

        int targetCount = Math.Max(1, maxCount);
        int targetExamples = Math.Max(128, maxExamples);
        var result = new List<List<TrainingExample>>();
        int exampleCount = 0;
        var difficulties = Enum.GetValues<DifficultyLevel>().Cast<int>().ToArray();
        var byDifficulty = copy
            .Select(seq => (Seq: seq, Diff: TryGetSequenceDifficulty(seq) ?? -1))
            .Where(x => x.Diff >= 0)
            .GroupBy(x => x.Diff)
            .ToDictionary(
                g => g.Key,
                g => g.Select(x => x.Seq).ToList());
        int perDiffCountCap = Math.Max(1, (int)Math.Ceiling(targetCount / (double)Math.Max(1, difficulties.Length)));
        int perDiffExampleCap = Math.Max(64, (int)Math.Ceiling(targetExamples / (double)Math.Max(1, difficulties.Length)));
        var diffCounts = difficulties.ToDictionary(d => d, _ => 0);
        var diffExamples = difficulties.ToDictionary(d => d, _ => 0);

        foreach (int diff in difficulties)
        {
            if (!byDifficulty.TryGetValue(diff, out var seqs))
                continue;

            foreach (var seq in seqs)
            {
                if (result.Count >= targetCount)
                    break;
                if (exampleCount + seq.Count > targetExamples && result.Count > 0)
                    continue;
                if (diffCounts[diff] >= perDiffCountCap || diffExamples[diff] + seq.Count > perDiffExampleCap)
                    continue;

                result.Add(seq);
                exampleCount += seq.Count;
                diffCounts[diff]++;
                diffExamples[diff] += seq.Count;
                break;
            }
        }

        foreach (var seq in copy)
        {
            if (result.Count >= targetCount)
                break;
            if (result.Contains(seq))
                continue;
            if (result.Count > 0 && exampleCount + seq.Count > targetExamples)
                continue;

            int? diff = TryGetSequenceDifficulty(seq);
            if (diff is not null && diffCounts.TryGetValue(diff.Value, out int diffCount))
            {
                if (diffCount >= perDiffCountCap * 2)
                    continue;
                if (diffExamples[diff.Value] + seq.Count > perDiffExampleCap * 2)
                    continue;
                diffCounts[diff.Value] = diffCount + 1;
                diffExamples[diff.Value] += seq.Count;
            }

            result.Add(seq);
            exampleCount += seq.Count;
        }

        return result.Count > 0 ? result : copy.Take(targetCount).ToList();
    }

    /// <summary>Fisher-Yates in-place shuffle of a copy, using the given seed.</summary>
    private static List<TrainingExample> FisherYatesShuffle(List<TrainingExample> source, int seed)
    {
        var copy = new List<TrainingExample>(source);
        var rng  = new Random(seed);
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }
        return copy;
    }

    private static bool ShouldAcceptSyntheticRefresh(
        double coreQuality,
        double fullQuality,
        ValidationDiagnostics diagnostics)
    {
        if (diagnostics.SuccessfulPairs < Math.Max(4, diagnostics.TotalPairs / 3))
            return false;
        if (diagnostics.TotalSyntheticExamples <= 0)
            return false;
        if (coreQuality < 0.20 || fullQuality < 0.20)
            return false;
        if (diagnostics.AvgCandidateSurvival < 0.01 || diagnostics.AvgDecodeRate < 0.10)
            return false;
        if (diagnostics.AvgCandidateSurvival > 0.28 || diagnostics.AvgDecodeRate > 1.70)
            return false;
        if (diagnostics.AvgSyntheticExamplesPerAcceptedPair > 700)
            return false;
        foreach (var kv in diagnostics.AvgGeneratedNotesByDifficulty)
        {
            double ceiling = kv.Key switch
            {
                DifficultyLevel.Easy => 180,
                DifficultyLevel.Normal => 320,
                DifficultyLevel.Hard => 520,
                DifficultyLevel.Expert => 800,
                DifficultyLevel.ExpertPlus => 1200,
                _ => 800,
            };
            if (kv.Value > ceiling)
                return false;
        }
        return true;
    }

    private static SyntheticBudget ComputeSyntheticBudget(
        IReadOnlyList<IReadOnlyList<TrainingExample>> realTrainSequences,
        int stableRefreshStreak)
    {
        int realSeqCount = realTrainSequences.Count;
        int realExampleCount = realTrainSequences.Sum(s => s.Count);

        double fraction = stableRefreshStreak switch
        {
            <= 0 => 0.015,
            1 => 0.020,
            2 => 0.030,
            3 => 0.040,
            4 => 0.055,
            _ => 0.070,
        };

        int seqCap = stableRefreshStreak switch
        {
            <= 0 => Math.Max(8, realSeqCount / 40),
            1 => Math.Max(10, realSeqCount / 32),
            2 => Math.Max(12, realSeqCount / 28),
            3 => Math.Max(16, realSeqCount / 24),
            4 => Math.Max(20, realSeqCount / 20),
            _ => Math.Max(24, realSeqCount / 16),
        };

        int exampleCap = Math.Max(512, (int)Math.Round(realExampleCount * fraction));
        return new SyntheticBudget(seqCap, exampleCap, fraction, stableRefreshStreak);
    }

    private static NegativeSupervisionBudget ComputeBadNegativeBudget(
        IReadOnlyList<IReadOnlyList<TrainingExample>> realTrainSequences)
    {
        int realSeqCount = realTrainSequences.Count;
        int realExampleCount = realTrainSequences.Sum(s => s.Count);
        // Reduced from 12% to 8%: after filtering to only clearly-bad positions, the
        // pool is smaller but cleaner, so a modest budget is appropriate.
        double fraction = 0.08;
        int seqCap = Math.Max(8, realSeqCount / 10);
        int exampleCap = Math.Max(256, (int)Math.Round(realExampleCount * fraction));
        return new NegativeSupervisionBudget(seqCap, exampleCap, fraction);
    }

    private static SequenceDifficultySnapshot SummarizeSequences(IReadOnlyList<IReadOnlyList<TrainingExample>> sequences)
    {
        var counts = Enum.GetValues<DifficultyLevel>()
            .ToDictionary(d => (int)d, _ => 0);
        int exampleCount = 0;
        int sequenceCount = 0;

        foreach (var seq in sequences)
        {
            if (seq.Count == 0)
                continue;

            sequenceCount++;
            exampleCount += seq.Count;
            int diff = seq[0].DifficultyLevel;
            if (!counts.TryAdd(diff, seq.Count))
                counts[diff] += seq.Count;
        }

        string mix = string.Join("  ",
            counts.Where(kv => kv.Value > 0)
                  .OrderBy(kv => kv.Key)
                  .Select(kv => $"{(DifficultyLevel)kv.Key}={kv.Value}"));

        return new SequenceDifficultySnapshot(sequenceCount, exampleCount, mix.Length > 0 ? mix : "none");
    }

    private static int? TryGetSequenceDifficulty(IReadOnlyList<TrainingExample> sequence)
    {
        if (sequence.Count == 0)
            return null;
        return sequence[0].DifficultyLevel;
    }

    private static void ValidateDifficultyCoverageOrThrow(SequenceDifficultySnapshot snapshot)
    {
        if (snapshot.Mix == "none")
            throw new InvalidOperationException("Training epoch contains no examples.");
    }

    private static SongFolderInfo[] InspectSongFolders(
        IReadOnlyList<string> folders,
        TrainingConsoleDashboard? dashboard = null)
    {
        var result = new SongFolderInfo[folders.Count];
        int done = 0;
        var stopwatch = Stopwatch.StartNew();

        Parallel.For(0, folders.Count,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            i =>
            {
                var folder = folders[i];
                try
                {
                    var difficulties = BeatmapImporter.Import(folder)
                        .Select(m => m.Difficulty.Difficulty)
                        .Distinct()
                        .OrderBy(d => (int)d)
                        .ToArray();
                    result[i] = new SongFolderInfo(folder, difficulties);
                }
                catch
                {
                    result[i] = new SongFolderInfo(folder, []);
                }

                if (dashboard is not null)
                {
                    int completed = Interlocked.Increment(ref done);
                    if (completed % 25 == 0 || completed == folders.Count)
                    {
                        dashboard.OnStageProgress(
                            completed,
                            folders.Count,
                            stopwatch.Elapsed.TotalSeconds,
                            Path.GetFileName(folder));
                    }
                }
            });

        return result;
    }

    private string[] ResolveMapFolders(string path, string ingestOutputPath, string logPrefix)
    {
        if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path) && !File.Exists(path))
            return [];

        if (IsMapFolder(path))
        {
            if (_dashboard is not null) _dashboard.AddNotice($"{logPrefix} single folder {Path.GetFileName(path)}");
            else Console.WriteLine($"{logPrefix} Using single map folder '{path}'.");
            return [path];
        }

        if (ContainsUnpackedMaps(path))
        {
            var folders = Directory.GetDirectories(path)
                .Where(IsMapFolder)
                .ToArray();
            if (_dashboard is not null) _dashboard.AddNotice($"{logPrefix} found {folders.Length} unpacked folders");
            else Console.WriteLine($"{logPrefix} Found {folders.Length} unpacked map folders in '{path}'.");
            return folders;
        }

        if (_dashboard is not null) _dashboard.OnStageStarted("ingest", 1, Path.GetFileName(path));
        else Console.WriteLine($"{logPrefix} Ingesting map archives from '{path}'...");
        var summary = _ingestion.IngestFolder(path, ingestOutputPath, path);
        if (_dashboard is not null)
            _dashboard.AddNotice(
                $"{logPrefix} ingested {summary.Imported}, skipped {summary.Skipped}, dup {summary.Duplicate}");
        else
            Console.WriteLine(
                $"{logPrefix} Ingested {summary.Imported} maps, {summary.Skipped} skipped, " +
                $"{summary.Malformed} malformed, {summary.Duplicate} duplicates.");
        return Directory.Exists(ingestOutputPath)
            ? Directory.GetDirectories(ingestOutputPath).Where(IsMapFolder).ToArray()
            : [];
    }

    private static SplitPlan ResolveSplitPlan(int totalSongs, TrainingOptions options)
    {
        if (totalSongs <= 1)
            return new SplitPlan(totalSongs, 0, 0, 0, 0);

        int minVal = totalSongs >= 8 ? 1 : 0;
        int minTest = totalSongs >= 12 ? 1 : 0;

        int autoVal = Math.Clamp(
            (int)Math.Round(totalSongs * 0.14, MidpointRounding.AwayFromZero),
            minVal,
            Math.Max(minVal, totalSongs / 5));

        int autoTest = Math.Clamp(
            (int)Math.Round(totalSongs * 0.10, MidpointRounding.AwayFromZero),
            minTest,
            Math.Max(minTest, totalSongs / 6));

        int valPoolSize = options.ValidationCachePoolSize > 0
            ? options.ValidationCachePoolSize
            : autoVal;

        int testSongCount = autoTest;
        int trainSongCount = totalSongs - valPoolSize - testSongCount;

        int minTrain = Math.Max(1, (int)Math.Ceiling(totalSongs * 0.70));
        while (trainSongCount < minTrain && (valPoolSize > minVal || testSongCount > minTest))
        {
            if (valPoolSize >= testSongCount && valPoolSize > minVal)
                valPoolSize--;
            else if (testSongCount > minTest)
                testSongCount--;
            else
                break;

            trainSongCount = totalSongs - valPoolSize - testSongCount;
        }

        if (trainSongCount <= 0)
            trainSongCount = Math.Max(1, totalSongs - valPoolSize - testSongCount);

        valPoolSize = Math.Max(0, Math.Min(valPoolSize, totalSongs - trainSongCount - testSongCount));
        int maxSongsPerEpoch = Math.Min(valPoolSize, 16);
        int songsPerEpoch = options.ValidationSongsPerEpoch > 0
            ? options.ValidationSongsPerEpoch
            : (valPoolSize <= 0 ? 0 : Math.Clamp((int)Math.Ceiling(Math.Sqrt(valPoolSize)), 1, maxSongsPerEpoch));

        if (valPoolSize > 0)
            songsPerEpoch = Math.Clamp(songsPerEpoch, 1, valPoolSize);

        return new SplitPlan(
            TrainSongs: trainSongCount,
            ValidationSongs: valPoolSize,
            TestSongs: testSongCount,
            ValidationSongsPerEpoch: songsPerEpoch,
            ValidationCachePoolSize: Math.Max(valPoolSize > 0 ? 1 : 0, Math.Max(songsPerEpoch, valPoolSize)));
    }

    private static (string[] TrainFolders, string[] ValidationFolders, string[] TestFolders)
        CreateDifficultyAwareSplit(
            IReadOnlyList<SongFolderInfo> folderInfos,
            SplitPlan plan,
            long seed)
    {
        var train = new List<string>(plan.TrainSongs);
        var val = new List<string>(plan.ValidationSongs);
        var test = new List<string>(plan.TestSongs);

        var rng = new Random((int)(seed & int.MaxValue));
        var buckets = folderInfos
            .GroupBy(info => info.HighestDifficulty)
            .OrderBy(_ => rng.Next())
            .ToList();

        int remainingSongs = folderInfos.Count;
        int remainingTrain = plan.TrainSongs;
        int remainingVal = plan.ValidationSongs;
        int remainingTest = plan.TestSongs;

        foreach (var bucket in buckets)
        {
            var shuffled = bucket.OrderBy(_ => rng.Next()).ToList();
            int bucketCount = shuffled.Count;
            if (bucketCount == 0)
                continue;

            int trainTake = AllocateBucketShare(bucketCount, remainingTrain, remainingSongs);
            int valTake = AllocateBucketShare(bucketCount, remainingVal, remainingSongs);
            trainTake = Math.Min(trainTake, bucketCount);
            valTake = Math.Min(valTake, bucketCount - trainTake);
            int testTake = Math.Min(bucketCount - trainTake - valTake, remainingTest);

            int assigned = trainTake + valTake + testTake;
            while (assigned < bucketCount)
            {
                if (remainingTrain - trainTake >= remainingVal - valTake &&
                    remainingTrain - trainTake >= remainingTest - testTake)
                    trainTake++;
                else if (remainingVal - valTake >= remainingTest - testTake)
                    valTake++;
                else
                    testTake++;

                assigned++;
            }

            train.AddRange(shuffled.Take(trainTake).Select(x => x.FolderPath));
            val.AddRange(shuffled.Skip(trainTake).Take(valTake).Select(x => x.FolderPath));
            test.AddRange(shuffled.Skip(trainTake + valTake).Take(testTake).Select(x => x.FolderPath));

            remainingSongs -= bucketCount;
            remainingTrain -= trainTake;
            remainingVal -= valTake;
            remainingTest -= testTake;
        }

        return (train.ToArray(), val.ToArray(), test.ToArray());
    }

    private static (string[] TrainFolders, string[] ValidationFolders, string[] TestFolders, IReadOnlyList<string> Moves)
        EnsureTrainingDifficultyCoverage(
            IReadOnlyList<SongFolderInfo> folderInfos,
            string[] trainFolders,
            string[] validationFolders,
            string[] testFolders)
    {
        var byFolder = folderInfos.ToDictionary(x => x.FolderPath, StringComparer.OrdinalIgnoreCase);
        var train = new List<string>(trainFolders);
        var val = new List<string>(validationFolders);
        var test = new List<string>(testFolders);
        var moves = new List<string>();

        var allDiffs = folderInfos
            .SelectMany(x => x.Difficulties)
            .Distinct()
            .OrderBy(x => (int)x)
            .ToArray();

        foreach (var diff in allDiffs)
        {
            if (train.Any(folder => byFolder.TryGetValue(folder, out var info) && info.Difficulties.Contains(diff)))
                continue;

            string? donor = val.FirstOrDefault(folder => byFolder.TryGetValue(folder, out var info) && info.Difficulties.Contains(diff))
                ?? test.FirstOrDefault(folder => byFolder.TryGetValue(folder, out var info) && info.Difficulties.Contains(diff));
            if (donor is null)
                continue;

            if (val.Remove(donor) || test.Remove(donor))
            {
                train.Add(donor);
                moves.Add($"moved '{Path.GetFileName(donor)}' into train to cover {diff}");
            }
        }

        return (train.ToArray(), val.ToArray(), test.ToArray(), moves);
    }

    private static int AllocateBucketShare(int bucketCount, int remainingTarget, int remainingSongs)
    {
        if (bucketCount <= 0 || remainingTarget <= 0 || remainingSongs <= 0)
            return 0;

        return (int)Math.Round(
            bucketCount * (remainingTarget / (double)remainingSongs),
            MidpointRounding.AwayFromZero);
    }

    private static string FormatDifficultyMix(IReadOnlyList<string> folders)
    {
        if (folders.Count == 0)
            return "none";

        var counts = Enum.GetValues<DifficultyLevel>()
            .ToDictionary(d => d, _ => 0);

        foreach (var folder in folders)
        {
            try
            {
                foreach (var diff in BeatmapImporter.Import(folder)
                    .Select(m => m.Difficulty.Difficulty)
                    .Distinct())
                    counts[diff]++;
            }
            catch
            {
            }
        }

        return string.Join("  ",
            counts.Where(kv => kv.Value > 0)
                  .OrderBy(kv => (int)kv.Key)
                  .Select(kv => $"{kv.Key}={kv.Value}"));
    }

    /// <summary>
    /// Build training examples from a set of map folders.
    /// Used to produce the held-out test set from test-split folders.
    /// </summary>
    private static IReadOnlyList<TrainingExample> BuildExamplesFromFolders(
        string[] folders, object printLock, ValidationAudioCache? audioCache = null)
    {
        var result   = new List<TrainingExample>();
        var listLock = new object();

        Parallel.ForEach(
            folders,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => new List<TrainingExample>(),
            (folder, _, localList) =>
            {
                try
                {
                    var maps    = BeatmapImporter.Import(folder);
                    var builder = new TrainingExampleBuilder();

                    string? audioPath = FindAudioFile(folder);
                    AudioAnalysisResult? audio = null;
                    if (audioPath != null)
                    {
                        audio = audioCache?.TryLoad(audioPath);
                        if (audio is null)
                        {
                            audio = new AudioFeatureExtractor().Extract(audioPath);
                            audioCache?.Store(audioPath, audio);
                        }
                    }

                    foreach (var map in maps)
                        localList.AddRange(builder.Build(map, audio ?? MakeStubAudio(map)));
                }
                catch (Exception ex)
                {
                    lock (printLock)
                        Console.WriteLine($"[Training] Warning (test): '{folder}': {ex.Message}");
                }
                return localList;
            },
            localList => { lock (listLock) result.AddRange(localList); });

        return result;
    }

    private static (IReadOnlyList<List<TrainingExample>> Sequences, int FilteredSeqs, int FallbackSeqs)
        BuildNegativeExamplesFromFolders(
        string[] folders,
        object printLock,
        ValidationAudioCache? audioCache,
        double baseNegativeWeight,
        TrainingConsoleDashboard? dashboard = null)
    {
        var result = new List<List<TrainingExample>>();
        var listLock = new object();
        int done = 0;
        int filteredSeqs = 0;   // sequences where ≥1 note passed IsObviouslyBadPlacement
        int fallbackSeqs = 0;   // sequences that used the all-notes-at-half-weight fallback
        var stopwatch = Stopwatch.StartNew();

        Parallel.ForEach(
            folders,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => new List<List<TrainingExample>>(),
            (folder, _, localSeqs) =>
            {
                try
                {
                    var maps = BeatmapImporter.Import(folder);
                    if (maps.Count == 0)
                        return localSeqs;

                    string? audioPath = FindAudioFile(folder);
                    AudioAnalysisResult? audio = null;
                    if (audioPath != null)
                    {
                        audio = audioCache?.TryLoad(audioPath);
                        if (audio is null)
                        {
                            audio = new AudioFeatureExtractor().Extract(audioPath);
                            audioCache?.Store(audioPath, audio);
                        }
                    }

                    var builder = new TrainingExampleBuilder();
                    foreach (var map in maps)
                    {
                        var built = builder.Build(map, audio ?? MakeStubAudio(map));
                        if (built.Count == 0)
                            continue;

                        var negativeSeq = built
                            .Where(ex => ex.HasNote && IsObviouslyBadPlacement(ex))
                            .Select(ex => ex with
                            {
                                HasNote = false,
                                NoteHand = -1,
                                NoteLane = -1,
                                NoteRow = -1,
                                NoteCutDir = -1,
                                Weight = ComputeBadNegativeWeight(ex, baseNegativeWeight),
                                IsNegativeSupervision = true,
                            })
                            .OrderBy(ex => ex.Beat)
                            .ToList();

                        bool usedFallback = false;
                        if (negativeSeq.Count == 0)
                        {
                            // Fallback: if no note in this map triggered a measurable
                            // violation, include all note positions at half weight so the
                            // map isn't silently dropped. Reduced weight signals lower
                            // confidence (we know the map is bad but can't identify why).
                            negativeSeq = built
                                .Where(ex => ex.HasNote)
                                .Select(ex => ex with
                                {
                                    HasNote = false,
                                    NoteHand = -1,
                                    NoteLane = -1,
                                    NoteRow = -1,
                                    NoteCutDir = -1,
                                    Weight = Math.Max(1.0, baseNegativeWeight * 0.5),
                                    IsNegativeSupervision = true,
                                })
                                .OrderBy(ex => ex.Beat)
                                .ToList();
                            usedFallback = true;
                        }

                        if (negativeSeq.Count > 0)
                        {
                            localSeqs.Add(negativeSeq);
                            if (usedFallback) Interlocked.Increment(ref fallbackSeqs);
                            else              Interlocked.Increment(ref filteredSeqs);
                        }
                    }
                }
                catch (Exception ex)
                {
                    lock (printLock)
                    {
                        if (dashboard is null)
                            Console.WriteLine($"[Training] Warning (bad-lib): '{folder}': {ex.Message}");
                        else
                            dashboard.AddNotice($"Bad-lib warning {Path.GetFileName(folder)}");
                    }
                }

                int completed = Interlocked.Increment(ref done);
                if (dashboard is not null && (completed % 25 == 0 || completed == folders.Length))
                {
                    dashboard.OnStageProgress(
                        completed,
                        folders.Length,
                        stopwatch.Elapsed.TotalSeconds,
                        Path.GetFileName(folder));
                }

                return localSeqs;
            },
            localSeqs =>
            {
                lock (listLock) result.AddRange(localSeqs);
            });

        return (result, filteredSeqs, fallbackSeqs);
    }

    /// <summary>
    /// Returns true if this note position in a bad-lib map has a clear, measurable quality
    /// violation that makes it a useful negative training example.
    ///
    /// Priority 1 — Audio weakness (most transferable): bad mappers often place notes
    ///   at positions with no real musical event. Audio features are identical at training
    ///   and inference time, so this signal transfers perfectly.
    ///
    /// Priority 2 — Geometric/playability violations: vision blocks, broken parity chains,
    ///   or reset pressure. Thresholds are intentionally lenient (0.30) to capture early-
    ///   sequence notes before the sliding-window history has fully built up.
    /// </summary>
    private static bool IsObviouslyBadPlacement(TrainingExample ex) =>
        // Audio-weak position: note placed with no real onset and low energy.
        // This transfers directly — the model won't have onset/energy at inference either.
        (ex.OnsetStrength < 0.25 && ex.EnergyLevel < 0.30) ||
        // Geometric violations (lenient thresholds to catch early-sequence notes)
        ex.CurrentBeatVisionBlockRisk > 0.30 ||
        Math.Max(ex.LeftParityBreakRate8, ex.RightParityBreakRate8) > 0.30 ||
        ex.ResetPressure > 0.30;

    private static double ComputeBadNegativeWeight(TrainingExample ex, double baseNegativeWeight)
    {
        double severity =
            1.0 +
            0.60 * Math.Clamp(ex.CurrentBeatVisionBlockRisk, 0.0, 1.0) +
            0.50 * Math.Max(
                Math.Clamp(ex.LeftParityBreakRate8, 0.0, 1.0),
                Math.Clamp(ex.RightParityBreakRate8, 0.0, 1.0)) +
            0.45 * Math.Clamp(ex.ResetPressure, 0.0, 1.0) +
            0.35 * (1.0 - Math.Clamp(ex.RecentRestRatio8, 0.0, 1.0)) +
            0.25 * Math.Clamp(ex.NotesAtCurrentBeatSoFar / 2.0, 0.0, 1.0) +
            0.25 * Math.Clamp(ex.LocalNps / 8.0, 0.0, 1.0);

        return Math.Clamp(baseNegativeWeight * severity, Math.Max(1.5, baseNegativeWeight), 8.5);
    }

    /// <summary>
    /// Returns a random sample of <paramref name="k"/> items from <paramref name="source"/>
    /// using a partial Fisher-Yates shuffle (no full copy needed).
    /// </summary>
    private static T[] FisherYatesSample<T>(T[] source, int k, Random rng)
    {
        var arr = (T[])source.Clone();
        int n   = arr.Length;
        k       = Math.Min(k, n);
        for (int i = 0; i < k; i++)
        {
            int j = rng.Next(i, n);
            (arr[i], arr[j]) = (arr[j], arr[i]);
        }
        return arr[..k];
    }

    private static bool IsMapFolder(string path) =>
        File.Exists(Path.Combine(path, "Info.dat")) ||
        File.Exists(Path.Combine(path, "info.dat")) ||
        File.Exists(Path.Combine(path, "info.json"));

    private static bool ContainsUnpackedMaps(string path) =>
        Directory.Exists(path) &&
        Directory.GetDirectories(path).Any(IsMapFolder);

    /// <summary>
    /// Locate the audio file for a map folder.
    /// Reads the declared filename from Info.dat, then falls back to a file-extension scan.
    /// </summary>
    private static string? FindAudioFile(string folder)
    {
        string? infoPath =
            File.Exists(Path.Combine(folder, "Info.dat")) ? Path.Combine(folder, "Info.dat") :
            File.Exists(Path.Combine(folder, "info.dat")) ? Path.Combine(folder, "info.dat") :
            null;

        if (infoPath is not null)
        {
            try
            {
                var root = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(infoPath));
                // V1/V2 use _songFilename, V3 uses songFilename
                string? fn = root.TryGetProperty("_songFilename", out var v2) ? v2.GetString()
                           : root.TryGetProperty("songFilename",  out var v3) ? v3.GetString()
                           : null;
                if (!string.IsNullOrEmpty(fn))
                {
                    string candidate = Path.Combine(folder, fn);
                    if (File.Exists(candidate)) return candidate;
                }
            }
            catch { }
        }

        foreach (string pattern in new[] { "*.ogg", "*.egg", "*.mp3", "*.wav" })
        {
            var files = Directory.GetFiles(folder, pattern);
            if (files.Length > 0) return files[0];
        }
        return null;
    }

    /// <summary>
    /// Minimal stub used when the audio file is missing.
    /// Produces zero audio features; only beat-position and placement context survive.
    /// </summary>
    private static AudioAnalysisResult MakeStubAudio(CanonicalBeatmap map)
    {
        double bpm      = map.Song.BeatsPerMinute > 0 ? map.Song.BeatsPerMinute : 120.0;
        double lastBeat = map.Notes.Count > 0 ? map.Notes[^1].Beat : 0;
        double[] onsets = map.Notes
            .Select(n => MathHelpers.BeatToSeconds(n.Beat, bpm))
            .Distinct().OrderBy(t => t).ToArray();
        return new AudioAnalysisResult
        {
            EstimatedBpm      = bpm,
            DurationSeconds   = MathHelpers.BeatToSeconds(lastBeat + 4.0, bpm),
            FrameRateHz       = 43.0,
            SampleRate        = 22050,
            OnsetTimesSeconds = onsets
        };
    }

    /// <summary>
    /// Ordinary-least-squares slope of a small value series.
    /// Returns 0 if fewer than 2 points are available.
    /// Negative slope = values are decreasing (good for loss).
    /// Positive slope = values are increasing (good for genQ).
    /// </summary>
    private static double ComputeLinearSlope(List<double> values)
    {
        int n = values.Count;
        if (n < 2) return 0.0;
        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        for (int i = 0; i < n; i++)
        {
            sumX  += i;
            sumY  += values[i];
            sumXY += i * values[i];
            sumXX += i * i;
        }
        double denom = n * sumXX - sumX * sumX;
        return denom == 0.0 ? 0.0 : (n * sumXY - sumX * sumY) / denom;
    }
}
