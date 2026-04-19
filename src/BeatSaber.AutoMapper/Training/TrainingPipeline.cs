using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json;
using BeatSaber.AutoMapper.Audio;
using BeatSaber.AutoMapper.Audio.Features;
using BeatSaber.AutoMapper.Beatmap;
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

    private sealed record SequenceDifficultySnapshot(
        int SequenceCount,
        int ExampleCount,
        string Mix);

    private sealed record ValidationPairOutcome(
        DifficultyLevel Difficulty,
        double Score,
        int CandidateCount,
        int ProposedCount,
        int DecodedNoteCount,
        int RepairedNoteCount,
        int SyntheticExampleCount,
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
        double AvgRepairedNotes,
        double AvgScoreAllPairs,
        Dictionary<DifficultyLevel, double> AvgGeneratedNotesByDifficulty,
        Dictionary<DifficultyLevel, double> DiffScores);

    public void Run(TrainingOptions options)
    {
        Guard.NotNull(options, nameof(options));
        Directory.CreateDirectory(options.ArtifactsOutputPath);

        // Warm-start from existing checkpoint
        bool warmStarted = _placementTrainer.TryLoadCheckpoint(options.ArtifactsOutputPath);
        if (warmStarted)
            Console.WriteLine("[Training] Warm-start: loaded existing checkpoint. LR reduced 0.3x for fine-tuning.");

        // ----------------------------------------------------------------
        // 1. Resolve map folders
        // ----------------------------------------------------------------
        string[] mapFolders;

        if (IsMapFolder(options.DatasetPath))
        {
            mapFolders = [options.DatasetPath];
            Console.WriteLine($"[Training] Using single map folder '{options.DatasetPath}'.");
        }
        else if (ContainsUnpackedMaps(options.DatasetPath))
        {
            mapFolders = Directory.GetDirectories(options.DatasetPath)
                .Where(IsMapFolder).ToArray();
            Console.WriteLine(
                $"[Training] Found {mapFolders.Length} unpacked map folders in '{options.DatasetPath}'.");
        }
        else
        {
            string libraryPath = Path.Combine(options.ArtifactsOutputPath, "library");
            Console.WriteLine($"[Training] Ingesting corpus from '{options.DatasetPath}'...");
            var summary = _ingestion.IngestFolder(options.DatasetPath, libraryPath, options.DatasetPath);
            Console.WriteLine(
                $"[Training] Ingested {summary.Imported} maps, {summary.Skipped} skipped, " +
                $"{summary.Malformed} malformed, {summary.Duplicate} duplicates.");

            if (!Directory.Exists(libraryPath))
            {
                Console.WriteLine("[Training] No maps found.");
                return;
            }
            mapFolders = Directory.GetDirectories(libraryPath);
        }

        // ----------------------------------------------------------------
        // 2. Song-level train / val / test split
        //    Splits are difficulty-aware and remain song-separated:
        //    one map folder and all its difficulties stay in exactly one split.
        // ----------------------------------------------------------------
        int totalSongs = mapFolders.Length;
        var folderInfos = InspectSongFolders(mapFolders);
        var splitPlan = ResolveSplitPlan(totalSongs, options);

        if (splitPlan.TrainSongs <= 0)
        {
            Console.WriteLine("[Training] Not enough songs to split into train / val / test. Aborting.");
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
            Console.WriteLine($"[Training] Coverage fix: {move}");

        Console.WriteLine(
            $"[Training] Split: {trainFolders.Length} train / " +
            $"{valFolders.Length} val / {testFolders.Length} test songs " +
            $"(total {totalSongs}, val-songs/epoch={splitPlan.ValidationSongsPerEpoch}, " +
            $"validation-cache-size={splitPlan.ValidationCachePoolSize}).");

        Console.WriteLine(
            $"[Training] Difficulty mix: train[{FormatDifficultyMix(trainFolders)}]  " +
            $"val[{FormatDifficultyMix(valFolders)}]  " +
            $"test[{FormatDifficultyMix(testFolders)}]");

        // ----------------------------------------------------------------
        // 3. Build training examples — parallel across trainFolders only
        // Each (folder, difficulty) pair becomes one sequence (beat-ordered).
        // ----------------------------------------------------------------
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
                        Console.WriteLine(
                            $"[Training]   {n}/{trainFolders.Length} folders " +
                            $"({100.0 * n / trainFolders.Length:F0}%)" +
                            $" — {elapsed:F0}s elapsed" +
                            (eta > 1 ? $", ~{eta:F0}s remaining" : string.Empty));
                }

                return localSeqs;
            },
            localSeqs =>
            {
                lock (listLock) allSequences.AddRange(localSeqs);
            });

        int totalExamples = allSequences.Sum(s => s.Count);
        Console.WriteLine(
            $"[Training] Built {totalExamples} training examples in {allSequences.Count} sequences.");
        if (allSequences.Count == 0) return;

        var trainSeqs = allSequences;
        var realTrainSnapshot = SummarizeSequences(trainSeqs);
        Console.WriteLine(
            $"[Training] Real train mix: seq={realTrainSnapshot.SequenceCount}  ex={realTrainSnapshot.ExampleCount}  " +
            $"{realTrainSnapshot.Mix}");

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
                            Console.WriteLine(
                                $"[Training]   Val {n}/{poolGroups.Count}" +
                                $" — {Path.GetFileName(group.AudioPath)}" +
                                $" [{(hit ? "cache" : "fresh")}]" +
                                $" {elapsed:F0}s elapsed" +
                                (eta > 1 ? $" ~{eta:F0}s remaining" : string.Empty));
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

            Console.WriteLine(
                $"[Training] Validation schedule: core={validationPlan.CorePairs.Length} pair(s) / {validationPlan.CoreSongCount} song(s), " +
                $"full-cap={validationPlan.FullPairs.Length} pair(s) / {validationPlan.FullSongCap} song(s), " +
                $"stage-start={validationPlan.StageStartEpoch}, stage-step={validationPlan.StageStepEpochs}.");
        }

        // ----------------------------------------------------------------
        // 5. Placement model — epoch loop with shuffle + parallel mini-batch GD
        // ----------------------------------------------------------------
        double lr = warmStarted ? options.InitialLearningRate * 0.3 : options.InitialLearningRate;
        const double MinLr   = 1e-5;
        const double LrDecay = 0.5;
        // Reduce LR after this many consecutive stagnation epochs.
        // Default: patience/5 → faster LR drops than the old patience/3.
        int lrPatience = options.LrPatience > 0
            ? options.LrPatience
            : Math.Max(1, options.EarlyStopPatience / 5);

        Console.WriteLine(
            $"[Training] Training GRU placement model {BeatSaberMappingNet.InputDim}→GRU({BeatSaberMappingNet.GruHiddenDim}×{BeatSaberMappingNet.GruLayers})→{BeatSaberMappingNet.MlpHidden} " +
            $"({options.Epochs} epochs max, Adam lr={lr:G4}, patience={options.EarlyStopPatience}, lr-patience={lrPatience})...[warmStart={warmStarted}]");

        double bestQuality     = -1.0;
        double smoothedQuality = -1.0;
        const double EmaAlpha  = 0.4;
        int    stagnationEpochs = 0;
        int    lrReductions     = 0;

        const int SlopeWindow = 10;
        var lossHistory = new List<double>(SlopeWindow + 1);
        var genQHistory = new List<double>(SlopeWindow + 1);

        _placementTrainer.SaveBestWeights();

        var epochRng = new Random((int)(options.RandomSeed & int.MaxValue));

        // Self-supervised synthetic examples — refreshed each epoch after warmup.
        // Stored as single-step sequences (sorted by beat within each synthetic batch).
        var currentSyntheticSeqs = new List<List<TrainingExample>>();

        for (int epoch = 0; epoch < options.Epochs; epoch++)
        {
            var epochStopwatch = Stopwatch.StartNew();

            // Shuffle sequences (not individual examples) each epoch
            var shuffledSeqs = FisherYatesShuffleSeqs(trainSeqs, epochRng.Next());
            bool syntheticMerged = false;

            if (currentSyntheticSeqs.Count > 0 && epoch >= options.SelfSupervisedWarmupEpochs)
            {
                shuffledSeqs = MergeSequences(
                    shuffledSeqs,
                    LimitSyntheticSequencesBalanced(
                        currentSyntheticSeqs,
                        shuffledSeqs.Count / 2,
                        Math.Max(512, shuffledSeqs.Sum(s => s.Count) / 4),
                        epochRng.Next()),
                    epochRng.Next());
                syntheticMerged = true;
            }

            var epochMix = SummarizeSequences(shuffledSeqs);
            ValidateDifficultyCoverageOrThrow(epochMix);

            var trainStopwatch = Stopwatch.StartNew();
            var (loss, plBce) = _placementTrainer.TrainEpoch(shuffledSeqs, lr);
            double trainSeconds = trainStopwatch.Elapsed.TotalSeconds;

            double coreQualityScore = 0;
            double fullQualityScore = 0;
            double validationSeconds = 0;
            int activeValidationPairs = 0;
            int activeValidationSongs = 0;
            ValidationDiagnostics? diagnostics = null;
            if (validationPlan is not null)
            {
                var validationStopwatch = Stopwatch.StartNew();
                int absoluteEpoch = resumeEpochOffset + epoch + 1;
                activeValidationSongs = validationPlan.ActiveFullSongCountForEpoch(absoluteEpoch);
                activeValidationPairs = CountPairsForSongs(validationPlan.FullPairs, activeValidationSongs);
                var activePairs = validationPlan.FullPairs.Take(activeValidationPairs).ToArray();
                int corePairCount = Math.Min(validationPlan.CorePairs.Length, activePairs.Length);
                var (coreScore, fullScore, newSynthetics, newDiagnostics) =
                    RunGenerationValidationWithFeedback(activePairs, corePairCount, options);
                coreQualityScore = coreScore;
                fullQualityScore = fullScore;
                diagnostics = newDiagnostics;
                validationSeconds = validationStopwatch.Elapsed.TotalSeconds;

                // Per-difficulty breakdown (logged after the main epoch line)
                if (diagnostics.DiffScores.Count > 1)
                {
                    var parts = diagnostics.DiffScores
                        .OrderBy(kv => (int)kv.Key)
                        .Select(kv => $"{kv.Key}={kv.Value:F3}");
                    Console.WriteLine($"[Training]   genQ by diff: {string.Join("  ", parts)}");
                }

                // Refresh synthetic pool on the configured cadence.
                // Each batch from a generated map becomes one sequence (sorted by beat).
                if ((epoch + 1) % Math.Max(1, options.SelfSupervisedEveryNEpochs) == 0)
                {
                    if (ShouldAcceptSyntheticRefresh(coreQualityScore, fullQualityScore, diagnostics))
                    {
                        currentSyntheticSeqs = new List<List<TrainingExample>>(newSynthetics);
                    }
                    else
                    {
                        Console.WriteLine(
                            $"[Training]   synthetic refresh skipped: coreQ={coreQualityScore:F3}  " +
                            $"fullQ={fullQualityScore:F3}  okPairs={diagnostics.SuccessfulPairs}/{diagnostics.TotalPairs}");
                    }
                }
            }

            // ── Improvement check (raw quality) ──────────────────────────────────
            // neural_placement.pt is ONLY written when a new best is reached.
            bool isNewBest = coreQualityScore > bestQuality + 1e-4;
            if (isNewBest)
            {
                bestQuality = coreQualityScore;
                _placementTrainer.SaveBestWeights();
                _placementTrainer.SaveWeights(options.ArtifactsOutputPath);
            }

            // EMA for display only (no longer drives LR/stop decisions)
            smoothedQuality = smoothedQuality < 0
                ? coreQualityScore
                : EmaAlpha * coreQualityScore + (1 - EmaAlpha) * smoothedQuality;

            // ── Trend detection — linear regression over last SlopeWindow epochs ─
            lossHistory.Add(loss);
            if (lossHistory.Count > SlopeWindow) lossHistory.RemoveAt(0);
            genQHistory.Add(coreQualityScore);
            if (genQHistory.Count > SlopeWindow) genQHistory.RemoveAt(0);

            double lossSlope = ComputeLinearSlope(lossHistory); // negative = loss improving
            double genQSlope = ComputeLinearSlope(genQHistory);  // positive = quality improving

            // Stagnation: neither loss nor quality is trending in the right direction.
            // lossSlope < -0.003 → loss still declining >0.003/epoch (cold-start / active learning)
            // genQSlope > 0.001  → quality genuinely trending upward
            bool isLossImproving = lossSlope < -0.003;
            bool isGenQImproving = genQSlope > 0.001;
            bool isStagnating    = !isLossImproving && !isGenQImproving;

            if (isStagnating)
                stagnationEpochs++;
            else
                stagnationEpochs = Math.Max(0, stagnationEpochs - 1);

            // ── Epoch log ────────────────────────────────────────────────────────
            string statusTag = isNewBest
                ? "  ** NEW BEST **"
                : stagnationEpochs > 0
                    ? $"  [stag: {stagnationEpochs}/{options.EarlyStopPatience}]"
                    : string.Empty;

            Console.WriteLine(
                $"[Training] Epoch {epoch + 1}/{options.Epochs}: " +
                $"loss={loss:F4}  plBCE={plBce:F4}  " +
                $"coreQ={coreQualityScore:F3}  fullQ={fullQualityScore:F3}  best={bestQuality:F3}  sm={smoothedQuality:F3}  lr={lr:G4}" +
                $"  lSlp={lossSlope:+0.0000;-0.0000}  qSlp={genQSlope:+0.0000;-0.0000}" +
                $"  train={trainSeconds:F1}s  val={validationSeconds:F1}s  epoch={epochStopwatch.Elapsed.TotalSeconds:F1}s" +
                (activeValidationPairs > 0 ? $"  valPairs={activeValidationPairs}" : string.Empty) +
                (currentSyntheticSeqs.Count > 0 ? $"  synthEx={currentSyntheticSeqs.Sum(s => s.Count)}" : string.Empty) +
                statusTag);

            Console.WriteLine(
                $"[Training]   train mix: seq={epochMix.SequenceCount}  ex={epochMix.ExampleCount}  " +
                $"synthetic={(syntheticMerged ? "on" : "off")}  {epochMix.Mix}");

            if (validationPlan is not null)
            {
                Console.WriteLine(
                    $"[Training]   validation stage: coreSongs={validationPlan.CoreSongCount}  activeFullSongs={activeValidationSongs}/{validationPlan.FullSongCap}");
                if (diagnostics is not null)
                {
                    var noteParts = diagnostics.AvgGeneratedNotesByDifficulty.Count > 0
                        ? string.Join("  ", diagnostics.AvgGeneratedNotesByDifficulty
                            .OrderBy(kv => (int)kv.Key)
                            .Select(kv => $"{kv.Key}Notes={kv.Value:F1}"))
                        : "none";
                    Console.WriteLine(
                        $"[Training]   val diag: ok={diagnostics.SuccessfulPairs}/{diagnostics.TotalPairs}  " +
                        $"coreOk={diagnostics.CoreSuccessfulPairs}/{Math.Max(1, diagnostics.CoreSuccessfulPairs + diagnostics.CoreFailedPairs)}  " +
                        $"avgAll={diagnostics.AvgScoreAllPairs:F3}  cand->prop={diagnostics.AvgCandidateSurvival:F3}  " +
                        $"prop->notes={diagnostics.AvgDecodeRate:F3}  repairedNotes={diagnostics.AvgRepairedNotes:F1}  " +
                        $"newSynthEx={diagnostics.TotalSyntheticExamples}");
                    Console.WriteLine($"[Training]   val notes by diff: {noteParts}");
                }
            }

            // ── Stagnation-driven LR reduction and early stop ────────────────────
            if (stagnationEpochs > 0)
            {
                // Periodic checkpoint marker (no file write)
                if (options.CheckpointEveryNEpochs > 0
                    && (epoch + 1) % options.CheckpointEveryNEpochs == 0)
                {
                    Console.WriteLine(
                        $"[Training] Checkpoint epoch {epoch + 1} — best so far: {bestQuality:F3}");
                }

                // Reduce LR when stagnation counter hits the lr-patience threshold
                if (stagnationEpochs % lrPatience == 0 && lr > MinLr * 1.1)
                {
                    lr = Math.Max(lr * LrDecay, MinLr);
                    lrReductions++;
                    Console.WriteLine(
                        $"[Training] Stagnation plateau — reducing lr to {lr:G4} (reduction #{lrReductions})");
                }

                if (options.EarlyStopPatience > 0 && stagnationEpochs >= options.EarlyStopPatience)
                {
                    Console.WriteLine(
                        $"[Training] Early stopping at epoch {epoch + 1} " +
                        $"(stagnation for {options.EarlyStopPatience} epochs). " +
                        $"Best genQuality={bestQuality:F3}");
                    break;
                }
            }

            if (validationPlan is not null)
                SaveValidationSchedule(validationStatePath, options.RandomSeed, validationPlan, resumeEpochOffset + epoch + 1);
        }

        _placementTrainer.RestoreBestWeights();

        // ----------------------------------------------------------------
        // 6. Save artifacts
        // ----------------------------------------------------------------
        _placementTrainer.SaveWeights(options.ArtifactsOutputPath);

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
        }

        Console.WriteLine(
            $"[Training] Done. Best generation quality={bestQuality:F3}. " +
            $"Artifacts in '{options.ArtifactsOutputPath}'.");
    }

    // ----------------------------------------------------------------
    // Per-epoch generation validation + self-supervised example generation
    // Both happen in the same generation pass — no extra cost.
    // ----------------------------------------------------------------

    private (double CoreQuality, double FullQuality, IReadOnlyList<List<TrainingExample>> Synthetics,
             ValidationDiagnostics Diagnostics)
        RunGenerationValidationWithFeedback(CachedPair[] cachedPairs, int corePairCount, TrainingOptions options)
    {
        var outcomes = new ValidationPairOutcome[cachedPairs.Length];
        var synthBag = new ConcurrentBag<List<TrainingExample>>();
        int validationParallelism = _placementTrainer.UsesGpuInference ? 1 : Environment.ProcessorCount;
        const double MinSyntheticQuality = 0.24;

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
                        RandomSeed:       options.RandomSeed,
                        UseLearned:       true,
                        ArtifactsPath:    null);

                    var result = svc.Generate(
                        cached.Audio,
                        cached.Pair.ReferenceMap.Song,
                        settings,
                        placementScorer: _placementTrainer);

                    double score = eval.Evaluate(result.Beatmap, cached.Pair.ReferenceMap).OverallScore;

                    // Derive synthetic training examples from this generation pass
                    var synthetics = gen.Generate(
                        result.Beatmap,
                        cached.Audio,
                        (int)settings.TargetDifficulty,
                        options.SelfSupervisedPositiveWeight,
                        options.SelfSupervisedNegativeWeight);

                    if (score >= MinSyntheticQuality && synthetics.Count > 0)
                        synthBag.Add(synthetics.OrderBy(s => s.Beat).ToList());

                    outcomes[i] = new ValidationPairOutcome(
                        cached.Pair.ReferenceMap.Difficulty.Difficulty,
                        score,
                        result.Telemetry.CandidateCount,
                        result.Telemetry.ProposedCount,
                        result.Telemetry.DecodedNoteCount,
                        result.Telemetry.RepairedNoteCount,
                        score >= MinSyntheticQuality ? synthetics.Count : 0,
                        Failed: false);
                }
                catch
                {
                    outcomes[i] = new ValidationPairOutcome(
                        cached.Pair.ReferenceMap.Difficulty.Difficulty,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        Failed: true);
                }
            });

        var valid = outcomes.Where(o => !o.Failed && o.Score > 0).ToArray();
        double fullQuality = valid.Length > 0 ? valid.Average(x => x.Score) : 0;

        var coreOutcomes = outcomes.Take(Math.Min(corePairCount, outcomes.Length)).ToArray();
        var coreValid = coreOutcomes.Where(o => !o.Failed && o.Score > 0).ToArray();
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
        double avgRepairedNotes = valid.Length > 0 ? valid.Average(x => x.RepairedNoteCount) : 0;
        double avgScoreAllPairs = outcomes.Length > 0 ? outcomes.Average(x => x.Score) : 0;
        var avgGeneratedNotesByDifficulty = valid
            .GroupBy(x => x.Difficulty)
            .ToDictionary(g => g.Key, g => g.Average(x => x.RepairedNoteCount));

        var diagnostics = new ValidationDiagnostics(
            TotalPairs: outcomes.Length,
            SuccessfulPairs: outcomes.Count(x => !x.Failed && x.Score > 0),
            FailedPairs: outcomes.Count(x => x.Failed || x.Score <= 0),
            CoreSuccessfulPairs: coreOutcomes.Count(x => !x.Failed && x.Score > 0),
            CoreFailedPairs: coreOutcomes.Count(x => x.Failed || x.Score <= 0),
            TotalSyntheticExamples: outcomes.Sum(x => x.SyntheticExampleCount),
            AvgCandidateSurvival: avgCandidateSurvival,
            AvgDecodeRate: avgDecodeRate,
            AvgRepairedNotes: avgRepairedNotes,
            AvgScoreAllPairs: avgScoreAllPairs,
            AvgGeneratedNotesByDifficulty: avgGeneratedNotesByDifficulty,
            DiffScores: diffScores);

        return (coreQuality, fullQuality, synthBag.ToList(), diagnostics);
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

        var result = new List<List<TrainingExample>>();
        int exampleCount = 0;
        var uncoveredDiffs = Enum.GetValues<DifficultyLevel>().Cast<int>().ToHashSet();

        foreach (var seq in copy)
        {
            if (uncoveredDiffs.Count == 0)
                break;

            int? diff = TryGetSequenceDifficulty(seq);
            if (diff is null || !uncoveredDiffs.Contains(diff.Value))
                continue;
            if (result.Count >= Math.Max(1, maxCount))
                break;
            if (result.Count > 0 && exampleCount + seq.Count > maxExamples)
                continue;

            result.Add(seq);
            exampleCount += seq.Count;
            uncoveredDiffs.Remove(diff.Value);
        }

        foreach (var seq in copy)
        {
            if (result.Count >= Math.Max(1, maxCount))
                break;
            if (result.Contains(seq))
                continue;
            if (result.Count > 0 && exampleCount + seq.Count > maxExamples)
                continue;

            result.Add(seq);
            exampleCount += seq.Count;
        }

        return result.Count > 0 ? result : copy.Take(Math.Max(1, maxCount)).ToList();
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
        return true;
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

    private static SongFolderInfo[] InspectSongFolders(IReadOnlyList<string> folders)
    {
        var result = new SongFolderInfo[folders.Count];

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
            });

        return result;
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
