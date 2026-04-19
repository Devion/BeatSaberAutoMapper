using System.Collections.Concurrent;
using System.Diagnostics;
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

        var (trainFolders, valFolders, testFolders) = CreateDifficultyAwareSplit(
            folderInfos,
            splitPlan,
            options.RandomSeed);

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

        var poolGroups = new ValidationSongFinder()
            .Find(valFolders, splitPlan.ValidationCachePoolSize, options.RandomSeed);

        CachedPair[] cachedPairs = [];
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

            // Randomly sample songsPerEpoch unique songs from the pool (without replacement).
            // This sample is fixed for the entire run so genQ is comparable across epochs.
            var sampledSongs = pool.Length <= songsPerEpoch
                ? pool
                : FisherYatesSample(pool, songsPerEpoch, new Random((int)(options.RandomSeed >> 1)));

            // Flatten: one CachedPair per (song, difficulty) combination.
            cachedPairs = sampledSongs
                .SelectMany(cs => cs.Group.Difficulties.Select(diff =>
                    new CachedPair(
                        new ValidationSongFinder.ValidationPair(
                            cs.Group.FolderPath, cs.Group.AudioPath, diff),
                        cs.Audio)))
                .ToArray();

            Console.WriteLine(
                $"[Training] Using {cachedPairs.Length} validation pair(s) across " +
                $"{sampledSongs.Length} song(s) this run (pool size {pool.Length}).");
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

            if (currentSyntheticSeqs.Count > 0 && epoch >= options.SelfSupervisedWarmupEpochs)
                shuffledSeqs = MergeSequences(
                    shuffledSeqs,
                    LimitSyntheticSequences(currentSyntheticSeqs, shuffledSeqs.Count / 2, epochRng.Next()),
                    epochRng.Next());

            var trainStopwatch = Stopwatch.StartNew();
            var (loss, plBce) = _placementTrainer.TrainEpoch(shuffledSeqs, lr);
            double trainSeconds = trainStopwatch.Elapsed.TotalSeconds;

            double qualityScore = 0;
            double validationSeconds = 0;
            if (cachedPairs.Length > 0)
            {
                var validationStopwatch = Stopwatch.StartNew();
                var (score, newSynthetics, diffScores) =
                    RunGenerationValidationWithFeedback(cachedPairs, options);
                qualityScore = score;
                validationSeconds = validationStopwatch.Elapsed.TotalSeconds;

                // Per-difficulty breakdown (logged after the main epoch line)
                if (diffScores.Count > 1)
                {
                    var parts = diffScores
                        .OrderBy(kv => (int)kv.Key)
                        .Select(kv => $"{kv.Key}={kv.Value:F3}");
                    Console.WriteLine($"[Training]   genQ by diff: {string.Join("  ", parts)}");
                }

                // Refresh synthetic pool on the configured cadence.
                // Each batch from a generated map becomes one sequence (sorted by beat).
                if ((epoch + 1) % Math.Max(1, options.SelfSupervisedEveryNEpochs) == 0)
                    currentSyntheticSeqs = new List<List<TrainingExample>>(newSynthetics);
            }

            // ── Improvement check (raw quality) ──────────────────────────────────
            // neural_placement.pt is ONLY written when a new best is reached.
            bool isNewBest = qualityScore > bestQuality + 1e-4;
            if (isNewBest)
            {
                bestQuality = qualityScore;
                _placementTrainer.SaveBestWeights();
                _placementTrainer.SaveWeights(options.ArtifactsOutputPath);
            }

            // EMA for display only (no longer drives LR/stop decisions)
            smoothedQuality = smoothedQuality < 0
                ? qualityScore
                : EmaAlpha * qualityScore + (1 - EmaAlpha) * smoothedQuality;

            // ── Trend detection — linear regression over last SlopeWindow epochs ─
            lossHistory.Add(loss);
            if (lossHistory.Count > SlopeWindow) lossHistory.RemoveAt(0);
            genQHistory.Add(qualityScore);
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
                $"genQ={qualityScore:F3}  best={bestQuality:F3}  sm={smoothedQuality:F3}  lr={lr:G4}" +
                $"  lSlp={lossSlope:+0.0000;-0.0000}  qSlp={genQSlope:+0.0000;-0.0000}" +
                $"  train={trainSeconds:F1}s  val={validationSeconds:F1}s  epoch={epochStopwatch.Elapsed.TotalSeconds:F1}s" +
                (currentSyntheticSeqs.Count > 0 ? $"  synthEx={currentSyntheticSeqs.Sum(s => s.Count)}" : string.Empty) +
                statusTag);

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

    private (double Quality, IReadOnlyList<List<TrainingExample>> Synthetics,
             Dictionary<DifficultyLevel, double> DiffScores)
        RunGenerationValidationWithFeedback(CachedPair[] cachedPairs, TrainingOptions options)
    {
        var scores   = new double[cachedPairs.Length];
        var synthBag = new ConcurrentBag<List<TrainingExample>>();
        int validationParallelism = _placementTrainer.UsesGpuInference ? 1 : Environment.ProcessorCount;

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

                    scores[i] = eval.Evaluate(result.Beatmap, cached.Pair.ReferenceMap).OverallScore;

                    // Derive synthetic training examples from this generation pass
                    var synthetics = gen.Generate(
                        result.Beatmap,
                        cached.Audio,
                        (int)settings.TargetDifficulty,
                        options.SelfSupervisedPositiveWeight,
                        options.SelfSupervisedNegativeWeight);

                    if (synthetics.Count > 0)
                        synthBag.Add(synthetics.OrderBy(s => s.Beat).ToList());
                }
                catch { scores[i] = 0; }
            });

        var valid = scores.Where(s => s > 0).ToArray();
        double quality = valid.Length > 0 ? valid.Average() : 0;

        // Per-difficulty breakdown
        var diffScores = scores
            .Select((s, i) => (Score: s, Diff: cachedPairs[i].Pair.ReferenceMap.Difficulty.Difficulty))
            .Where(x => x.Score > 0)
            .GroupBy(x => x.Diff)
            .ToDictionary(g => g.Key, g => g.Average(x => x.Score));

        return (quality, synthBag.ToList(), diffScores);
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

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

    private static List<List<TrainingExample>> LimitSyntheticSequences(
        List<List<TrainingExample>> synthetics,
        int maxCount,
        int seed)
    {
        if (synthetics.Count <= maxCount) return synthetics;
        var copy = new List<List<TrainingExample>>(synthetics);
        var rng = new Random(seed);
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }
        return copy.Take(Math.Max(1, maxCount)).ToList();
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
