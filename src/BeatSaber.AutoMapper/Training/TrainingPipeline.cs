using System.Collections.Concurrent;
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
        //    Shuffle all map folders deterministically, then partition so that
        //    validation and test songs are never seen during training.
        // ----------------------------------------------------------------
        var splitRng      = new Random((int)(options.RandomSeed & int.MaxValue));
        var shuffledFolders = FisherYatesShuffleFolders(mapFolders, splitRng);
        int totalSongs    = shuffledFolders.Length;

        // Number of songs to reserve for the validation pool.
        // Prefer the explicit ValidationCachePoolSize if given; otherwise use the fraction.
        int valPoolSize = options.ValidationCachePoolSize > 0
            ? options.ValidationCachePoolSize
            : Math.Max(
                options.ValidationSongsPerEpoch,
                (int)(totalSongs * options.ValidationFraction));
        valPoolSize = Math.Min(valPoolSize, totalSongs / 4);   // cap at 25% of songs

        int testSongCount  = Math.Max(0, (int)(totalSongs * options.TestFraction));
        int trainSongCount = totalSongs - valPoolSize - testSongCount;

        if (trainSongCount <= 0)
        {
            Console.WriteLine("[Training] Not enough songs to split into train / val / test. Aborting.");
            return;
        }

        var trainFolders = shuffledFolders[..trainSongCount];
        var valFolders   = shuffledFolders[trainSongCount..(trainSongCount + valPoolSize)];
        var testFolders  = shuffledFolders[(trainSongCount + valPoolSize)..];

        Console.WriteLine(
            $"[Training] Split: {trainFolders.Length} train / " +
            $"{valFolders.Length} val / {testFolders.Length} test songs " +
            $"(total {totalSongs}).");

        // ----------------------------------------------------------------
        // 3. Build training examples — parallel across trainFolders only
        // ----------------------------------------------------------------
        Console.WriteLine(
            $"[Training] Building examples from {trainFolders.Length} train folders " +
            $"using {Environment.ProcessorCount} threads...");

        var allExamples  = new List<TrainingExample>();
        var printLock    = new object();
        int doneCount    = 0;
        var listLock     = new object();

        Parallel.ForEach(
            trainFolders,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            () => new List<TrainingExample>(),       // thread-local list
            (folder, _, localList) =>
            {
                try
                {
                    var maps    = BeatmapImporter.Import(folder);
                    var builder = new TrainingExampleBuilder();
                    foreach (var map in maps)
                    {
                        double bpm      = map.Song.BeatsPerMinute > 0 ? map.Song.BeatsPerMinute : 120.0;
                        double lastBeat = map.Notes.Count > 0 ? map.Notes[^1].Beat : 0;

                        // Synthesise pseudo-onsets from note positions so wOnset receives a
                        // meaningful gradient: beats where notes exist ≈ musical events.
                        double[] onsets = map.Notes
                            .Select(n => MathHelpers.BeatToSeconds(n.Beat, bpm))
                            .Distinct()
                            .OrderBy(t => t)
                            .ToArray();

                        localList.AddRange(builder.Build(map, new AudioAnalysisResult
                        {
                            EstimatedBpm      = bpm,
                            DurationSeconds   = MathHelpers.BeatToSeconds(lastBeat + 4.0, bpm),
                            FrameRateHz       = 43.0,
                            SampleRate        = 22050,
                            OnsetTimesSeconds = onsets   // pseudo-onsets from note positions
                        }));
                    }
                }
                catch (Exception ex)
                {
                    lock (printLock)
                        Console.WriteLine($"[Training] Warning: '{folder}': {ex.Message}");
                }

                int n = Interlocked.Increment(ref doneCount);
                if (n % 25 == 0 || n == trainFolders.Length)
                    lock (printLock)
                        Console.WriteLine($"[Training]   {n}/{trainFolders.Length} folders processed...");

                return localList;
            },
            localList =>
            {
                lock (listLock) allExamples.AddRange(localList);
            });

        Console.WriteLine($"[Training] Built {allExamples.Count} training examples.");
        if (allExamples.Count == 0) return;

        var trainEx = allExamples;

        // Build test examples from held-out test folders (used only for final evaluation)
        var testEx = testFolders.Length > 0
            ? BuildExamplesFromFolders(testFolders, printLock)
            : (IReadOnlyList<TrainingExample>)[];

        // ----------------------------------------------------------------
        // 4. Pre-cache audio analysis for validation songs (runs ONCE, reused every epoch)
        //
        // Only songs from valFolders are used — training songs are never evaluated.
        // Each song contributes ALL its valid difficulties as separate pairs.
        // 'songsPerEpoch' is sampled from the pool once and fixed for the whole run.
        // ----------------------------------------------------------------
        int songsPerEpoch = Math.Max(1, options.ValidationSongsPerEpoch);

        var poolGroups = new ValidationSongFinder()
            .Find(valFolders, valFolders.Length, options.RandomSeed);

        CachedPair[] cachedPairs = [];
        if (poolGroups.Count > 0)
        {
            var audioCache  = new ValidationAudioCache(options.ArtifactsOutputPath);
            var poolTmp     = new CachedSong?[poolGroups.Count];
            int cacheHits   = 0;
            int cacheMisses = 0;

            Parallel.For(0, poolGroups.Count,
                new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                i =>
                {
                    var group = poolGroups[i];
                    try
                    {
                        // One audio cache entry per unique song (shared across all its difficulties)
                        var audio = audioCache.TryLoad(group.AudioPath);
                        if (audio is not null)
                        {
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
                });

            var pool = poolTmp.Where(s => s is not null).Select(s => s!).ToArray();
            int poolPairCount = pool.Sum(s => s.Group.Difficulties.Count);
            Console.WriteLine(
                $"[Training] Validation pool: {pool.Length}/{poolGroups.Count} unique songs ready " +
                $"({poolPairCount} total pairs, {cacheHits} from cache, {cacheMisses} freshly analysed).");

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
            $"[Training] Training neural placement model {BeatSaberMappingNet.InputDim}→1024→512→256→128 ({options.Epochs} epochs max, " +
            $"Adam lr={lr:G4}, patience={options.EarlyStopPatience}, lr-patience={lrPatience})...[warmStart={warmStarted}]");

        double bestQuality     = -1.0;
        double smoothedQuality = -1.0;
        const double EmaAlpha  = 0.4;   // for display only
        int    stagnationEpochs = 0;    // trend-based; only increments when both loss+genQ are flat
        int    lrReductions     = 0;

        // Sliding window for trend detection (last SlopeWindow epochs)
        const int SlopeWindow = 10;
        var lossHistory = new List<double>(SlopeWindow + 1);
        var genQHistory = new List<double>(SlopeWindow + 1);

        _placementTrainer.SaveBestWeights();

        var epochRng = new Random((int)(options.RandomSeed & int.MaxValue));

        // Self-supervised synthetic examples — refreshed each epoch after warmup
        var currentSynthetics = new List<TrainingExample>();

        for (int epoch = 0; epoch < options.Epochs; epoch++)
        {
            // Shuffle real examples and optionally merge with synthetic pool
            var shuffled = FisherYatesShuffle(trainEx, epochRng.Next());

            if (currentSynthetics.Count > 0 && epoch >= options.SelfSupervisedWarmupEpochs)
                shuffled = MergeWithSynthetics(shuffled, currentSynthetics, epochRng.Next());

            var (loss, plBce) = _placementTrainer.TrainEpoch(shuffled, lr);

            double qualityScore = 0;
            if (cachedPairs.Length > 0)
            {
                var (score, newSynthetics, diffScores) =
                    RunGenerationValidationWithFeedback(cachedPairs, options);
                qualityScore = score;

                // Per-difficulty breakdown (logged after the main epoch line)
                if (diffScores.Count > 1)
                {
                    var parts = diffScores
                        .OrderBy(kv => (int)kv.Key)
                        .Select(kv => $"{kv.Key}={kv.Value:F3}");
                    Console.WriteLine($"[Training]   genQ by diff: {string.Join("  ", parts)}");
                }

                // Refresh synthetic pool on the configured cadence
                if ((epoch + 1) % Math.Max(1, options.SelfSupervisedEveryNEpochs) == 0)
                    currentSynthetics = new List<TrainingExample>(newSynthetics);
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
                (currentSynthetics.Count > 0 ? $"  synthEx={currentSynthetics.Count}" : string.Empty) +
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

    private (double Quality, IReadOnlyList<TrainingExample> Synthetics,
             Dictionary<DifficultyLevel, double> DiffScores)
        RunGenerationValidationWithFeedback(CachedPair[] cachedPairs, TrainingOptions options)
    {
        var scores   = new double[cachedPairs.Length];
        var synthBag = new ConcurrentBag<TrainingExample>();

        Parallel.For(0, cachedPairs.Length,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
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
                        UseLearned:       false,
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

                    foreach (var s in synthetics) synthBag.Add(s);
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

    /// <summary>Fisher-Yates shuffle of a string array, returning a new shuffled copy.</summary>
    private static string[] FisherYatesShuffleFolders(string[] source, Random rng)
    {
        var arr = (string[])source.Clone();
        for (int i = arr.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (arr[i], arr[j]) = (arr[j], arr[i]);
        }
        return arr;
    }

    /// <summary>
    /// Build training examples from a set of map folders.
    /// Used to produce the held-out test set from test-split folders.
    /// </summary>
    private static IReadOnlyList<TrainingExample> BuildExamplesFromFolders(
        string[] folders, object printLock)
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
                    foreach (var map in maps)
                    {
                        double bpm      = map.Song.BeatsPerMinute > 0 ? map.Song.BeatsPerMinute : 120.0;
                        double lastBeat = map.Notes.Count > 0 ? map.Notes[^1].Beat : 0;
                        double[] onsets = map.Notes
                            .Select(n => MathHelpers.BeatToSeconds(n.Beat, bpm))
                            .Distinct().OrderBy(t => t).ToArray();
                        localList.AddRange(builder.Build(map, new AudioAnalysisResult
                        {
                            EstimatedBpm      = bpm,
                            DurationSeconds   = MathHelpers.BeatToSeconds(lastBeat + 4.0, bpm),
                            FrameRateHz       = 43.0,
                            SampleRate        = 22050,
                            OnsetTimesSeconds = onsets
                        }));
                    }
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

    /// <summary>
    /// Merge real and synthetic examples into a single shuffled list.
    /// Synthetic examples carry elevated Weight, so interleaving ensures
    /// their gradients are spread across all batches rather than concentrated
    /// at the end.
    /// </summary>
    private static List<TrainingExample> MergeWithSynthetics(
        List<TrainingExample> real,
        IReadOnlyList<TrainingExample> synthetics,
        int seed)
    {
        var merged = new List<TrainingExample>(real.Count + synthetics.Count);
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

    private static bool IsMapFolder(string path) =>
        File.Exists(Path.Combine(path, "Info.dat")) ||
        File.Exists(Path.Combine(path, "info.dat")) ||
        File.Exists(Path.Combine(path, "info.json"));

    private static bool ContainsUnpackedMaps(string path) =>
        Directory.Exists(path) &&
        Directory.GetDirectories(path).Any(IsMapFolder);

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
