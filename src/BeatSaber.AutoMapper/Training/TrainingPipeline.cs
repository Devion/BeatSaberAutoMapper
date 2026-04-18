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
    private readonly AttributeModelTrainer  _attributeTrainer = new();
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
        // 2. Build training examples — parallel across map folders
        // ----------------------------------------------------------------
        Console.WriteLine(
            $"[Training] Building examples from {mapFolders.Length} map folders " +
            $"using {Environment.ProcessorCount} threads...");

        var allExamples  = new List<TrainingExample>();
        var printLock    = new object();
        int doneCount    = 0;
        var listLock     = new object();

        Parallel.ForEach(
            mapFolders,
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
                if (n % 25 == 0 || n == mapFolders.Length)
                    lock (printLock)
                        Console.WriteLine($"[Training]   {n}/{mapFolders.Length} folders processed...");

                return localList;
            },
            localList =>
            {
                lock (listLock) allExamples.AddRange(localList);
            });

        Console.WriteLine($"[Training] Built {allExamples.Count} training examples.");
        if (allExamples.Count == 0) return;

        int trainN  = (int)(allExamples.Count * options.TrainFraction);
        var trainEx = allExamples[..trainN];
        var testEx  = allExamples[trainN..];

        // ----------------------------------------------------------------
        // 3. Pre-cache audio analysis for validation songs (runs ONCE, reused every epoch)
        //
        // Pool model:
        //   poolSize = max(ValidationSongsPerEpoch, ValidationCachePoolSize)
        //   → find 'poolSize' unique songs and cache one audio entry per song
        //   → each song contributes ALL its valid difficulties as separate pairs
        //   → randomly sample 'songsPerEpoch' songs from the pool each run;
        //     total pairs per epoch ≈ songsPerEpoch × avg_difficulties_per_song
        //
        // On the first run the cache is empty — all poolSize songs are analysed and saved.
        // On subsequent restarts only new / invalidated entries are re-analysed.
        // ----------------------------------------------------------------
        int songsPerEpoch = Math.Max(1, options.ValidationSongsPerEpoch);
        int poolSize      = Math.Max(songsPerEpoch,
            options.ValidationCachePoolSize > 0 ? options.ValidationCachePoolSize : songsPerEpoch);

        var poolGroups = new ValidationSongFinder()
            .Find(mapFolders, poolSize, options.RandomSeed);

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
            // Each sampled song contributes all its difficulties, so total pairs per epoch
            // is typically 2–4× songsPerEpoch.
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
        // 4. Attribute model — one-shot parallel frequency counting
        // ----------------------------------------------------------------
        Console.WriteLine("[Training] Building attribute model...");
        _attributeTrainer.Train(trainEx);

        // ----------------------------------------------------------------
        // 5. Placement model — epoch loop with shuffle + parallel mini-batch GD
        // ----------------------------------------------------------------
        double lr = warmStarted ? options.InitialLearningRate * 0.3 : options.InitialLearningRate;
        const double MinLr      = 1e-5;
        const double LrDecay    = 0.5;   // multiply LR by this when plateaued
        // Reduce LR after this many no-improvement epochs (before full early-stop)
        int lrPatience = Math.Max(1, options.EarlyStopPatience / 3);

        Console.WriteLine(
            $"[Training] Training neural placement model 27→1024→512→256→128 ({options.Epochs} epochs max, " +
            $"Adam lr={lr:G4}, patience={options.EarlyStopPatience}, lr-patience={lrPatience})...[warmStart={warmStarted}]");
        double bestQuality         = -1.0;
        double bestSmoothedQuality = -1.0;   // EMA-smoothed, used for LR/early-stop
        double smoothedQuality     = -1.0;
        const double EmaAlpha      = 0.4;    // weight of current epoch vs history
        int    epochsNoImprovement = 0;
        int    lrReductions        = 0;
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

            var (loss, acc) = _placementTrainer.TrainEpoch(shuffled, lr);

            double qualityScore = 0;
            if (cachedPairs.Length > 0)
            {
                var (score, newSynthetics) =
                    RunGenerationValidationWithFeedback(cachedPairs, options);
                qualityScore = score;

                // Refresh synthetic pool on the configured cadence
                if ((epoch + 1) % Math.Max(1, options.SelfSupervisedEveryNEpochs) == 0)
                    currentSynthetics = new List<TrainingExample>(newSynthetics);
            }

            // EMA smoothing — stabilises LR/early-stop signals across noisy epochs
            smoothedQuality = smoothedQuality < 0
                ? qualityScore
                : EmaAlpha * qualityScore + (1 - EmaAlpha) * smoothedQuality;

            Console.WriteLine(
                $"[Training] Epoch {epoch + 1}/{options.Epochs}: " +
                $"loss={loss:F4}  acc={acc:P2}  genQuality={qualityScore:F3}  lr={lr:G4}" +
                (currentSynthetics.Count > 0
                    ? $"  synthEx={currentSynthetics.Count}"
                    : string.Empty));

            // Save best weights on RAW quality peaks (catches genuine highs even in noisy epochs)
            if (qualityScore > bestQuality + 1e-4)
            {
                bestQuality = qualityScore;
                _placementTrainer.SaveBestWeights();
                _placementTrainer.SaveWeights(options.ArtifactsOutputPath);
                _attributeTrainer.SaveModel(options.ArtifactsOutputPath);
            }

            // Use EMA-smoothed quality for plateau/early-stop decisions
            if (smoothedQuality > bestSmoothedQuality + 1e-4)
            {
                bestSmoothedQuality = smoothedQuality;
                epochsNoImprovement = 0;
            }
            else
            {
                epochsNoImprovement++;

                // Periodic checkpoint (even if not a new best)
                if (options.CheckpointEveryNEpochs > 0
                    && (epoch + 1) % options.CheckpointEveryNEpochs == 0)
                {
                    _placementTrainer.SaveWeights(options.ArtifactsOutputPath);
                    _attributeTrainer.SaveModel(options.ArtifactsOutputPath);
                    Console.WriteLine(
                        $"[Training] Checkpoint saved at epoch {epoch + 1} (bestQuality={bestQuality:F3})");
                }

                // Reduce LR on plateau before triggering full early stop
                if (epochsNoImprovement > 0 && epochsNoImprovement % lrPatience == 0
                    && lr > MinLr * 1.1)
                {
                    lr = Math.Max(lr * LrDecay, MinLr);
                    lrReductions++;
                    Console.WriteLine(
                        $"[Training] Plateau detected — reducing lr to {lr:G4} (reduction #{lrReductions})");
                }

                if (options.EarlyStopPatience > 0 && epochsNoImprovement >= options.EarlyStopPatience)
                {
                    Console.WriteLine(
                        $"[Training] Early stopping at epoch {epoch + 1} " +
                        $"(no improvement for {options.EarlyStopPatience} epochs). " +
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
        _attributeTrainer.SaveModel(options.ArtifactsOutputPath);

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

    private (double Quality, IReadOnlyList<TrainingExample> Synthetics)
        RunGenerationValidationWithFeedback(CachedPair[] cachedPairs, TrainingOptions options)
    {
        var scores  = new double[cachedPairs.Length];
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
                        placementScorer: _placementTrainer,
                        attributeModel:  _attributeTrainer);

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
        return (quality, synthBag.ToList());
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
}
