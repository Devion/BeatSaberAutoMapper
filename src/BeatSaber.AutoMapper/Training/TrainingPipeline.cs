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

public sealed class TrainingPipeline
{
    private readonly CorpusIngestionService _ingestion        = new();
    private readonly TrainingExampleBuilder _exampleBuilder   = new();
    private readonly DatasetManifestBuilder _manifestBuilder  = new();
    private readonly TorchPlacementTrainer   _placementTrainer = new();
    private readonly AttributeModelTrainer  _attributeTrainer = new();
    private readonly EvaluationRunner       _evaluator        = new();

    // Pre-analysed audio kept alive for all epochs (analysis is expensive, result is immutable)
    private sealed record CachedPair(
        ValidationSongFinder.ValidationPair Pair,
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
        // ----------------------------------------------------------------
        int songsPerEpoch = Math.Max(1, options.ValidationSongsPerEpoch);
        var validationPairs = new ValidationSongFinder()
            .Find(mapFolders, songsPerEpoch, options.RandomSeed);

        CachedPair[] cachedPairs = [];
        if (validationPairs.Count > 0)
        {
            Console.WriteLine(
                $"[Training] Pre-analysing {validationPairs.Count} validation song(s) in parallel...");

            var tmp = new CachedPair?[validationPairs.Count];
            Parallel.For(0, validationPairs.Count,
                new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                i =>
                {
                    var pair = validationPairs[i];
                    try
                    {
                        // New extractor per thread — AudioFeatureExtractor has mutable inner state
                        var audio = new AudioFeatureExtractor().Extract(pair.AudioPath);
                        tmp[i] = new CachedPair(pair, audio);
                    }
                    catch (Exception ex)
                    {
                        lock (printLock)
                            Console.WriteLine(
                                $"[Training] Warning: could not analyse '{pair.AudioPath}': {ex.Message}");
                    }
                });

            cachedPairs = tmp.Where(p => p is not null).Select(p => p!).ToArray();
            Console.WriteLine($"[Training] {cachedPairs.Length} validation song(s) ready.");
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

        Console.WriteLine(
            $"[Training] Training neural placement model 23→1024→512→256→128 ({options.Epochs} epochs max, " +
            $"Adam lr={lr:G4}, patience={options.EarlyStopPatience})...[warmStart={warmStarted}]");
        double bestQuality         = -1.0;
        int    epochsNoImprovement = 0;
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

            Console.WriteLine(
                $"[Training] Epoch {epoch + 1}/{options.Epochs}: " +
                $"loss={loss:F4}  acc={acc:P2}  genQuality={qualityScore:F3}" +
                (currentSynthetics.Count > 0
                    ? $"  synthEx={currentSynthetics.Count}"
                    : string.Empty));

            if (qualityScore > bestQuality + 1e-4)
            {
                bestQuality = qualityScore;
                epochsNoImprovement = 0;
                _placementTrainer.SaveBestWeights();
            }
            else
            {
                epochsNoImprovement++;
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
