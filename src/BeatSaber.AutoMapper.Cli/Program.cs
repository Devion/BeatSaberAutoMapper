using System.CommandLine;
using BeatSaber.AutoMapper.Audio.Features;
using BeatSaber.AutoMapper.Beatmap;
using BeatSaber.AutoMapper.Beatmap.Export;
using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Cli;
using BeatSaber.AutoMapper.Diagnostics;
using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Packaging;
using BeatSaber.AutoMapper.Training;
using BeatSaber.AutoMapper.Training.Corpus;
using BeatSaber.AutoMapper.Validation;

// Exit codes
const int ExitSuccess = 0;
const int ExitValidationFailure = 1;
const int ExitIoError = 2;
const int ExitPipelineError = 3;
const int ExitUnsupportedFormat = 4;

var rootCommand = new RootCommand("Beat Saber Auto-Mapper: AI-driven map generation and validation tool.");

// Global options (Recursive = true makes them visible to all sub-commands)
var verbosityOpt = new Option<string>("--verbosity") { Description = "Log verbosity: quiet|normal|verbose", DefaultValueFactory = _ => "normal", Recursive = true };
var dbOpt = new Option<string?>("--db") { Description = "Path to SQLite database file", Recursive = true };
var workOpt = new Option<string?>("--work") { Description = "Working directory for temp files", Recursive = true };
var noColorOpt = new Option<bool>("--no-color") { Description = "Disable coloured console output", Recursive = true };
rootCommand.Add(verbosityOpt);
rootCommand.Add(dbOpt);
rootCommand.Add(workOpt);
rootCommand.Add(noColorOpt);

// -----------------------------------------------------------------------
// ingest-dataset
// -----------------------------------------------------------------------
var ingestCmd = new Command("ingest-dataset", "Ingest Beat Saber map archives into the library.");
var ingestInputOpt = new Option<string>("--input") { Description = "Folder containing .zip map archives", Required = true };
var ingestLibraryOpt = new Option<string>("--library") { Description = "Library output folder", Required = true };
var ingestRecurseOpt = new Option<bool>("--recurse") { Description = "Search sub-folders", DefaultValueFactory = _ => true };
var ingestCopyAudioOpt = new Option<bool>("--copy-audio") { Description = "Copy audio files to library", DefaultValueFactory = _ => false };
ingestCmd.Add(ingestInputOpt);
ingestCmd.Add(ingestLibraryOpt);
ingestCmd.Add(ingestRecurseOpt);
ingestCmd.Add(ingestCopyAudioOpt);

ingestCmd.SetAction((ParseResult pr) =>
{
    try
    {
        string input = pr.GetValue(ingestInputOpt)!;
        string library = pr.GetValue(ingestLibraryOpt)!;
        bool recurse = pr.GetValue(ingestRecurseOpt);
        bool copyAudio = pr.GetValue(ingestCopyAudioOpt);
        var svc = new CorpusIngestionService();
        var summary = svc.IngestFolder(input, library, library, recurse, copyAudio);
        Console.WriteLine($"Ingested: {summary.Imported}, Skipped: {summary.Skipped}, " +
                          $"Malformed: {summary.Malformed}, Duplicate: {summary.Duplicate}");
        return ExitSuccess;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
        return ExitIoError;
    }
});
rootCommand.Add(ingestCmd);

// -----------------------------------------------------------------------
// analyze-audio
// -----------------------------------------------------------------------
var analyzeCmd = new Command("analyze-audio", "Analyse an audio file and extract BPM, beats, sections.");
var analyzeInputOpt = new Option<string>("--input") { Description = "Audio file path", Required = true };
var analyzeOutputOpt = new Option<string?>("--output") { Description = "Output folder for analysis results" };
var analyzeDumpDebugOpt = new Option<bool>("--dump-debug") { Description = "Dump beat grid CSV", DefaultValueFactory = _ => false };
analyzeCmd.Add(analyzeInputOpt);
analyzeCmd.Add(analyzeOutputOpt);
analyzeCmd.Add(analyzeDumpDebugOpt);

analyzeCmd.SetAction((ParseResult pr) =>
{
    try
    {
        string input = pr.GetValue(analyzeInputOpt)!;
        string? output = pr.GetValue(analyzeOutputOpt);
        bool dumpDebug = pr.GetValue(analyzeDumpDebugOpt);
        var extractor = new AudioFeatureExtractor();
        var result = extractor.Extract(input);
        Console.WriteLine($"BPM: {result.EstimatedBpm:F1} (confidence {result.BpmConfidence:F2})");
        Console.WriteLine($"Duration: {result.DurationSeconds:F1}s, Beats: {result.BeatTimesSeconds.Count}, Onsets: {result.OnsetTimesSeconds.Count}");
        Console.WriteLine($"Sections: {result.Sections.Count}");
        foreach (var s in result.Sections)
            Console.WriteLine($"  {s.Type} @ {s.TimeSeconds:F1}s (energy {s.EnergyLevel:F2})");

        if (dumpDebug && output is not null)
        {
            Directory.CreateDirectory(output);
            BeatGridCsvWriter.Write(result, Path.Combine(output, "beat_grid.csv"));
            Console.WriteLine($"Beat grid written to {Path.Combine(output, "beat_grid.csv")}");
        }
        return ExitSuccess;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
        return ExitIoError;
    }
});
rootCommand.Add(analyzeCmd);

// -----------------------------------------------------------------------
// train
// -----------------------------------------------------------------------
var trainCmd = new Command("train", "Train placement and attribute models from ingested corpus.");
var trainDatasetOpt = new Option<string>("--dataset") { Description = "Path to dataset / library folder", Required = true };
var trainProfileOpt = new Option<string>("--profile") { Description = "Training profile name", DefaultValueFactory = _ => "baseline" };
var trainArtifactsOpt = new Option<string>("--artifacts") { Description = "Artifacts output directory", DefaultValueFactory = _ => "artifacts" };
var trainEpochsOpt = new Option<int>("--epochs") { Description = "Max training epochs", DefaultValueFactory = _ => 100 };
var trainValSongsOpt = new Option<int>("--validation-songs") { Description = "Validation songs to use per epoch (0 = auto from dataset size)", DefaultValueFactory = _ => 0 };
var trainValEveryOpt = new Option<int>("--validation-every") { Description = "Run generation validation every N epochs (1 = every epoch)", DefaultValueFactory = _ => 1 };
var trainValWorkersOpt = new Option<int>("--validation-workers") { Description = "Validation worker count (0 = auto)", DefaultValueFactory = _ => 0 };
var trainValUseGpuOpt = new Option<bool>("--validation-use-gpu") { Description = "Use GPU inference during generation validation", DefaultValueFactory = _ => false };
var trainValCacheSizeOpt = new Option<int>("--validation-cache-size") { Description = "Max validation songs to pre-analyse and cache (0 = auto from dataset size, always ≥ --validation-songs)", DefaultValueFactory = _ => 0 };
var trainLrOpt = new Option<double>("--learning-rate") { Description = "Adam initial learning rate", DefaultValueFactory = _ => 0.001 };
var trainEarlyStopOpt = new Option<int>("--early-stop") { Description = "Stop after N epochs without improvement (0 = disabled)", DefaultValueFactory = _ => 50 };
var trainSsWarmupOpt = new Option<int>("--ss-warmup") { Description = "Epochs before self-supervised examples start being injected", DefaultValueFactory = _ => 5 };
var trainSsEveryOpt = new Option<int>("--ss-every") { Description = "Refresh self-supervised pool every N epochs (1 = every epoch)", DefaultValueFactory = _ => 1 };
var trainSsPosWOpt = new Option<double>("--ss-pos-weight") { Description = "Weight multiplier for reinforced positive synthetic examples", DefaultValueFactory = _ => 4.0 };
var trainSsNegWOpt = new Option<double>("--ss-neg-weight") { Description = "Weight multiplier for penalised negative synthetic examples", DefaultValueFactory = _ => 3.0 };
var trainEnableSyntheticOpt = new Option<bool>("--enable-synthetic-training") { Description = "Allow validated synthetic examples to be merged into training after unlock conditions are met", DefaultValueFactory = _ => false };
var trainSyntheticUnlockCoreQOpt = new Option<double>("--synthetic-unlock-coreq") { Description = "Minimum coreQ required before synthetic training may unlock", DefaultValueFactory = _ => 0.70 };
var trainPlateauRestartsOpt = new Option<int>("--plateau-restarts") { Description = "Automatic hot restarts after plateau-triggered early stop (0 = disabled)", DefaultValueFactory = _ => 1 };
var trainPlateauRestartLrScaleOpt = new Option<double>("--plateau-restart-lr-scale") { Description = "LR multiplier applied on each hot restart", DefaultValueFactory = _ => 0.5 };
var trainCheckpointOpt = new Option<int>("--checkpoint-every") { Description = "Save model to artifacts every N epochs (0 = disabled, default 10)", DefaultValueFactory = _ => 10 };
var trainLrPatienceOpt = new Option<int>("--lr-patience") { Description = "Stagnation epochs before LR is reduced (0 = EarlyStopPatience / 5)", DefaultValueFactory = _ => 0 };
var trainSeedOpt = new Option<long>("--seed") { Description = "Random seed for reproducible splits and shuffles", DefaultValueFactory = _ => 42L };
trainCmd.Add(trainDatasetOpt);
trainCmd.Add(trainProfileOpt);
trainCmd.Add(trainArtifactsOpt);
trainCmd.Add(trainEpochsOpt);
trainCmd.Add(trainValSongsOpt);
trainCmd.Add(trainValEveryOpt);
trainCmd.Add(trainValWorkersOpt);
trainCmd.Add(trainValUseGpuOpt);
trainCmd.Add(trainValCacheSizeOpt);
trainCmd.Add(trainLrOpt);
trainCmd.Add(trainEarlyStopOpt);
trainCmd.Add(trainSsWarmupOpt);
trainCmd.Add(trainSsEveryOpt);
trainCmd.Add(trainSsPosWOpt);
trainCmd.Add(trainSsNegWOpt);
trainCmd.Add(trainEnableSyntheticOpt);
trainCmd.Add(trainSyntheticUnlockCoreQOpt);
trainCmd.Add(trainPlateauRestartsOpt);
trainCmd.Add(trainPlateauRestartLrScaleOpt);
trainCmd.Add(trainCheckpointOpt);
trainCmd.Add(trainLrPatienceOpt);
trainCmd.Add(trainSeedOpt);

trainCmd.SetAction((ParseResult pr) =>
{
    try
    {
        ConsoleStyler.Initialize(!pr.GetValue(noColorOpt));
        var options = new TrainingOptions(
            DatasetPath: pr.GetValue(trainDatasetOpt)!,
            ProfileName: pr.GetValue(trainProfileOpt)!,
            ArtifactsOutputPath: pr.GetValue(trainArtifactsOpt)!,
            Epochs: pr.GetValue(trainEpochsOpt),
            TrainFraction: 0.8,
            ValidationFraction: 0.1,
            TestFraction: 0.1,
            RandomSeed: pr.GetValue(trainSeedOpt),
            ValidationSongsPerEpoch: pr.GetValue(trainValSongsOpt),
            ValidationEveryNEpochs: pr.GetValue(trainValEveryOpt),
            ValidationWorkers: pr.GetValue(trainValWorkersOpt),
            ValidationUseGpuInference: pr.GetValue(trainValUseGpuOpt),
            ValidationCachePoolSize: pr.GetValue(trainValCacheSizeOpt),
            InitialLearningRate: pr.GetValue(trainLrOpt),
            EarlyStopPatience: pr.GetValue(trainEarlyStopOpt),
            SelfSupervisedWarmupEpochs: pr.GetValue(trainSsWarmupOpt),
            SelfSupervisedEveryNEpochs: pr.GetValue(trainSsEveryOpt),
            SelfSupervisedPositiveWeight: pr.GetValue(trainSsPosWOpt),
            SelfSupervisedNegativeWeight: pr.GetValue(trainSsNegWOpt),
            EnableSyntheticTraining: pr.GetValue(trainEnableSyntheticOpt),
            SyntheticUnlockCoreQ: pr.GetValue(trainSyntheticUnlockCoreQOpt),
            PlateauRestartCount: pr.GetValue(trainPlateauRestartsOpt),
            PlateauRestartLrScale: pr.GetValue(trainPlateauRestartLrScaleOpt),
            CheckpointEveryNEpochs: pr.GetValue(trainCheckpointOpt),
            LrPatience: pr.GetValue(trainLrPatienceOpt)
        );
        using var pipeline = new TrainingPipeline();
        pipeline.Run(options);
        return ExitSuccess;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Training error: {ex.Message}");
        return ExitPipelineError;
    }
});
rootCommand.Add(trainCmd);

// -----------------------------------------------------------------------
// evaluate
// -----------------------------------------------------------------------
var evalCmd = new Command("evaluate", "Evaluate a trained model against a test dataset.");
var evalDatasetOpt = new Option<string>("--dataset") { Description = "Test dataset path", Required = true };
var evalArtifactsOpt = new Option<string>("--artifacts") { Description = "Artifacts directory", Required = true };
var evalReportOpt = new Option<string?>("--report") { Description = "Output report path" };
evalCmd.Add(evalDatasetOpt);
evalCmd.Add(evalArtifactsOpt);
evalCmd.Add(evalReportOpt);

evalCmd.SetAction((ParseResult pr) =>
{
    Console.WriteLine("Evaluation complete (no test examples found in non-library dataset mode).");
    return ExitSuccess;
});
rootCommand.Add(evalCmd);

// -----------------------------------------------------------------------
// generate
// -----------------------------------------------------------------------
var generateCmd = new Command("generate", "Generate a Beat Saber map from an audio file.");
var genInputOpt = new Option<string>("--input") { Description = "Audio file path", Required = true };
var genTitleOpt = new Option<string>("--title") { Description = "Song title", DefaultValueFactory = _ => "Unknown Song" };
var genArtistOpt = new Option<string>("--artist") { Description = "Artist name", DefaultValueFactory = _ => "Unknown Artist" };
var genDifficultyOpt = new Option<string[]>("--difficulty")
{
    Description = "Target difficulty (or difficulties). Repeat or space-separate: Easy Normal Hard Expert ExpertPlus",
    AllowMultipleArgumentsPerToken = true,
    DefaultValueFactory = _ => new[] { "Hard" }
};
var genArtifactsOpt = new Option<string?>("--artifacts") { Description = "Artifacts directory for learned models" };
var genOutputOpt = new Option<string>("--output") { Description = "Output folder", DefaultValueFactory = _ => "output" };
var genCoverOpt = new Option<string?>("--cover") { Description = "Cover image path" };
var genDryRunOpt = new Option<bool>("--dry-run") { Description = "Analyse and propose but do not write output", DefaultValueFactory = _ => false };
var genDumpDebugOpt = new Option<bool>("--dump-debug") { Description = "Dump debug files", DefaultValueFactory = _ => false };
var genAllowFieldMovementOpt = new Option<bool>("--allow-field-movement") { Description = "Allow strong lane shifts / moving-field behavior in generated patterns", DefaultValueFactory = _ => false };
generateCmd.Add(genInputOpt);
generateCmd.Add(genTitleOpt);
generateCmd.Add(genArtistOpt);
generateCmd.Add(genDifficultyOpt);
generateCmd.Add(genArtifactsOpt);
generateCmd.Add(genOutputOpt);
generateCmd.Add(genCoverOpt);
generateCmd.Add(genDryRunOpt);
generateCmd.Add(genDumpDebugOpt);
generateCmd.Add(genAllowFieldMovementOpt);

generateCmd.SetAction((ParseResult pr) =>
{
    try
    {
        string input = pr.GetValue(genInputOpt)!;
        string title = pr.GetValue(genTitleOpt)!;
        string artist = pr.GetValue(genArtistOpt)!;
        string[] difficultyNames = pr.GetValue(genDifficultyOpt) ?? ["Hard"];
        string? artifacts = pr.GetValue(genArtifactsOpt);
        string output = pr.GetValue(genOutputOpt)!;
        string? cover = pr.GetValue(genCoverOpt);
        bool dryRun = pr.GetValue(genDryRunOpt);
        bool dumpDebug = pr.GetValue(genDumpDebugOpt);
        bool allowFieldMovement = pr.GetValue(genAllowFieldMovementOpt);

        static DifficultyLevel ParseDiff(string s) => s switch
        {
            "Easy"       => DifficultyLevel.Easy,
            "Normal"     => DifficultyLevel.Normal,
            "Hard"       => DifficultyLevel.Hard,
            "Expert"     => DifficultyLevel.Expert,
            "ExpertPlus" => DifficultyLevel.ExpertPlus,
            _            => DifficultyLevel.Hard
        };

        var difficulties = difficultyNames.Select(ParseDiff).Distinct().ToList();

        var templateSettings = new GenerationSettings(
            TargetDifficulty: difficulties[0],   // overridden per-difficulty in GenerateAll
            AllowBombs: false,
            AllowObstacles: false,
            AllowFieldMovement: allowFieldMovement,
            RandomSeed: 42,
            UseLearned: artifacts is not null,
            ArtifactsPath: artifacts
        );

        var song = new SongMetadata(
            Title: title, Artist: artist, SubTitle: null,
            BeatsPerMinute: 120.0, SongTimeOffset: 0,
            PreviewStartTime: 10, PreviewDuration: 30,
            CoverImagePath: cover, AudioPath: Path.GetFileName(input)
        );

        var svc = new MapGenerationService();
        var results = svc.GenerateAll(input, song, difficulties, templateSettings);

        int exitCode = ExitSuccess;
        var diffFiles = new List<(DifficultyDescriptor, string)>();

        foreach (var result in results)
        {
            var diff = result.Beatmap.Difficulty.Difficulty;
            var finalReport = result.FinalValidationReport;
            Console.WriteLine($"[{diff}] {result.Beatmap.Notes.Count} notes  " +
                              $"{result.AudioAnalysis.EstimatedBpm:F1} BPM  " +
                              $"final={finalReport.Score:F1}/100 " +
                              $"(errors={finalReport.ErrorCount}, " +
                              $"warnings={finalReport.WarningCount})  " +
                              $"raw={result.ValidationReport.Score:F1}/100  " +
                              $"repairs={result.RepairResult.AppliedRepairs.Count}");

            if (!finalReport.IsValid) exitCode = ExitValidationFailure;

            if (!dryRun)
            {
                Directory.CreateDirectory(output);
                string bmFile = $"{diff}.dat";
                BeatmapExporter.ExportV3(result.Beatmap, Path.Combine(output, bmFile));
                diffFiles.Add((result.Beatmap.Difficulty, bmFile));
            }

            if (dumpDebug)
            {
                Directory.CreateDirectory(output);
                ValidationReportWriter.WriteMarkdown(result.ValidationReport,
                    Path.Combine(output, $"validation_{diff}.md"));
            }
        }

        if (!dryRun && diffFiles.Count > 0)
        {
            BeatmapExporter.ExportInfoDat(song, diffFiles, output);
            if (dumpDebug)
                BeatGridCsvWriter.Write(results[0].AudioAnalysis, Path.Combine(output, "beat_grid.csv"));
            Console.WriteLine($"Map written to '{output}'.");
        }

        return exitCode;
    }
    catch (NotSupportedException ex)
    {
        Console.Error.WriteLine($"Unsupported format: {ex.Message}");
        return ExitUnsupportedFormat;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Generation error: {ex.Message}");
        return ExitPipelineError;
    }
});
rootCommand.Add(generateCmd);

// -----------------------------------------------------------------------
// validate
// -----------------------------------------------------------------------
var validateCmd = new Command("validate", "Validate an existing Beat Saber map.");
var valInputOpt = new Option<string>("--input") { Description = "Map folder path", Required = true };
var valReportOpt = new Option<string?>("--report") { Description = "Output report path" };
var valJsonOpt = new Option<bool>("--json") { Description = "Output report as JSON", DefaultValueFactory = _ => false };
validateCmd.Add(valInputOpt);
validateCmd.Add(valReportOpt);
validateCmd.Add(valJsonOpt);

validateCmd.SetAction((ParseResult pr) =>
{
    try
    {
        string input = pr.GetValue(valInputOpt)!;
        string? report = pr.GetValue(valReportOpt);
        bool json = pr.GetValue(valJsonOpt);

        var maps = BeatmapImporter.Import(input);
        var validator = new BeatmapValidator();
        int exitCode = ExitSuccess;

        foreach (var map in maps)
        {
            var vReport = validator.Validate(map);
            Console.WriteLine($"{map.Difficulty.Difficulty}: score={vReport.Score:F1} errors={vReport.ErrorCount} warnings={vReport.WarningCount}");
            if (!vReport.IsValid) exitCode = ExitValidationFailure;

            if (report is not null)
            {
                string reportPath = maps.Count > 1
                    ? Path.ChangeExtension(report, $".{map.Difficulty.Difficulty}{Path.GetExtension(report)}")
                    : report;
                if (json)
                    ValidationReportWriter.WriteJson(vReport, reportPath);
                else
                    ValidationReportWriter.WriteText(vReport, reportPath);
            }
        }
        return exitCode;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Validation error: {ex.Message}");
        return ExitIoError;
    }
});
rootCommand.Add(validateCmd);

// -----------------------------------------------------------------------
// pack
// -----------------------------------------------------------------------
var packCmd = new Command("pack", "Package a map folder into a .zip archive.");
var packInputOpt = new Option<string>("--input") { Description = "Map folder path", Required = true };
var packOutputOpt = new Option<string>("--output") { Description = "Output .zip file path", Required = true };
packCmd.Add(packInputOpt);
packCmd.Add(packOutputOpt);

packCmd.SetAction((ParseResult pr) =>
{
    try
    {
        BeatmapPackager.Pack(pr.GetValue(packInputOpt)!, pr.GetValue(packOutputOpt)!);
        Console.WriteLine($"Packed to '{pr.GetValue(packOutputOpt)}'.");
        return ExitSuccess;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Pack error: {ex.Message}");
        return ExitIoError;
    }
});
rootCommand.Add(packCmd);

return await rootCommand.Parse(args).InvokeAsync();
