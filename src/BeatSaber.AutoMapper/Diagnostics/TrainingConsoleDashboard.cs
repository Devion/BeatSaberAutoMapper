using System.Text;

namespace BeatSaber.AutoMapper.Diagnostics;

public sealed class TrainingConsoleDashboard : IDisposable
{
    public sealed record Config(
        string Device,
        string ValidationDevice,
        int InputDim,
        int HiddenDim,
        int Layers,
        int MlpHidden,
        double LearningRate,
        int Epochs,
        int ValidationEvery,
        int ValidationSongs,
        int ValidationPairs,
        int PlateauRestarts,
        double PlateauRestartScale,
        string CheckpointKind);

    public sealed record EpochSummary(
        int Epoch,
        int Epochs,
        bool ValidationRan,
        double Loss,
        double PlaceBce,
        double CoreQ,
        double FullQ,
        double BestQ,
        double SmoothedQ,
        double LearningRate,
        double LossSlope,
        double QualitySlope,
        double TrainSeconds,
        double ValidationSeconds,
        double EpochSeconds,
        int ValidationPairs,
        int SyntheticExamples,
        int StagnationEpochs,
        int EarlyStopPatience,
        string TrainMix,
        string ValidationInfo,
        string ValidationDelta,
        string DiffScores,
        string Status);

    private const int Width = 80;
    private const int Height = 25;
    private readonly object _lock = new();
    private readonly string[] _frame = new string[Height];
    private readonly Queue<string> _notices = new();
    private readonly List<double> _qualityHistory = [];
    private bool _disposed;

    private Config? _config;
    private int _currentEpoch;
    private int _totalEpochs;
    private string _phase = "init";
    private int _trainBatchesDone;
    private int _trainBatchesTotal;
    private int _trainChunks;
    private double _trainElapsed;
    private int _validationDone;
    private int _validationTotal;
    private int _validationSongs;
    private int _validationWorkers;
    private double _validationElapsed;
    private EpochSummary? _lastEpoch;
    private EpochSummary? _bestEpoch;
    private int _epochsCompleted;
    private double _sumLoss;
    private double _sumTrainSeconds;
    private double _sumValidationSeconds;
    private double _sumEpochSeconds;
    private int _validatedEpochs;
    private double _sumCoreQ;
    private double _sumFullQ;

    public static TrainingConsoleDashboard? TryCreate()
    {
        if (Console.IsOutputRedirected)
            return null;

        try
        {
            return new TrainingConsoleDashboard();
        }
        catch
        {
            return null;
        }
    }

    private TrainingConsoleDashboard()
    {
        EnsureConsoleSize();
        Console.CursorVisible = false;
        Console.Clear();
        Array.Fill(_frame, string.Empty);
        Render();
    }

    public void SetConfig(Config config)
    {
        lock (_lock)
        {
            _config = config;
            _totalEpochs = config.Epochs;
            Render();
        }
    }

    public void AddNotice(string message)
    {
        lock (_lock)
        {
            if (_notices.Count >= 4)
                _notices.Dequeue();
            _notices.Enqueue(message);
            Render();
        }
    }

    public void BeginEpoch(int epoch, int totalEpochs)
    {
        lock (_lock)
        {
            _currentEpoch = epoch;
            _totalEpochs = totalEpochs;
            _phase = "train";
            _trainBatchesDone = 0;
            _trainBatchesTotal = 0;
            _trainChunks = 0;
            _trainElapsed = 0;
            _validationDone = 0;
            _validationTotal = 0;
            _validationElapsed = 0;
            Render();
        }
    }

    public void OnTrainStarted(int totalBatches)
    {
        lock (_lock)
        {
            _phase = "train";
            _trainBatchesTotal = totalBatches;
            Render();
        }
    }

    public void OnTrainProgress(int completedBatches, int totalBatches, int chunks, double elapsedSeconds)
    {
        lock (_lock)
        {
            _phase = "train";
            _trainBatchesDone = completedBatches;
            _trainBatchesTotal = totalBatches;
            _trainChunks = chunks;
            _trainElapsed = elapsedSeconds;
            Render();
        }
    }

    public void OnValidationStarted(int pairs, int songs, int workers)
    {
        lock (_lock)
        {
            _phase = "validate";
            _validationDone = 0;
            _validationTotal = pairs;
            _validationSongs = songs;
            _validationWorkers = workers;
            _validationElapsed = 0;
            Render();
        }
    }

    public void OnValidationProgress(int completedPairs, int totalPairs, double elapsedSeconds)
    {
        lock (_lock)
        {
            _phase = "validate";
            _validationDone = completedPairs;
            _validationTotal = totalPairs;
            _validationElapsed = elapsedSeconds;
            Render();
        }
    }

    public void CompleteEpoch(EpochSummary summary)
    {
        lock (_lock)
        {
            _phase = "idle";
            _lastEpoch = summary;
            _epochsCompleted++;
            _sumLoss += summary.Loss;
            _sumTrainSeconds += summary.TrainSeconds;
            _sumValidationSeconds += summary.ValidationSeconds;
            _sumEpochSeconds += summary.EpochSeconds;

            if (summary.ValidationRan)
            {
                _validatedEpochs++;
                _sumCoreQ += summary.CoreQ;
                _sumFullQ += summary.FullQ;
                _qualityHistory.Add(summary.CoreQ);
                if (_qualityHistory.Count > 32)
                    _qualityHistory.RemoveAt(0);

                if (_bestEpoch is null || summary.CoreQ >= _bestEpoch.CoreQ)
                    _bestEpoch = summary;
            }

            Render();
        }
    }

    public void MarkComplete(string message)
    {
        AddNotice(message);
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
                return;
            _disposed = true;
            try
            {
                Console.SetCursorPosition(0, Height);
                Console.CursorVisible = true;
            }
            catch
            {
            }
        }
    }

    private void EnsureConsoleSize()
    {
        if (!OperatingSystem.IsWindows())
            return;

        try
        {
            if (Console.BufferWidth < Width)
                Console.BufferWidth = Width;
            if (Console.BufferHeight < Height + 1)
                Console.BufferHeight = Height + 1;
            if (Console.WindowWidth < Width)
                Console.WindowWidth = Width;
            if (Console.WindowHeight < Height)
                Console.WindowHeight = Height;
        }
        catch
        {
        }
    }

    private void Render()
    {
        var lines = BuildFrame();
        for (int i = 0; i < Height; i++)
        {
            string next = Fit(lines[i]);
            if (_frame[i] == next)
                continue;

            Console.SetCursorPosition(0, i);
            Console.Write(next);
            _frame[i] = next;
        }
    }

    private string[] BuildFrame()
    {
        var lines = new string[Height];
        lines[0] = Center("Beat Saber AutoMapper Training");
        lines[1] = Fit($"Epoch {FmtEpoch()}  Phase {(_phase ?? "init"),-8}  Time {DateTime.Now:HH:mm:ss}");

        if (_config is not null)
        {
            lines[2] = Fit(
                $"Model {_config.InputDim}->GRU({_config.HiddenDim}x{_config.Layers})->{_config.MlpHidden}  LR {_config.LearningRate:G4}");
            lines[3] = Fit(
                $"Device {_config.Device,-10}  Val {_config.ValidationDevice,-3}  Ckpt {_config.CheckpointKind,-7}");
            lines[4] = Fit(
                $"Val every {_config.ValidationEvery,2}  songs {_config.ValidationSongs,2}  pairs {_config.ValidationPairs,3}  Restarts {_config.PlateauRestarts}");
        }
        else
        {
            lines[2] = string.Empty;
            lines[3] = string.Empty;
            lines[4] = string.Empty;
        }

        lines[5] = Divider();
        lines[6] = Fit($"Current  {ProgressBar(_trainBatchesDone, Math.Max(1, _trainBatchesTotal), 30)}  train");
        lines[7] = Fit($"Batches {_trainBatchesDone,3}/{Math.Max(0, _trainBatchesTotal),-3}  Chunks {_trainChunks,4}  Elapsed {_trainElapsed,6:F1}s");
        lines[8] = Fit($"Validate {ProgressBar(_validationDone, Math.Max(1, _validationTotal), 30)}  val");
        lines[9] = Fit($"Pairs {_validationDone,3}/{Math.Max(0, _validationTotal),-3}  Songs {_validationSongs,3}  Workers {_validationWorkers,2}  Elapsed {_validationElapsed,6:F1}s");

        lines[10] = Divider();
        lines[11] = Fit("Last Epoch");
        lines[12] = _lastEpoch is null
            ? Fit("  n/a")
            : Fit($"  loss {_lastEpoch.Loss:F4}  plBCE {_lastEpoch.PlaceBce:F4}  coreQ {FmtQ(_lastEpoch.CoreQ, _lastEpoch.ValidationRan)}  fullQ {FmtQ(_lastEpoch.FullQ, _lastEpoch.ValidationRan)}");
        lines[13] = _lastEpoch is null
            ? string.Empty
            : Fit($"  train {_lastEpoch.TrainSeconds:F1}s  val {_lastEpoch.ValidationSeconds:F1}s  epoch {_lastEpoch.EpochSeconds:F1}s  lr {_lastEpoch.LearningRate:G4}");
        lines[14] = _lastEpoch is null ? string.Empty : Fit($"  {_lastEpoch.Status}  {_lastEpoch.ValidationInfo}");

        lines[15] = Divider();
        lines[16] = Fit(_bestEpoch is null
            ? "Best Epoch  n/a"
            : $"Best Epoch  #{_bestEpoch.Epoch,3}  coreQ {_bestEpoch.CoreQ:F3}  fullQ {_bestEpoch.FullQ:F3}  loss {_bestEpoch.Loss:F4}");
        lines[17] = Fit(
            $"Avg        loss {Avg(_sumLoss, _epochsCompleted):F4}  train {Avg(_sumTrainSeconds, _epochsCompleted):F1}s  val {Avg(_sumValidationSeconds, _epochsCompleted):F1}s  epoch {Avg(_sumEpochSeconds, _epochsCompleted):F1}s");
        lines[18] = Fit(
            $"Avg Q      coreQ {FmtAvgQ(Avg(_sumCoreQ, _validatedEpochs), _validatedEpochs)}  fullQ {FmtAvgQ(Avg(_sumFullQ, _validatedEpochs), _validatedEpochs)}");

        lines[19] = Divider();
        lines[20] = Fit($"CoreQ Trend {Graph(_qualityHistory, 64)}");

        lines[21] = Divider();
        lines[22] = Fit($"Notice 1: {GetNotice(0)}");
        lines[23] = Fit($"Notice 2: {GetNotice(1)}");
        lines[24] = Fit($"Notice 3: {GetNotice(2)}");
        return lines;
    }

    private string FmtEpoch()
    {
        if (_totalEpochs <= 0)
            return _currentEpoch.ToString();
        return $"{_currentEpoch,3}/{_totalEpochs,-3}";
    }

    private static string ProgressBar(int done, int total, int width)
    {
        total = Math.Max(1, total);
        done = Math.Clamp(done, 0, total);
        int filled = (int)Math.Round(done / (double)total * width);
        return $"[{new string('#', filled)}{new string('.', Math.Max(0, width - filled))}]";
    }

    private static double Avg(double total, int count) => count > 0 ? total / count : 0.0;

    private static string FmtQ(double value, bool hasValue) => hasValue ? value.ToString("F3") : "skip";

    private static string FmtAvgQ(double value, int count) => count > 0 ? value.ToString("F3") : "n/a";

    private static string Graph(IReadOnlyList<double> values, int width)
    {
        if (values.Count == 0)
            return "(no validation yet)";

        const string ramp = " .:-=+*#%@";
        double min = values.Min();
        double max = values.Max();
        double span = Math.Max(1e-6, max - min);
        var sb = new StringBuilder(width);

        int start = Math.Max(0, values.Count - width);
        for (int i = start; i < values.Count; i++)
        {
            double normalized = (values[i] - min) / span;
            int idx = (int)Math.Round(normalized * (ramp.Length - 1));
            sb.Append(ramp[Math.Clamp(idx, 0, ramp.Length - 1)]);
        }

        if (sb.Length < width)
            sb.Insert(0, new string(' ', width - sb.Length));
        return sb.ToString();
    }

    private string GetNotice(int index)
    {
        var arr = _notices.Reverse().ToArray();
        return index < arr.Length ? arr[index] : string.Empty;
    }

    private static string Divider() => new string('-', Width);

    private static string Center(string text)
    {
        if (text.Length >= Width)
            return text[..Width];
        int left = (Width - text.Length) / 2;
        return new string(' ', left) + text;
    }

    private static string Fit(string value)
    {
        value ??= string.Empty;
        if (value.Length > Width)
            return value[..Width];
        if (value.Length < Width)
            return value.PadRight(Width);
        return value;
    }
}
