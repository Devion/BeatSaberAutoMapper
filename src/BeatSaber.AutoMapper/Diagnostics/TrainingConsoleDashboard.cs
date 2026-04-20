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

    private const int MinWidth = 80;
    private const int MinHeight = 25;
    private readonly object _lock = new();
    private readonly Queue<string> _notices = new();
    private readonly List<double> _qualityHistory = [];
    private string[] _frame;
    private bool _disposed;
    private int _width;
    private int _height;

    private Config? _config;
    private int _currentEpoch;
    private int _totalEpochs;
    private string _phase = "init";
    private int _trainBatchesDone;
    private int _trainBatchesTotal;
    private int _trainChunks;
    private double _trainElapsed;
    private string _stageLabel = "idle";
    private string _stageDetail = string.Empty;
    private int _stageDone;
    private int _stageTotal;
    private double _stageElapsed;
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
        (_width, _height) = ResolveConsoleSize();
        _frame = Enumerable.Repeat(string.Empty, _height).ToArray();
        Console.CursorVisible = false;
        Console.Clear();
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
            if (_notices.Count >= 6)
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
            _stageLabel = "epoch";
            _stageDetail = string.Empty;
            _stageDone = 0;
            _stageTotal = 0;
            _stageElapsed = 0;
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

    public void OnStageStarted(string label, int total, string? detail = null)
    {
        lock (_lock)
        {
            _phase = "prep";
            _stageLabel = label;
            _stageDetail = detail ?? string.Empty;
            _stageDone = 0;
            _stageTotal = Math.Max(0, total);
            _stageElapsed = 0;
            Render();
        }
    }

    public void OnStageProgress(int completed, int total, double elapsedSeconds, string? detail = null)
    {
        lock (_lock)
        {
            _phase = "prep";
            _stageDone = Math.Max(0, completed);
            _stageTotal = Math.Max(0, total);
            _stageElapsed = Math.Max(0, elapsedSeconds);
            if (!string.IsNullOrWhiteSpace(detail))
                _stageDetail = detail!;
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
                if (_qualityHistory.Count > Math.Max(20, InnerWidth - 12))
                    _qualityHistory.RemoveAt(0);

                if (_bestEpoch is null || summary.CoreQ >= _bestEpoch.CoreQ)
                    _bestEpoch = summary;
            }

            Render();
        }
    }

    public void MarkComplete(string message) => AddNotice(message);

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
                return;
            _disposed = true;
            try
            {
                Console.SetCursorPosition(0, _height);
                Console.CursorVisible = true;
            }
            catch
            {
            }
        }
    }

    private int InnerWidth => _width - 4;

    private void Render()
    {
        ResizeIfNeeded();

        var lines = BuildFrame();
        if (_frame.Length != _height)
            _frame = Enumerable.Repeat(string.Empty, _height).ToArray();

        for (int i = 0; i < _height; i++)
        {
            string next = FitVisible(i < lines.Length ? lines[i] : string.Empty, _width);
            if (_frame[i] == next)
                continue;

            Console.SetCursorPosition(0, i);
            Console.Write(next);
            _frame[i] = next;
        }
    }

    private void ResizeIfNeeded()
    {
        var (width, height) = ResolveConsoleSize();
        if (width == _width && height == _height)
            return;

        _width = width;
        _height = height;
        _frame = Enumerable.Repeat(string.Empty, _height).ToArray();
        Console.Clear();
    }

    private static (int Width, int Height) ResolveConsoleSize()
    {
        int width = 100;
        int height = 30;

        try
        {
            width = Math.Max(MinWidth, Console.WindowWidth);
            height = Math.Max(MinHeight, Console.WindowHeight - 1);
        }
        catch
        {
        }

        return (width, height);
    }

    private string[] BuildFrame()
    {
        var lines = new List<string>(_height);

        lines.Add(Border($" Beat Saber AutoMapper :: Training Dashboard "));
        lines.Add(Box(
            $"{Accent("Epoch")} {Value(FmtEpoch()),-12} " +
            $"{Accent("Phase")} {PhaseChip(),-18} " +
            $"{Accent("Clock")} {DateTime.Now:HH:mm:ss}"));

        if (_config is not null)
        {
            lines.Add(Box(
                $"{Accent("Model")} {_config.InputDim}->{_config.HiddenDim}x{_config.Layers}->{_config.MlpHidden}  " +
                $"{Accent("LR")} {Value(_config.LearningRate.ToString("G4"))}  " +
                $"{Accent("Ckpt")} {CheckpointChip(_config.CheckpointKind)}"));
            lines.Add(Box(
                $"{Accent("Device")} {DeviceChip(_config.Device),-12} " +
                $"{Accent("Val")} {DeviceChip(_config.ValidationDevice),-12} " +
                $"{Accent("Every")} {_config.ValidationEvery,2}  " +
                $"{Accent("Songs")} {_config.ValidationSongs,3}  " +
                $"{Accent("Pairs")} {_config.ValidationPairs,4}  " +
                $"{Accent("Restarts")} {_config.PlateauRestarts}x{_config.PlateauRestartScale:F2}"));
        }
        else
        {
            lines.Add(Box("Initializing model configuration..."));
            lines.Add(Box(string.Empty));
        }

        lines.Add(Border(" Pipeline "));
        if (_phase == "prep")
        {
            lines.Add(Box(
                $"{Accent("Setup")} {ProgressBar(_stageDone, _stageTotal, 28, ConsoleColor.Cyan)} " +
                $"{Value($"{_stageDone}/{Math.Max(0, _stageTotal)}"),8}  " +
                $"{Accent("Elapsed")} {FmtSeconds(_stageElapsed),7}  " +
                $"{Accent("ETA")} {FmtEta(_stageDone, _stageTotal, _stageElapsed),7}"));
            lines.Add(Box($"{Accent("Stage")} {TrimRaw(_stageLabel, InnerWidth - 6)}"));
            lines.Add(Box($"{Accent("Detail")} {TrimRaw(_stageDetail, InnerWidth - 7)}"));
        }
        else
        {
            lines.Add(Box(
                $"{Accent("Train")} {ProgressBar(_trainBatchesDone, _trainBatchesTotal, 28, ConsoleColor.Green)} " +
                $"{Value($"{_trainBatchesDone}/{Math.Max(0, _trainBatchesTotal)}"),8}  " +
                $"{Accent("Chunks")} {_trainChunks,4}  " +
                $"{Accent("Elapsed")} {FmtSeconds(_trainElapsed),7}"));
            lines.Add(Box(
                $"{Accent("Valid")} {ProgressBar(_validationDone, _validationTotal, 28, ConsoleColor.Yellow)} " +
                $"{Value($"{_validationDone}/{Math.Max(0, _validationTotal)}"),8}  " +
                $"{Accent("Songs")} {_validationSongs,3}  " +
                $"{Accent("Wrk")} {_validationWorkers,2}  " +
                $"{Accent("Elapsed")} {FmtSeconds(_validationElapsed),7}"));
            lines.Add(Box($"{Accent("Flow")} {CurrentDetailLine()}"));
        }

        lines.Add(Border(" Last Epoch "));
        if (_lastEpoch is null)
        {
            lines.Add(Box("No completed epoch yet."));
            lines.Add(Box(string.Empty));
            lines.Add(Box(string.Empty));
        }
        else
        {
            lines.Add(Box(
                $"{Accent("Loss")} {Value(_lastEpoch.Loss.ToString("F4"))}  " +
                $"{Accent("plBCE")} {Value(_lastEpoch.PlaceBce.ToString("F4"))}  " +
                $"{Accent("coreQ")} {QualityChip(_lastEpoch.CoreQ, _lastEpoch.ValidationRan)}  " +
                $"{Accent("fullQ")} {QualityChip(_lastEpoch.FullQ, _lastEpoch.ValidationRan)}  " +
                $"{Accent("best")} {QualityChip(_lastEpoch.BestQ, _lastEpoch.ValidationRan)}"));
            lines.Add(Box(
                $"{Accent("Train")} {FmtSeconds(_lastEpoch.TrainSeconds),7}  " +
                $"{Accent("Val")} {FmtSeconds(_lastEpoch.ValidationSeconds),7}  " +
                $"{Accent("Epoch")} {FmtSeconds(_lastEpoch.EpochSeconds),7}  " +
                $"{Accent("Synth")} {_lastEpoch.SyntheticExamples,5}  " +
                $"{Accent("Status")} {StatusChip(_lastEpoch.Status)}"));
            lines.Add(Box(
                $"{Accent("Info")} {TrimRaw(_lastEpoch.ValidationInfo, 30)}  " +
                $"{Accent("Delta")} {TrimRaw(_lastEpoch.ValidationDelta, Math.Max(0, InnerWidth - 46))}"));
        }

        lines.Add(Border(" Best / Average "));
        lines.Add(Box(
            _bestEpoch is null
                ? "Best epoch unavailable."
                : $"{Accent("Best")} #{_bestEpoch.Epoch,3}  " +
                  $"{Accent("coreQ")} {QualityChip(_bestEpoch.CoreQ, true)}  " +
                  $"{Accent("fullQ")} {QualityChip(_bestEpoch.FullQ, true)}  " +
                  $"{Accent("Loss")} {Value(_bestEpoch.Loss.ToString("F4"))}  " +
                  $"{Accent("Val")} {FmtSeconds(_bestEpoch.ValidationSeconds),7}"));
        lines.Add(Box(
            $"{Accent("Avg loss")} {Value(Avg(_sumLoss, _epochsCompleted).ToString("F4"))}  " +
            $"{Accent("Avg train")} {FmtSeconds(Avg(_sumTrainSeconds, _epochsCompleted)),7}  " +
            $"{Accent("Avg val")} {FmtSeconds(Avg(_sumValidationSeconds, _epochsCompleted)),7}  " +
            $"{Accent("Avg ep")} {FmtSeconds(Avg(_sumEpochSeconds, _epochsCompleted)),7}"));
        lines.Add(Box(
            $"{Accent("Avg coreQ")} {AvgQualityChip(Avg(_sumCoreQ, _validatedEpochs), _validatedEpochs)}  " +
            $"{Accent("Avg fullQ")} {AvgQualityChip(Avg(_sumFullQ, _validatedEpochs), _validatedEpochs)}  " +
            $"{Accent("Validated")} {_validatedEpochs,4}"));

        lines.Add(Border(" Quality Trend "));
        lines.Add(Box($"{Accent("CoreQ")} {Graph(_qualityHistory, Math.Max(20, InnerWidth - 8))}"));

        lines.Add(Border(" Activity "));
        int noticeLines = Math.Max(3, _height - lines.Count - 1);
        for (int i = 0; i < noticeLines; i++)
            lines.Add(Box($"{Accent((i + 1).ToString("00"))} {TrimRaw(GetNotice(i), InnerWidth - 4)}"));

        while (lines.Count < _height - 1)
            lines.Add(Box(string.Empty));

        lines.Add(Border(string.Empty));
        return lines.Take(_height).ToArray();
    }

    private string CurrentDetailLine()
    {
        if (_phase == "validate")
            return $"{TrimRaw(_stageDetail, Math.Max(0, InnerWidth - 30))}";
        if (_phase == "train")
            return _lastEpoch?.TrainMix is { Length: > 0 } mix
                ? TrimRaw(mix, Math.Max(0, InnerWidth - 8))
                : "training in progress";
        return TrimRaw(_stageDetail, Math.Max(0, InnerWidth - 8));
    }

    private string FmtEpoch()
    {
        if (_totalEpochs <= 0)
            return _currentEpoch.ToString();
        return $"{_currentEpoch,3}/{_totalEpochs,-3}";
    }

    private static double Avg(double total, int count) => count > 0 ? total / count : 0.0;

    private static string FmtEta(int done, int total, double elapsed)
    {
        if (done <= 0 || total <= done || elapsed <= 0)
            return "--";
        double eta = elapsed / done * (total - done);
        return FmtSeconds(eta);
    }

    private static string FmtSeconds(double seconds)
    {
        if (seconds <= 0)
            return "--";
        if (seconds < 100)
            return $"{seconds,5:F1}s";
        if (seconds < 3600)
            return $"{seconds,5:F0}s";
        var ts = TimeSpan.FromSeconds(seconds);
        return $"{(int)ts.TotalHours}:{ts.Minutes:00}:{ts.Seconds:00}";
    }

    private static string Graph(IReadOnlyList<double> values, int width)
    {
        if (values.Count == 0)
            return TrimRaw("(no validation history)", width);

        const string ramp = " .:-=+*#%@";
        double min = values.Min();
        double max = values.Max();
        double span = Math.Max(1e-6, max - min);
        var sb = new StringBuilder();
        int start = Math.Max(0, values.Count - width);

        for (int i = start; i < values.Count; i++)
        {
            double normalized = (values[i] - min) / span;
            int idx = (int)Math.Round(normalized * (ramp.Length - 1));
            sb.Append(ramp[Math.Clamp(idx, 0, ramp.Length - 1)]);
        }

        return sb.ToString().PadLeft(width);
    }

    private static string ProgressBar(int done, int total, int width, ConsoleColor color)
    {
        total = Math.Max(1, total);
        done = Math.Clamp(done, 0, total);
        int filled = (int)Math.Round(done / (double)total * width);
        var raw = new StringBuilder(width + 2);
        raw.Append('[');
        if (filled <= 0)
        {
            raw.Append(new string('.', width));
        }
        else if (filled >= width)
        {
            raw.Append(new string('=', width));
        }
        else
        {
            raw.Append(new string('=', Math.Max(0, filled - 1)));
            raw.Append('>');
            raw.Append(new string('.', Math.Max(0, width - filled)));
        }
        raw.Append(']');
        return ConsoleStyler.Colorize(raw.ToString(), color);
    }

    private string Border(string title)
    {
        string normalized = string.IsNullOrWhiteSpace(title) ? string.Empty : $" {title.Trim()} ";
        int fill = Math.Max(0, _width - 2 - normalized.Length);
        int left = fill / 2;
        int right = fill - left;
        string raw = "+" + new string('=', left) + normalized + new string('=', right) + "+";
        return ConsoleStyler.Colorize(raw, ConsoleColor.Cyan);
    }

    private string Box(string content)
    {
        return ConsoleStyler.Colorize("| ", ConsoleColor.Cyan)
             + FitVisible(content, InnerWidth)
             + ConsoleStyler.Colorize(" |", ConsoleColor.Cyan);
    }

    private static string Accent(string text) => ConsoleStyler.Colorize(text, ConsoleColor.Cyan);

    private static string Value(string text) => ConsoleStyler.Colorize(text, ConsoleColor.White);

    private string PhaseChip()
    {
        var color = _phase switch
        {
            "prep" => ConsoleColor.Cyan,
            "train" => ConsoleColor.Green,
            "validate" => ConsoleColor.Yellow,
            "idle" => ConsoleColor.Magenta,
            _ => ConsoleColor.White
        };
        return ConsoleStyler.Colorize(_phase.ToUpperInvariant(), color);
    }

    private static string DeviceChip(string text)
    {
        var color = text.Contains("GPU", StringComparison.OrdinalIgnoreCase) ||
                    text.Contains("CUDA", StringComparison.OrdinalIgnoreCase)
            ? ConsoleColor.Green
            : ConsoleColor.Yellow;
        return ConsoleStyler.Colorize(text, color);
    }

    private static string CheckpointChip(string text)
    {
        var color = text switch
        {
            "Current" => ConsoleColor.Green,
            "Best" => ConsoleColor.Cyan,
            "Legacy" => ConsoleColor.Magenta,
            "None" => ConsoleColor.Yellow,
            _ => ConsoleColor.White
        };
        return ConsoleStyler.Colorize(text, color);
    }

    private static string StatusChip(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Value("idle");

        var color =
            text.Contains("best", StringComparison.OrdinalIgnoreCase) ? ConsoleColor.Green :
            text.Contains("restart", StringComparison.OrdinalIgnoreCase) ? ConsoleColor.Magenta :
            text.Contains("stop", StringComparison.OrdinalIgnoreCase) ? ConsoleColor.Red :
            text.Contains("stagn", StringComparison.OrdinalIgnoreCase) ? ConsoleColor.Yellow :
            ConsoleColor.White;
        return ConsoleStyler.Colorize(TrimRaw(text, 22), color);
    }

    private static string QualityChip(double value, bool hasValue)
    {
        if (!hasValue)
            return ConsoleStyler.Colorize("skip", ConsoleColor.Yellow);
        var color = value >= 0.45 ? ConsoleColor.Green
            : value >= 0.30 ? ConsoleColor.Yellow
            : ConsoleColor.Red;
        return ConsoleStyler.Colorize(value.ToString("F3"), color);
    }

    private static string AvgQualityChip(double value, int count)
    {
        if (count <= 0)
            return ConsoleStyler.Colorize("n/a", ConsoleColor.Yellow);
        return QualityChip(value, true);
    }

    private string GetNotice(int index)
    {
        var arr = _notices.Reverse().ToArray();
        return index < arr.Length ? arr[index] : string.Empty;
    }

    private static string TrimRaw(string value, int width)
    {
        value ??= string.Empty;
        return value.Length <= width ? value : value[..Math.Max(0, width)];
    }

    private static string FitVisible(string value, int width)
    {
        value ??= string.Empty;
        if (width <= 0)
            return string.Empty;

        var sb = new StringBuilder();
        int visible = 0;

        for (int i = 0; i < value.Length && visible < width; i++)
        {
            if (value[i] == '\u001b')
            {
                int end = i;
                while (end < value.Length && value[end] != 'm')
                    end++;
                if (end < value.Length)
                {
                    sb.Append(value, i, end - i + 1);
                    i = end;
                }
                continue;
            }

            sb.Append(value[i]);
            visible++;
        }

        if (visible < width)
            sb.Append(' ', width - visible);

        return sb.ToString();
    }
}
