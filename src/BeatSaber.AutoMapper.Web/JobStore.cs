using System.Collections.Concurrent;

namespace BeatSaber.AutoMapper.Web;

/// <summary>In-memory registry of completed generation jobs (GUID → zip file path).</summary>
public sealed class JobStore
{
    private readonly ConcurrentDictionary<string, string> _jobs = new();

    public string Add(string zipPath)
    {
        string id = Guid.NewGuid().ToString("N");
        _jobs[id] = zipPath;
        return id;
    }

    public bool TryGet(string id, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? zipPath)
        => _jobs.TryGetValue(id, out zipPath);

    /// <summary>
    /// Remove jobs older than <paramref name="maxAge"/> to avoid filling temp storage.
    /// Call periodically (e.g. on each new upload).
    /// </summary>
    public void Evict(TimeSpan maxAge)
    {
        var cutoff = DateTime.UtcNow - maxAge;
        foreach (var (id, path) in _jobs)
        {
            try
            {
                if (File.Exists(path) && File.GetLastWriteTimeUtc(path) < cutoff)
                {
                    File.Delete(path);
                    _jobs.TryRemove(id, out _);
                }
            }
            catch { /* best-effort */ }
        }
    }
}

/// <summary>Stats for a single generated difficulty — passed via TempData as JSON.</summary>
public sealed record DifficultyStats(
    string Difficulty,
    int    NoteCount,
    double Bpm,
    double ValidationScore,
    int    ErrorCount,
    int    WarningCount,
    int    RepairCount);
