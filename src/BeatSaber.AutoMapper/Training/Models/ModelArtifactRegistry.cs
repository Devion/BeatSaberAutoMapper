namespace BeatSaber.AutoMapper.Training.Models;

public sealed class ModelArtifactRegistry
{
    private static readonly JsonSerializerOptions _opts = new() { WriteIndented = true };

    private sealed record RegistryEntry(
        string ArtifactType,
        string Version,
        string FilePath,
        string RegisteredAt
    );

    public void Register(string artifactsPath, string artifactType, string version, string filePath)
    {
        Guard.NotNullOrEmpty(artifactsPath, nameof(artifactsPath));
        Guard.NotNullOrEmpty(artifactType, nameof(artifactType));
        Guard.NotNullOrEmpty(version, nameof(version));
        Guard.NotNullOrEmpty(filePath, nameof(filePath));

        var entries = LoadRegistry(artifactsPath);
        entries.Add(new RegistryEntry(artifactType, version, filePath, DateTime.UtcNow.ToString("O")));
        SaveRegistry(artifactsPath, entries);
    }

    public string? GetLatest(string artifactsPath, string artifactType)
    {
        var entries = LoadRegistry(artifactsPath);
        return entries
            .Where(e => e.ArtifactType == artifactType)
            .OrderByDescending(e => e.RegisteredAt)
            .FirstOrDefault()
            ?.FilePath;
    }

    public IReadOnlyList<string> List(string artifactsPath)
    {
        var entries = LoadRegistry(artifactsPath);
        return entries.Select(e => $"{e.ArtifactType} v{e.Version}: {e.FilePath}").ToList();
    }

    private static List<RegistryEntry> LoadRegistry(string artifactsPath)
    {
        string path = RegistryPath(artifactsPath);
        if (!File.Exists(path)) return [];
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<List<RegistryEntry>>(json) ?? [];
    }

    private static void SaveRegistry(string artifactsPath, List<RegistryEntry> entries)
    {
        Directory.CreateDirectory(artifactsPath);
        File.WriteAllText(RegistryPath(artifactsPath), JsonSerializer.Serialize(entries, _opts));
    }

    private static string RegistryPath(string artifactsPath) =>
        Path.Combine(artifactsPath, "registry.json");
}
