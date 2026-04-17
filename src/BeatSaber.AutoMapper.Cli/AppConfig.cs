using System.Text.Json;

namespace BeatSaber.AutoMapper.Cli;

public sealed class AppConfig
{
    public string? DbPath { get; set; }
    public string? WorkPath { get; set; }
    public string? FfmpegPath { get; set; }
    public string? DefaultArtifactsPath { get; set; }
    public string Verbosity { get; set; } = "normal";

    private static readonly JsonSerializerOptions _opts = new() { WriteIndented = true };

    public static AppConfig Load(string? configPath = null)
    {
        string path = configPath ?? DefaultConfigPath();
        if (!File.Exists(path)) return new AppConfig();
        try
        {
            string json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<AppConfig>(json) ?? new AppConfig();
        }
        catch
        {
            return new AppConfig();
        }
    }

    public void Save(string configPath)
    {
        string dir = Path.GetDirectoryName(configPath) ?? ".";
        Directory.CreateDirectory(dir);
        File.WriteAllText(configPath, JsonSerializer.Serialize(this, _opts));
    }

    private static string DefaultConfigPath() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "BeatSaberAutoMapper",
            "config.json");
}
