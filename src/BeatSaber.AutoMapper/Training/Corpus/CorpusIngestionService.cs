using BeatSaber.AutoMapper.Beatmap;
using BeatSaber.AutoMapper.Beatmap.Export;

namespace BeatSaber.AutoMapper.Training.Corpus;

public sealed class CorpusIngestionService
{
    public sealed record IngestionSummary(
        int Imported,
        int Skipped,
        int Malformed,
        int Duplicate,
        string? ReportPath
    );

    public IngestionSummary IngestFolder(
        string inputFolder,
        string libraryPath,
        string dbPath,
        bool recurse = true,
        bool copyAudio = false)
    {
        Guard.NotNullOrEmpty(inputFolder, nameof(inputFolder));
        Guard.NotNullOrEmpty(libraryPath, nameof(libraryPath));
        Directory.CreateDirectory(libraryPath);

        var option = recurse ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var archives = Directory.GetFiles(inputFolder, "*.zip", option)
            .Concat(Directory.GetFiles(inputFolder, "*.bsmap", option))
            .ToList();

        int imported = 0, skipped = 0, malformed = 0, duplicate = 0;
        var seen = new HashSet<string>();

        foreach (var archive in archives)
        {
            var (success, error) = TryIngestArchive(archive, libraryPath);
            if (success) imported++;
            else if (error == "duplicate") duplicate++;
            else if (error == "malformed") malformed++;
            else skipped++;
        }

        return new IngestionSummary(imported, skipped, malformed, duplicate, null);
    }

    private (bool success, string? error) TryIngestArchive(string archivePath, string libraryPath)
    {
        try
        {
            string name = Path.GetFileNameWithoutExtension(archivePath);
            string dest = Path.Combine(libraryPath, name);
            if (Directory.Exists(dest)) return (false, "duplicate");

            BeatmapPackager.Unpack(archivePath, dest);

            // Quick validation: check that Info.dat exists
            if (!File.Exists(Path.Combine(dest, "Info.dat")) &&
                !File.Exists(Path.Combine(dest, "info.dat")))
            {
                Directory.Delete(dest, recursive: true);
                return (false, "malformed");
            }

            return (true, null);
        }
        catch
        {
            return (false, "malformed");
        }
    }
}
