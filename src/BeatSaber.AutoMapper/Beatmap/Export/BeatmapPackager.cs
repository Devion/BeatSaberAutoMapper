using System.IO.Compression;

namespace BeatSaber.AutoMapper.Beatmap.Export;

public sealed class BeatmapPackager
{
    /// <summary>Packages a Beat Saber map folder into a .zip archive.</summary>
    public static void Pack(string mapFolderPath, string outputZipPath)
    {
        Guard.NotNullOrEmpty(mapFolderPath, nameof(mapFolderPath));
        Guard.NotNullOrEmpty(outputZipPath, nameof(outputZipPath));

        if (File.Exists(outputZipPath)) File.Delete(outputZipPath);
        ZipFile.CreateFromDirectory(mapFolderPath, outputZipPath, CompressionLevel.Optimal, false);
    }

    /// <summary>Unpacks a Beat Saber map .zip archive to a folder.</summary>
    public static void Unpack(string zipPath, string outputFolderPath)
    {
        Guard.NotNullOrEmpty(zipPath, nameof(zipPath));
        Guard.NotNullOrEmpty(outputFolderPath, nameof(outputFolderPath));

        Directory.CreateDirectory(outputFolderPath);
        ZipFile.ExtractToDirectory(zipPath, outputFolderPath, overwriteFiles: true);
    }
}
