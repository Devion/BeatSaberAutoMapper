namespace BeatSaber.AutoMapper.Training.Features;

public sealed record DatasetManifest(
    string Name,
    string Version,
    IReadOnlyList<string> TrainFiles,
    IReadOnlyList<string> ValidationFiles,
    IReadOnlyList<string> TestFiles,
    int TotalExamples
);

public sealed class DatasetManifestBuilder
{
    private static readonly JsonSerializerOptions _opts = new() { WriteIndented = true };

    public DatasetManifest Build(
        IReadOnlyList<string> exampleFiles,
        double trainFrac,
        double valFrac,
        double testFrac,
        long seed = 42)
    {
        Guard.NotNull(exampleFiles, nameof(exampleFiles));
        var rng = new Random((int)seed);
        var shuffled = exampleFiles.OrderBy(_ => rng.Next()).ToList();

        int total = shuffled.Count;
        int trainN = (int)(total * trainFrac);
        int valN = (int)(total * valFrac);

        return new DatasetManifest(
            Name: "dataset",
            Version: "1.0",
            TrainFiles: shuffled[..trainN],
            ValidationFiles: shuffled[trainN..(trainN + valN)],
            TestFiles: shuffled[(trainN + valN)..],
            TotalExamples: total
        );
    }

    public void SaveManifest(DatasetManifest manifest, string outputPath)
    {
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));
        string json = JsonSerializer.Serialize(manifest, _opts);
        File.WriteAllText(outputPath, json, Encoding.UTF8);
    }

    public DatasetManifest LoadManifest(string path)
    {
        Guard.NotNullOrEmpty(path, nameof(path));
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<DatasetManifest>(json)
               ?? throw new InvalidDataException("Failed to deserialise DatasetManifest.");
    }
}
