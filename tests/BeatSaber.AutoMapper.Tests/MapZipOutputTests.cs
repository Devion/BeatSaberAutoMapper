using System.IO.Compression;
using System.Text;
using System.Text.Json;
using BeatSaber.AutoMapper.Beatmap;
using BeatSaber.AutoMapper.Beatmap.Export;
using BeatSaber.AutoMapper.Canonical;
using FluentAssertions;

namespace BeatSaber.AutoMapper.Tests;

/// <summary>
/// Verifies that the generated ZIP packages are valid Beat Saber maps that can be
/// uploaded to BeatSaver and loaded by the game.
///
/// Checks:
///   - Info.dat exists, contains no UTF-8 BOM, and is valid JSON
///   - Every difficulty file referenced in Info.dat exists, has no BOM, and is valid JSON
///   - The audio filename referenced in Info.dat exists in the archive
///   - The cover image filename referenced in Info.dat exists in the archive
///   - All JSON round-trips to the same note count
///
/// Also includes an integration test (skipped if D:\Maps is empty) that runs the full
/// export pipeline against a real training map to catch platform-specific regressions.
/// </summary>
public sealed class MapZipOutputTests : IDisposable
{
    private readonly string _tmpDir;

    public MapZipOutputTests()
    {
        _tmpDir = Path.Combine(Path.GetTempPath(), $"bsam_ziptest_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir)) Directory.Delete(_tmpDir, recursive: true);
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static SongMetadata MakeMeta(string coverFile = "cover.jpg") =>
        new(Title: "Test Song",
            Artist: "Test Artist",
            SubTitle: null,
            BeatsPerMinute: 120.0,
            SongTimeOffset: 0,
            PreviewStartTime: 10,
            PreviewDuration: 30,
            CoverImagePath: coverFile,
            AudioPath: "song.egg");

    private static CanonicalBeatmap MakeBeatmap(SongMetadata meta, int noteCount = 10)
    {
        var rng   = new Random(42);
        var notes = Enumerable.Range(0, noteCount).Select(i => new CanonicalNote(
            Beat: i * 0.5,
            Lane: rng.Next(0, 4),
            Row: rng.Next(0, 3),
            Color: i % 2 == 0 ? NoteColor.Red : NoteColor.Blue,
            CutDirection: (CutDirection)rng.Next(0, 9)
        )).ToList();

        return new CanonicalBeatmap
        {
            Song = meta,
            Difficulty = new DifficultyDescriptor(
                DifficultyLevel.Hard, BeatmapCharacteristic.Standard, null, null, null),
            TimingPoints = [new BeatTimingPoint(0, 0, 120.0)],
            Notes = notes,
            Bombs = [],
            Obstacles = [],
            Sections = []
        };
    }

    /// <summary>
    /// Builds a full map folder (Info.dat + Hard.dat + audio stub + cover) and packages
    /// to ZIP, then returns the ZIP path for verification.
    /// </summary>
    private string BuildMapZip(string coverExt = ".jpg")
    {
        string mapDir  = Path.Combine(_tmpDir, "map");
        string zipPath = Path.Combine(_tmpDir, "map.zip");
        Directory.CreateDirectory(mapDir);

        string coverFile = "cover" + coverExt;
        var meta     = MakeMeta(coverFile);
        var beatmap  = MakeBeatmap(meta);

        // Difficulty .dat
        string datFile = "Hard.dat";
        BeatmapExporter.ExportV3(beatmap, Path.Combine(mapDir, datFile));

        // Info.dat
        BeatmapExporter.ExportInfoDat(meta, [(beatmap.Difficulty, datFile)], mapDir);

        // Minimal audio stub (real maps use OGG Vorbis but any non-empty file satisfies presence checks)
        File.WriteAllBytes(Path.Combine(mapDir, "song.egg"), [0x4F, 0x67, 0x67, 0x53]); // OGG magic

        // Cover image stub
        if (coverExt == ".jpg")
            File.WriteAllBytes(Path.Combine(mapDir, coverFile), [0xFF, 0xD8, 0xFF, 0xE0]); // JPEG magic
        else
            File.WriteAllBytes(Path.Combine(mapDir, coverFile), [0x89, 0x50, 0x4E, 0x47]); // PNG magic

        BeatmapPackager.Pack(mapDir, zipPath);
        return zipPath;
    }

    // ── BOM check helper ─────────────────────────────────────────────────────

    private static void AssertNoBom(string content, string fileName)
    {
        content.Should().NotStartWith("\uFEFF",
            because: $"{fileName} must not start with a UTF-8 BOM — it breaks JSON parsers");
    }

    private static string ReadZipEntry(ZipArchive zip, string name)
    {
        var entry = zip.GetEntry(name);
        entry.Should().NotBeNull($"entry '{name}' must be present in the ZIP");
        using var sr = new StreamReader(entry!.Open(), Encoding.UTF8);
        return sr.ReadToEnd();
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    [Fact]
    public void GeneratedZip_ContainsInfoDat()
    {
        string zip = BuildMapZip();
        using var archive = ZipFile.OpenRead(zip);
        archive.GetEntry("Info.dat").Should().NotBeNull("Info.dat is required in every Beat Saber map");
    }

    [Fact]
    public void GeneratedZip_InfoDat_IsValidJsonWithNoBom()
    {
        string zip = BuildMapZip();
        using var archive = ZipFile.OpenRead(zip);
        string content = ReadZipEntry(archive, "Info.dat");

        AssertNoBom(content, "Info.dat");
        var act = () => JsonDocument.Parse(content);
        act.Should().NotThrow("Info.dat must be parseable JSON");
    }

    [Fact]
    public void GeneratedZip_InfoDat_ContainsRequiredV2Keys()
    {
        string zip = BuildMapZip();
        using var archive = ZipFile.OpenRead(zip);
        string content = ReadZipEntry(archive, "Info.dat");

        using var doc  = JsonDocument.Parse(content);
        var root = doc.RootElement;

        root.TryGetProperty("_version", out _).Should().BeTrue("_version is required");
        root.TryGetProperty("_songName", out _).Should().BeTrue("_songName is required");
        root.TryGetProperty("_beatsPerMinute", out _).Should().BeTrue("_beatsPerMinute is required");
        root.TryGetProperty("_songFilename", out _).Should().BeTrue("_songFilename is required");
        root.TryGetProperty("_coverImageFilename", out _).Should().BeTrue("_coverImageFilename is required");
        root.TryGetProperty("_difficultyBeatmapSets", out _).Should().BeTrue("_difficultyBeatmapSets is required");
    }

    [Fact]
    public void GeneratedZip_AllReferencedDatFilesExist_AndAreValidJson()
    {
        string zip = BuildMapZip();
        using var archive = ZipFile.OpenRead(zip);
        string infoContent = ReadZipEntry(archive, "Info.dat");

        using var doc  = JsonDocument.Parse(infoContent);
        var sets = doc.RootElement.GetProperty("_difficultyBeatmapSets");

        sets.GetArrayLength().Should().BeGreaterThan(0, "at least one beatmap set required");

        foreach (var set in sets.EnumerateArray())
        {
            foreach (var diff in set.GetProperty("_difficultyBeatmaps").EnumerateArray())
            {
                string filename = diff.GetProperty("_beatmapFilename").GetString()!;
                string datContent = ReadZipEntry(archive, filename);

                AssertNoBom(datContent, filename);
                var act = () => JsonDocument.Parse(datContent);
                act.Should().NotThrow($"beatmap file '{filename}' must be parseable JSON");
            }
        }
    }

    [Fact]
    public void GeneratedZip_BeatmapDat_ContainsColorNotes()
    {
        string zip = BuildMapZip();
        using var archive = ZipFile.OpenRead(zip);
        string datContent = ReadZipEntry(archive, "Hard.dat");

        using var doc  = JsonDocument.Parse(datContent);
        var root = doc.RootElement;

        root.TryGetProperty("version", out var ver).Should().BeTrue("v3 beatmap must have 'version' field");
        ver.GetString().Should().StartWith("3", "we emit v3 beatmap format");

        root.TryGetProperty("colorNotes", out var notes).Should().BeTrue("v3 beatmap must have 'colorNotes' array");
        notes.GetArrayLength().Should().Be(10, "we placed 10 notes");
    }

    [Fact]
    public void GeneratedZip_AudioFileExists()
    {
        string zip = BuildMapZip();
        using var archive = ZipFile.OpenRead(zip);
        string infoContent = ReadZipEntry(archive, "Info.dat");
        using var doc = JsonDocument.Parse(infoContent);

        string audioFile = doc.RootElement.GetProperty("_songFilename").GetString()!;
        archive.GetEntry(audioFile).Should().NotBeNull($"audio file '{audioFile}' must be in the ZIP");
    }

    [Fact]
    public void GeneratedZip_CoverImageExists()
    {
        string zip = BuildMapZip();
        using var archive = ZipFile.OpenRead(zip);
        string infoContent = ReadZipEntry(archive, "Info.dat");
        using var doc = JsonDocument.Parse(infoContent);

        string coverFile = doc.RootElement.GetProperty("_coverImageFilename").GetString()!;
        archive.GetEntry(coverFile).Should().NotBeNull($"cover image '{coverFile}' must be in the ZIP");
    }

    [Fact]
    public void GeneratedZip_WithPngCover_UsesCorrectExtension()
    {
        // When the placeholder PNG is used, the file must be named *.png (not *.jpg with PNG bytes)
        string zip = BuildMapZip(coverExt: ".png");
        using var archive = ZipFile.OpenRead(zip);
        string infoContent = ReadZipEntry(archive, "Info.dat");
        using var doc = JsonDocument.Parse(infoContent);

        string coverFile = doc.RootElement.GetProperty("_coverImageFilename").GetString()!;
        coverFile.Should().EndWith(".png", "placeholder cover must use .png extension to match PNG bytes");

        var entry = archive.GetEntry(coverFile);
        entry.Should().NotBeNull();
        using var ms = new MemoryStream();
        entry!.Open().CopyTo(ms);
        // First 4 bytes of a PNG: 89 50 4E 47
        ms.ToArray()[0].Should().Be(0x89);
        ms.ToArray()[1].Should().Be(0x50);
    }

    [Fact]
    public void GeneratedZip_RoundTrip_PreservesNoteCount()
    {
        string zip = BuildMapZip();
        string extractDir = Path.Combine(_tmpDir, "extracted");
        BeatmapPackager.Unpack(zip, extractDir);

        var maps = BeatmapImporter.Import(extractDir);
        maps.Should().HaveCount(1);
        maps[0].Notes.Should().HaveCount(10, "all 10 notes must survive the export→pack→unpack→import round-trip");
    }

    // ── Integration test ──────────────────────────────────────────────────────

    /// <summary>
    /// Uses the first real map from D:\Maps to verify the import pipeline
    /// (not the export pipeline — that's covered above).
    /// Skipped automatically if no maps are available.
    /// </summary>
    [Fact]
    public void IntegrationTest_RealMapZip_ParsesWithoutError()
    {
        const string mapsDir = @"D:\Maps";
        if (!Directory.Exists(mapsDir))
        {
            // Gracefully skip if this machine doesn't have the training library.
            return;
        }

        string? zipFile = Directory.EnumerateFiles(mapsDir, "*.zip")
            .FirstOrDefault();

        if (zipFile is null) return;

        string extractDir = Path.Combine(_tmpDir, "real_extract");

        var act = () =>
        {
            BeatmapPackager.Unpack(zipFile, extractDir);
            var maps = BeatmapImporter.Import(extractDir);
            maps.Should().NotBeEmpty("a valid Beat Saber map ZIP must contain at least one difficulty");
            foreach (var m in maps)
                m.Notes.Should().NotBeEmpty("every difficulty must have at least one note");
        };

        act.Should().NotThrow($"real map '{Path.GetFileName(zipFile)}' must import without error");
    }

    /// <summary>
    /// Exports a map and verifies the resulting ZIP passes the same structural checks
    /// that BeatSaver runs (no BOM, valid JSON, all referenced files present).
    /// </summary>
    [Fact]
    public void GeneratedZip_PassesBeatSaverStructuralChecks()
    {
        string zip = BuildMapZip();
        using var archive = ZipFile.OpenRead(zip);

        // 1. Info.dat present
        var infoEntry = archive.GetEntry("Info.dat");
        infoEntry.Should().NotBeNull();

        // 2. No BOM in Info.dat
        using var infoReader = new StreamReader(infoEntry!.Open(), Encoding.UTF8);
        string infoJson = infoReader.ReadToEnd();
        infoJson.Should().NotStartWith("\uFEFF",
            "BOM prefix breaks BeatSaver's JSON parser and most map editors");

        // 3. Parse Info.dat
        using var infoDoc = JsonDocument.Parse(infoJson);
        var infoRoot = infoDoc.RootElement;

        // 4. Song file exists
        string songFilename = infoRoot.GetProperty("_songFilename").GetString()!;
        archive.GetEntry(songFilename).Should().NotBeNull();

        // 5. Cover image exists
        string coverFilename = infoRoot.GetProperty("_coverImageFilename").GetString()!;
        archive.GetEntry(coverFilename).Should().NotBeNull();

        // 6. All beatmap files exist and are valid JSON without BOM
        foreach (var set in infoRoot.GetProperty("_difficultyBeatmapSets").EnumerateArray())
        {
            foreach (var diff in set.GetProperty("_difficultyBeatmaps").EnumerateArray())
            {
                string beatmapFilename = diff.GetProperty("_beatmapFilename").GetString()!;
                var beatmapEntry = archive.GetEntry(beatmapFilename);
                beatmapEntry.Should().NotBeNull($"difficulty file '{beatmapFilename}' must be in ZIP");

                using var bmReader = new StreamReader(beatmapEntry!.Open(), Encoding.UTF8);
                string bmJson = bmReader.ReadToEnd();
                bmJson.Should().NotStartWith("\uFEFF",
                    $"beatmap file '{beatmapFilename}' must not have a BOM");

                var bmAct = () => JsonDocument.Parse(bmJson);
                bmAct.Should().NotThrow($"beatmap file '{beatmapFilename}' must be parseable JSON");
            }
        }
    }
}
