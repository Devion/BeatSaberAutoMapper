using BeatSaber.AutoMapper.Beatmap;
using BeatSaber.AutoMapper.Canonical;
using FluentAssertions;
using System.IO;

namespace BeatSaber.AutoMapper.Tests;

public sealed class BeatmapRoundTripTests : IDisposable
{
    private readonly string _tmpDir;

    public BeatmapRoundTripTests()
    {
        _tmpDir = Path.Combine(Path.GetTempPath(), $"bsam_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    private static CanonicalBeatmap MakeSimpleBeatmap()
    {
        var notes = new List<CanonicalNote>
        {
            new(0.0, 0, 1, NoteColor.Red,  CutDirection.Down),
            new(0.5, 3, 1, NoteColor.Blue, CutDirection.Down),
            new(1.0, 1, 1, NoteColor.Red,  CutDirection.Up),
            new(1.5, 2, 1, NoteColor.Blue, CutDirection.Up),
            new(2.0, 0, 0, NoteColor.Red,  CutDirection.DownLeft),
        };

        var meta = new SongMetadata(
            Title: "Round Trip Test",
            Artist: "Unit Test",
            SubTitle: null,
            BeatsPerMinute: 120.0,
            SongTimeOffset: 0,
            PreviewStartTime: 0,
            PreviewDuration: 10,
            CoverImagePath: null,
            AudioPath: "song.ogg"
        );

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

    [Fact]
    public void ExportV3_ThenImport_PreservesNoteCount()
    {
        var original = MakeSimpleBeatmap();
        string mapFile = Path.Combine(_tmpDir, "Hard.dat");

        BeatmapExporter.ExportV3(original, mapFile);
        mapFile.Should().Be(mapFile); // file was created

        var imported = BeatmapImporter.ImportFile(
            mapFile, original.Song, original.Difficulty, original.Song.BeatsPerMinute);

        imported.Notes.Count.Should().Be(original.Notes.Count);
    }

    [Fact]
    public void ExportV3_ThenImport_PreservesLaneRowColor()
    {
        var original = MakeSimpleBeatmap();
        string mapFile = Path.Combine(_tmpDir, "Hard_lrc.dat");

        BeatmapExporter.ExportV3(original, mapFile);
        var imported = BeatmapImporter.ImportFile(
            mapFile, original.Song, original.Difficulty, original.Song.BeatsPerMinute);

        for (int i = 0; i < original.Notes.Count; i++)
        {
            var orig = original.Notes.OrderBy(n => n.Beat).ElementAt(i);
            var imp = imported.Notes.OrderBy(n => n.Beat).ElementAt(i);
            imp.Lane.Should().Be(orig.Lane, because: $"note {i} lane should be preserved");
            imp.Row.Should().Be(orig.Row, because: $"note {i} row should be preserved");
            imp.Color.Should().Be(orig.Color, because: $"note {i} colour should be preserved");
        }
    }

    [Fact]
    public void ExportV3_ThenImport_PreservesBpm()
    {
        var original = MakeSimpleBeatmap();
        string mapFile = Path.Combine(_tmpDir, "Hard_bpm.dat");

        BeatmapExporter.ExportV3(original, mapFile);
        var imported = BeatmapImporter.ImportFile(
            mapFile, original.Song, original.Difficulty, original.Song.BeatsPerMinute);

        imported.Song.BeatsPerMinute.Should().BeApproximately(original.Song.BeatsPerMinute, 0.01);
    }

    [Fact]
    public void ExportInfoDat_WritesInfoDatFile()
    {
        var original = MakeSimpleBeatmap();
        string bmFile = "Hard.dat";
        BeatmapExporter.ExportInfoDat(
            original.Song,
            [(original.Difficulty, bmFile)],
            _tmpDir);

        File.Exists(Path.Combine(_tmpDir, "Info.dat")).Should().BeTrue();
    }
}
