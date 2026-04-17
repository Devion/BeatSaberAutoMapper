using BeatSaber.AutoMapper.Beatmap;
using BeatSaber.AutoMapper.Canonical;
using FluentAssertions;

namespace BeatSaber.AutoMapper.Tests;

/// <summary>Tests that v1-format maps (info.json + flat difficultyLevels) can be imported.</summary>
public sealed class BeatmapV1ImportTests
{
    private const string V1InfoJson = """
        {
          "songName": "Test Song",
          "songSubName": "Sub",
          "authorName": "TestAuthor",
          "beatsPerMinute": 120,
          "previewStartTime": 10,
          "previewDuration": 5,
          "coverImagePath": "cover.jpg",
          "environmentName": "DefaultEnvironment",
          "difficultyLevels": [
            {
              "difficulty": "Expert",
              "difficultyRank": 3,
              "audioPath": "song.ogg",
              "jsonPath": "Expert.json",
              "offset": 0,
              "noteJumpSpeed": 14
            }
          ]
        }
        """;

    private const string V1DifficultyJson = """
        {
          "_version": "1.5.0",
          "_beatsPerMinute": 120,
          "_beatsPerBar": 16,
          "_noteJumpSpeed": 14,
          "_shuffle": 0,
          "_shufflePeriod": 0.5,
          "_events": [],
          "_notes": [
            { "_time": 1.0, "_lineIndex": 0, "_lineLayer": 1, "_type": 0, "_cutDirection": 1 },
            { "_time": 1.0, "_lineIndex": 3, "_lineLayer": 1, "_type": 1, "_cutDirection": 1 },
            { "_time": 2.0, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 0 },
            { "_time": 3.0, "_lineIndex": 2, "_lineLayer": 0, "_type": 3, "_cutDirection": 0 }
          ],
          "_obstacles": [
            { "_time": 4.0, "_lineIndex": 0, "_type": 0, "_duration": 2.0, "_width": 2 }
          ]
        }
        """;

    [Fact]
    public void V1Info_Parsed_Correctly()
    {
        using var tmp = new TempFolder();
        File.WriteAllText(Path.Combine(tmp.Path, "info.json"),    V1InfoJson);
        File.WriteAllText(Path.Combine(tmp.Path, "Expert.json"), V1DifficultyJson);

        var maps = BeatmapImporter.Import(tmp.Path);

        maps.Should().HaveCount(1);
        var map = maps[0];

        map.Song.Title.Should().Be("Test Song");
        map.Song.Artist.Should().Be("TestAuthor");
        map.Song.BeatsPerMinute.Should().Be(120);
        map.Difficulty.Difficulty.Should().Be(DifficultyLevel.Expert);
        map.Difficulty.Characteristic.Should().Be(BeatmapCharacteristic.Standard);
    }

    [Fact]
    public void V1Notes_Imported_With_Correct_Count_And_Types()
    {
        using var tmp = new TempFolder();
        File.WriteAllText(Path.Combine(tmp.Path, "info.json"),    V1InfoJson);
        File.WriteAllText(Path.Combine(tmp.Path, "Expert.json"), V1DifficultyJson);

        var map = BeatmapImporter.Import(tmp.Path)[0];

        // 3 notes (type 0 and 1), 1 bomb (type 3)
        map.Notes.Should().HaveCount(3);
        map.Bombs.Should().HaveCount(1);
    }

    [Fact]
    public void V1Notes_Colors_Correct()
    {
        using var tmp = new TempFolder();
        File.WriteAllText(Path.Combine(tmp.Path, "info.json"),    V1InfoJson);
        File.WriteAllText(Path.Combine(tmp.Path, "Expert.json"), V1DifficultyJson);

        var notes = BeatmapImporter.Import(tmp.Path)[0].Notes;

        notes.Should().ContainSingle(n => n.Beat == 1.0 && n.Color == NoteColor.Red);
        notes.Should().ContainSingle(n => n.Beat == 1.0 && n.Color == NoteColor.Blue);
        notes.Should().ContainSingle(n => n.Beat == 2.0 && n.Color == NoteColor.Red);
    }

    [Fact]
    public void V1Obstacles_Imported()
    {
        using var tmp = new TempFolder();
        File.WriteAllText(Path.Combine(tmp.Path, "info.json"),    V1InfoJson);
        File.WriteAllText(Path.Combine(tmp.Path, "Expert.json"), V1DifficultyJson);

        var map = BeatmapImporter.Import(tmp.Path)[0];

        map.Obstacles.Should().HaveCount(1);
        map.Obstacles[0].Beat.Should().Be(4.0);
        map.Obstacles[0].Duration.Should().Be(2.0);
    }

    [Fact]
    public void V1_Multiple_Difficulties()
    {
        const string infoWithTwo = """
            {
              "songName": "Two Diffs",
              "songSubName": "",
              "authorName": "Auth",
              "beatsPerMinute": 100,
              "previewStartTime": 0,
              "previewDuration": 10,
              "coverImagePath": "cover.jpg",
              "environmentName": "DefaultEnvironment",
              "difficultyLevels": [
                { "difficulty": "Hard",       "difficultyRank": 2, "audioPath": "song.ogg", "jsonPath": "Hard.json",   "offset": 0, "noteJumpSpeed": 12 },
                { "difficulty": "ExpertPlus", "difficultyRank": 4, "audioPath": "song.ogg", "jsonPath": "Expert.json", "offset": 0, "noteJumpSpeed": 18 }
              ]
            }
            """;

        const string emptyDiff = """
            {"_version":"1.5.0","_beatsPerMinute":100,"_beatsPerBar":16,
             "_noteJumpSpeed":12,"_shuffle":0,"_shufflePeriod":0.5,
             "_events":[],"_notes":[],"_obstacles":[]}
            """;

        using var tmp = new TempFolder();
        File.WriteAllText(Path.Combine(tmp.Path, "info.json"),    infoWithTwo);
        File.WriteAllText(Path.Combine(tmp.Path, "Hard.json"),   emptyDiff);
        File.WriteAllText(Path.Combine(tmp.Path, "Expert.json"), emptyDiff);

        var maps = BeatmapImporter.Import(tmp.Path);
        maps.Should().HaveCount(2);
        maps.Select(m => m.Difficulty.Difficulty).Should()
            .Contain(DifficultyLevel.Hard).And.Contain(DifficultyLevel.ExpertPlus);
    }
}

/// <summary>Simple RAII temp folder for tests.</summary>
internal sealed class TempFolder : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"bsam_{Guid.NewGuid():N}");
    public TempFolder() => Directory.CreateDirectory(Path);
    public void Dispose() { if (Directory.Exists(Path)) Directory.Delete(Path, recursive: true); }
}
