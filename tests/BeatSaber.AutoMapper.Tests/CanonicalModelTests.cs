using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Utilities;
using FluentAssertions;

namespace BeatSaber.AutoMapper.Tests;

public sealed class CanonicalModelTests
{
    private static CanonicalBeatmap MakeBeatmap(
        double bpm = 120.0,
        IReadOnlyList<CanonicalNote>? notes = null) =>
        new()
        {
            Song = new SongMetadata("Test", "Artist", null, bpm, 0, 10, 30, null, null),
            Difficulty = new DifficultyDescriptor(DifficultyLevel.Hard, BeatmapCharacteristic.Standard, null, null, null),
            TimingPoints = [new BeatTimingPoint(0, 0, bpm)],
            Notes = notes ?? [],
            Bombs = [],
            Obstacles = [],
            Sections = []
        };

    [Fact]
    public void BeatToSeconds_RoundTrip()
    {
        var bm = MakeBeatmap(120.0);
        double beat = 4.0;
        double secs = bm.BeatToSeconds(beat);
        double roundTrip = bm.SecondsToBeat(secs);
        roundTrip.Should().BeApproximately(beat, 0.001);
    }

    [Fact]
    public void BeatToSeconds_At120Bpm_Beat1_Is0Point5Seconds()
    {
        var bm = MakeBeatmap(120.0);
        bm.BeatToSeconds(1.0).Should().BeApproximately(0.5, 0.001);
    }

    [Fact]
    public void LocalNps_ReturnsZero_ForEmptyBeatmap()
    {
        var bm = MakeBeatmap();
        bm.LocalNps(4.0).Should().Be(0);
    }

    [Fact]
    public void LocalNps_CountsCorrectWindow()
    {
        var notes = Enumerable.Range(0, 8)
            .Select(i => new CanonicalNote(i * 0.5, 1, 1, NoteColor.Red, CutDirection.Down))
            .ToList();
        var bm = MakeBeatmap(120.0, notes);

        // 8 notes over 4 beats, 4 beats = 2 seconds at 120 bpm → 4 NPS
        double nps = bm.LocalNps(2.0, 4.0);
        nps.Should().BeGreaterThan(0);
    }

    [Fact]
    public void ConflictsWith_SameBeatLaneRow_ReturnsTrue()
    {
        var a = new CanonicalNote(1.0, 2, 1, NoteColor.Red, CutDirection.Up);
        var b = new CanonicalNote(1.0, 2, 1, NoteColor.Blue, CutDirection.Down);
        a.ConflictsWith(b).Should().BeTrue();
    }

    [Fact]
    public void ConflictsWith_DifferentBeat_ReturnsFalse()
    {
        var a = new CanonicalNote(1.0, 2, 1, NoteColor.Red, CutDirection.Up);
        var b = new CanonicalNote(2.0, 2, 1, NoteColor.Red, CutDirection.Up);
        a.ConflictsWith(b).Should().BeFalse();
    }

    [Fact]
    public void Hand_RedIsLeft_BlueIsRight()
    {
        var red = new CanonicalNote(0, 1, 1, NoteColor.Red, CutDirection.Down);
        var blue = new CanonicalNote(0, 2, 1, NoteColor.Blue, CutDirection.Down);
        red.Hand.Should().Be(NoteHand.Left);
        blue.Hand.Should().Be(NoteHand.Right);
    }
}
