using BeatSaber.AutoMapper.Beatmap.Raw;
using BeatSaber.AutoMapper.Utilities;

namespace BeatSaber.AutoMapper.Beatmap.Import;

internal sealed class V3BeatmapAdapter
{
    public static CanonicalBeatmap Convert(
        V3Beatmap raw,
        SongMetadata meta,
        DifficultyDescriptor difficulty)
    {
        var notes = raw.ColorNotes.Select(n => new CanonicalNote(
            n.Beat, n.Lane, n.Row,
            n.Color == 0 ? NoteColor.Red : NoteColor.Blue,
            (CutDirection)n.CutDirection
        )).OrderBy(n => n.Beat).ToList();

        var bombs = raw.BombNotes.Select(b => new CanonicalBomb(b.Beat, b.Lane, b.Row))
                       .OrderBy(b => b.Beat).ToList();

        var obstacles = raw.Obstacles.Select(o => new CanonicalObstacle(
            o.Beat, o.Lane, o.Duration, o.Width, o.Height, o.Row
        )).OrderBy(o => o.Beat).ToList();

        // Build timing points from BPM events
        var timingPoints = new List<BeatTimingPoint>();
        double baseBpm = meta.BeatsPerMinute;

        if (raw.BpmEvents.Count == 0)
        {
            timingPoints.Add(new BeatTimingPoint(0.0, 0.0, baseBpm));
        }
        else
        {
            double currentBpm = baseBpm;
            double currentBeat = 0.0;
            double currentTime = 0.0;
            timingPoints.Add(new BeatTimingPoint(0.0, 0.0, baseBpm));

            foreach (var ev in raw.BpmEvents.OrderBy(e => e.Beat))
            {
                double deltaBeat = ev.Beat - currentBeat;
                double deltaTime = MathHelpers.BeatToSeconds(deltaBeat, currentBpm);
                currentTime += deltaTime;
                currentBeat = ev.Beat;
                currentBpm = ev.Bpm;
                timingPoints.Add(new BeatTimingPoint(currentBeat, currentTime, currentBpm));
            }
        }

        return new CanonicalBeatmap
        {
            Song = meta,
            Difficulty = difficulty,
            TimingPoints = timingPoints,
            Notes = notes,
            Bombs = bombs,
            Obstacles = obstacles,
            Sections = []
        };
    }
}
