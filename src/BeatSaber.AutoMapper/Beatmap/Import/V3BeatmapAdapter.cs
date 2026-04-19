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
        var normalizer = GridNormalizer.FromCoordinates(
            raw.ColorNotes.Select(n => (n.Lane, n.Row))
                .Concat(raw.BombNotes.Select(b => (b.Lane, b.Row)))
                .Concat(raw.Obstacles.SelectMany(o => new[]
                {
                    (o.Lane, o.Row),
                    (o.Lane + Math.Max(1, o.Width) - 1, o.Row + Math.Max(1, o.Height) - 1)
                })));

        var notes = raw.ColorNotes
            .Where(n => IsSupportedColorNote(n.Color, n.CutDirection))
            .Select(n => new CanonicalNote(
                n.Beat, normalizer.NormalizeLane(n.Lane), normalizer.NormalizeRow(n.Row),
                n.Color == 0 ? NoteColor.Red : NoteColor.Blue,
                (CutDirection)n.CutDirection
            ))
            .OrderBy(n => n.Beat)
            .ToList();

        var bombs = raw.BombNotes
            .Select(b =>
            {
                var (lane, row) = normalizer.NormalizeCell(b.Lane, b.Row);
                return new CanonicalBomb(b.Beat, lane, row);
            })
            .OrderBy(b => b.Beat)
            .ToList();

        var obstacles = raw.Obstacles.Select(o =>
        {
            var (lane, width) = normalizer.NormalizeLaneSpan(o.Lane, o.Width);
            int row = normalizer.NormalizeRow(o.Row);
            return new CanonicalObstacle(
                o.Beat, lane, o.Duration, width, o.Height, row
            );
        })
        .OrderBy(o => o.Beat)
        .ToList();

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

    private static bool IsSupportedColorNote(int color, int cutDirection) =>
        (color == 0 || color == 1) && cutDirection is >= 0 and <= 8;
}
