using BeatSaber.AutoMapper.Beatmap.Raw;
using BeatSaber.AutoMapper.Utilities;

namespace BeatSaber.AutoMapper.Beatmap.Import;

internal sealed class V2BeatmapAdapter
{
    public static CanonicalBeatmap Convert(
        V2Beatmap raw,
        SongMetadata meta,
        DifficultyDescriptor difficulty,
        double bpm)
    {
        var normalizer = GridNormalizer.FromCoordinates(
            raw.Notes.Select(n => (n.LineIndex, n.LineLayer))
                .Concat(raw.Obstacles.SelectMany(o =>
                {
                    int startRow = o.Type == 0 ? 0 : 2;
                    return new[] { (o.LineIndex, startRow), (o.LineIndex + Math.Max(1, o.Width) - 1, startRow) };
                })));

        var notes = new List<CanonicalNote>();
        var bombs = new List<CanonicalBomb>();

        foreach (var n in raw.Notes)
        {
            if (n.Type == 3) // bomb
            {
                var (lane, row) = normalizer.NormalizeCell(n.LineIndex, n.LineLayer);
                bombs.Add(new CanonicalBomb(n.Time, lane, row));
            }
            else
            {
                if (!IsSupportedColorNote(n.Type, n.CutDirection))
                    continue;

                var (lane, row) = normalizer.NormalizeCell(n.LineIndex, n.LineLayer);
                var color = n.Type == 0 ? NoteColor.Red : NoteColor.Blue;
                var cut = (CutDirection)n.CutDirection;
                notes.Add(new CanonicalNote(n.Time, lane, row, color, cut));
            }
        }

        var obstacles = raw.Obstacles.Select(o =>
        {
            int height = o.Type == 0 ? 5 : 3;
            int startRow = o.Type == 0 ? 0 : 2;
            var (lane, width) = normalizer.NormalizeLaneSpan(o.LineIndex, o.Width);
            int row = normalizer.NormalizeRow(startRow);
            return new CanonicalObstacle(o.Time, lane, o.Duration, width, height, row);
        })
        .ToList();

        // v2 beatmaps have a single BPM from Info.dat; no embedded timing changes
        var timingPoints = new List<BeatTimingPoint>
        {
            new BeatTimingPoint(0.0, 0.0, bpm)
        };

        return new CanonicalBeatmap
        {
            Song = meta,
            Difficulty = difficulty,
            TimingPoints = timingPoints,
            Notes = notes.OrderBy(n => n.Beat).ToList(),
            Bombs = bombs.OrderBy(b => b.Beat).ToList(),
            Obstacles = obstacles.OrderBy(o => o.Beat).ToList(),
            Sections = []
        };
    }

    private static bool IsSupportedColorNote(int type, int cutDirection) =>
        (type == 0 || type == 1) && cutDirection is >= 0 and <= 8;
}
