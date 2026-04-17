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
        var notes = new List<CanonicalNote>();
        var bombs = new List<CanonicalBomb>();

        foreach (var n in raw.Notes)
        {
            if (n.Type == 3) // bomb
            {
                bombs.Add(new CanonicalBomb(n.Time, n.LineIndex, n.LineLayer));
            }
            else
            {
                var color = n.Type == 0 ? NoteColor.Red : NoteColor.Blue;
                var cut = (CutDirection)n.CutDirection;
                notes.Add(new CanonicalNote(n.Time, n.LineIndex, n.LineLayer, color, cut));
            }
        }

        var obstacles = raw.Obstacles.Select(o =>
        {
            int height = o.Type == 0 ? 5 : 3;
            int startRow = o.Type == 0 ? 0 : 2;
            return new CanonicalObstacle(o.Time, o.LineIndex, o.Duration, o.Width, height, startRow);
        }).ToList();

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
}
