using BeatSaber.AutoMapper.Utilities;

namespace BeatSaber.AutoMapper.Canonical;

public sealed class CanonicalBeatmap
{
    public required SongMetadata Song { get; init; }
    public required DifficultyDescriptor Difficulty { get; init; }
    public required IReadOnlyList<BeatTimingPoint> TimingPoints { get; init; }
    public required IReadOnlyList<CanonicalNote> Notes { get; init; }
    public required IReadOnlyList<CanonicalBomb> Bombs { get; init; }
    public required IReadOnlyList<CanonicalObstacle> Obstacles { get; init; }
    public required IReadOnlyList<SectionMarker> Sections { get; init; }

    /// <summary>Convert a beat number to wall-clock seconds using the timing point table.</summary>
    public double BeatToSeconds(double beat)
    {
        if (TimingPoints.Count == 0)
            return MathHelpers.BeatToSeconds(beat, Song.BeatsPerMinute);

        // Find the last timing point at or before the beat
        var tp = TimingPoints[0];
        foreach (var p in TimingPoints)
        {
            if (p.Beat <= beat) tp = p;
            else break;
        }
        double deltaBeat = beat - tp.Beat;
        return tp.TimeSeconds + MathHelpers.BeatToSeconds(deltaBeat, tp.BeatsPerMinute);
    }

    /// <summary>Convert wall-clock seconds to a beat number using the timing point table.</summary>
    public double SecondsToBeat(double seconds)
    {
        if (TimingPoints.Count == 0)
            return MathHelpers.SecondsToBeat(seconds, Song.BeatsPerMinute);

        var tp = TimingPoints[0];
        foreach (var p in TimingPoints)
        {
            if (p.TimeSeconds <= seconds) tp = p;
            else break;
        }
        double deltaSec = seconds - tp.TimeSeconds;
        return tp.Beat + MathHelpers.SecondsToBeat(deltaSec, tp.BeatsPerMinute);
    }

    /// <summary>Notes-per-second calculated over a rolling beat window centred on <paramref name="beat"/>.</summary>
    public double LocalNps(double beat, double windowBeats = 4.0)
    {
        double half = windowBeats / 2.0;
        int count = Notes.Count(n => n.Beat >= beat - half && n.Beat <= beat + half);
        double windowSec = BeatToSeconds(beat + half) - BeatToSeconds(beat - half);
        return windowSec > 0 ? count / windowSec : 0;
    }
}
