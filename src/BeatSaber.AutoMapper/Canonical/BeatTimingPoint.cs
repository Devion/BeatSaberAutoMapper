namespace BeatSaber.AutoMapper.Canonical;

public sealed record BeatTimingPoint(
    double Beat,
    double TimeSeconds,
    double BeatsPerMinute
);
