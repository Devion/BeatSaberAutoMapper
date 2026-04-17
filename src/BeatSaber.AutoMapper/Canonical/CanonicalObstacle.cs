namespace BeatSaber.AutoMapper.Canonical;

public sealed record CanonicalObstacle(
    double Beat,
    int Lane,
    double Duration,
    int Width,
    int Height,
    int StartRow
);
