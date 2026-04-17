namespace BeatSaber.AutoMapper.Canonical;

public sealed record SectionMarker(
    double Beat,
    double TimeSeconds,
    string Label,
    SectionType Type,
    double EnergyLevel  // 0-1 normalised energy
);
