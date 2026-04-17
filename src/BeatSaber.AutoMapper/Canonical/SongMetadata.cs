namespace BeatSaber.AutoMapper.Canonical;

public sealed record SongMetadata(
    string Title,
    string Artist,
    string? SubTitle,
    double BeatsPerMinute,
    double SongTimeOffset,
    double PreviewStartTime,
    double PreviewDuration,
    string? CoverImagePath,
    string? AudioPath
);
