namespace BeatSaber.AutoMapper.Beatmap.Import;

internal static class V4BeatmapAdapter
{
    public static CanonicalBeatmap Convert(string beatmapFilePath, SongMetadata meta, DifficultyDescriptor difficulty)
    {
        throw new NotSupportedException(
            "Beat Saber v4 beatmap format is not yet supported. " +
            "Please convert the map to v3 format using the Beat Saber Map Converter tool before importing.");
    }
}
