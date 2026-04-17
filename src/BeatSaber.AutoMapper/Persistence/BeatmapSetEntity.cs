namespace BeatSaber.AutoMapper.Persistence;

public sealed class BeatmapSetEntity
{
    public long Id { get; set; }
    public long SongId { get; set; }
    public string Characteristic { get; set; } = "Standard";
    public string FolderPath { get; set; } = "";

    public SongEntity? Song { get; set; }
    public List<DifficultyEntity> Difficulties { get; set; } = [];
}
