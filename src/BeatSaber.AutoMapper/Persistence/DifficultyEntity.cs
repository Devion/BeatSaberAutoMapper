namespace BeatSaber.AutoMapper.Persistence;

public sealed class DifficultyEntity
{
    public long Id { get; set; }
    public long BeatmapSetId { get; set; }
    public string Difficulty { get; set; } = "Easy";
    public int DifficultyRank { get; set; }
    public string BeatmapFilename { get; set; } = "";
    public int? NoteJumpMovementSpeed { get; set; }
    public double? NoteJumpStartBeatOffset { get; set; }
    public int NoteCount { get; set; }
    public double QualityScore { get; set; }

    public BeatmapSetEntity? BeatmapSet { get; set; }
    public List<ValidationRunEntity> ValidationRuns { get; set; } = [];
}
