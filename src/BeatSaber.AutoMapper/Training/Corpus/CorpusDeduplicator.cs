using System.Security.Cryptography;

namespace BeatSaber.AutoMapper.Training.Corpus;

public sealed class CorpusDeduplicator
{
    /// <summary>Compute a hash of a beatmap based on its note sequence.</summary>
    public string ComputeMapHash(CanonicalBeatmap beatmap)
    {
        Guard.NotNull(beatmap, nameof(beatmap));

        var sb = new StringBuilder();
        foreach (var n in beatmap.Notes.OrderBy(n => n.Beat))
            sb.Append($"{n.Beat:F3}:{n.Lane}:{n.Row}:{(int)n.Color}:{(int)n.CutDirection};");

        byte[] bytes = Encoding.UTF8.GetBytes(sb.ToString());
        byte[] hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash);
    }

    public bool IsDuplicate(string hash, IReadOnlyList<string> existingHashes) =>
        existingHashes.Contains(hash, StringComparer.OrdinalIgnoreCase);
}
