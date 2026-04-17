namespace BeatSaber.AutoMapper.Diagnostics;

public static class GenerationDiffWriter
{
    /// <summary>
    /// Compare a generated beatmap to an optional reference map and write a diff report.
    /// </summary>
    public static void Write(CanonicalBeatmap generated, CanonicalBeatmap? reference, string outputPath)
    {
        Guard.NotNull(generated, nameof(generated));
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));

        var sb = new StringBuilder();
        sb.AppendLine("# Generation Diff Report");
        sb.AppendLine();
        sb.AppendLine($"## Generated Map: {generated.Song.Title} — {generated.Difficulty.Difficulty}");
        sb.AppendLine($"- Notes: {generated.Notes.Count}");
        sb.AppendLine($"- Bombs: {generated.Bombs.Count}");
        sb.AppendLine($"- Obstacles: {generated.Obstacles.Count}");

        if (reference is null)
        {
            sb.AppendLine();
            sb.AppendLine("*No reference map provided.*");
            File.WriteAllText(outputPath, sb.ToString(), Encoding.UTF8);
            return;
        }

        sb.AppendLine();
        sb.AppendLine($"## Reference Map: {reference.Song.Title} — {reference.Difficulty.Difficulty}");
        sb.AppendLine($"- Notes: {reference.Notes.Count}");

        sb.AppendLine();
        sb.AppendLine("## Comparison");

        int genCount = generated.Notes.Count;
        int refCount = reference.Notes.Count;
        sb.AppendLine($"- Note count delta: {genCount - refCount:+#;-#;0}");

        // Notes in generated but not in reference (within ±0.05 beat tolerance)
        int matched = 0;
        foreach (var gn in generated.Notes)
        {
            if (reference.Notes.Any(rn =>
                    Math.Abs(rn.Beat - gn.Beat) < 0.05 &&
                    rn.Lane == gn.Lane &&
                    rn.Row == gn.Row &&
                    rn.Color == gn.Color))
                matched++;
        }

        double precision = genCount > 0 ? matched / (double)genCount : 0;
        double recall = refCount > 0 ? matched / (double)refCount : 0;
        sb.AppendLine($"- Precision (matched/generated): {precision:P1}");
        sb.AppendLine($"- Recall (matched/reference):    {recall:P1}");

        File.WriteAllText(outputPath, sb.ToString(), Encoding.UTF8);
    }
}
