namespace BeatSaber.AutoMapper.Validation.Repair;

public sealed class RepairEngine
{
    private readonly BeatmapValidator _validator;

    public RepairEngine() { _validator = new BeatmapValidator(); }

    public sealed record RepairResult(
        CanonicalBeatmap RepairedBeatmap,
        IReadOnlyList<string> AppliedRepairs,
        ValidationReport FinalReport
    );

    public RepairResult Repair(CanonicalBeatmap beatmap)
    {
        Guard.NotNull(beatmap, nameof(beatmap));

        var applied = new List<string>();
        const int maxIterations = 10;

        CanonicalBeatmap current = beatmap;
        for (int iter = 0; iter < maxIterations; iter++)
        {
            var report = _validator.Validate(current);
            if (report.IsValid) break;

            bool anyRepaired = false;
            foreach (var issue in report.Issues.Where(i => i.Severity == IssueSeverity.Error))
            {
                CanonicalBeatmap? repaired = null;
                if (issue.RuleName == "Parity")
                {
                    repaired = TryRepairParityBreak(current, issue, out string? desc);
                    if (repaired is not null) applied.Add(desc ?? "Parity repair");
                }
                else if (issue.RuleName is "DoubleDirectional" or "SwingSpeed" or "HitboxPath")
                {
                    repaired = TryRemoveOffendingNote(current, issue, out string? desc3);
                    if (repaired is not null) applied.Add(desc3 ?? "Flow repair");
                }
                else if (issue.RuleName == "Density")
                {
                    repaired = TryRepairDensitySpike(current, issue, out string? desc2);
                    if (repaired is not null) applied.Add(desc2 ?? "Density repair");
                }
                else if (issue.RuleName is "ExcessiveDouble" or "Handclap" or "VisionBlock")
                {
                    repaired = TryThinBeatPattern(current, issue, out string? desc4);
                    if (repaired is not null) applied.Add(desc4 ?? "Chord thinning repair");
                }

                if (repaired is not null)
                {
                    current = repaired;
                    anyRepaired = true;
                    break; // restart after each repair
                }
            }

            if (!anyRepaired) break;
        }

        return new RepairResult(current, applied, _validator.Validate(current));
    }

    private CanonicalBeatmap? TryRepairParityBreak(
        CanonicalBeatmap beatmap, ValidationIssue issue, out string? description)
    {
        description = null;
        if (issue.Beat is null) return null;

        double beat = issue.Beat.Value;
        var hand = issue.Hand;

        // Strategy: remove the offending note
        var newNotes = beatmap.Notes
            .Where(n => !(Math.Abs(n.Beat - beat) < 0.001 && n.Hand == hand))
            .ToList();

        if (newNotes.Count == beatmap.Notes.Count) return null;

        description = $"Removed parity-breaking note at beat {beat:F2} ({hand} hand).";
        return CloneWith(beatmap, newNotes);
    }

    private CanonicalBeatmap? TryRepairDensitySpike(
        CanonicalBeatmap beatmap, ValidationIssue issue, out string? description)
    {
        description = null;
        if (issue.Beat is null) return null;

        double center = issue.Beat.Value;
        double window = 2.0; // half-window in beats

        // Thin notes in the window: keep every other note
        var inWindow = beatmap.Notes
            .Where(n => Math.Abs(n.Beat - center) <= window)
            .OrderBy(n => n.Beat)
            .ToList();

        var toRemove = new HashSet<CanonicalNote>();
        for (int i = 1; i < inWindow.Count; i += 2)
            toRemove.Add(inWindow[i]);

        if (toRemove.Count == 0) return null;

        var newNotes = beatmap.Notes.Where(n => !toRemove.Contains(n)).ToList();
        description = $"Thinned {toRemove.Count} notes near beat {center:F2} to reduce density spike.";
        return CloneWith(beatmap, newNotes);
    }

    private CanonicalBeatmap? TryRemoveOffendingNote(
        CanonicalBeatmap beatmap, ValidationIssue issue, out string? description)
    {
        description = null;
        if (issue.Beat is null)
            return null;

        double beat = issue.Beat.Value;
        var candidate = beatmap.Notes
            .Where(n => Math.Abs(n.Beat - beat) < PlayabilityHeuristics.BeatTolerance &&
                        (issue.Hand is null || n.Hand == issue.Hand))
            .OrderByDescending(n => n.Row)
            .ThenByDescending(n => Math.Abs(n.Lane - 1.5))
            .FirstOrDefault();

        if (candidate is null)
            return null;

        var newNotes = beatmap.Notes.Where(n => n != candidate).ToList();
        description = $"Removed note at beat {candidate.Beat:F2} for {issue.RuleName}.";
        return CloneWith(beatmap, newNotes);
    }

    private CanonicalBeatmap? TryThinBeatPattern(
        CanonicalBeatmap beatmap, ValidationIssue issue, out string? description)
    {
        description = null;
        if (issue.Beat is null)
            return null;

        double beat = issue.Beat.Value;
        var notesAtBeat = beatmap.Notes
            .Where(n => Math.Abs(n.Beat - beat) < PlayabilityHeuristics.BeatTolerance)
            .ToList();

        var toRemove = PlayabilityHeuristics.PickNoteToRemoveAtBeat(notesAtBeat);
        if (toRemove is null)
            return null;

        var newNotes = beatmap.Notes.Where(n => n != toRemove).ToList();
        description = $"Removed one note from beat {beat:F2} to resolve {issue.RuleName}.";
        return CloneWith(beatmap, newNotes);
    }

    private static CanonicalBeatmap CloneWith(CanonicalBeatmap original, List<CanonicalNote> notes) =>
        new()
        {
            Song = original.Song,
            Difficulty = original.Difficulty,
            TimingPoints = original.TimingPoints,
            Notes = notes,
            Bombs = original.Bombs,
            Obstacles = original.Obstacles,
            Sections = original.Sections
        };
}
