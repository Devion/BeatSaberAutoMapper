using BeatSaber.AutoMapper.Canonical.Derived;

namespace BeatSaber.AutoMapper.Diagnostics;

public static class ParityTraceWriter
{
    /// <summary>
    /// Writes a parity trace CSV.
    /// Columns: beat, hand, lane, row, cut_direction, parity_class, transition, reason
    /// </summary>
    public static void Write(CanonicalBeatmap beatmap, string outputPath)
    {
        Guard.NotNull(beatmap, nameof(beatmap));
        Guard.NotNullOrEmpty(outputPath, nameof(outputPath));

        using var sw = new StreamWriter(outputPath, false, Encoding.UTF8);
        sw.WriteLine("beat,hand,lane,row,cut_direction,parity_class,transition,reason");

        foreach (NoteHand hand in new[] { NoteHand.Left, NoteHand.Right })
        {
            var notes = beatmap.Notes.Where(n => n.Hand == hand).OrderBy(n => n.Beat).ToList();
            var ctx = new SwingContext(hand);
            CanonicalNote? prev = null;

            foreach (var note in notes)
            {
                var result = ParityAnalyzer.ClassifyTransition(prev, note, ctx);
                string parityClass = ctx.CurrentParity.ToString();
                sw.WriteLine(
                    $"{note.Beat:F3},{hand},{note.Lane},{note.Row}," +
                    $"{note.CutDirection},{parityClass}," +
                    $"{result.Transition},{EscapeCsv(result.Reason)}");

                ctx.Update(note);
                prev = note;
            }
        }
    }

    private static string EscapeCsv(string? s)
    {
        if (s is null) return "";
        if (s.Contains(',') || s.Contains('"') || s.Contains('\n'))
            return $"\"{s.Replace("\"", "\"\"")}\"";
        return s;
    }
}
