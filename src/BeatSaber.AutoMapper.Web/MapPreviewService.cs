using System.IO.Compression;
using System.Text.Json;

namespace BeatSaber.AutoMapper.Web;

/// <summary>
/// Reads a generated map ZIP and returns a lightweight JSON payload for the
/// in-browser canvas preview: BPM, total duration, and notes per difficulty.
/// </summary>
internal static class MapPreviewService
{
    public static IResult GetPreview(string zipPath)
    {
        try
        {
            using var archive = ZipFile.OpenRead(zipPath);

            // Info.dat (case-insensitive — some tools lowercase it)
            var infoEntry = archive.Entries.FirstOrDefault(
                e => e.Name.Equals("Info.dat", StringComparison.OrdinalIgnoreCase));
            if (infoEntry is null)
                return Results.Problem("Info.dat not found in archive.");

            string infoJson;
            using (var sr = new StreamReader(infoEntry.Open()))
                infoJson = sr.ReadToEnd();

            using var infoDoc = JsonDocument.Parse(infoJson);
            var infoRoot = infoDoc.RootElement;

            double bpm = infoRoot.GetProperty("_beatsPerMinute").GetDouble();
            if (bpm <= 0) bpm = 120;

            var diffs  = new List<object>();
            double maxBeat = 0;

            foreach (var set in infoRoot.GetProperty("_difficultyBeatmapSets").EnumerateArray())
            {
                // Only Standard characteristic for the preview
                string charName = set.TryGetProperty("_beatmapCharacteristicName", out var cn)
                    ? cn.GetString() ?? "Standard" : "Standard";
                if (!charName.Equals("Standard", StringComparison.OrdinalIgnoreCase))
                    continue;

                foreach (var diff in set.GetProperty("_difficultyBeatmaps").EnumerateArray())
                {
                    string filename = diff.GetProperty("_beatmapFilename").GetString()!;
                    string diffName = diff.GetProperty("_difficulty").GetString() ?? filename;

                    var bmEntry = archive.Entries.FirstOrDefault(
                        e => e.FullName.Equals(filename, StringComparison.OrdinalIgnoreCase));
                    if (bmEntry is null) continue;

                    string bmJson;
                    using (var sr = new StreamReader(bmEntry.Open()))
                        bmJson = sr.ReadToEnd();

                    var notes = ParseNotes(bmJson);
                    if (notes.Length > 0)
                        maxBeat = Math.Max(maxBeat, notes[^1].b);

                    diffs.Add(new { name = diffName, notes });
                }
            }

            if (diffs.Count == 0)
                return Results.Problem("No Standard difficulties found.");

            // Estimate duration from last note + small tail (8 beats)
            double duration = maxBeat > 0 ? (maxBeat + 8.0) * 60.0 / bpm : 60.0;

            return Results.Json(new { bpm, duration, difficulties = diffs });
        }
        catch (Exception ex)
        {
            return Results.Problem($"Preview error: {ex.Message}");
        }
    }

    // ── note parsing ──────────────────────────────────────────────────────────

    // Compact note record — serialises to {b,x,y,c,d} which the JS reads directly.
    private record struct NoteDto(double b, int x, int y, int c, int d);

    private static NoteDto[] ParseNotes(string json)
    {
        try
        {
            using var doc  = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // v3 format: colorNotes array with {b,x,y,c,d}
            if (root.TryGetProperty("colorNotes", out var v3))
            {
                return v3.EnumerateArray()
                    .Select(n => new NoteDto(
                        b: n.GetProperty("b").GetDouble(),
                        x: n.GetProperty("x").GetInt32(),
                        y: n.GetProperty("y").GetInt32(),
                        c: n.GetProperty("c").GetInt32(),
                        d: n.GetProperty("d").GetInt32()))
                    .OrderBy(n => n.b)
                    .ToArray();
            }

            // v2 format: _notes with {_time, _lineIndex, _lineLayer, _type, _cutDirection}
            if (root.TryGetProperty("_notes", out var v2))
            {
                return v2.EnumerateArray()
                    .Select(n => (
                        time: n.GetProperty("_time").GetDouble(),
                        x:    n.GetProperty("_lineIndex").GetInt32(),
                        y:    n.GetProperty("_lineLayer").GetInt32(),
                        type: n.GetProperty("_type").GetInt32(),
                        dir:  n.GetProperty("_cutDirection").GetInt32()))
                    .Where(n => n.type == 0 || n.type == 1)   // skip bombs (_type=3)
                    .Select(n => new NoteDto(b: n.time, x: n.x, y: n.y, c: n.type, d: n.dir))
                    .OrderBy(n => n.b)
                    .ToArray();
            }
        }
        catch { /* malformed beatmap — return empty */ }

        return [];
    }
}
