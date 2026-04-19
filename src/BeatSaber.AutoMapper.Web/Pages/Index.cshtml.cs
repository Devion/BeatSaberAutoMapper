using System.Text.Json;
using BeatSaber.AutoMapper.Audio.Features;
using BeatSaber.AutoMapper.Beatmap;
using BeatSaber.AutoMapper.Beatmap.Export;
using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Generation;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace BeatSaber.AutoMapper.Web.Pages;

public class IndexModel(JobStore jobStore, IConfiguration config) : PageModel
{
    [BindProperty] public string Title     { get; set; } = "My Song";
    [BindProperty] public string Artist    { get; set; } = "Unknown Artist";
    [BindProperty] public IFormFile? AudioFile { get; set; }
    [BindProperty] public List<string> Difficulties { get; set; } = ["Hard"];

    public string? Error { get; private set; }

    public void OnGet() { }

    public async Task<IActionResult> OnPostAsync()
    {
        // Evict stale jobs (> 1 hour) to prevent unbounded temp file growth
        jobStore.Evict(TimeSpan.FromHours(1));

        if (AudioFile is null || AudioFile.Length == 0)
        {
            Error = "Please select an audio file.";
            return Page();
        }

        if (Difficulties.Count == 0)
        {
            Error = "Select at least one difficulty.";
            return Page();
        }

        // --- Save uploaded audio to a temp directory -------------------------
        string tempDir  = Path.Combine(Path.GetTempPath(), "bsam_" + Guid.NewGuid().ToString("N"));
        string mapDir   = Path.Combine(tempDir, "map");
        string zipPath  = Path.Combine(tempDir, "map.zip");
        Directory.CreateDirectory(mapDir);

        // Save original upload to temp so we can analyse it before encoding.
        string origExt  = Path.GetExtension(AudioFile.FileName).ToLowerInvariant();
        string origPath = Path.Combine(tempDir, "upload" + origExt);

        await using (var fs = System.IO.File.Create(origPath))
            await AudioFile.CopyToAsync(fs);

        try
        {
            // --- Extract cover art from audio tags; fall back to a placeholder ---
            string coverFile;
            string coverPath;
            // Try embedded ID3 art first — preserve whatever extension/format is stored in tags.
            string jpgPath = Path.Combine(mapDir, "cover.jpg");
            if (TryExtractCoverArt(origPath, jpgPath))
            {
                coverFile = "cover.jpg";
                coverPath = jpgPath;
            }
            else
            {
                // Fallback: minimal PNG (valid image, correct extension)
                coverFile = "cover.png";
                coverPath = Path.Combine(mapDir, coverFile);
                await System.IO.File.WriteAllBytesAsync(coverPath, CreatePlaceholderPng());
            }

            // --- Analyse audio (get BPM etc.) from original file -------------
            var extractor = new AudioFeatureExtractor();
            var audio     = extractor.Extract(origPath);

            // --- Convert / copy audio to song.egg (OGG Vorbis) ---------------
            string audioEgg = Path.Combine(mapDir, "song.egg");
            var oggBytes    = await Task.Run(() => AudioConverter.ToOgg(origPath));
            await System.IO.File.WriteAllBytesAsync(audioEgg, oggBytes);

            double bpm = audio.EstimatedBpm > 0 ? audio.EstimatedBpm : 120.0;

            var song = new SongMetadata(
                Title:            Title,
                Artist:           Artist,
                SubTitle:         null,
                BeatsPerMinute:   bpm,
                SongTimeOffset:   0,
                PreviewStartTime: 10,
                PreviewDuration:  30,
                CoverImagePath:   coverFile,
                AudioPath:        "song.egg");

            var diffs = Difficulties
                .Select(ParseDifficulty)
                .Distinct()
                .ToList();

            var artifactsPath = config["BeatSaber:ArtifactsPath"] ?? @"D:\bsamartifacts";
            var useLearned    = Directory.Exists(artifactsPath) &&
                                Directory.GetFiles(artifactsPath, "*.pt").Length > 0;

            var templateSettings = new GenerationSettings(
                TargetDifficulty: diffs[0],
                AllowBombs:       false,
                AllowObstacles:   false,
                AllowFieldMovement: false,
                RandomSeed:       42,
                UseLearned:       useLearned,
                ArtifactsPath:    useLearned ? artifactsPath : null);

            // --- Generate all requested difficulties -------------------------
            var svc     = new MapGenerationService();
            var results = svc.GenerateAll(audio, song, diffs, templateSettings);

            // --- Write beatmap files ----------------------------------------
            var diffFiles = new List<(DifficultyDescriptor, string)>();
            var stats     = new List<DifficultyStats>();

            foreach (var r in results)
            {
                string diffName = r.Beatmap.Difficulty.Difficulty.ToString();
                string datFile  = $"{diffName}.dat";
                BeatmapExporter.ExportV3(r.Beatmap, Path.Combine(mapDir, datFile));
                diffFiles.Add((r.Beatmap.Difficulty, datFile));
                int totalNotes = Math.Max(1, r.Beatmap.Notes.Count);
                int leftCount = r.Beatmap.Notes.Count(n => n.Color == NoteColor.Red);
                int rightCount = r.Beatmap.Notes.Count(n => n.Color == NoteColor.Blue);
                int centerCount = r.Beatmap.Notes.Count(n => n.Lane is 1 or 2);
                int topRowCount = r.Beatmap.Notes.Count(n => n.Row == 2);
                int dotCount = r.Beatmap.Notes.Count(n => n.CutDirection == CutDirection.Dot);
                int diagonalCount = r.Beatmap.Notes.Count(n =>
                    n.CutDirection is CutDirection.UpLeft or CutDirection.UpRight or CutDirection.DownLeft or CutDirection.DownRight);

                stats.Add(new DifficultyStats(
                    Difficulty:      diffName,
                    NoteCount:       r.Beatmap.Notes.Count,
                    Bpm:             audio.EstimatedBpm,
                    ValidationScore: r.ValidationReport.Score,
                    ErrorCount:      r.ValidationReport.ErrorCount,
                    WarningCount:    r.ValidationReport.WarningCount,
                    RepairCount:     r.RepairResult.AppliedRepairs.Count,
                    LeftHandPercent: leftCount / (double)totalNotes,
                    RightHandPercent: rightCount / (double)totalNotes,
                    CenterLanePercent: centerCount / (double)totalNotes,
                    TopRowPercent: topRowCount / (double)totalNotes,
                    DotPercent: dotCount / (double)totalNotes,
                    DiagonalPercent: diagonalCount / (double)totalNotes));
            }

            BeatmapExporter.ExportInfoDat(song, diffFiles, mapDir);

            // --- Package to zip ----------------------------------------------
            BeatmapPackager.Pack(mapDir, zipPath);

            string jobId = jobStore.Add(zipPath);

            TempData["Stats"] = JsonSerializer.Serialize(stats);
            return RedirectToPage("Result", new { id = jobId });
        }
        catch (Exception ex)
        {
            Error = $"Generation failed: {ex.Message}";
            return Page();
        }
    }

    private static DifficultyLevel ParseDifficulty(string s) => s switch
    {
        "Easy"       => DifficultyLevel.Easy,
        "Normal"     => DifficultyLevel.Normal,
        "Hard"       => DifficultyLevel.Hard,
        "Expert"     => DifficultyLevel.Expert,
        "ExpertPlus" => DifficultyLevel.ExpertPlus,
        _            => DifficultyLevel.Hard
    };

    // --- Cover art helpers ---------------------------------------------------

    private static bool TryExtractCoverArt(string audioPath, string destPath)
    {
        try
        {
            using var tags = TagLib.File.Create(audioPath);
            var pic = tags.Tag.Pictures.FirstOrDefault();
            if (pic?.Data?.Data is { Length: > 0 } bytes)
            {
                System.IO.File.WriteAllBytes(destPath, bytes);
                return true;
            }
        }
        catch { /* Unsupported format, missing tags, corrupt file, etc. */ }
        return false;
    }

    /// <summary>Generates a minimal solid dark-navy PNG without any extra packages.</summary>
    private static byte[] CreatePlaceholderPng(int size = 256)
    {
        // Raw image: per row → filter byte (0 = None) + RGB pixels
        int rowLen = 1 + size * 3;
        byte[] raw = new byte[size * rowLen];
        for (int row = 0; row < size; row++)
        {
            int off = row * rowLen;
            for (int col = 0; col < size; col++)
            {
                raw[off + 1 + col * 3] = 0x1a; // R
                raw[off + 2 + col * 3] = 0x1a; // G
                raw[off + 3 + col * 3] = 0x50; // B  →  dark navy
            }
        }

        byte[] idat;
        using (var ms = new MemoryStream())
        {
            using (var zlib = new System.IO.Compression.ZLibStream(
                       ms, System.IO.Compression.CompressionLevel.Fastest, leaveOpen: true))
                zlib.Write(raw);
            idat = ms.ToArray();
        }

        using var result = new MemoryStream(1024);
        result.Write(stackalloc byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }); // PNG signature

        byte[] ihdr = new byte[13];
        ihdr[0] = (byte)(size >> 24); ihdr[1] = (byte)(size >> 16);
        ihdr[2] = (byte)(size >> 8);  ihdr[3] = (byte)size; // width
        ihdr[4] = (byte)(size >> 24); ihdr[5] = (byte)(size >> 16);
        ihdr[6] = (byte)(size >> 8);  ihdr[7] = (byte)size; // height
        ihdr[8] = 8; ihdr[9] = 2;    // bit depth 8, colour type RGB
        WritePngChunk(result, "IHDR"u8, ihdr);
        WritePngChunk(result, "IDAT"u8, idat);
        WritePngChunk(result, "IEND"u8, []);
        return result.ToArray();
    }

    private static void WritePngChunk(MemoryStream ms, ReadOnlySpan<byte> type, byte[] data)
    {
        int len = data.Length;
        ms.WriteByte((byte)(len >> 24)); ms.WriteByte((byte)(len >> 16));
        ms.WriteByte((byte)(len >> 8));  ms.WriteByte((byte)len);
        ms.Write(type);
        ms.Write(data);
        uint crc = PngCrc32(0xFFFF_FFFFu, type);
        crc = PngCrc32(crc, data);
        crc ^= 0xFFFF_FFFFu;
        ms.WriteByte((byte)(crc >> 24)); ms.WriteByte((byte)(crc >> 16));
        ms.WriteByte((byte)(crc >> 8));  ms.WriteByte((byte)crc);
    }

    private static uint PngCrc32(uint crc, ReadOnlySpan<byte> data)
    {
        foreach (byte b in data)
        {
            crc ^= b;
            for (int i = 0; i < 8; i++)
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB8_8320u : crc >> 1;
        }
        return crc;
    }
}
