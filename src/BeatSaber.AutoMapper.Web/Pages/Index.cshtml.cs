using System.Text.Json;
using BeatSaber.AutoMapper.Audio.Features;
using BeatSaber.AutoMapper.Beatmap;
using BeatSaber.AutoMapper.Beatmap.Export;
using BeatSaber.AutoMapper.Canonical;
using BeatSaber.AutoMapper.Generation;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace BeatSaber.AutoMapper.Web.Pages;

public class IndexModel(JobStore jobStore) : PageModel
{
    [BindProperty] public string Title     { get; set; } = "My Song";
    [BindProperty] public string Artist    { get; set; } = "Unknown Artist";
    [BindProperty] public IFormFile? AudioFile { get; set; }
    [BindProperty] public List<string> Difficulties { get; set; } = ["Hard"];
    [BindProperty] public string? ArtifactsPath { get; set; }

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
                CoverImagePath:   null,
                AudioPath:        "song.egg");

            var diffs = Difficulties
                .Select(ParseDifficulty)
                .Distinct()
                .ToList();

            var templateSettings = new GenerationSettings(
                TargetDifficulty: diffs[0],
                AllowBombs:       false,
                AllowObstacles:   false,
                RandomSeed:       42,
                UseLearned:       !string.IsNullOrWhiteSpace(ArtifactsPath),
                ArtifactsPath:    string.IsNullOrWhiteSpace(ArtifactsPath) ? null : ArtifactsPath);

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

                stats.Add(new DifficultyStats(
                    Difficulty:      diffName,
                    NoteCount:       r.Beatmap.Notes.Count,
                    Bpm:             audio.EstimatedBpm,
                    ValidationScore: r.ValidationReport.Score,
                    ErrorCount:      r.ValidationReport.ErrorCount,
                    WarningCount:    r.ValidationReport.WarningCount,
                    RepairCount:     r.RepairResult.AppliedRepairs.Count));
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
}
