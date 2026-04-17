using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace BeatSaber.AutoMapper.Web.Pages;

public class ResultModel(JobStore jobStore) : PageModel
{
    public string?  JobId        { get; private set; }
    public string?  ArcViewerUrl { get; private set; }
    public List<DifficultyStats> Stats { get; private set; } = [];

    public IActionResult OnGet(string id)
    {
        if (string.IsNullOrWhiteSpace(id) || !jobStore.TryGet(id, out _))
            return RedirectToPage("Index");

        JobId = id;

        // Build the local download URL – ArcViewer is served from the same origin,
        // so the zip fetch is same-origin (no Private Network Access issues).
        string downloadUrl = $"{Request.Scheme}://{Request.Host}/api/download/{id}";
        ArcViewerUrl = $"/arcviewer/?zip={Uri.EscapeDataString(downloadUrl)}";

        // Deserialise per-difficulty stats passed via TempData from Index
        if (TempData["Stats"] is string json)
        {
            try { Stats = JsonSerializer.Deserialize<List<DifficultyStats>>(json) ?? []; }
            catch { /* TempData may be missing on a direct refresh — just show no stats */ }
        }

        return Page();
    }
}
