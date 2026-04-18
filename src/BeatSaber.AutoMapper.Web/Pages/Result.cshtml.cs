using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace BeatSaber.AutoMapper.Web.Pages;

public class ResultModel(JobStore jobStore) : PageModel
{
    public string? JobId { get; private set; }
    public List<DifficultyStats> Stats { get; private set; } = [];

    public IActionResult OnGet(string id)
    {
        if (string.IsNullOrWhiteSpace(id) || !jobStore.TryGet(id, out _))
            return RedirectToPage("Index");

        JobId = id;

        if (TempData["Stats"] is string json)
        {
            try { Stats = JsonSerializer.Deserialize<List<DifficultyStats>>(json) ?? []; }
            catch { /* TempData may be missing on a direct refresh — show no stats */ }
        }

        return Page();
    }
}

