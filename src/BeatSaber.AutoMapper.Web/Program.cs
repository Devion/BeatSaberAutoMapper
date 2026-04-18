using BeatSaber.AutoMapper.Web;
using Microsoft.AspNetCore.StaticFiles;
using System.IO.Compression;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages();
builder.Services.AddSingleton<JobStore>();

// Increase upload limit to 200 MB (large audio files)
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(o =>
{
    o.MultipartBodyLengthLimit = 200 * 1024 * 1024;
});
builder.WebHost.ConfigureKestrel(o =>
    o.Limits.MaxRequestBodySize = 200 * 1024 * 1024);

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}

// ---------------------------------------------------------------------------
// Static files — .egg is an Ogg-encoded audio file used by Beat Saber.
// ---------------------------------------------------------------------------
var mimeProvider = new FileExtensionContentTypeProvider();
mimeProvider.Mappings[".egg"] = "audio/ogg";

app.UseStaticFiles(new StaticFileOptions { ContentTypeProvider = mimeProvider });
app.UseRouting();

// Serve generated zip files.
app.MapGet("/api/download/{id}", async (string id, JobStore jobs, HttpResponse response) =>
{
    if (!jobs.TryGet(id, out var zipPath) || !File.Exists(zipPath))
        return Results.NotFound();

    response.ContentType = "application/zip";
    response.Headers.Append("Content-Disposition", "attachment; filename=\"beatsaber-map.zip\"");
    await response.SendFileAsync(zipPath!);
    return Results.Empty;
});

// Map preview data endpoint — returns BPM, duration, and note list per difficulty.
app.MapGet("/api/preview/{id}", (string id, JobStore jobs) =>
{
    if (!jobs.TryGet(id, out var zipPath) || !File.Exists(zipPath))
        return Results.NotFound();
    return MapPreviewService.GetPreview(zipPath!);
});

// Audio stream endpoint — extracts song.egg from the map ZIP.
// Range processing enabled so the browser <audio> element can seek.
app.MapGet("/api/audio/{id}", (string id, JobStore jobs) =>
{
    if (!jobs.TryGet(id, out var zipPath) || !File.Exists(zipPath))
        return Results.NotFound();

    using var archive = ZipFile.OpenRead(zipPath!);
    var entry = archive.Entries.FirstOrDefault(
        e => e.Name.Equals("song.egg", StringComparison.OrdinalIgnoreCase));
    if (entry is null) return Results.NotFound();

    var ms = new MemoryStream((int)entry.Length);
    using (var es = entry.Open()) es.CopyTo(ms);
    ms.Position = 0;
    return Results.File(ms, "audio/ogg", "song.ogg", enableRangeProcessing: true);
});

app.MapRazorPages();

app.Run();

