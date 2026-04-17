using BeatSaber.AutoMapper.Web;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages();
builder.Services.AddSingleton<JobStore>();
builder.Services.AddHttpClient();

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

app.UseStaticFiles();
app.UseRouting();

// CORS for /api – kept in case ArcViewer is accessed externally.
app.Use(async (ctx, next) =>
{
    if (ctx.Request.Method == HttpMethods.Options &&
        ctx.Request.Path.StartsWithSegments("/api"))
    {
        ctx.Response.Headers.Append("Access-Control-Allow-Origin", "*");
        ctx.Response.Headers.Append("Access-Control-Allow-Private-Network", "true");
        ctx.Response.Headers.Append("Access-Control-Allow-Methods", "GET, OPTIONS");
        ctx.Response.StatusCode = StatusCodes.Status204NoContent;
        return;
    }
    await next();
});

// Serve generated zip files.
app.MapGet("/api/download/{id}", async (string id, JobStore jobs, HttpResponse response) =>
{
    if (!jobs.TryGet(id, out var zipPath) || !File.Exists(zipPath))
        return Results.NotFound();

    response.Headers.Append("Access-Control-Allow-Origin", "*");
    response.Headers.Append("Access-Control-Allow-Private-Network", "true");
    response.ContentType = "application/zip";
    response.Headers.Append("Content-Disposition", "attachment; filename=\"beatsaber-map.zip\"");
    await response.SendFileAsync(zipPath!);
    return Results.Empty;
});

// ---------------------------------------------------------------------------
// ArcViewer proxy – mirrors allpoland.github.io/ArcViewer through this server
// so the iframe and the zip download share the same origin (localhost).
// Large build assets are cached on disk after the first download.
// ---------------------------------------------------------------------------
string arcCacheDir = Path.Combine(app.Environment.ContentRootPath, ".arcviewer-cache");

// Reuse a single HttpClient (handles connection pooling).
var arcClient = app.Services.GetRequiredService<IHttpClientFactory>().CreateClient();
arcClient.Timeout = TimeSpan.FromMinutes(15);
arcClient.DefaultRequestHeaders.UserAgent.ParseAdd("BeatSaberAutoMapper-ArcViewerProxy/1.0");

// Redirect bare /arcviewer → /arcviewer/ so relative URLs resolve correctly.
app.MapGet("/arcviewer", ctx =>
{
    ctx.Response.Redirect("/arcviewer/", permanent: false);
    return Task.CompletedTask;
});

app.MapGet("/arcviewer/{**path}", async (string? path, HttpContext ctx) =>
{
    // Sanitise path segments – prevent directory traversal in cache key.
    path = string.Join("/",
        (path ?? "").Split('/', StringSplitOptions.RemoveEmptyEntries)
                    .Where(s => s is not ".." and not "."));

    bool isRoot = string.IsNullOrEmpty(path) || path == "index.html";
    string remoteUrl = isRoot
        ? "https://allpoland.github.io/ArcViewer/"
        : $"https://allpoland.github.io/ArcViewer/{path}";

    // index.html is always fetched fresh; everything else is cached on disk.
    string? cachePath = isRoot
        ? null
        : Path.Combine(arcCacheDir, path.Replace('/', Path.DirectorySeparatorChar));

    if (cachePath is not null && File.Exists(cachePath))
    {
        ctx.Response.ContentType = ArcMime(path);
        ctx.Response.Headers.CacheControl = "public, max-age=604800";
        await ctx.Response.SendFileAsync(cachePath);
        return;
    }

    try
    {
        using var upstream = await arcClient.GetAsync(
            remoteUrl, HttpCompletionOption.ResponseHeadersRead, ctx.RequestAborted);

        if (!upstream.IsSuccessStatusCode)
        {
            ctx.Response.StatusCode = (int)upstream.StatusCode;
            return;
        }

        ctx.Response.ContentType = ArcMime(path);
        if (upstream.Content.Headers.ContentLength.HasValue)
            ctx.Response.ContentLength = upstream.Content.Headers.ContentLength.Value;

        await using var upStream = await upstream.Content.ReadAsStreamAsync(ctx.RequestAborted);

        if (cachePath is not null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
            string tmp = cachePath + ".tmp";
            try
            {
                await using var cacheFs = File.Create(tmp);
                byte[] buf = new byte[81920];
                int n;
                while ((n = await upStream.ReadAsync(buf, ctx.RequestAborted)) > 0)
                {
                    await ctx.Response.Body.WriteAsync(buf.AsMemory(0, n), ctx.RequestAborted);
                    await cacheFs.WriteAsync(buf.AsMemory(0, n), ctx.RequestAborted);
                }
                await cacheFs.FlushAsync(ctx.RequestAborted);
                File.Move(tmp, cachePath, overwrite: true);
            }
            catch
            {
                if (File.Exists(tmp)) File.Delete(tmp);
                throw;
            }
        }
        else
        {
            await upStream.CopyToAsync(ctx.Response.Body, ctx.RequestAborted);
        }
    }
    catch (OperationCanceledException) { /* client disconnected */ }
    catch (HttpRequestException ex)
    {
        if (!ctx.Response.HasStarted)
        {
            ctx.Response.StatusCode = 502;
            await ctx.Response.WriteAsync($"Could not load ArcViewer asset: {ex.Message}");
        }
    }
});

static string ArcMime(string path) =>
    Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".wasm" => "application/wasm",
        ".js"   => "text/javascript",
        ".css"  => "text/css",
        ".html" => "text/html; charset=utf-8",
        ".ico"  => "image/x-icon",
        ".png"  => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        ".wav"  => "audio/wav",
        _       => "application/octet-stream"
    };

app.MapRazorPages();
app.Run();

