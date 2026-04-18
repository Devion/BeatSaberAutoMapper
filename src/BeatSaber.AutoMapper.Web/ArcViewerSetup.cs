namespace BeatSaber.AutoMapper.Web;

/// <summary>
/// Downloads the ArcViewer Unity WebGL build files from GitHub Pages into wwwroot/arcviewer/
/// so they can be served locally without any external requests at preview time.
///
/// Files are downloaded once and cached; subsequent starts skip the download if the marker
/// file <c>wwwroot/arcviewer/.ready</c> is present.
///
/// Total download: ~80 MB (wasm ~53 MB, data ~26 MB, framework ~0.5 MB, etc.)
/// </summary>
internal static class ArcViewerSetup
{
    private const string BaseUrl    = "https://allpoland.github.io/ArcViewer";
    private const string MarkerFile = ".ready";

    private static readonly string[] Files =
    [
        "index.html",
        "Build/ArcViewer.loader.js",
        "Build/ArcViewer.data",
        "Build/ArcViewer.wasm",
        "Build/ArcViewer.framework.js",
        "TemplateData/style.css",
        "TemplateData/favicon.ico",
        "TemplateData/Scripts/oggdecode.js",
    ];

    /// <summary>
    /// Returns true if the local WebGL build is already present and complete.
    /// </summary>
    public static bool IsReady(string wwwrootPath) =>
        File.Exists(Path.Combine(wwwrootPath, "arcviewer", MarkerFile));

    /// <summary>
    /// Downloads all ArcViewer WebGL files into <paramref name="wwwrootPath"/>/arcviewer/.
    /// Safe to call concurrently – a second call while a download is in progress will return
    /// immediately once the first completes.
    /// </summary>
    public static async Task EnsureDownloadedAsync(
        string wwwrootPath, HttpClient http, ILogger logger,
        CancellationToken ct = default)
    {
        string arcDir = Path.Combine(wwwrootPath, "arcviewer");
        string marker = Path.Combine(arcDir, MarkerFile);

        if (File.Exists(marker)) return;

        Directory.CreateDirectory(arcDir);

        logger.LogInformation("[ArcViewer] Downloading WebGL build (~80 MB, one-time)…");

        foreach (string file in Files)
        {
            ct.ThrowIfCancellationRequested();

            string dest = Path.Combine(arcDir,
                file.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(dest)!);

            string url  = $"{BaseUrl}/{file}";
            string tmp  = dest + ".tmp";

            try
            {
                logger.LogDebug("[ArcViewer] Fetching {File}", file);
                using var response = await http.GetAsync(url,
                    HttpCompletionOption.ResponseHeadersRead, ct);
                response.EnsureSuccessStatusCode();

                await using var src = await response.Content.ReadAsStreamAsync(ct);
                await using var dst = File.Create(tmp);
                await src.CopyToAsync(dst, ct);
            }
            catch (Exception ex)
            {
                if (File.Exists(tmp)) File.Delete(tmp);
                logger.LogError(ex, "[ArcViewer] Failed to download {File}: {Msg}",
                    file, ex.Message);
                throw;
            }

            File.Move(tmp, dest, overwrite: true);
        }

        // Mark as complete so future startups skip the download.
        await File.WriteAllTextAsync(marker,
            $"Downloaded {DateTime.UtcNow:O} from {BaseUrl}", ct);

        logger.LogInformation("[ArcViewer] WebGL build downloaded successfully.");
    }
}
