namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Placement model loaded from placement_weights.json artifacts.
/// Implements the same logistic regression as PlacementModelTrainer.
/// </summary>
public sealed class PlacementModel : IPlacementScorer
{
    private sealed record Weights(
        double w0, double wOnset, double wEnergy, double wSubdiv,
        double wNps, double wBeatStrength, double wMeasureBeat, double wDifficulty);

    private readonly Weights _w;

    private PlacementModel(Weights w) => _w = w;

    public static PlacementModel? TryLoad(string artifactsPath)
    {
        string path = Path.Combine(artifactsPath, "placement_weights.json");
        if (!File.Exists(path)) return null;
        try
        {
            var json = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(path));
            var w = new Weights(
                w0:            GetDouble(json, "w0"),
                wOnset:        GetDouble(json, "wOnset"),
                wEnergy:       GetDouble(json, "wEnergy"),
                wSubdiv:       GetDouble(json, "wSubdiv"),
                wNps:          GetDouble(json, "wNps"),
                wBeatStrength: GetDouble(json, "wBeatStrength", 0.8),
                wMeasureBeat:  GetDouble(json, "wMeasureBeat", 0.2),
                wDifficulty:   GetDouble(json, "wDifficulty", 1.0));
            return new PlacementModel(w);
        }
        catch { return null; }
    }

    public double ScorePlacement(double onset, double energy, double subdiv,
                                 double localNps, double beatStrength, int measureBeat,
                                 int difficultyLevel)
    {
        double z = _w.w0
            + _w.wOnset        * onset
            + _w.wEnergy       * energy
            + _w.wSubdiv       * subdiv
            + _w.wNps          * localNps
            + _w.wBeatStrength * beatStrength
            + _w.wMeasureBeat  * (measureBeat / 4.0)
            + _w.wDifficulty   * (difficultyLevel / 4.0);
        return Sigmoid(z);
    }

    private static double GetDouble(JsonElement el, string key, double fallback = 0.0) =>
        el.TryGetProperty(key, out var v) ? v.GetDouble() : fallback;

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
}
