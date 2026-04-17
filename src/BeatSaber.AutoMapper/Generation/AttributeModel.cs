namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Attribute model loaded from attribute_model.json artifacts.
/// Uses conditional probability tables for cut direction and position.
/// </summary>
public sealed class AttributeModel : IAttributeModel
{
    private readonly Dictionary<string, double[]> _transitions;
    private readonly Dictionary<string, double[]> _positions;

    private AttributeModel(
        Dictionary<string, double[]> transitions,
        Dictionary<string, double[]> positions)
    {
        _transitions = transitions;
        _positions   = positions;
    }

    public static AttributeModel? TryLoad(string artifactsPath)
    {
        string path = Path.Combine(artifactsPath, "attribute_model.json");
        if (!File.Exists(path)) return null;
        try
        {
            var root = JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(path));
            var transitions = DeserializeTable(root, "Transitions");
            var positions   = DeserializeTable(root, "Positions");
            if (transitions is null || positions is null) return null;
            return new AttributeModel(transitions, positions);
        }
        catch { return null; }
    }

    public CutDirection SampleCutDirection(int prevCutDir, double beatStrength, int hand,
                                           int difficultyLevel, Random rng)
    {
        int bsBin = BeatStrengthBin(beatStrength);
        string key = $"{(prevCutDir >= 0 ? prevCutDir : 8)}_{bsBin}_{hand}_{difficultyLevel}";
        if (!_transitions.TryGetValue(key, out var probs))
        {
            string fallback = $"{(prevCutDir >= 0 ? prevCutDir : 8)}_{bsBin}_{hand}_2";
            probs = _transitions.TryGetValue(fallback, out var p2) ? p2 : UniformProbs(9);
        }
        return (CutDirection)SampleFromProbs(probs, rng);
    }

    public (int Lane, int Row) SamplePosition(int cutDir, int hand, double beatStrength,
                                              double spectralCentroid, int difficultyLevel, Random rng)
    {
        int bsBin = BeatStrengthBin(beatStrength);
        string key = $"{(cutDir >= 0 ? cutDir : 8)}_{hand}_{bsBin}_{difficultyLevel}";
        if (!_positions.TryGetValue(key, out var rawProbs))
        {
            string fallback = $"{(cutDir >= 0 ? cutDir : 8)}_{hand}_{bsBin}_2";
            rawProbs = _positions.TryGetValue(fallback, out var p2) ? p2 : UniformProbs(12);
        }
        double[] probs = (double[])rawProbs.Clone();
        ApplyCentroidBias(probs, spectralCentroid);
        int idx = SampleFromProbs(probs, rng);
        return (idx / 3, idx % 3);
    }

    // -----------------------------------------------------------------------

    private static Dictionary<string, double[]>? DeserializeTable(JsonElement root, string prop)
    {
        if (!root.TryGetProperty(prop, out var obj)) return null;
        var dict = new Dictionary<string, double[]>();
        foreach (var kv in obj.EnumerateObject())
        {
            var arr = kv.Value.EnumerateArray().Select(e => e.GetDouble()).ToArray();
            dict[kv.Name] = arr;
        }
        return dict;
    }

    private static void ApplyCentroidBias(double[] probs, double centroid)
    {
        if (centroid <= 0) return;
        double topBoost    = centroid > 0.6 ? 1.5 : 1.0;
        double bottomBoost = centroid < 0.4 ? 1.5 : 1.0;
        for (int lane = 0; lane < 4; lane++)
        {
            probs[lane * 3 + 0] *= topBoost;
            probs[lane * 3 + 2] *= bottomBoost;
        }
        double sum = probs.Sum();
        if (sum > 0)
            for (int i = 0; i < probs.Length; i++) probs[i] /= sum;
    }

    private static int BeatStrengthBin(double s) => s >= 0.75 ? 0 : s >= 0.5 ? 1 : 2;

    private static double[] UniformProbs(int size)
    {
        double p = 1.0 / size;
        var arr = new double[size];
        Array.Fill(arr, p);
        return arr;
    }

    private static int SampleFromProbs(double[] probs, Random rng)
    {
        double r = rng.NextDouble(), cum = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (r < cum) return i;
        }
        return probs.Length - 1;
    }
}
