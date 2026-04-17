using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Training.Features;

namespace BeatSaber.AutoMapper.Training.Models;

/// <summary>
/// Attribute model trained via frequency counting over note examples.
/// Learns conditional probability tables for cut direction transitions and positions.
/// Implements IAttributeModel so it can be passed directly to the generation pipeline.
/// </summary>
public sealed class AttributeModelTrainer : IAttributeModel
{
    // Raw counts: [prevDir_bsBin_hand] → int[9]  (9 cut directions)
    private readonly Dictionary<string, int[]> _transitionCounts = new();
    // Raw counts: [cutDir_hand_bsBin] → int[12]  (4 lanes × 3 rows)
    private readonly Dictionary<string, int[]> _positionCounts   = new();

    // Normalized probability tables (built once during Train())
    private Dictionary<string, double[]> _transitions = new();
    private Dictionary<string, double[]> _positions   = new();

    private double _leftHandFraction = 0.5;

    // -----------------------------------------------------------------------
    // Training (one-shot frequency counting)
    // -----------------------------------------------------------------------

    public void Train(IReadOnlyList<TrainingExample> examples)
    {
        Guard.NotNull(examples, nameof(examples));

        var notes = examples.AsParallel()
            .Where(e => e.HasNote && e.NoteHand >= 0)
            .ToList();

        _leftHandFraction = notes.Count > 0
            ? notes.AsParallel().Count(e => e.NoteHand == 0) / (double)notes.Count
            : 0.5;

        _transitionCounts.Clear();
        _positionCounts.Clear();

        // Each thread builds its own local count tables then we merge
        int P = Environment.ProcessorCount;
        int chunkSize = Math.Max(1, (notes.Count + P - 1) / P);

        var localTrans = new Dictionary<string, int[]>[P];
        var localPos   = new Dictionary<string, int[]>[P];
        for (int t = 0; t < P; t++)
        {
            localTrans[t] = new Dictionary<string, int[]>();
            localPos[t]   = new Dictionary<string, int[]>();
        }

        Parallel.For(0, P, new ParallelOptions { MaxDegreeOfParallelism = P }, t =>
        {
            int start = t * chunkSize;
            int end   = Math.Min(start + chunkSize, notes.Count);
            var trans = localTrans[t];
            var pos   = localPos[t];

            for (int i = start; i < end; i++)
            {
                var ex     = notes[i];
                int bsBin  = BeatStrengthBin(ex.BeatStrength);
                int hand   = ex.NoteHand;
                int diff   = ex.DifficultyLevel;
                int prevDir = (hand == 0 ? ex.PreviousLeftCutDir : ex.PreviousRightCutDir);
                if (prevDir < 0) prevDir = 8;

                string tKey = $"{prevDir}_{bsBin}_{hand}_{diff}";
                if (!trans.TryGetValue(tKey, out var tArr))
                    trans[tKey] = tArr = new int[9];
                int cutDir = ex.NoteCutDir >= 0 ? ex.NoteCutDir : 8;
                if (cutDir < 9) tArr[cutDir]++;

                if (ex.NoteLane >= 0 && ex.NoteRow >= 0)
                {
                    string pKey = $"{cutDir}_{hand}_{bsBin}_{diff}";
                    if (!pos.TryGetValue(pKey, out var pArr))
                        pos[pKey] = pArr = new int[12];
                    int posIdx = ex.NoteLane * 3 + ex.NoteRow;
                    if (posIdx < 12) pArr[posIdx]++;
                }
            }
        });

        // Merge thread-local tables into shared tables
        foreach (var dict in localTrans)
            foreach (var (key, arr) in dict)
            {
                if (!_transitionCounts.TryGetValue(key, out var existing))
                    _transitionCounts[key] = existing = new int[arr.Length];
                for (int i = 0; i < arr.Length; i++) existing[i] += arr[i];
            }

        foreach (var dict in localPos)
            foreach (var (key, arr) in dict)
            {
                if (!_positionCounts.TryGetValue(key, out var existing))
                    _positionCounts[key] = existing = new int[arr.Length];
                for (int i = 0; i < arr.Length; i++) existing[i] += arr[i];
            }

        _transitions = Normalize(_transitionCounts, 9);
        _positions   = Normalize(_positionCounts,   12);
    }

    // -----------------------------------------------------------------------
    // IAttributeModel
    // -----------------------------------------------------------------------

    public CutDirection SampleCutDirection(int prevCutDir, double beatStrength, int hand,
                                           int difficultyLevel, Random rng)
    {
        int bsBin = BeatStrengthBin(beatStrength);
        string key = $"{(prevCutDir >= 0 ? prevCutDir : 8)}_{bsBin}_{hand}_{difficultyLevel}";
        // Fall back to difficulty-agnostic key if no data for this difficulty
        if (!_transitions.TryGetValue(key, out var probs))
        {
            string fallback = $"{(prevCutDir >= 0 ? prevCutDir : 8)}_{bsBin}_{hand}_2"; // Hard as default
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

        // Bias rows toward top for bright audio, bottom for bass-heavy audio
        if (spectralCentroid > 0)
        {
            double topBoost    = spectralCentroid > 0.6 ? 1.5 : 1.0;
            double bottomBoost = spectralCentroid < 0.4 ? 1.5 : 1.0;
            for (int lane = 0; lane < 4; lane++)
            {
                probs[lane * 3 + 0] *= topBoost;
                probs[lane * 3 + 2] *= bottomBoost;
            }
            double sum = probs.Sum();
            if (sum > 0)
                for (int i = 0; i < probs.Length; i++) probs[i] /= sum;
        }

        int idx = SampleFromProbs(probs, rng);
        return (idx / 3, idx % 3);
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    public void SaveModel(string artifactsPath)
    {
        Directory.CreateDirectory(artifactsPath);
        var artifact = new
        {
            LeftHandFraction = _leftHandFraction,
            Transitions      = _transitions,
            Positions        = _positions
        };
        File.WriteAllText(
            Path.Combine(artifactsPath, "attribute_model.json"),
            JsonSerializer.Serialize(artifact, new JsonSerializerOptions { WriteIndented = true }));
    }

    // -----------------------------------------------------------------------

    private static Dictionary<string, double[]> Normalize(Dictionary<string, int[]> raw, int size)
    {
        var result = new Dictionary<string, double[]>(raw.Count);
        foreach (var (key, arr) in raw)
        {
            double total = arr.Sum() + size; // Laplace +1 per bin
            result[key] = arr.Select(c => (c + 1.0) / total).ToArray();
        }
        return result;
    }

    private static int BeatStrengthBin(double s) => s >= 0.75 ? 0 : s >= 0.5 ? 1 : 2;

    private static double[] UniformProbs(int size)
    {
        var arr = new double[size];
        Array.Fill(arr, 1.0 / size);
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

