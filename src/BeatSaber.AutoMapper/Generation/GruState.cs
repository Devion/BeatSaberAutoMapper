namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Mutable GRU hidden state passed through the generation pipeline.
/// The scorer reads and updates <see cref="H"/> in-place at each beat position,
/// allowing the GRU to carry temporal context across the full song.
/// One instance per song/generation pass — not thread-safe; create one per parallel task.
/// </summary>
public sealed class GruState
{
    /// <summary>Flattened hidden state: [NumLayers × HiddenDim] in row-major order.</summary>
    public float[] H { get; }

    public GruState(int numLayers, int hiddenDim)
    {
        H = new float[numLayers * hiddenDim];
    }

    private GruState(float[] state) => H = state;

    /// <summary>Reset hidden state to zeros (start of a new song).</summary>
    public void Reset() => Array.Clear(H, 0, H.Length);

    public GruState Clone() => new((float[])H.Clone());
}
