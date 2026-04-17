namespace BeatSaber.AutoMapper.Generation;

/// <summary>
/// Samples note attributes (cut direction, lane, row) given context.
/// Implemented by both the in-memory trainer and the loaded AttributeModel.
/// </summary>
public interface IAttributeModel
{
    /// <summary>
    /// Sample a cut direction for the next note.
    /// </summary>
    /// <param name="prevCutDir">Previous cut direction for this hand (int cast of CutDirection, or -1 if first note).</param>
    /// <param name="beatStrength">Beat strength 0-1 (1=downbeat).</param>
    /// <param name="hand">0=left, 1=right.</param>
    /// <param name="difficultyLevel">Difficulty level 0=Easy … 4=ExpertPlus.</param>
    /// <param name="rng">Random source.</param>
    CutDirection SampleCutDirection(int prevCutDir, double beatStrength, int hand,
                                    int difficultyLevel, Random rng);

    /// <summary>
    /// Sample a (lane, row) position for the next note, optionally biased by spectral centroid.
    /// </summary>
    /// <param name="cutDir">The resolved cut direction (int cast of CutDirection).</param>
    /// <param name="hand">0=left, 1=right.</param>
    /// <param name="beatStrength">Beat strength 0-1.</param>
    /// <param name="spectralCentroid">Normalised spectral centroid 0-1 (0=no signal), higher = brighter.</param>
    /// <param name="difficultyLevel">Difficulty level 0=Easy … 4=ExpertPlus.</param>
    /// <param name="rng">Random source.</param>
    (int Lane, int Row) SamplePosition(int cutDir, int hand, double beatStrength,
                                       double spectralCentroid, int difficultyLevel, Random rng);
}
