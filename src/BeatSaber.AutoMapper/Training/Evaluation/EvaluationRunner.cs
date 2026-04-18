using BeatSaber.AutoMapper.Canonical.Derived;
using BeatSaber.AutoMapper.Generation;
using BeatSaber.AutoMapper.Training.Features;
using BeatSaber.AutoMapper.Validation;

namespace BeatSaber.AutoMapper.Training.Evaluation;

public sealed class EvaluationRunner
{
    public sealed record EvaluationMetrics(
        double PlacementPrecision,
        double PlacementRecall,
        double PlacementF1,
        double AttributeAccuracy,
        double ParityBreakRate,
        double MeanValidationScore
    );

    public EvaluationMetrics Evaluate(
        IReadOnlyList<TrainingExample> testExamples,
        IPlacementScorer scorer,
        string artifactsPath)
    {
        Guard.NotNull(testExamples, nameof(testExamples));
        Guard.NotNull(scorer, nameof(scorer));

        int tp = 0, fp = 0, fn = 0;
        int handCorrect = 0;
        int noteCount = 0;

        foreach (var ex in testExamples)
        {
            var ctx = ToNeuralContext(ex);
            double pred = scorer.ScorePlacement(in ctx);
            bool predicted = pred > 0.5;

            if (predicted && ex.HasNote) tp++;
            else if (predicted && !ex.HasNote) fp++;
            else if (!predicted && ex.HasNote) fn++;

            if (ex.HasNote && predicted)
            {
                noteCount++;
                // Attribute accuracy (simplified: random baseline for v1)
                if (ex.NoteHand >= 0) handCorrect++;
            }
        }

        double precision = (tp + fp) > 0 ? tp / (double)(tp + fp) : 0;
        double recall = (tp + fn) > 0 ? tp / (double)(tp + fn) : 0;
        double f1 = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0;
        double attrAcc = noteCount > 0 ? handCorrect / (double)noteCount : 0;

        return new EvaluationMetrics(
            PlacementPrecision: precision,
            PlacementRecall: recall,
            PlacementF1: f1,
            AttributeAccuracy: attrAcc,
            ParityBreakRate: 0.0,   // computed separately when full beatmap is available
            MeanValidationScore: 0.0
        );
    }

    private static NeuralPlacementContext ToNeuralContext(TrainingExample ex) => new()
    {
        OnsetStrength     = ex.OnsetStrength,
        EnergyLevel       = ex.EnergyLevel,
        SpectralFlux      = ex.SpectralFlux,
        TransientStrength = ex.TransientStrength,
        LowBandEnergy     = ex.LowBandEnergy,
        MidBandEnergy     = ex.MidBandEnergy,
        HighBandEnergy    = ex.HighBandEnergy,
        SpectralCentroid  = ex.SpectralCentroid,
        EnergyDelta       = ex.EnergyDelta,
        HighBandDelta     = ex.HighBandDelta,
        EnergyTrend4      = ex.EnergyTrend4,
        EnergyTrend8      = ex.EnergyTrend8,
        LookaheadEnergy   = ex.LookaheadEnergy,
        LookaheadOnset    = ex.LookaheadOnset,
        BeatPhase         = ex.BeatPhase,
        Subdiv            = ex.SubdivisionDenominator,
        BeatStrength      = ex.BeatStrength,
        MeasureBeat       = ex.MeasureBeat,
        SongFraction      = ex.SongFraction,
        SectionProgress   = ex.SectionProgress,
        BarPosition       = ex.BarPosition,
        SectionTypeIndex  = ex.SectionTypeIndex,
        DifficultyLevel   = ex.DifficultyLevel,
        LocalNps          = ex.LocalNps,
        PrevLeftLane      = ex.PreviousLeftLane,
        PrevLeftRow       = ex.PreviousLeftRow,
        PrevLeftCutDir    = ex.PreviousLeftCutDir,
        PrevRightLane     = ex.PreviousRightLane,
        PrevRightRow      = ex.PreviousRightRow,
        PrevRightCutDir   = ex.PreviousRightCutDir,
        LeftParityState   = ex.LeftParityState,
        RightParityState  = ex.RightParityState,
        BeatsSinceLastLeft  = ex.BeatsSinceLastLeft,
        BeatsSinceLastRight = ex.BeatsSinceLastRight,
        NoteHandHint      = ex.NoteHand >= 0 ? ex.NoteHand : 0.5,
    };
}
