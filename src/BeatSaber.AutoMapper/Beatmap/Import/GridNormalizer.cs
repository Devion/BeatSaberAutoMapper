namespace BeatSaber.AutoMapper.Beatmap.Import;

internal readonly record struct GridNormalizer(
    int MinLane,
    int MaxLane,
    int MinRow,
    int MaxRow)
{
    public static GridNormalizer FromCoordinates(IEnumerable<(int Lane, int Row)> coordinates)
    {
        int minLane = 0, maxLane = 3, minRow = 0, maxRow = 2;
        bool any = false;

        foreach (var (lane, row) in coordinates)
        {
            if (!any)
            {
                minLane = maxLane = lane;
                minRow = maxRow = row;
                any = true;
                continue;
            }

            minLane = Math.Min(minLane, lane);
            maxLane = Math.Max(maxLane, lane);
            minRow = Math.Min(minRow, row);
            maxRow = Math.Max(maxRow, row);
        }

        if (!any)
            return new GridNormalizer(0, 3, 0, 2);

        return new GridNormalizer(minLane, maxLane, minRow, maxRow);
    }

    public int NormalizeLane(int lane) => NormalizeAxis(lane, MinLane, MaxLane, 4);

    public int NormalizeRow(int row) => NormalizeAxis(row, MinRow, MaxRow, 3);

    public (int Lane, int Row) NormalizeCell(int lane, int row) =>
        (NormalizeLane(lane), NormalizeRow(row));

    public (int Lane, int Width) NormalizeLaneSpan(int lane, int width)
    {
        int safeWidth = Math.Max(1, width);
        int start = NormalizeLane(lane);
        int end = NormalizeLane(lane + safeWidth - 1);
        if (end < start)
            (start, end) = (end, start);
        return (start, Math.Max(1, end - start + 1));
    }

    private static int NormalizeAxis(int value, int min, int max, int targetSize)
    {
        if (targetSize <= 1)
            return 0;
        if (min >= 0 && max < targetSize)
            return Math.Clamp(value, 0, targetSize - 1);
        if (max <= min)
            return targetSize / 2;

        double normalized = (value - min) / (double)(max - min);
        int mapped = (int)Math.Round(normalized * (targetSize - 1), MidpointRounding.AwayFromZero);
        return Math.Clamp(mapped, 0, targetSize - 1);
    }
}
