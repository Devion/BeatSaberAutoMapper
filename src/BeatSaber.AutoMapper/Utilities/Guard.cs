namespace BeatSaber.AutoMapper.Utilities;

public static class Guard
{
    public static T NotNull<T>(T? value, string name) where T : class
    {
        if (value is null)
            throw new ArgumentNullException(name);
        return value;
    }

    public static string NotNullOrEmpty(string? value, string name)
    {
        if (string.IsNullOrEmpty(value))
            throw new ArgumentException($"'{name}' must not be null or empty.", name);
        return value;
    }

    public static int InRange(int value, int min, int max, string name)
    {
        if (value < min || value > max)
            throw new ArgumentOutOfRangeException(name, value, $"'{name}' must be between {min} and {max}.");
        return value;
    }
}
