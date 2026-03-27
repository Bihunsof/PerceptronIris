using System.Globalization;
using PerceptronIris.Models;

namespace PerceptronIris.Data;

public sealed class IrisCsvLoader
{
    public List<Observation> Load(string path, params int[] featureIndexes)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Ścieżka do pliku nie może być pusta.", nameof(path));

        if (!File.Exists(path))
            throw new FileNotFoundException("Nie znaleziono pliku iris.csv.", path);

        if (featureIndexes == null || featureIndexes.Length == 0)
            featureIndexes = new[] { 0, 1, 2, 3 };

        if (featureIndexes.Any(index => index < 0 || index > 3))
            throw new ArgumentOutOfRangeException(nameof(featureIndexes), "Dozwolone indeksy cech: 0..3.");

        var dataset = new List<Observation>();

        foreach (string rawLine in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(rawLine))
                continue;

            string line = rawLine.Trim();
            string[] parts = line.Split(',', StringSplitOptions.TrimEntries);

            if (parts.Length < 5)
                continue;

            if (!double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double sepalLength))
                continue; // np. nagłówek

            double sepalWidth = ParseDouble(parts[1]);
            double petalLength = ParseDouble(parts[2]);
            double petalWidth = ParseDouble(parts[3]);

            string species = NormalizeSpecies(parts[4]);
            int? label = MapLabel(species);

            if (!label.HasValue)
                continue; // ignorujemy virginica

            double[] allFeatures = { sepalLength, sepalWidth, petalLength, petalWidth };
            double[] selectedFeatures = featureIndexes.Select(index => allFeatures[index]).ToArray();

            dataset.Add(new Observation(selectedFeatures, label.Value, species));
        }

        return dataset;
    }

    private static double ParseDouble(string value)
    {
        return double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    private static string NormalizeSpecies(string species)
    {
        return species.Trim().ToLowerInvariant() switch
        {
            "iris-setosa" => "setosa",
            "iris-versicolor" => "versicolor",
            "iris-virginica" => "virginica",
            var other => other
        };
    }

    private static int? MapLabel(string species)
    {
        return species switch
        {
            "setosa" => 1,
            "versicolor" => 0,
            "virginica" => null,
            _ => null
        };
    }
}