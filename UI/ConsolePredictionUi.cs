using System.Globalization;
using PerceptronIris.ML;

namespace PerceptronIris.UI;

public sealed class ConsolePredictionUi
{
    private readonly Perceptron _model;

    public ConsolePredictionUi(Perceptron model)
    {
        _model = model;
    }

    public void Run()
    {
        Console.WriteLine("=== Tryb predykcji użytkownika ===");
        Console.WriteLine("Podawaj 2 cechy w tej kolejności:");
        Console.WriteLine("1. petal length");
        Console.WriteLine("2. petal width");
        Console.WriteLine("Wpisz q aby zakończyć.");
        Console.WriteLine();

        while (true)
        {
            Console.Write("Petal length (q = koniec): ");
            string? first = Console.ReadLine();

            if (string.Equals(first, "q", StringComparison.OrdinalIgnoreCase))
                break;

            if (!TryParseNumber(first, out double petalLength))
            {
                Console.WriteLine("Błędna liczba.");
                Console.WriteLine();
                continue;
            }

            Console.Write("Petal width: ");
            string? second = Console.ReadLine();

            if (!TryParseNumber(second, out double petalWidth))
            {
                Console.WriteLine("Błędna liczba.");
                Console.WriteLine();
                continue;
            }

            int prediction = _model.Predict(new[] { petalLength, petalWidth });
            string species = prediction == 1 ? "setosa" : "versicolor";

            Console.WriteLine($"Predykcja: {species} ({prediction})");
            Console.WriteLine();
        }
    }

    private static bool TryParseNumber(string? text, out double value)
    {
        value = 0.0;

        if (string.IsNullOrWhiteSpace(text))
            return false;

        string normalized = text.Trim().Replace(',', '.');

        return double.TryParse(
            normalized,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out value);
    }
}