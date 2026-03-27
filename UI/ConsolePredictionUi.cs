using System.Globalization;
using PerceptronIris.ML;

namespace PerceptronIris.UI;

public sealed class ConsolePredictionUi
{
    private readonly Perceptron _model;
    private readonly string[] _featureNames;
    private readonly string _positiveClassName;
    private readonly string _negativeClassName;

    public ConsolePredictionUi(
        Perceptron model,
        IReadOnlyList<string> featureNames,
        string positiveClassName,
        string negativeClassName)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _positiveClassName = positiveClassName ?? throw new ArgumentNullException(nameof(positiveClassName));
        _negativeClassName = negativeClassName ?? throw new ArgumentNullException(nameof(negativeClassName));

        if (featureNames == null || featureNames.Count == 0)
            throw new ArgumentException("Lista nazw cech nie może być pusta.", nameof(featureNames));

        if (featureNames.Count != model.Dimension)
            throw new ArgumentException("Liczba nazw cech musi być zgodna z dimension perceptronu.", nameof(featureNames));

        _featureNames = featureNames.ToArray();
    }

    public void Run()
    {
        Console.WriteLine("=== Tryb predykcji użytkownika ===");
        Console.WriteLine($"Model oczekuje {_model.Dimension} cech.");
        Console.WriteLine("Wpisz q w dowolnym momencie, aby zakończyć.");
        Console.WriteLine();

        while (true)
        {
            double[] input = new double[_model.Dimension];

            for (int i = 0; i < _model.Dimension; i++)
            {
                while (true)
                {
                    Console.Write($"{_featureNames[i]} (q = koniec): ");
                    string? text = Console.ReadLine();

                    if (string.Equals(text, "q", StringComparison.OrdinalIgnoreCase))
                        return;

                    if (TryParseNumber(text, out double value))
                    {
                        input[i] = value;
                        break;
                    }

                    Console.WriteLine("Błędna liczba. Spróbuj ponownie.");
                }
            }

            int prediction = _model.Predict(input);
            string species = prediction == 1 ? _positiveClassName : _negativeClassName;

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