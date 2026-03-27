using PerceptronIris.Data;
using PerceptronIris.Evaluation;
using PerceptronIris.ML;
using PerceptronIris.UI;

namespace PerceptronIris;

internal static class Program
{
    private static void Main()
    {
        string dataPath = Path.Combine(AppContext.BaseDirectory, "Data", "iris.csv");

        var loader = new IrisCsvLoader();
        
        var dataset = loader.Load(dataPath, 2, 3);

        var splitter = new PrepareDataset();
        var split = splitter.TrainTestSplit(dataset, trainRatio: 0.7, seed: 42);

        double[][] trainInputs = split.Training.Select(item => item.Features).ToArray();
        int[] trainLabels = split.Training.Select(item => item.Label).ToArray();

        double[][] testInputs = split.Test.Select(item => item.Features).ToArray();
        int[] testLabels = split.Test.Select(item => item.Label).ToArray();

        var perceptron = new Perceptron(dimension: 2, seed: 42);
        perceptron.Train(trainInputs, trainLabels, alpha: 0.1, beta: 0.1, maxEpochs: 100);

        int[] predictedTestLabels = testInputs
            .Select(perceptron.Predict)
            .ToArray();

        var metrics = new EvaluationMetrics();
        double testAccuracy = metrics.MeasureAccuracy(testLabels, predictedTestLabels);

        Console.WriteLine("=== Podsumowanie eksperymentu ===");
        Console.WriteLine($"Liczba rekordów po odfiltrowaniu virginica: {dataset.Count}");
        Console.WriteLine($"Train: {split.Training.Count}");
        Console.WriteLine($"Test: {split.Test.Count}");
        Console.WriteLine($"Liczba epok: {perceptron.EpochsCompleted}");
        Console.WriteLine($"Accuracy testowe: {testAccuracy:P2}");
        Console.WriteLine();

        Console.WriteLine("=== Accuracy po epokach ===");
        for (int i = 0; i < perceptron.AccuracyHistory.Count; i++)
        {
            Console.WriteLine(
                $"Epoka {i + 1,2}: accuracy = {perceptron.AccuracyHistory[i]:P2}, błędy = {perceptron.ErrorHistory[i]}");
        }

        Console.WriteLine();

        var ui = new ConsolePredictionUi(perceptron);
        ui.Run();
    }
}