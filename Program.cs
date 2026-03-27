using PerceptronIris.Data;
using PerceptronIris.Evaluation;
using PerceptronIris.ML;
using PerceptronIris.UI;
using PerceptronIris.Visualization;

namespace PerceptronIris;

internal static class Program
{
    private static void Main()
    {
        string dataPath = Path.Combine(AppContext.BaseDirectory, "Data", "iris.csv");
        string outputDirectory = Path.Combine(AppContext.BaseDirectory, "output");

        var loader = new IrisCsvLoader();
        var splitter = new PrepareDataset();
        var metrics = new EvaluationMetrics();
        var plotter = new SvgPlotter();
        
        var dataset4D = loader.Load(dataPath, 0, 1, 2, 3);

        var split4D = splitter.TrainTestSplit(dataset4D, trainRatio: 0.7, seed: 42);

        double[][] trainInputs4D = split4D.Training
            .Select(item => item.Features)
            .ToArray();

        int[] trainLabels4D = split4D.Training
            .Select(item => item.Label)
            .ToArray();

        double[][] testInputs4D = split4D.Test
            .Select(item => item.Features)
            .ToArray();

        int[] testLabels4D = split4D.Test
            .Select(item => item.Label)
            .ToArray();

        var perceptron4D = new Perceptron(
            dimension: 4,
            initialWeights: new[] { 5.0, 5.0, 5.0, 5.0 },
            initialThreshold: 2.0);

        perceptron4D.Train(trainInputs4D, trainLabels4D, alpha: 0.01, beta: 0.01, maxEpochs: 100);

        int[] predictedTestLabels4D = testInputs4D
            .Select(perceptron4D.Predict)
            .ToArray();

        double testAccuracy4D = metrics.MeasureAccuracy(testLabels4D, predictedTestLabels4D);
        
        var dataset2D = loader.Load(dataPath, 2, 3);

        var split2D = splitter.TrainTestSplit(dataset2D, trainRatio: 0.7, seed: 42);

        double[][] trainInputs2D = split2D.Training
            .Select(item => item.Features)
            .ToArray();

        int[] trainLabels2D = split2D.Training
            .Select(item => item.Label)
            .ToArray();

        var perceptron2D = new Perceptron(
            dimension: 2,
            initialWeights: new[] { 5.0, 5.0 },
            initialThreshold: 2.0);

        perceptron2D.Train(trainInputs2D, trainLabels2D, alpha: 0.01, beta: 0.01, maxEpochs: 100);
        
        Console.WriteLine("=== Podsumowanie eksperymentu ===");
        Console.WriteLine($"Liczba rekordów po odfiltrowaniu virginica: {dataset4D.Count}");
        Console.WriteLine($"Train 4D: {split4D.Training.Count}");
        Console.WriteLine($"Test 4D: {split4D.Test.Count}");
        Console.WriteLine($"Liczba epok modelu 4D: {perceptron4D.EpochsCompleted}");
        Console.WriteLine($"Accuracy testowe modelu 4D: {testAccuracy4D:P2}");
        Console.WriteLine();

        Console.WriteLine("=== Accuracy modelu 4D po epokach ===");
        for (int i = 0; i < perceptron4D.AccuracyHistory.Count; i++)
        {
            Console.WriteLine(
                $"Epoka {i + 1,2}: accuracy = {perceptron4D.AccuracyHistory[i]:P2}, błędy = {perceptron4D.ErrorHistory[i]}");
        }
        
        Directory.CreateDirectory(outputDirectory);

        string accuracyPlotPath = Path.Combine(outputDirectory, "accuracy_4d.svg");
        string decisionBoundaryPath = Path.Combine(outputDirectory, "decision_boundary_2d.svg");

        plotter.ExportAccuracyChart(perceptron4D.AccuracyHistory, accuracyPlotPath);

        plotter.ExportDecisionBoundary(
            split2D.Test,
            perceptron2D.Weights,
            perceptron2D.Threshold,
            decisionBoundaryPath,
            "Petal length",
            "Petal width");

        Console.WriteLine();
        Console.WriteLine("Wygenerowane pliki:");
        Console.WriteLine(accuracyPlotPath);
        Console.WriteLine(decisionBoundaryPath);
        Console.WriteLine();
        
        var ui = new ConsolePredictionUi(
            perceptron4D,
            new[]
            {
                "Sepal length",
                "Sepal width",
                "Petal length",
                "Petal width"
            },
            positiveClassName: "setosa",
            negativeClassName: "versicolor");

        ui.Run();
    }
}