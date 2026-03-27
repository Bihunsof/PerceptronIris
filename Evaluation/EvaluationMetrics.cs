namespace PerceptronIris.Evaluation;

public sealed class EvaluationMetrics
{
    public double MeasureAccuracy(IReadOnlyList<int> realClasses, IReadOnlyList<int> predictedClasses)
    {
        if (realClasses == null)
            throw new ArgumentNullException(nameof(realClasses));

        if (predictedClasses == null)
            throw new ArgumentNullException(nameof(predictedClasses));

        if (realClasses.Count == 0 || predictedClasses.Count == 0)
            throw new ArgumentException("Listy klas nie mogą być puste.");

        if (realClasses.Count != predictedClasses.Count)
            throw new ArgumentException("Listy muszą mieć ten sam rozmiar.");

        int correct = 0;

        for (int i = 0; i < realClasses.Count; i++)
        {
            if (realClasses[i] == predictedClasses[i])
            {
                correct++;
            }
        }

        return correct / (double)realClasses.Count;
    }

    public double measureAccuracy(IReadOnlyList<int> realClasses, IReadOnlyList<int> predictedClasses)
    {
        return MeasureAccuracy(realClasses, predictedClasses);
    }
}