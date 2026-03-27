namespace PerceptronIris.ML;

public sealed class Perceptron
{
    private readonly Random _random;

    public int Dimension { get; }
    public double[] Weights { get; }
    public double Threshold { get; private set; }
    public double Alpha { get; private set; }
    public double Beta { get; private set; }
    public int EpochsCompleted { get; private set; }

    public List<double> AccuracyHistory { get; } = new();
    public List<int> ErrorHistory { get; } = new();

    public Perceptron(int dimension, int seed = 42)
    {
        if (dimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension musi być > 0.");

        Dimension = dimension;
        Weights = new double[dimension];
        _random = new Random(seed);

        for (int i = 0; i < dimension; i++)
        {
            Weights[i] = _random.NextDouble() - 0.5;
        }

        Threshold = _random.NextDouble() - 0.5;
    }

    public void Train(double[][] inputs, int[] labels, double alpha, double beta, int maxEpochs = 1000)
    {
        ValidateTrainingData(inputs, labels);

        if (alpha <= 0)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha musi być > 0.");

        if (beta <= 0)
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta musi być > 0.");

        if (maxEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxEpochs), "maxEpochs musi być > 0.");

        Alpha = alpha;
        Beta = beta;
        EpochsCompleted = 0;
        AccuracyHistory.Clear();
        ErrorHistory.Clear();

        for (int epoch = 1; epoch <= maxEpochs; epoch++)
        {
            int errors = 0;

            for (int sampleIndex = 0; sampleIndex < inputs.Length; sampleIndex++)
            {
                int predicted = Predict(inputs[sampleIndex]);
                int error = labels[sampleIndex] - predicted;

                if (error == 0)
                    continue;

                errors++;

                for (int featureIndex = 0; featureIndex < Dimension; featureIndex++)
                {
                    Weights[featureIndex] += Alpha * error * inputs[sampleIndex][featureIndex];
                }

                //treshold - (w*x)
                Threshold -= Beta * error;
            }

            AccuracyHistory.Add(CalculateAccuracy(inputs, labels));
            ErrorHistory.Add(errors);
            EpochsCompleted = epoch;

            if (errors == 0)
                break;
        }
    }
    
    public void train(double[][] inputs, int[] labels, double alpha, double beta, int maxEpochs = 1000)
    {
        Train(inputs, labels, alpha, beta, maxEpochs);
    }

    public int Predict(double[] inputs)
    {
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        if (inputs.Length != Dimension)
            throw new ArgumentException("Nieprawidłowy wymiar wejścia.", nameof(inputs));

        double activation = 0.0;

        for (int i = 0; i < Dimension; i++)
        {
            activation += Weights[i] * inputs[i];
        }

        activation -= Threshold;

        return activation >= 0.0 ? 1 : 0;
    }
    
    public int predict(double[] inputs)
    {
        return Predict(inputs);
    }

    private void ValidateTrainingData(double[][] inputs, int[] labels)
    {
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        if (labels == null)
            throw new ArgumentNullException(nameof(labels));

        if (inputs.Length == 0 || labels.Length == 0)
            throw new ArgumentException("Dane treningowe i etykiety nie mogą być puste.");

        if (inputs.Length != labels.Length)
            throw new ArgumentException("Liczba obserwacji musi być równa liczbie etykiet.");

        foreach (double[] vector in inputs)
        {
            if (vector.Length != Dimension)
                throw new ArgumentException("Każdy wektor wejściowy musi mieć wymiar zgodny z perceptronem.");
        }
    }

    private double CalculateAccuracy(double[][] inputs, int[] labels)
    {
        int correct = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            if (Predict(inputs[i]) == labels[i])
            {
                correct++;
            }
        }

        return correct / (double)inputs.Length;
    }
}