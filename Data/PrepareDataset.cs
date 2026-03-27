using PerceptronIris.Models;

namespace PerceptronIris.Data;

public sealed class PrepareDataset
{
    public DatasetSplit TrainTestSplit(IReadOnlyList<Observation> dataset, double trainRatio = 0.7, int seed = 42)
    {
        if (dataset == null || dataset.Count == 0)
            throw new ArgumentException("Dataset nie może być pusty.", nameof(dataset));

        if (trainRatio <= 0 || trainRatio >= 1)
            throw new ArgumentOutOfRangeException(nameof(trainRatio), "trainRatio musi należeć do przedziału (0,1).");

        var random = new Random(seed);
        var training = new List<Observation>();
        var test = new List<Observation>();

        foreach (var group in dataset.GroupBy(item => item.Label))
        {
            var classItems = group.ToList();
            Shuffle(classItems, random);

            int trainCount = (int)Math.Round(classItems.Count * trainRatio, MidpointRounding.AwayFromZero);

            training.AddRange(classItems.Take(trainCount));
            test.AddRange(classItems.Skip(trainCount));
        }

        Shuffle(training, random);
        Shuffle(test, random);

        return new DatasetSplit(training, test);
    }

    public DatasetSplit trainTestSplit(IReadOnlyList<Observation> dataset)
    {
        return TrainTestSplit(dataset);
    }

    private static void Shuffle<T>(IList<T> items, Random random)
    {
        for (int i = items.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }
}