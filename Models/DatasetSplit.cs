namespace PerceptronIris.Models;

public sealed record DatasetSplit(
    IReadOnlyList<Observation> Training,
    IReadOnlyList<Observation> Test
);