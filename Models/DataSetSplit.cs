namespace PerceptronIris.Models;

public sealed record DataSetSplit(
    IReadOnlyList<Observation> Training,
    IReadOnlyList<Observation Test);