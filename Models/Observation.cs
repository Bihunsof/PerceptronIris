namespace PerceptronIris.Models;

public sealed record Observation(double[] Features, int Label, string Species);