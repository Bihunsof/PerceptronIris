using System.Globalization;
using System.Text;
using PerceptronIris.Models;

namespace PerceptronIris.Visualization;

public sealed class SvgPlotter
{
    private const int Width = 900;
    private const int Height = 600;
    private const int MarginLeft = 90;
    private const int MarginRight = 40;
    private const int MarginTop = 50;
    private const int MarginBottom = 80;

    public void ExportAccuracyChart(IReadOnlyList<double> accuracies, string outputPath)
    {
        if (accuracies == null || accuracies.Count == 0)
            throw new ArgumentException("Lista accuracy nie może być pusta.", nameof(accuracies));

        EnsureDirectory(outputPath);

        double minX = 1;
        double maxX = accuracies.Count == 1 ? 2 : accuracies.Count;
        double minY = 0;
        double maxY = 1;

        var sb = new StringBuilder();
        AppendSvgStart(sb, "Accuracy vs Epoch");

        AppendAxesAndGrid(sb, minX, maxX, minY, maxY, "Epoka", "Accuracy");

        var polylinePoints = new List<string>();

        for (int i = 0; i < accuracies.Count; i++)
        {
            double epoch = i + 1;
            double x = MapX(epoch, minX, maxX);
            double y = MapY(accuracies[i], minY, maxY);
            polylinePoints.Add($"{F(x)},{F(y)}");
        }

        sb.AppendLine(
            $"<polyline fill='none' stroke='#1f77b4' stroke-width='3' points='{string.Join(" ", polylinePoints)}' />");

        for (int i = 0; i < accuracies.Count; i++)
        {
            double epoch = i + 1;
            double x = MapX(epoch, minX, maxX);
            double y = MapY(accuracies[i], minY, maxY);

            sb.AppendLine($"<circle cx='{F(x)}' cy='{F(y)}' r='4' fill='#1f77b4' />");
            sb.AppendLine(
                $"<text x='{F(x + 6)}' y='{F(y - 8)}' font-size='11' fill='#333'>{F(accuracies[i] * 100)}%</text>");
        }

        AppendSvgEnd(sb);
        File.WriteAllText(outputPath, sb.ToString());
    }

    public void ExportDecisionBoundary(
        IReadOnlyList<Observation> testSet,
        double[] weights,
        double threshold,
        string outputPath,
        string xLabel,
        string yLabel)
    {
        if (testSet == null || testSet.Count == 0)
            throw new ArgumentException("Zbiór testowy nie może być pusty.", nameof(testSet));

        if (weights == null || weights.Length != 2)
            throw new ArgumentException("Wizualizacja granicy decyzyjnej wymaga dokładnie 2 wag.", nameof(weights));

        EnsureDirectory(outputPath);

        double rawMinX = testSet.Min(item => item.Features[0]);
        double rawMaxX = testSet.Max(item => item.Features[0]);
        double rawMinY = testSet.Min(item => item.Features[1]);
        double rawMaxY = testSet.Max(item => item.Features[1]);

        double paddingX = (rawMaxX - rawMinX) * 0.15;
        double paddingY = (rawMaxY - rawMinY) * 0.15;

        if (paddingX == 0) paddingX = 1;
        if (paddingY == 0) paddingY = 1;

        double minX = rawMinX - paddingX;
        double maxX = rawMaxX + paddingX;
        double minY = rawMinY - paddingY;
        double maxY = rawMaxY + paddingY;

        var sb = new StringBuilder();
        AppendSvgStart(sb, "Decision Boundary");

        AppendAxesAndGrid(sb, minX, maxX, minY, maxY, xLabel, yLabel);

        foreach (Observation item in testSet)
        {
            string color = item.Label == 1 ? "#2ca02c" : "#d62728";
            double x = MapX(item.Features[0], minX, maxX);
            double y = MapY(item.Features[1], minY, maxY);

            sb.AppendLine($"<circle cx='{F(x)}' cy='{F(y)}' r='6' fill='{color}' fill-opacity='0.85' />");
        }

        var linePoints = GetDecisionBoundarySegment(weights, threshold, minX, maxX, minY, maxY);

        if (linePoints != null)
        {
            var (p1, p2) = linePoints.Value;
            double x1 = MapX(p1.X, minX, maxX);
            double y1 = MapY(p1.Y, minY, maxY);
            double x2 = MapX(p2.X, minX, maxX);
            double y2 = MapY(p2.Y, minY, maxY);

            sb.AppendLine(
                $"<line x1='{F(x1)}' y1='{F(y1)}' x2='{F(x2)}' y2='{F(y2)}' stroke='#111' stroke-width='3' />");
        }

        AppendLegend(sb);
        AppendSvgEnd(sb);

        File.WriteAllText(outputPath, sb.ToString());
    }

    private static (Point P1, Point P2)? GetDecisionBoundarySegment(
        double[] weights,
        double threshold,
        double minX,
        double maxX,
        double minY,
        double maxY)
    {
        const double eps = 1e-10;
        var points = new List<Point>();

        double w1 = weights[0];
        double w2 = weights[1];

        if (Math.Abs(w2) > eps)
        {
            double yAtMinX = (threshold - w1 * minX) / w2;
            double yAtMaxX = (threshold - w1 * maxX) / w2;

            if (yAtMinX >= minY && yAtMinX <= maxY)
                points.Add(new Point(minX, yAtMinX));

            if (yAtMaxX >= minY && yAtMaxX <= maxY)
                points.Add(new Point(maxX, yAtMaxX));
        }

        if (Math.Abs(w1) > eps)
        {
            double xAtMinY = (threshold - w2 * minY) / w1;
            double xAtMaxY = (threshold - w2 * maxY) / w1;

            if (xAtMinY >= minX && xAtMinY <= maxX)
                points.Add(new Point(xAtMinY, minY));

            if (xAtMaxY >= minX && xAtMaxY <= maxX)
                points.Add(new Point(xAtMaxY, maxY));
        }

        var uniquePoints = points
            .GroupBy(p => $"{Math.Round(p.X, 6)}|{Math.Round(p.Y, 6)}")
            .Select(g => g.First())
            .ToList();

        if (uniquePoints.Count < 2)
            return null;

        return (uniquePoints[0], uniquePoints[1]);
    }

    private static void AppendSvgStart(StringBuilder sb, string title)
    {
        sb.AppendLine($"<svg xmlns='http://www.w3.org/2000/svg' width='{Width}' height='{Height}'>");
        sb.AppendLine("<rect width='100%' height='100%' fill='white' />");
        sb.AppendLine(
            $"<text x='{Width / 2}' y='28' text-anchor='middle' font-size='22' font-family='Arial' fill='#111'>{title}</text>");
    }

    private static void AppendSvgEnd(StringBuilder sb)
    {
        sb.AppendLine("</svg>");
    }

    private static void AppendAxesAndGrid(
        StringBuilder sb,
        double minX,
        double maxX,
        double minY,
        double maxY,
        string xLabel,
        string yLabel)
    {
        int plotLeft = MarginLeft;
        int plotRight = Width - MarginRight;
        int plotTop = MarginTop;
        int plotBottom = Height - MarginBottom;

        sb.AppendLine(
            $"<rect x='{plotLeft}' y='{plotTop}' width='{plotRight - plotLeft}' height='{plotBottom - plotTop}' fill='none' stroke='#aaa' />");

        for (int i = 0; i <= 5; i++)
        {
            double xValue = minX + i * (maxX - minX) / 5.0;
            double yValue = minY + i * (maxY - minY) / 5.0;

            double x = MapX(xValue, minX, maxX);
            double y = MapY(yValue, minY, maxY);

            sb.AppendLine(
                $"<line x1='{F(x)}' y1='{plotTop}' x2='{F(x)}' y2='{plotBottom}' stroke='#eee' stroke-width='1' />");
            sb.AppendLine(
                $"<line x1='{plotLeft}' y1='{F(y)}' x2='{plotRight}' y2='{F(y)}' stroke='#eee' stroke-width='1' />");

            sb.AppendLine(
                $"<text x='{F(x)}' y='{plotBottom + 22}' text-anchor='middle' font-size='12' fill='#444'>{F(xValue)}</text>");
            sb.AppendLine(
                $"<text x='{plotLeft - 12}' y='{F(y + 4)}' text-anchor='end' font-size='12' fill='#444'>{F(yValue)}</text>");
        }

        sb.AppendLine(
            $"<text x='{(plotLeft + plotRight) / 2}' y='{Height - 20}' text-anchor='middle' font-size='16' fill='#111'>{xLabel}</text>");

        sb.AppendLine(
            $"<text x='28' y='{Height / 2}' text-anchor='middle' font-size='16' fill='#111' transform='rotate(-90 28 {Height / 2})'>{yLabel}</text>");
    }

    private static void AppendLegend(StringBuilder sb)
    {
        int x = Width - 200;
        int y = 70;

        sb.AppendLine($"<rect x='{x}' y='{y}' width='150' height='70' fill='white' stroke='#ccc' />");
        sb.AppendLine($"<circle cx='{x + 18}' cy='{y + 22}' r='6' fill='#2ca02c' />");
        sb.AppendLine($"<text x='{x + 32}' y='{y + 27}' font-size='13' fill='#111'>setosa (1)</text>");
        sb.AppendLine($"<circle cx='{x + 18}' cy='{y + 48}' r='6' fill='#d62728' />");
        sb.AppendLine($"<text x='{x + 32}' y='{y + 53}' font-size='13' fill='#111'>versicolor (0)</text>");
    }

    private static double MapX(double value, double minX, double maxX)
    {
        int plotLeft = MarginLeft;
        int plotRight = Width - MarginRight;

        return plotLeft + (value - minX) / (maxX - minX) * (plotRight - plotLeft);
    }

    private static double MapY(double value, double minY, double maxY)
    {
        int plotTop = MarginTop;
        int plotBottom = Height - MarginBottom;

        return plotBottom - (value - minY) / (maxY - minY) * (plotBottom - plotTop);
    }

    private static void EnsureDirectory(string outputPath)
    {
        string? directory = Path.GetDirectoryName(outputPath);

        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }
    }

    private static string F(double value)
    {
        return value.ToString("0.###", CultureInfo.InvariantCulture);
    }

    private readonly record struct Point(double X, double Y);
}