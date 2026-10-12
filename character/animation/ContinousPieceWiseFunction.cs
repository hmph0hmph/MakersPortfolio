using Godot;
using System;
using System.Collections.Generic;

[GlobalClass]
public partial class ContinousPieceWiseFunction : Resource
{
    [Export] public float[] PointHeights;       // Must have N+1 values for N segments
    [Export] public int[] IsConcaveUp;    // N values; 0 = ease-out segment (vertex at the higher point), 1 = ease-in segment
    [Export] public float[] SegmentLengths; // Relative lengths (normalized in SetUp); length N

    private bool setUp = false;
    float TotalSegmentLengths;
    private float[] normalizedSegmentLengthA;
    private float[] normalizedSegmentLengthB;

    public void SetUp()
    {
        setUp = true;
        int count = SegmentLengths.Length;
        normalizedSegmentLengthA = new float[count];
        normalizedSegmentLengthB = new float[count];

        for (int i = 0; i < SegmentLengths.Length; i++)
        {
            TotalSegmentLengths += SegmentLengths[i];
        }

        float cumulative = 0f;
        for (int i = 0; i < SegmentLengths.Length; i++)
        {
            normalizedSegmentLengthA[i] = SegmentLengths[i] / TotalSegmentLengths;
            cumulative += normalizedSegmentLengthA[i];
            normalizedSegmentLengthB[i] = cumulative;
        }
    }

    public float InOutNormalized(float normalizedX)
    {
        if (!setUp)
        {
            SetUp();
        }
        float X = Mathf.Clamp(normalizedX, 0f, 1f);
        int PointBPointer = 0;
        for (int i = 0; i < normalizedSegmentLengthB.Length; i++)
        {
            if (normalizedSegmentLengthB[i] >= X)
            {
                PointBPointer = i;
                break;
            }
        }
        float temp = 0;
        if (PointBPointer != 0)
        {
            temp = normalizedSegmentLengthB[PointBPointer - 1];
        }
        Vector2 pointA = new Vector2(temp, PointHeights[PointBPointer]);
        Vector2 pointB = new Vector2(normalizedSegmentLengthB[PointBPointer], PointHeights[PointBPointer + 1]);
        return Utilities.MathFunctions.EvaluateHalfParabola(pointA, pointB, X, IsConcaveUp[PointBPointer] == 0);
    }
    public float InOut(float X, float minX, float maxX)
    {
        return InOutNormalized(Mathf.InverseLerp(minX, maxX, X));
    }
    public float getNormalizedA(int index)
    {
        return normalizedSegmentLengthA[index];
    }
    public float getNormalizedB(int index)
    {
        return normalizedSegmentLengthB[index];
    }
}
