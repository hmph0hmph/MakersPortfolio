using Godot;
using System;

public class PlaneClass
{
    public Vector3 Normal { get; private set; }
    public Vector3 PointOnPlane { get; private set; }
    public Vector3? SliceDirection { get; private set; }
    public float UpMagnitude { get; private set; }

    public PlaneClass(Vector3 normal, Vector3 point, Vector3? direction = null, float upMagnitude = 1f)
    {
        Normal = normal.Normalized();
        PointOnPlane = point;
        UpMagnitude = upMagnitude;
        
        if (direction.HasValue)
        {
            // Project direction vector strictly onto the plane to ensure perfectly coplanar evaluations
            Vector3 dir = direction.Value;
            Vector3 projectedDir = dir - (Normal * dir.Dot(Normal));
            if (projectedDir.LengthSquared() < 0.0001f)
            {
                SliceDirection = null;
            }
            else
            {
                SliceDirection = projectedDir.Normalized();
            }
        }
        else
        {
            SliceDirection = null;
        }
    }

    // --- Mathematical Utilities ---

    /// <summary>
    /// Projects a directional physics vector (e.g., velocity) identically along the plane.
    /// (Skips planar offset computation, purely flattens direction).
    /// </summary>
    public Vector3 ProjectVector(Vector3 vector)
    {
        return vector - (Normal * vector.Dot(Normal));
    }

    /// <summary>
    /// Draws visual debug coordinates representing the exact target orientation on screen.
    /// </summary>
    public void DrawDebugGizmos()
    {
        Vector3 targetUp = (Normal * UpMagnitude).Normalized();
        // Drawing a cyan line plotting exactly where the palm should technically point globally
        Utilities.Debug.DrawLine(PointOnPlane, PointOnPlane + (targetUp * 0.5f), Colors.Cyan);
    }
}
