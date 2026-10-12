using Godot;
using System;

[GlobalClass]
public partial class SliceSectorDefinition : Resource
{
    [Export] public string SectorName = "New Sector";
    
    [ExportGroup("Sector Bounds (Local X/Y from Chest)")]
    [Export] public Vector2 MinBounds = new Vector2(-1.0f, -1.0f);
    [Export] public Vector2 MaxBounds = new Vector2(1.0f, 1.0f);

    [ExportGroup("Blocked Slice Angles (Degrees)")]
    // Angles: 0 is Right, 90 is Up, 180/-180 is Left, -90 is Down.
    // E.g. (-180, 0) blocks all downward slices.
    [Export] public Godot.Collections.Array<Vector2> BlockedAngleRanges = new Godot.Collections.Array<Vector2>();

    [ExportGroup("Mode Overrides")]
    [Export] public int DefaultIKMode = 1;
    [Export] public Godot.Collections.Array<SliceModeOverride> ModeOverrides = new Godot.Collections.Array<SliceModeOverride>();

    public bool IsPointInSector(Vector2 localPoint)
    {
        return localPoint.X >= MinBounds.X && localPoint.X <= MaxBounds.X &&
               localPoint.Y >= MinBounds.Y && localPoint.Y <= MaxBounds.Y;
    }

    public bool IsSliceBlocked(float angleDegrees)
    {
        float normalizedAngle = angleDegrees;
        if (normalizedAngle < 0) normalizedAngle += 360f;

        if (BlockedAngleRanges == null) return false;

        foreach (var range in BlockedAngleRanges)
        {
            float min = range.X;
            float max = range.Y;
            if (min < 0) min += 360f;
            if (max < 0) max += 360f;

            if (min > max)
            {
                if (normalizedAngle >= min || normalizedAngle <= max) return true;
            }
            else
            {
                if (normalizedAngle >= min && normalizedAngle <= max) return true;
            }
        }
        return false;
    }

    public int GetModeForSlice(float angleDegrees)
    {
        float normalizedAngle = angleDegrees;
        if (normalizedAngle < 0) normalizedAngle += 360f;

        if (ModeOverrides != null)
        {
            foreach (var over in ModeOverrides)
            {
                if (over == null) continue;
                float min = over.AngleRange.X;
                float max = over.AngleRange.Y;
                if (min < 0) min += 360f;
                if (max < 0) max += 360f;
                
                if (min > max)
                {
                    if (normalizedAngle >= min || normalizedAngle <= max) return over.IKMode;
                }
                else
                {
                    if (normalizedAngle >= min && normalizedAngle <= max) return over.IKMode;
                }
            }
        }
        return DefaultIKMode;
    }
}

[GlobalClass]
public partial class SliceModeOverride : Resource
{
    [Export] public string OverrideName = "Mode Override";
    [Export] public Vector2 AngleRange = new Vector2(0, 90);
    [Export] public int IKMode = 2;
}
