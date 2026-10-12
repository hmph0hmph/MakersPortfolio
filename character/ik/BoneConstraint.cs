using Godot;

[GlobalClass]
public partial class BoneConstraint : Resource
{
    [ExportGroup("Rotation Limits (Degrees)")]
    [Export(PropertyHint.Range, "-180,180")]
    public float MinYaw = -90;  // rotation about Y (left/right)

    [Export(PropertyHint.Range, "-180,180")]
    public float MaxYaw = 90;

    [Export(PropertyHint.Range, "-180,180")]
    public float MinPitch = -90;   // elevation (up/down)

    [Export(PropertyHint.Range, "-180,180")]
    public float MaxPitch = 90;

    [ExportGroup("Constraint Self Rotation")]
    [Export] public bool conicalPitch = true;
    [Export(PropertyHint.Range, "-180,180")]
    public float DirectionChangeAmount = 0;
    // DirectionAxis (0,1,0) swings the direction sideways (positive = left); (1,0,0) tilts it up/down (positive = up).
    [Export] public Vector3 DirectionAxis = Vector3.Up; // Axis around which the direction can change
    [Export(PropertyHint.Range, "-180,180")]
    public float DirectionChangeAmount2 = 0;
    [Export] public Vector3 DirectionAxis2 = Vector3.Left; // Axis around which the direction can change
    public BoneConstraint() { }

    public bool Valid(Vector3 localPos)
    {
        if (localPos == Vector3.Zero)
        {
            return false;
        }

        Vector3 dir = localPos.Normalized();

        if (DirectionAxis.LengthSquared() > 0.0001f && DirectionChangeAmount != 0)
        {
            Basis rotation = new Basis(DirectionAxis.Normalized(), Mathf.DegToRad(DirectionChangeAmount));
            dir = rotation * dir;
            if (DirectionAxis2.LengthSquared() > 0.0001f && DirectionChangeAmount2 != 0)
            {
                Basis rotation2 = new Basis(DirectionAxis2.Normalized(), Mathf.DegToRad(DirectionChangeAmount2));
                dir = rotation2 * dir;
            }
        }

        float yaw = Mathf.RadToDeg(Mathf.Atan2(dir.X, dir.Z));
        Vector2 yzProjection = new Vector2(dir.Z, dir.Y).Normalized();
        float pitch;
        if (conicalPitch)
        {
            pitch = Mathf.RadToDeg(Mathf.Atan2(yzProjection.Y, yzProjection.X));
        }
        else
        {
            pitch = Mathf.RadToDeg(Mathf.Asin(dir.Y));
        }

        bool inYaw = yaw >= MinYaw && yaw <= MaxYaw;
        bool inPitch = pitch >= MinPitch && pitch <= MaxPitch;

        if (inYaw && inPitch)
        {
            return true;
        }

        return false;
    }
}
