using Godot;

public partial class IKSolverSinglePosition : IKSolverBase
{
    [Export] public Skeleton3D Skeleton;
    [Export] public Node3D target;
    [Export] public string bone = "bone";
    Vector3 basePoint = new Vector3(0, 0, 0);
    int boneId;

    private Vector3 originalTargetPosition;
    public override void _Ready()
    {
        if (target == null || Skeleton == null)
        {
            GD.PrintErr("Target or skeleton is not set.");
            return;
        }
        originalTargetPosition = target.GlobalPosition;
        boneId = Skeleton.FindBone(bone);
        if (boneId == -1)
        {
            GD.PrintErr($"Bone '{bone}' not found!");
            return;
        }
        basePoint = Skeleton.GetBonePose(boneId).Origin;
    }

    public override void IKProcess(float delta)
    {
        Vector3 currentTargetPos = target.GlobalPosition;
        Vector3 targetChange = currentTargetPos - originalTargetPosition;
        Vector3 newBonePosition = basePoint + targetChange;
        Skeleton.SetBonePosePosition(boneId, newBonePosition);
    }
    public void SetTargetPosition(Vector3 targetPosition)
	{
		target.GlobalPosition = targetPosition;
	}
}
