using Godot;

// IK = inverse kinematics: given where a hand or foot should end up, work out the joint rotations that
// put it there. Each solver in this folder is one way of doing that.
public abstract partial class IKSolverBase : Node
{
    [Export] public bool Enabled = true; // IKSolverCaller skips this solver when false (e.g. arm IK handed to momentum control)
    public abstract void IKProcess(float delta);
}
