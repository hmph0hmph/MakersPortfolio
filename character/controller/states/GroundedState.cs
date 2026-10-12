using Godot;
public abstract class GroundedState : CharacterState
{
    public GroundedState(PlayerController controller) : base(controller) { }

    public override void Enter()
    {
        controller.Velocity = new Vector3(controller.Velocity.X, 0, controller.Velocity.Z);
    }
    public override void Update(float delta)
    {
        // Shared checks for all ground states. The jump (-> PreJumpState) and dash (-> DashState) entries are
        // currently disabled.
    }
    public override void PhysicsUpdate(float delta)
    {
        // Gravity is applied centrally in PlayerController._PhysicsProcess (world Y, after the animation's
        // horizontal velocity is rebuilt from localVelocity). Applying it here writes controller.Velocity.Y,
        // which that rebuild would immediately overwrite -- so it lives in one place instead.
        controller.DecayRunVar(delta);
    }
}
