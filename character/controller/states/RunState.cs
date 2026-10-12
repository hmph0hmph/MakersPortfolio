using Godot;
public class RunState : GroundedState
{
    public RunState(PlayerController controller) : base(controller) { }

    public override void Enter()
    {
        controller.isRunning = true;
    }

    public override void Update(float delta)
    {
        base.Update(delta);
        if (controller.MoveInput == Vector2.Zero)
        {
            controller.TransitionTo(new BrakeState(controller));
            return;
        }
        if (controller.RunVar < controller.RunThreshold)
        {
            controller.TransitionTo(new WalkState(controller));
            return;
        }
    }
    public override void PhysicsUpdate(float delta)
    {
        controller.ApplyMovement(delta, controller.RunVar, controller.lastMoveDirWithInput);
        base.PhysicsUpdate(delta);
    }
    public override void Exit()
    {
        controller.isRunning = false; // Reset running state
    }
}
