using Godot;
public class DashState : GroundedState
{
    private float dashTimer = 0f;
    private float dashDuration = 0.1f;

    public DashState(PlayerController controller) : base(controller) { }

    public override void Enter()
    {
        dashDuration = controller.dashDuration; // Get the dash duration from the controller
        dashTimer = dashDuration;
        controller.isRunning = true; // Set running state to true during dash
        controller.isDashing = true; // Set dashing state to true
    }

    public override void Update(float delta)
    {
        base.Update(delta);
        controller.GetCameraRelativeDirection(controller.MoveInput, true);
        if (controller.HasMoveInput())
        {
            dashDuration = controller.dashDuration;
        }
        else
        {
            dashDuration = controller.dashDuration * 2; // Hold the dash twice as long when there is no input
        }
        if (dashTimer >= dashDuration)
        {
            controller.TransitionTo(new RunState(controller));
            return;
        }
    }
    public override void PhysicsUpdate(float delta)
    {
        controller.ApplyMovement(delta, controller.RunVar, controller.lastMoveDirWithInput);
        base.PhysicsUpdate(delta);
        dashTimer += delta;
    }

    public override void Exit()
    {
        controller.isDashing = false;
        controller.isRunning = false; // Reset running state after dash
    }
}
