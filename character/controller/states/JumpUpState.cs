using Godot;
public class JumpUpState : AerialState
{
    public JumpUpState(PlayerController controller) : base(controller) { }
    public int jumpsPressed = 0;
    public override void Enter()
    {
        controller.Velocity = new Vector3(controller.aerialCalcVelocity.X/60, (controller.jumpVelocity * jumpsPressed)/60, controller.aerialCalcVelocity.Z/60);
    }

    public override void Update(float delta)
    {
        if(controller.Velocity.Y < 0)
        {
            controller.TransitionTo(new FallState(controller));
            return;
        }
    }

    public override void PhysicsUpdate(float delta)
    {
        controller.Velocity = new Vector3(
                                controller.aerialCalcVelocity.X * delta,
                                Mathf.Clamp(controller.Velocity.Y - (controller.Gravity * delta * controller.jumpFloatGravityMultiplier),-controller.MaxFallSpeed,controller.MaxFallSpeed),
                                controller.aerialCalcVelocity.Z * delta);
        controller.MoveAndSlide();
    }

}
